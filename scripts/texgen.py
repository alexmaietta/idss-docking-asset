"""Procedural, tileable PBR texture sets (numpy only; run with the venv Python, not Blender).

Every set is periodic, so it tiles seamlessly, and is written to
<out>/<set>/<set>_<map>.png with these maps:
- basecolor (sRGB)
- roughness, metallic (linear greyscale)
- normal (OpenGL, +Y up)
- opacity, only for cut-outs

The meshes carry UVs in metres, and a set's `tile_m` is the real-world size of one tile. Materials
divide by it, so textures land at true scale.

Asset configs declare which sets they need as (generator name, kwargs); see `generate()`.
"""
import json
import math
import os
import struct
import zlib

import numpy as np

# ---------------------------------------------------------------- noise primitives


def _rng(seed):
    return np.random.default_rng(seed)


def pnoise(n, feature_px, seed, octaves=4, persistence=0.5):
    """Periodic fractal noise in [0, 1] via FFT band-limited white noise."""
    rng = _rng(seed)
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    out = np.zeros((n, n))
    amp, fp, total = 1.0, feature_px, 0.0
    for _ in range(octaves):
        w = rng.standard_normal((n, n))
        f0 = 1.0 / max(fp, 1.0)
        band = np.exp(-((f - f0) ** 2) / (2 * (0.6 * f0) ** 2))
        layer = np.real(np.fft.ifft2(np.fft.fft2(w) * band))
        layer /= layer.std() + 1e-9
        out += amp * layer
        total += amp
        amp *= persistence
        fp /= 2.0
    out /= total
    return np.clip(0.5 + out * 0.18, 0, 1)


def streaks(n, seed, along=1, length_px=900, width_px=1.2):
    """Periodic anisotropic streaks (brushing / rolling marks) along axis `along` (1 = U)."""
    rng = _rng(seed)
    w = rng.standard_normal((n, n))
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    fa, fc = (fx, fy) if along == 1 else (fy, fx)
    band = np.exp(-(fa * length_px) ** 2) * np.exp(-(np.abs(fc) * width_px) ** 2 / 50)
    out = np.real(np.fft.ifft2(np.fft.fft2(w) * band))
    out /= out.std() + 1e-9
    return np.clip(0.5 + out * 0.2, 0, 1)


def scratches(n, seed, count=220, max_len=0.35, depth=1.0):
    """Periodic fine scratches as a height map (0 = surface, negative = groove)."""
    rng = _rng(seed)
    h = np.zeros((n, n))
    for _ in range(count):
        x0, y0 = rng.random(2) * n
        ang = rng.normal(0.3, 0.5)
        ln = rng.random() ** 2 * max_len * n
        steps = int(ln)
        if steps < 2:
            continue
        t = np.linspace(0, ln, steps)
        xs = (x0 + np.cos(ang) * t).astype(int) % n
        ys = (y0 + np.sin(ang) * t + np.sin(t / 60) * 3).astype(int) % n
        h[ys, xs] -= depth * (0.3 + 0.7 * rng.random())
    k = np.fft.fft2(np.pad(np.ones((2, 2)), ((0, n - 2), (0, n - 2))))
    return np.real(np.fft.ifft2(np.fft.fft2(h) * k)) / 4


def _upsample(a, n):
    """Periodic (wrap-around) bilinear upsampling of a square field to n x n."""
    from scipy.ndimage import zoom
    if a.shape[0] == n:
        return a
    return zoom(a, n / a.shape[0], order=1, mode="grid-wrap")


def crumple(n, seed, folds=90, kmax=7, sharp=1.0, warp_px=40.0):
    """See _crumple; computed at <= 1024 px and upsampled (the folds are low-frequency)."""
    m = min(n, 1024)
    return _upsample(_crumple(m, seed, folds, kmax, sharp, warp_px * m / n), n)


def _crumple(n, seed, folds=90, kmax=7, sharp=1.0, warp_px=40.0):
    """Tileable crumpled-foil height field: piecewise-planar facets separated by sharp creases.

    It is a sum of tent waves |fract(k·p + phi) - 0.5| with integer wave vectors k, so it is
    periodic. Each fold's amplitude is modulated by smooth noise (folds die out locally), and the
    domain is warped slightly so the creases are not dead straight.
    """
    rng = _rng(seed)
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    wx = (pnoise(n, warp_px * 4, seed + 1, octaves=2) - 0.5) * warp_px / n
    wy = (pnoise(n, warp_px * 4, seed + 2, octaves=2) - 0.5) * warp_px / n
    x, y = x + wx, y + wy
    h = np.zeros((n, n))
    for i in range(folds):
        while True:
            kx, ky = rng.integers(-kmax, kmax + 1, 2)
            if kx or ky:
                break
        kl = math.hypot(kx, ky)
        phase = rng.random()
        t = np.abs(np.mod(kx * x + ky * y + phase, 1.0) - 0.5)
        env = pnoise(n, n / (1.5 + rng.random() * 2), seed + 10 + i, octaves=1)
        env = np.clip((env - 0.42) * 6, 0, 1)
        h += t * env * (1.0 / kl) ** sharp
    return (h - h.min()) / (np.ptp(h) + 1e-9)


def facets(n, seed, cells, tilt_deg, stretch=1.0):
    """Tileable faceted-normal field: periodic Voronoi cells, each a flat facet with a random tilt.

    Returns (normal xyz array in [-1, 1], cell id, distance-to-crease in px). `stretch` > 1
    elongates the facets along U, as MLI wrinkles that run along the wrap direction do.
    """
    from scipy.spatial import cKDTree
    rng = _rng(seed)
    pts = rng.random((cells, 2)) * n
    tree = cKDTree(pts * [1 / stretch, 1], boxsize=[n / stretch, n])
    cid = np.empty((n, n), np.int32)
    edge = np.empty((n, n), np.float32)
    xs = (np.arange(n) + 0.5) / stretch % (n / stretch)
    rows = max(1, (1 << 20) // n)                    # query ~1M pixels at a time (memory)
    for r0 in range(0, n, rows):
        r1 = min(n, r0 + rows)
        yy, xx = np.meshgrid(np.arange(r0, r1) + 0.5, xs, indexing="ij")
        d, idx = tree.query(np.stack([xx.ravel(), yy.ravel() % n], 1), k=2)
        cid[r0:r1] = idx[:, 0].reshape(r1 - r0, n)
        edge[r0:r1] = ((d[:, 1] - d[:, 0]) * 0.5).reshape(r1 - r0, n)
    t = np.radians(tilt_deg) * np.sqrt(rng.random(cells))
    a = rng.random(cells) * 2 * math.pi
    nx, ny = np.sin(t) * np.cos(a), np.sin(t) * np.sin(a)
    nz = np.sqrt(1 - nx ** 2 - ny ** 2)
    nrm = np.stack([nx[cid], ny[cid], nz[cid]], -1).astype(np.float32)
    return nrm, cid, edge


def _encode_normal(nrm):
    nrm = nrm / np.linalg.norm(nrm, axis=-1, keepdims=True)
    return nrm * 0.5 + 0.5


def normal_from_height(h, strength):
    """Tangent-space OpenGL normal map from a periodic height field (strength in pixel units)."""
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * strength
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * strength
    nx, ny, nz = -dx, dy, np.ones_like(h)   # image rows go down, OpenGL +Y up
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / ln, ny / ln, nz / ln], -1) * 0.5 + 0.5


def tint(var, colour, amount):
    """Base colour = linear colour modulated by a [0, 1] variation field."""
    c = np.array(colour)[None, None, :]
    return c * (1 + (var[..., None] - 0.5) * 2 * amount)


def _grid_lines(n, cells, width_px, offset=0.0):
    """Distance-based anti-aliased grid line mask (1 on the lines), `cells` lines per tile per axis."""
    y, x = np.mgrid[0:n, 0:n] + 0.5
    p = n / cells
    dx = np.abs(np.mod(x / p + offset, 1.0) - 0.5) * p
    dy = np.abs(np.mod(y / p + offset, 1.0) - 0.5) * p
    d = np.minimum(p / 2 - dx, p / 2 - dy)
    return np.clip(width_px / 2 - d + 0.5, 0, 1), x, y


def _dots(n, pitch_px, r_px, offset=(0.5, 0.5), stagger=False):
    """Round dot mask on a square lattice (fastener heads, perforations); returns (mask, dome height)."""
    y, x = np.mgrid[0:n, 0:n] + 0.5
    u = x / pitch_px + offset[0]
    v = y / pitch_px + offset[1]
    if stagger:
        u = u + 0.5 * (np.floor(v) % 2)
    du = (np.mod(u, 1.0) - 0.5) * pitch_px
    dv = (np.mod(v, 1.0) - 0.5) * pitch_px
    d = np.sqrt(du * du + dv * dv)
    mask = np.clip(r_px - d + 0.5, 0, 1)
    dome = np.sqrt(np.clip(1 - (d / r_px) ** 2, 0, 1))
    return mask, dome


# ---------------------------------------------------------------- io

def _to8(a):
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def _srgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def write_png(path, a8):
    """Minimal PNG writer (8-bit greyscale or RGB) using only zlib."""
    h, w = a8.shape[:2]
    ch = 1 if a8.ndim == 2 else a8.shape[2]
    raw = np.concatenate([np.zeros((h, 1), np.uint8), a8.reshape(h, w * ch)], 1).tobytes()

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0 if ch == 1 else 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


def save_set(out, name, tile_m, **maps):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    for k, a in maps.items():
        if np.ndim(a) == 0:
            continue
        a8 = _to8(_srgb(a)) if k == "basecolor" else _to8(a)
        # set-prefixed names: USD exporters flatten every texture into one folder
        write_png(os.path.join(d, f"{name}_{k}.png"), a8)
    meta = {"tile_m": tile_m, "maps": sorted(k for k, a in maps.items() if np.ndim(a) != 0)}
    with open(os.path.join(d, "set.json"), "w") as f:
        json.dump(meta, f)
    return meta


# ---------------------------------------------------------------- material generators
# Colours are linear RGB. Every generator returns the maps dict plus tile_m.

def mli(n, seed, colour, tile_m=1.0, gloss=0.06, tape=True, crinkle=1.0):
    """Multi-layer insulation outer layer (aluminised Kapton / gold, copper or silver foil).

    Large hand-formed folds plus fine crinkle. Most facets are mirror-glossy; there are a few duller
    handled patches and, optionally, strips of matching tape on the blanket seams.
    """
    # facet sizes in metres -> cells per tile: ~12 cm folds, ~3 cm crinkles
    big, cid, e1 = facets(n, seed, int((tile_m / 0.20) ** 2 * 1.3), 14, stretch=1.25)
    fine, _, e2 = facets(n, seed + 1, int((tile_m / 0.045) ** 2 * 1.3 * crinkle), 12, stretch=1.15)
    micro, _, _ = facets(n, seed + 6, int((tile_m / 0.016) ** 2 * 1.3 * crinkle), 7, stretch=1.3)
    puff = pnoise(n, n / 3, seed + 3, octaves=2)
    pn = normal_from_height(puff, 6.0 * n / 2048) * 2 - 1
    folds = crumple(n, seed + 2, folds=60, kmax=7, sharp=0.9, warp_px=25)
    fn = normal_from_height(folds, 14.0 * n / 2048) * 2 - 1
    nrm = big * 0.8 + fine * 0.8 + micro * 0.45 + pn * 0.6 + fn * 1.3
    nrm[..., 2] = np.abs(nrm[..., 2]) + 0.3
    var = pnoise(n, n / 5, seed + 4, octaves=3)
    smudge = pnoise(n, n / 12, seed + 5, octaves=3)
    crease = np.clip(1.5 - e1 / (1.5 * n / 2048), 0, 1)
    fvar = ((np.sin(cid * 12.9898) * 43758.5453) % 1.0)
    base = tint(var * 0.6 + fvar * 0.4, colour, 0.08) * (1 - 0.08 * crease[..., None])
    rough = np.clip(gloss + (smudge > 0.6) * 0.08 + crease * 0.06 + fvar * 0.04, 0.02, 1)
    if tape:   # ~5 cm tape seams, one per tile in each direction, slightly matte and flat
        y, x = np.mgrid[0:n, 0:n] + 0.5
        w = 0.05 / tile_m * n
        seam = ((np.abs(x - n * 0.37) < w / 2) | (np.abs(y - n * 0.71) < w / 2)).astype(float)
        rough = np.clip(rough + seam * 0.10, 0, 1)
        base = base * (1 - 0.06 * seam[..., None])
        nrm = nrm * (1 - 0.75 * seam[..., None]) + np.array([0, 0, 1.0]) * 0.75 * seam[..., None]
    return dict(basecolor=base, roughness=rough, metallic=np.ones((n, n)), normal=_encode_normal(nrm)), tile_m


def beta_cloth(n, seed, colour, tile_m=0.6, rough0=0.75):
    """Teflon-coated beta-glass cloth outer blanket (white MLI): fine weave plus soft sags and wrinkles."""
    y, x = np.mgrid[0:n, 0:n]
    p = n / (tile_m / 0.0012)
    weave = np.abs(np.sin(math.pi * x / p)) * 0.5 + np.abs(np.sin(math.pi * y / p)) * 0.5
    sag = pnoise(n, n / 4, seed, octaves=3)
    wr = crumple(n, seed + 1, folds=50, kmax=8, sharp=1.0)
    h = sag * 2.0 + wr * 1.2 + weave * 0.04
    var = pnoise(n, n / 6, seed + 2)
    base = tint(var, colour, 0.03) * (0.95 + 0.05 * weave[..., None])
    rough = np.clip(rough0 + (var - 0.5) * 0.08, 0, 1)
    return dict(basecolor=base, roughness=rough, metallic=np.zeros((n, n)),
                normal=normal_from_height(h, 4.0 * n / 2048)), tile_m


def panel(n, seed, colour, tile_m=0.5, rough0=0.5, fastener_pitch=0.05, metallic=0.0, seams=1):
    """Painted / anodised honeycomb panel: edge fastener rows on a panel grid, faint orange peel."""
    peel = pnoise(n, 10, seed, octaves=3)
    var = pnoise(n, n / 4, seed + 1)
    grime = pnoise(n, n / 10, seed + 2, octaves=4)
    line, x, y = _grid_lines(n, seams, 2.0 * n / 2048, offset=0.0)
    # fastener rows run 2 cm inside each seam
    p = n / seams
    inset = 0.02 / tile_m * n
    dxs = np.abs(np.mod(x, p) - inset)
    dys = np.abs(np.mod(y, p) - inset)
    pitch = fastener_pitch / tile_m * n
    r = 0.0035 / tile_m * n
    fx = np.sqrt(dxs ** 2 + ((np.mod(y, pitch)) - pitch / 2) ** 2)
    fy = np.sqrt(dys ** 2 + ((np.mod(x, pitch)) - pitch / 2) ** 2)
    fd = np.minimum(fx, fy)
    head = np.clip(r - fd + 0.5, 0, 1)
    dome = np.sqrt(np.clip(1 - (fd / r) ** 2, 0, 1))
    base = tint(var, colour, 0.03) * (1 - 0.06 * (grime[..., None] - 0.5)) * (1 - 0.35 * line[..., None])
    base = base * (1 - head[..., None]) + np.array([0.55, 0.55, 0.56]) * head[..., None]
    rough = np.clip(rough0 + (peel - 0.5) * 0.1 + (var - 0.5) * 0.06 - head * 0.2, 0, 1)
    met = np.clip(metallic + head, 0, 1)
    h = peel * 0.3 - line * 1.5 + dome * 1.2
    return dict(basecolor=base, roughness=rough, metallic=met,
                normal=normal_from_height(h, 2.0 * n / 2048)), tile_m


def brushed(n, seed, colour, tile_m=0.4, rough0=0.3, along=1):
    """Brushed / rolled bare metal (aluminium, steel, titanium)."""
    st = streaks(n, seed, along)
    var = pnoise(n, n / 4, seed + 1)
    sc = scratches(n, seed + 2, count=160)
    base = tint(var, colour, 0.06) * (0.92 + 0.16 * st[..., None])
    rough = np.clip(rough0 + (st - 0.5) * 0.25 + (var - 0.5) * 0.1 - sc * 0.1, 0.05, 1)
    return dict(basecolor=base, roughness=rough, metallic=np.ones((n, n)),
                normal=normal_from_height(st * 0.6 + sc * 0.8, 1.5)), tile_m


def anodised(n, seed, colour, tile_m=0.3, rough0=0.45):
    """Bead-blasted / anodised aluminium: isotropic fine pitting, subtle mottle."""
    pit = pnoise(n, 4.0, seed, octaves=3)
    var = pnoise(n, n / 5, seed + 1)
    base = tint(var, colour, 0.07) * (0.94 + 0.12 * pit[..., None])
    rough = np.clip(rough0 + (pit - 0.5) * 0.3 + (var - 0.5) * 0.1, 0, 1)
    return dict(basecolor=base, roughness=rough, metallic=np.full((n, n), 0.9),
                normal=normal_from_height(pit, 2.0)), tile_m


def perforated(n, seed, colour, tile_m=0.2, hole_d=0.022, pitch=0.04, rough0=0.35):
    """Lightening-hole strut / truss web: round holes on a staggered lattice as an opacity cut-out."""
    pitch_px = pitch / tile_m * n
    hole, dome = _dots(n, pitch_px, hole_d / 2 / tile_m * n, stagger=False)
    st = streaks(n, seed, 1)
    var = pnoise(n, n / 4, seed + 1)
    rim, _ = _dots(n, pitch_px, (hole_d / 2 + 0.002) / tile_m * n)
    base = tint(var, colour, 0.05) * (0.94 + 0.1 * st[..., None]) * (1 - 0.2 * (rim - hole)[..., None])
    rough = np.clip(rough0 + (st - 0.5) * 0.2, 0, 1)
    h = st * 0.4 - (rim - hole) * 1.0
    return dict(basecolor=base, roughness=rough, metallic=np.ones((n, n)),
                normal=normal_from_height(h, 1.5), opacity=1 - hole), tile_m


def solar_cells(n, seed, tile_m=0.32, cell=(0.08, 0.04), gap=0.0015):
    """Triple-junction GaAs cells under cover glass: dark blue-black cells, silver grid fingers and
    bus bars, white substrate gaps. The tile holds an integer number of cells."""
    y, x = np.mgrid[0:n, 0:n] + 0.5
    cw, ch = cell[0] / tile_m * n, cell[1] / tile_m * n
    u, v = np.mod(x, cw), np.mod(y, ch)
    g = gap / tile_m * n
    in_cell = (u > g) & (u < cw - g) & (v > g) & (v < ch - g)
    fingers = (np.abs(np.mod(u, cw / 40) - cw / 80) < 0.35 * n / 2048) & in_cell
    bus = (np.abs(v - ch * 0.08) < 1.5 * n / 2048) & in_cell
    cid = np.floor(x / cw) * 7 + np.floor(y / ch) * 13
    var = (np.sin(cid * 12.9898) * 43758.5453) % 1.0
    cellc = np.array([0.018, 0.022, 0.055]) * (0.85 + 0.3 * var[..., None])
    base = np.where(in_cell[..., None], cellc, np.array([0.65, 0.63, 0.58]))
    base = np.where((fingers | bus)[..., None], np.array([0.55, 0.55, 0.57]), base)
    rough = np.where(in_cell, 0.08, 0.5) + 0.0 * x
    met = np.where(fingers | bus, 1.0, 0.0)
    h = in_cell * 0.5 + (fingers | bus) * 0.3
    return dict(basecolor=base, roughness=rough, metallic=met, normal=normal_from_height(h, 1.0)), tile_m


def osr(n, seed, tile_m=0.2, tile=0.04):
    """Optical solar reflector radiator: grid of second-surface silvered quartz mirror tiles."""
    line, x, y = _grid_lines(n, int(round(tile_m / tile)), 3.0 * n / 2048)
    tid = np.floor(x / (n * tile / tile_m)) * 3 + np.floor(y / (n * tile / tile_m)) * 17
    var = (np.sin(tid * 78.233) * 43758.5453) % 1.0
    base = np.array([0.80, 0.81, 0.82])[None, None, :] * (0.94 + 0.08 * var[..., None]) * (1 - 0.6 * line[..., None])
    rough = 0.03 + 0.04 * var + line * 0.5
    return dict(basecolor=base, roughness=rough, metallic=np.full((n, n), 0.95) - line * 0.6,
                normal=normal_from_height(-line, 2.0)), tile_m


def nozzle_metal(n, seed, colour, tile_m=0.3, rough0=0.4, bands=0):
    """Regeneratively cooled nozzle jacket: heat-tinted superalloy, faint axial rib lines (u = around,
    v = along the bell), mottled oxide."""
    y, x = np.mgrid[0:n, 0:n] + 0.5
    ox = pnoise(n, n / 6, seed, octaves=4)
    st = streaks(n, seed + 1, along=0, length_px=1200)
    ribs = 0.5 + 0.5 * np.cos(2 * math.pi * x / (n / max(bands, 1))) if bands else np.zeros((n, n))
    base = tint(ox, colour, 0.18) * (0.9 + 0.15 * st[..., None])
    rough = np.clip(rough0 + (ox - 0.5) * 0.3, 0, 1)
    return dict(basecolor=base, roughness=rough, metallic=np.full((n, n), 0.95),
                normal=normal_from_height(ribs * 0.4 + st * 0.4, 1.5)), tile_m


def composite(n, seed, colour, tile_m=0.15, rough0=0.35):
    """Carbon-fibre overwrap (COPVs, leg struts): +-45 deg filament bands under clear resin."""
    y, x = np.mgrid[0:n, 0:n] + 0.5
    k = 24
    a = np.sin(2 * math.pi * k * (x + y) / n)
    b = np.sin(2 * math.pi * k * (x - y) / n)
    band = (np.floor(2 * k * x / n) + np.floor(2 * k * y / n)) % 2
    h = np.where(band > 0, a, b) * 0.5 + 0.5
    var = pnoise(n, n / 5, seed)
    base = tint(var, colour, 0.08) * (0.85 + 0.2 * h[..., None])
    rough = np.clip(rough0 + (h - 0.5) * 0.15, 0, 1)
    return dict(basecolor=base, roughness=rough, metallic=np.zeros((n, n)), normal=normal_from_height(h, 1.2)), tile_m


def cloth(n, seed, colour, tile_m=0.1, rough0=0.85, threads=120):
    """Nylon / Nomex protective covers (remove-before-flight covers, GSE bags): plain weave."""
    y, x = np.mgrid[0:n, 0:n]
    p = n / threads
    u, v = x / p, y / p
    warp = np.abs(np.sin(math.pi * u)) * (0.6 + 0.4 * np.cos(math.pi * v) * np.sign(np.sin(math.pi * u)))
    weft = np.abs(np.sin(math.pi * v)) * (0.6 - 0.4 * np.cos(math.pi * u) * np.sign(np.sin(math.pi * v)))
    h = np.maximum(warp, weft)
    var = pnoise(n, n / 5, seed + 1)
    base = tint(var, colour, 0.05) * (0.86 + 0.14 * h[..., None])
    rough = np.clip(rough0 + (0.5 - h) * 0.1, 0, 1)
    return dict(basecolor=base, roughness=rough, metallic=np.zeros((n, n)), normal=normal_from_height(h, 0.9)), tile_m


def regolith(n, seed, colour=(0.13, 0.125, 0.12), tile_m=4.0):
    """Lunar regolith ground: pebbles, micro-craters, fine dust."""
    fine = pnoise(n, 3, seed, octaves=3)
    mid = pnoise(n, 30, seed + 1, octaves=4)
    big = pnoise(n, n / 4, seed + 2, octaves=3)
    peb, dome = _dots(n, n / 37.0, n / 600, stagger=True)
    jitter = pnoise(n, 20, seed + 3) > 0.62
    h = fine * 0.4 + mid * 1.0 + big * 1.5 + dome * jitter * 2.0
    base = tint(big * 0.6 + mid * 0.4, colour, 0.25) * (0.9 + 0.2 * fine[..., None])
    rough = np.clip(0.92 + (fine - 0.5) * 0.1, 0, 1)
    return dict(basecolor=base, roughness=rough, metallic=np.zeros((n, n)), normal=normal_from_height(h, 3.0)), tile_m


def _star_mask(h, w, cx, cy, r_out):
    """Boolean mask of a 5-point star centred at (cx, cy) in pixel coords, point up."""
    r_in = r_out * 0.382
    pts = []
    for k in range(10):
        a = math.pi / 2 + k * math.pi / 5
        r = r_out if k % 2 == 0 else r_in
        pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
    ys, xs = np.mgrid[0:h, 0:w] + 0.5
    inside = np.zeros((h, w), bool)
    for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
        cond = (ay > ys) != (by > ys)
        xint = ax + (ys - ay) * (bx - ax) / np.where(by - ay == 0, 1e-9, by - ay)
        inside ^= cond & (xint > xs)
    return inside


def flag(n, seed, tile_m=1.0):
    """US flag (Executive Order 10834 proportions) printed on a matte film decal. The flag fills the
    whole square image (the decal mesh has 0-1 UVs over a 1.9:1 rectangle, which restores the aspect)."""
    H, W = n, int(n * 1.9)
    red, white, blue = np.array([0.45, 0.02, 0.04]), np.array([0.80, 0.80, 0.78]), np.array([0.03, 0.04, 0.16])
    rows = (np.arange(H) // (H / 13)).astype(int)
    img = np.empty((H, W, 3))
    img[:] = white
    img[rows % 2 == 0] = red
    ch, cw = int(round(H * 7 / 13)), int(round(H * 0.76))
    img[:ch, :cw] = blue
    E = F = 0.054 * H
    G = 0.063 * H
    K = 0.0616 * H
    stars = np.zeros((ch, cw), bool)
    for r in range(9):
        for c in range(6 if r % 2 == 0 else 5):
            stars |= _star_mask(ch, cw, G + (2 * c + (r % 2)) * G, E + r * F, K / 2)
    img[:ch, :cw][stars] = white
    idx = (np.arange(n) * W / n).astype(int)        # squeeze 1.9:1 into the square texture
    img = img[:, idx]
    var = pnoise(n, n / 6, seed)
    base = img * (0.96 + 0.08 * var[..., None])
    rough = np.full((n, n), 0.55) + (var - 0.5) * 0.1
    return dict(basecolor=base, roughness=rough, metallic=np.zeros((n, n)),
                normal=normal_from_height(var, 0.5)), tile_m


def hard_anodize(n, seed, colour, tile_m=0.25, rough0=0.45, metallic=0.0, mottle=0.08):
    """Type III hard anodize / PTFE-impregnated hard anodize: a dark ceramic oxide over aluminium.
    Mostly dielectric (the oxide is ~50 um thick) with fine bead-blast pitting and handling mottle."""
    pit = pnoise(n, 3.0, seed, octaves=3)
    var = pnoise(n, n / 5, seed + 1)
    smudge = pnoise(n, n / 9, seed + 2, octaves=3)
    sc = scratches(n, seed + 3, count=60)
    base = tint(var * 0.7 + smudge * 0.3, colour, mottle) * (0.95 + 0.1 * pit[..., None])
    rough = np.clip(rough0 + (pit - 0.5) * 0.18 + (smudge - 0.5) * 0.12 - sc * 0.15, 0.05, 1)
    met = np.clip(metallic + sc * 0.5, 0, 1)
    return dict(basecolor=base, roughness=rough, metallic=met,
                normal=normal_from_height(pit * 0.8 - sc * 0.6, 1.6)), tile_m


def turned(n, seed, colour, tile_m=0.3, rough0=0.32, metallic=1.0):
    """Lathe-turned / faced aluminium: fine concentric tool marks along U (UVs run round the ring)."""
    y, x = np.mgrid[0:n, 0:n] + 0.5
    marks = 0.5 + 0.5 * np.sin(2 * math.pi * y / (n / (tile_m / 0.00035)))
    st = streaks(n, seed, along=1, length_px=1600, width_px=0.8)
    var = pnoise(n, n / 4, seed + 1)
    sc = scratches(n, seed + 2, count=90)
    base = tint(var, colour, 0.05) * (0.93 + 0.08 * marks[..., None] + 0.06 * st[..., None])
    rough = np.clip(rough0 + (marks - 0.5) * 0.08 + (st - 0.5) * 0.15 - sc * 0.1, 0.04, 1)
    return dict(basecolor=base, roughness=rough, metallic=np.full((n, n), metallic),
                normal=normal_from_height(marks * 0.25 + st * 0.5 + sc * 0.6, 1.2)), tile_m


def rubber(n, seed, colour, tile_m=0.1, rough0=0.62):
    """Moulded silicone / elastomer (seals): soft sheen, fine mould texture, faint dust."""
    fine = pnoise(n, 2.5, seed, octaves=2)
    var = pnoise(n, n / 6, seed + 1)
    base = tint(var, colour, 0.06)
    rough = np.clip(rough0 + (fine - 0.5) * 0.12, 0, 1)
    return dict(basecolor=base, roughness=rough, metallic=np.zeros((n, n)), normal=normal_from_height(fine, 0.6)), tile_m


def painted(n, seed, colour, tile_m=0.4, rough0=0.45):
    """Plain painted / coated part (targets, placards): faint orange peel, no fasteners."""
    peel = pnoise(n, 8, seed, octaves=3)
    var = pnoise(n, n / 5, seed + 1)
    base = tint(var, colour, 0.02)
    rough = np.clip(rough0 + (peel - 0.5) * 0.08, 0, 1)
    return dict(basecolor=base, roughness=rough, metallic=np.zeros((n, n)), normal=normal_from_height(peel, 0.5)), tile_m


GENERATORS = {f.__name__: f for f in (flag, mli, beta_cloth, panel, brushed, anodised, perforated, solar_cells, osr,
                                      nozzle_metal, composite, cloth, regolith, hard_anodize, turned,
                                      rubber, painted)}


def generate(out, sets, res=2048, only=None):
    """sets: {set_name: (generator, kwargs)}. Writes each set plus out/tiles.json. Cached by spec."""
    os.makedirs(out, exist_ok=True)
    tiles = {}
    for i, (name, (gen, kw)) in enumerate(sorted(sets.items())):
        kw = dict(kw)
        n = kw.pop("res", res)
        seed = kw.pop("seed", 1000 + i * 37)
        spec = json.dumps([gen, n, seed, kw], sort_keys=True, default=list)
        stamp = os.path.join(out, name, "spec.json")
        if (only is None or name in only) and not (os.path.exists(stamp) and open(stamp).read() == spec):
            maps, tile_m = GENERATORS[gen](n, seed, **kw)
            save_set(out, name, tile_m, **maps)
            with open(stamp, "w") as f:
                f.write(spec)
            print(f"  {name:24s} {gen:12s} {n}px  tile {tile_m} m")
        with open(os.path.join(out, name, "set.json")) as f:
            tiles[name] = json.load(f)
    with open(os.path.join(out, "tiles.json"), "w") as f:
        json.dump(tiles, f, indent=1)
    return tiles
