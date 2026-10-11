"""Composite the three rendered docking views into one annotated GIF (and an MP4 if ffmpeg is available).

Run: $PY scripts/make_gif.py [--frames build/anim_rtx] [--out docs/images/docking.gif] [--width 1440] [--gif-width 960]
     [--fps 12]

Each output frame, top to bottom:
  header     the docking phase, a one-line explanation of what the mechanism is doing, sim time and playback speed
  timeline   the nine phases of the sequence, done / current (with progress) / to come
  panels     wide view, petal/latch/striker close-up and hook station, each with a caption chip
  telemetry  stat tiles from the recorded Newton state of that frame: HCS gap (with its log-scale trace), lateral and
             angular misalignment, soft-capture ring load, capture latches, hook swing/pull and the total hook preload
             against the IDSS range
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "newton", "utils"))
from rtx import caption  # noqa: E402  (the small phase / sim-time chip, shared with the RTX viewer video)

VIEWS = [("wide", "Approach and overview", ""), ("capture", "Petal, capture latch and striker", "x-ray"),
         ("hooks", "Hard-capture hook station", "x-ray")]
PHASES = [("approach", "Approach"), ("contact", "Contact"), ("captured", "Captured"), ("attenuate", "Attenuate"),
          ("align", "Align"), ("retract", "Retract"), ("hook gang 1", "Hook gang 1"), ("hook gang 2", "Hook gang 2"),
          ("docked", "Docked")]
EXPLAIN = {
    "approach": "Vehicle closes at 8 cm/s with a 40 mm offset and 2° misalignment; NDS ring extended to ready-to-capture",
    "contact": "Guide petals slide over the IDA petals and strikers; the compliant ring aligns itself and is pressed home",
    "captured": "All three capture latches have snapped past the 2 mm striker steps: soft capture",
    "attenuate": "Soft, heavily damped actuators absorb the residual relative motion",
    "align": "Hexapod nulls the lateral and angular offsets, swinging the 13 t vehicle into line",
    "retract": "Ring retracts and reels the vehicle in; guide pins seat in their receptacles",
    "hook gang 1": "Hook gang 1 swings over the passive hooks, then pulls down to preload the Belleville stacks",
    "hook gang 2": "Hook gang 2 closes and preloads: the interface is clamped and the seals compressed",
    "docked": "Hard docked: 12 hooks preloaded, interface sealed",
}
PRELOAD_BAND = (12 * 31e3, 12 * 44e3)   # IDSS total hook preload range, 12 hooks at 31-44 kN [N]
GAP_RANGE = (1.0, 700.0)   # HCS gap trace, log scale [mm]

# colour tokens (dark surface; see the dataviz palette): text never wears the accent or status colours
SURFACE = (17, 17, 16)       # page
SURFACE_2 = (30, 30, 28)     # tiles, timeline track
TRACK = (56, 56, 53)         # unfilled meter / future phases
TEXT_1 = (240, 239, 236)     # primary ink
TEXT_2 = (195, 194, 183)     # secondary ink
TEXT_3 = (138, 137, 130)     # muted ink (labels, units)
ACCENT = (57, 135, 229)      # blue: progress, current value
ACCENT_SOFT = (35, 70, 115)  # lighter step of the accent ramp: done phases, target band
GOOD = (12, 163, 12)         # status good: latches locked, preload in range (always with a label)


# colours every GIF frame's palette keeps: the UI tokens (status and accent dots cover too few pixels to survive an
# adaptive palette) and the handrail yellow of the renders
RESERVED = [SURFACE, SURFACE_2, TRACK, TEXT_1, TEXT_2, TEXT_3, ACCENT, ACCENT_SOFT, GOOD,
            (174, 143, 42), (180, 149, 47), (186, 156, 51), (120, 98, 30)]


def to_palette(im, colors=192):
    """Adaptive palette for the frame, with the RESERVED colours forced in."""
    adaptive = im.quantize(colors=colors - len(RESERVED), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = adaptive.getpalette()[:3 * (colors - len(RESERVED))]
    pal += [c for rgb in RESERVED for c in rgb]
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(pal + [0] * (768 - len(pal)))
    return im.quantize(palette=pal_img, dither=Image.Dither.FLOYDSTEINBERG)


def font(size, weight="Regular"):
    for p in (f"/usr/share/fonts/opentype/inter/Inter-{weight}.otf",
              f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if weight != 'Regular' else ''}.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def text(d, xy, s, f, fill, anchor="la"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor, features=["tnum"])


def phase_spans(meta):
    """First and last frame of every phase, for the timeline progress."""
    spans = {}
    for k, m in enumerate(meta):
        a, _ = spans.get(m["phase"], (k, k))
        spans[m["phase"]] = (a, k)
    return spans


def playback_speed(meta, k, fps):
    a, b = max(0, k - 1), min(len(meta) - 1, k + 1)
    return (meta[b]["time"] - meta[a]["time"]) / max(1, b - a) * fps


def caption_chip(img, x, y, title, tag, F, u):
    """Caption on a translucent chip so it stays legible over bright renders."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    pad, gap = int(8 * u), int(8 * u)
    tw = od.textlength(title, font=F["cap"])
    tagw = od.textlength(tag, font=F["tag"]) + 2 * int(6 * u) if tag else 0
    h = F["cap"].size + 2 * int(6 * u)
    od.rounded_rectangle([x, y, x + pad + tw + (gap + tagw if tag else 0) + pad, y + h], radius=int(6 * u),
                         fill=(12, 12, 11, 190))
    img.alpha_composite(overlay) if img.mode == "RGBA" else img.paste(Image.alpha_composite(img.convert("RGBA"),
                                                                                            overlay).convert("RGB"))
    d = ImageDraw.Draw(img)
    text(d, (x + pad, y + h // 2), title, F["cap"], TEXT_1, "lm")
    if tag:
        tx = x + pad + tw + gap
        d.rounded_rectangle([tx, y + int(5 * u), tx + tagw, y + h - int(5 * u)], radius=int(4 * u), outline=TEXT_3,
                            width=1)
        text(d, (tx + tagw / 2, y + h // 2), tag, F["tag"], TEXT_2, "mm")


def header(d, meta, k, fps, W, h, F, u):
    m = meta[k]
    names = [p for p, _ in PHASES]
    i = names.index(m["phase"]) if m["phase"] in names else 0
    x0, xr = int(16 * u), W - int(16 * u)
    y1, y2 = int(10 * u), int(24 * u)
    y3 = y2 + F["title"].size + int(6 * u)
    text(d, (x0, y1), f"Step {i + 1} of {len(PHASES)}", F["label"], TEXT_3)
    text(d, (x0, y2), dict(PHASES).get(m["phase"], m["phase"]), F["title"], TEXT_1)
    text(d, (x0, y3), EXPLAIN.get(m["phase"], ""), F["body"], TEXT_2)
    text(d, (xr, y1), f"Sim time · {playback_speed(meta, k, fps):.1f}× speed", F["label"], TEXT_3, "ra")
    text(d, (xr, y2), f"{m['time']:.1f} s", F["title"], TEXT_1, "ra")


def timeline(d, meta, k, spans, y, W, F, u):
    x0, x1 = int(16 * u), W - int(16 * u)
    gap = int(4 * u)
    n = len(PHASES)
    seg = (x1 - x0 - gap * (n - 1)) / n
    names = [p for p, _ in PHASES]
    cur = names.index(meta[k]["phase"]) if meta[k]["phase"] in names else 0
    bar_h = max(2, int(5 * u))
    for j, (p, label) in enumerate(PHASES):
        a = x0 + j * (seg + gap)
        b = a + seg
        d.rounded_rectangle([a, y, b, y + bar_h], radius=bar_h // 2, fill=ACCENT_SOFT if j < cur else TRACK)
        if j == cur:
            s0, s1 = spans.get(p, (k, k))
            frac = 1.0 if s1 <= s0 else min(1.0, (k - s0 + 1) / (s1 - s0 + 1))
            d.rounded_rectangle([a, y, a + max(bar_h, seg * frac), y + bar_h], radius=bar_h // 2, fill=ACCENT)
        f = F["tl_cur"] if j == cur else F["tl"]
        text(d, (a, y + bar_h + int(5 * u)), label, f, TEXT_1 if j == cur else (TEXT_2 if j < cur else TEXT_3))


def tile(d, x, y, w, h, label, u, F):
    d.rounded_rectangle([x, y, x + w, y + h], radius=int(8 * u), fill=SURFACE_2)
    text(d, (x + int(12 * u), y + int(10 * u)), label, F["label"], TEXT_3)


def value(d, x, y, v, unit, F, u):
    text(d, (x, y), v, F["value"], TEXT_1, "ls")
    vw = d.textlength(v, font=F["value"], features=["tnum"])
    text(d, (x + vw + int(4 * u), y), unit, F["unit"], TEXT_3, "ls")


def telemetry(d, meta, k, y, W, h, F, u):
    m = meta[k]
    x0, gap = int(16 * u), int(8 * u)
    widths = [1.6, 1.0, 0.9, 1.0, 1.0, 1.6]   # relative tile widths
    total = W - 2 * x0 - gap * (len(widths) - 1)
    ws = [total * r / sum(widths) for r in widths]
    xs = [x0 + sum(ws[:i]) + gap * i for i in range(len(ws))]
    base = y + int(10 * u) + F["label"].size + int(8 * u) + F["value"].size
    pad = int(12 * u)

    # HCS gap, with its log-scale trace over the whole run (de-emphasised history, current point in the accent)
    x, w = xs[0], ws[0]
    tile(d, x, y, w, h, "HCS gap", u, F)
    gap_mm = m["gap"] * 1000.0
    value(d, x + pad, base, f"{gap_mm:.1f}" if gap_mm < 100 else f"{gap_mm:.0f}", "mm", F, u)
    sx0, sx1 = x + w * 0.52, x + w - pad
    sy0, sy1 = y + int(14 * u), y + h - int(14 * u)
    lo, hi = (math.log10(v) for v in GAP_RANGE)

    def gy(g):
        return sy1 - (math.log10(min(max(g * 1000.0, GAP_RANGE[0]), GAP_RANGE[1])) - lo) / (hi - lo) * (sy1 - sy0)
    n = len(meta)
    pts = [(sx0 + (sx1 - sx0) * i / (n - 1), gy(meta[i]["gap"])) for i in range(k + 1)]
    for ref in (1.0, 10.0, 100.0):   # recessive gridlines at 1, 10, 100 mm
        yy = gy(ref / 1000.0)
        d.line([(sx0, yy), (sx1, yy)], fill=(44, 44, 41), width=1)
        text(d, (sx0 - int(4 * u), yy), f"{ref:g}", F["tick"], TEXT_3, "rm")
    if len(pts) > 1:
        d.line(pts, fill=TEXT_3, width=max(1, int(2 * u)), joint="curve")
    r = int(3.5 * u)
    d.ellipse([pts[-1][0] - r, pts[-1][1] - r, pts[-1][0] + r, pts[-1][1] + r], fill=ACCENT)

    # misalignment
    x, w = xs[1], ws[1]
    tile(d, x, y, w, h, "Misalignment", u, F)
    value(d, x + pad, base, f"{m.get('lateral', 0) * 1000:.1f}", "mm", F, u)
    text(d, (x + pad, base + int(16 * u)), f"{m.get('tilt', 0):.2f}° tilt", F["unit"], TEXT_2, "ls")

    # ring load
    x, w = xs[2], ws[2]
    tile(d, x, y, w, h, "Ring load", u, F)
    value(d, x + pad, base, f"{abs(m.get('ring_load', 0)) / 1000:.2f}", "kN", F, u)

    # capture latches: three dots plus a label, never colour alone
    x, w = xs[3], ws[3]
    tile(d, x, y, w, h, "Capture latches", u, F)
    locked = int(m.get("latched", 0))
    r, cy = int(6 * u), base - F["value"].size * 0.35
    for j in range(3):
        cx = x + pad + r + j * (2 * r + int(4 * u))
        if j < locked:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=GOOD)
        else:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=TEXT_3, width=max(1, int(2 * u)))
    text(d, (x + pad + 6 * r + int(8 * u) + int(8 * u), cy), "Locked" if locked == 3 else f"{locked} of 3",
         F["unit_b"], TEXT_1 if locked == 3 else TEXT_2, "lm")

    # hooks
    x, w = xs[4], ws[4]
    tile(d, x, y, w, h, "Hook swing · pull", u, F)
    value(d, x + pad, base, f"{math.degrees(m.get('hook_swing', 0)):.1f}°", "", F, u)
    text(d, (x + pad, base + int(16 * u)), f"{m.get('hook_pull', 0) * 1000:.1f} mm pull-down", F["unit"], TEXT_2, "ls")

    # total hook preload, as a meter against the IDSS range
    x, w = xs[5], ws[5]
    tile(d, x, y, w, h, "Hook preload", u, F)
    pre = m["preload"]
    value(d, x + pad, base, f"{pre / 1000:.0f}", "kN", F, u)
    in_band = PRELOAD_BAND[0] <= pre <= PRELOAD_BAND[1]
    if in_band:
        tx = x + w - pad
        text(d, (tx, y + int(10 * u)), "In IDSS range", F["label_b"], TEXT_1, "ra")
        lw = d.textlength("In IDSS range", font=F["label_b"])
        rr = int(4 * u)
        cy = y + int(10 * u) + F["label_b"].size * 0.55
        d.ellipse([tx - lw - int(8 * u) - 2 * rr, cy - rr, tx - lw - int(8 * u), cy + rr], fill=GOOD)
    mx0, mx1 = x + pad, x + w - pad
    my = base + int(8 * u)
    mh = int(6 * u)
    full = PRELOAD_BAND[1] * 1.1

    def mxp(v):
        return mx0 + (mx1 - mx0) * min(1.0, v / full)
    d.rounded_rectangle([mx0, my, mx1, my + mh], radius=mh // 2, fill=TRACK)
    d.rectangle([mxp(PRELOAD_BAND[0]), my, mxp(PRELOAD_BAND[1]), my + mh], fill=ACCENT_SOFT)
    if pre > 0:
        d.rounded_rectangle([mx0, my, max(mx0 + mh, mxp(pre)), my + mh], radius=mh // 2, fill=ACCENT)
    text(d, (mxp(PRELOAD_BAND[0]), my + mh + int(4 * u)), f"{PRELOAD_BAND[0] / 1000:.0f}", F["tick"], TEXT_3, "ma")
    text(d, (mxp(PRELOAD_BAND[1]), my + mh + int(4 * u)), f"{PRELOAD_BAND[1] / 1000:.0f}", F["tick"], TEXT_3, "ma")


def compose(frames_dir, meta, k, W, u, F, spans, fps, plain=False):
    gap = max(2, round(2 * u))   # surface gap between panels
    panel_w = (W - gap * (len(VIEWS) - 1)) // len(VIEWS)
    panel_h = panel_w * 3 // 4
    W = panel_w * len(VIEWS) + gap * (len(VIEWS) - 1)
    head_h, tl_h, tel_h = (0, 0, 0) if plain else (int(74 * u), int(34 * u), int(96 * u))
    tile_h = tel_h - int(20 * u)   # 10 px above and below the tile row
    H = head_h + tl_h + panel_h + tel_h
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    if not plain:
        header(d, meta, k, fps, W, head_h, F, u)
        timeline(d, meta, k, spans, head_h, W, F, u)
    py = head_h + tl_h
    for j, (view, title, tag) in enumerate(VIEWS):
        p = Image.open(os.path.join(frames_dir, view, f"{k:04d}.png")).convert("RGB").resize((panel_w, panel_h),
                                                                                             Image.LANCZOS)
        px = j * (panel_w + gap)
        img.paste(p, (px, py))
        if not plain:
            caption_chip(img, px + int(10 * u), py + int(10 * u), title, tag, F, u)
    if not plain:
        telemetry(ImageDraw.Draw(img), meta, k, py + panel_h + int(10 * u), W, tile_h, F, u)
    else:
        img = caption(img, meta[k]["phase"], meta[k]["time"], scale=0.8 * u)
    return img


def compose_single(frames_dir, meta, k, W, view):
    """One view at full width with the small phase / sim-time chip. If <view>_xray was rendered too, the two are
    crossfaded by the frame's x-ray amount (frames.json "xray"), so the outer structure fades in and out."""
    p = Image.open(os.path.join(frames_dir, view, f"{k:04d}.png")).convert("RGB")
    xray = os.path.join(frames_dir, f"{view}_xray", f"{k:04d}.png")
    w = float(meta[k].get("xray", 0.0))
    if w > 0.0 and os.path.exists(xray):
        p = Image.blend(p, Image.open(xray).convert("RGB"), w)
    p = p.resize((W, round(p.height * W / p.width)), Image.LANCZOS)
    return caption(p, meta[k]["phase"], meta[k]["time"], scale=W / 1280.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="build/anim_rtx")
    ap.add_argument("--out", default="docs/images/docking.gif")
    ap.add_argument("--width", type=int, default=1920, help="Composite (and MP4) width; the layout is drawn for a "
                    "960 px display and scaled, so 1920 is 2x.")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--gif-width", type=int, default=960,
                    help="GIF width (README images display at ~900 px; the MP4 keeps the full --width).")
    ap.add_argument("--plain", action="store_true",
                    help="Panels with just the phase / sim-time chip: no header, timeline, captions or telemetry.")
    ap.add_argument("--single", default=None, help="Just this view (e.g. hero) at full width, with the phase chip.")
    ap.add_argument("--only", type=int, default=None, help="Write just this composite frame (PNG) and stop.")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.frames, "frames.json")))
    u = a.width / 960.0   # type sizes below are px on a 960 px wide display (GitHub shows README images at ~880)
    F = {"title": font(round(21 * u), "SemiBold"), "body": font(round(13 * u)), "cap": font(round(12 * u), "Medium"),
         "tag": font(round(10 * u), "Medium"), "label": font(round(11 * u), "Medium"),
         "label_b": font(round(11 * u), "SemiBold"), "value": font(round(21 * u), "SemiBold"),
         "unit": font(round(12 * u)), "unit_b": font(round(12 * u), "SemiBold"), "tick": font(round(9.5 * u)),
         "tl": font(round(10.5 * u)), "tl_cur": font(round(10.5 * u), "SemiBold")}
    spans = phase_spans(meta)
    comp = os.path.join(a.frames, f"composite_{a.single}" if a.single else
                        ("composite_plain" if a.plain else "composite"))

    def frame(k):
        if a.single:
            return compose_single(a.frames, meta, k, a.width, a.single)
        return compose(a.frames, meta, k, a.width, u, F, spans, a.fps, a.plain)
    os.makedirs(comp, exist_ok=True)
    if a.only is not None:
        out = os.path.join(comp, f"{a.only:04d}.png")
        frame(a.only).save(out)
        print(f"wrote {out}")
        return
    imgs = []
    for k in range(len(meta)):
        im = frame(k)
        im.save(os.path.join(comp, f"{k:04d}.png"))
        imgs.append(im)
    imgs += [imgs[-1]] * a.fps * 2   # hold the docked state
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    gw = min(a.gif_width, imgs[0].width)
    gh = round(imgs[0].height * gw / imgs[0].width)
    pal = [to_palette(im if gw == im.width else im.resize((gw, gh), Image.LANCZOS)) for im in imgs]
    pal[0].save(a.out, save_all=True, append_images=pal[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
    print(f"wrote {a.out} ({len(imgs)} frames, {os.path.getsize(a.out) / 1e6:.1f} MB)")
    if shutil.which("ffmpeg"):
        mp4 = os.path.splitext(a.out)[0] + ".mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i",
                        os.path.join(comp, "%04d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                        "-c:v", "libx264", "-crf", "20", mp4], check=False)
        if os.path.exists(mp4):
            print(f"wrote {mp4}")


if __name__ == "__main__":
    main()
