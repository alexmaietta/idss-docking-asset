"""International Docking Adapter (IDA-2 / IDA-3) builder: the passive IDSS port on ISS.

NDS IDD JSC 65795 Rev H Fig 3.1.2-1 / 4.2.1-2: the Energia IDA Primary Structure (EIPS = modified APAS
MAPAS + adapter ring EAR + passive hard-mate assembly PHMA), the passive soft-capture system (fixed
capture ring with three truss guide petals and three strikers), retro-reflectors, peripheral docking
targets, heaters, trunnions, MMOD shields, push-off separators and handrails, all under white beta-cloth
MLI. 642 mm from the PMA interface (z = 0) to the IDSS HCS mating plane; 1077 mm overall with petals
and the MAPAS guides; 526 kg.

Appearance from the KSC photos (refs/images/photos/ida): the hard-capture face is bare machined
aluminium; the capture ring and truss petals are dark hard anodize; the strikers bright steel; a
rolled white blanket wraps the flange, the shell blanket carries Velcro tabs, gold handrails and
white target plates with black rings on stand-off posts.

Links: ida_base (root: everything fixed), hook_a_00..11 (active hooks: the IDA can also hook the
visiting vehicle), hook_p_00..11 (passive hooks), separator_1..3 (retractable pushers), sensor_1..3.
"""
import math
import random

from mathutils import Vector

import geo as G
import idss as I
import nds
import params as P

UP = Vector((0, 0, 1))


def blanket_ring(name, link_ob, r_in, r_out, z0, z1, puff=0.004, seed=0, segs=192, mat="mli"):
    """Rolled MLI blanket: rounded-rectangle section swept round the axis, gently puffed."""
    prof = []
    rr = min(r_out - r_in, z1 - z0) * 0.45
    corners = [(r_out - rr, z1 - rr, 0.0), (r_in + rr, z1 - rr, 90.0), (r_in + rr, z0 + rr, 180.0), (r_out - rr, z0 + rr, 270.0)]
    for cx, cz, a0 in corners:
        for i in range(5):
            a = math.radians(a0 + 90.0 * i / 4)
            prof.append((cx + rr * math.cos(a), cz + rr * math.sin(a)))
    v, f, u = I.revolve(prof, segs=segs)
    rnd = random.Random(seed)
    waves = [(rnd.uniform(3, 9), rnd.uniform(0, math.tau), rnd.uniform(4, 16), rnd.uniform(0, math.tau)) for _ in range(5)]
    out = []
    for p in v:
        a = math.atan2(p.y, p.x)
        s = sum(math.sin(k * a + ph) * math.sin(kz * p.z + pz) for k, ph, kz, pz in waves) / 3
        rad = Vector((p.x, p.y, 0)).normalized()
        out.append(p + rad * puff * s)
    return I.part(name, (out, f, u), mat, link_ob, smooth=70.0, bev=0)


def shell_blanket(name, link_ob, r, z0, z1, seed=1):
    """Side MLI blanket over the shell: quilted panels with sags between the stitch lines."""
    segs, rows = 240, 24
    V, F, U = [], [], []
    rnd = random.Random(seed)
    for j in range(rows + 1):
        z = z0 + (z1 - z0) * j / rows
        for i in range(segs):
            a = math.tau * i / segs
            panel = (a * 8 / math.tau) % 1.0
            sag = math.sin(math.pi * panel) * math.sin(math.pi * j / rows) * 0.006
            bump = 0.0025 * math.sin(3 * a + 7 * z + seed) * math.sin(11 * a - 5 * z)
            rr = r + sag + bump
            V.append(Vector((rr * math.cos(a), rr * math.sin(a), z)))
            U.append((r * a, z))
    for j in range(rows):
        for i in range(segs):
            i1 = (i + 1) % segs
            F.append((j * segs + i, j * segs + i1, (j + 1) * segs + i1, (j + 1) * segs + i))
    return I.part(name, (V, F, U), "mli", link_ob, smooth=70.0, bev=0)


def handrail(name, link_ob, phi, z_lo, z_hi, r_surf):
    """ISS EVA handrail (gold anodize): rounded bar on two stand-offs, 0.05 m off the blanket."""
    out = I.radial(phi)
    t = I.tangent(phi)
    r_bar = r_surf + 0.055
    geo = []
    a, b = r_bar * out + UP * z_lo, r_bar * out + UP * z_hi
    geo.append(I.box((a + b) / 2, (0.025, 0.032, z_hi - z_lo + 0.03), (t, out, UP)))
    for z in (z_lo + 0.02, z_hi - 0.02):
        geo.append(I.box(out * (r_surf + 0.025) + UP * z, (0.03, 0.06, 0.028), (t, out, UP)))
    return I.part(name, I.merge(*geo), "handrail", link_ob, bev=0.003)


def target_plate(link_ob, k, phi, z, r_surf):
    """Peripheral docking target: white two-lobe ("dog-bone") plate with black ring markers and two
    stand-off posts carrying disc tops (KSC-20190620-PH_ILW01_0048/0050, ida1_ksc_2015)."""
    out, t = I.radial(phi), I.tangent(phi)
    c0 = (r_surf + 0.04) * out + UP * z
    lobes = [c0 + out * 0.12 + t * 0.10, c0 + out * 0.25 - t * 0.08]
    poly = []
    pts = []
    for i in range(48):
        a = math.tau * i / 48
        for lc in lobes:
            pts.append(lc + 0.11 * (math.cos(a) * out + math.sin(a) * t))
    # convex-ish outline in plate coordinates then a thin slab
    local = [((p - c0).dot(out), (p - c0).dot(t)) for p in pts] + [(-0.02, 0.10), (-0.02, -0.12)]
    hull = _hull2d(local)
    slab = I.prism(hull, c0, out, t, UP, 0.008)
    I.part(f"target{k}_plate", slab, "target_white", link_ob, bev=0.0015)
    rings = []
    for lc in lobes:
        for dx, dy in ((0.0, 0.0),):
            c = lc + UP * 0.0042
            rings.append(I.revolve([(0.035, -0.0002), (0.07, -0.0002), (0.07, 0.0003), (0.035, 0.0003)], segs=48, z0=0.0))
            v, f, u = rings[-1]
            rings[-1] = ([p + c for p in v], f, u)
    for i, (dx, dy) in enumerate(((0.05, 0.16), (0.20, -0.16), (0.30, 0.06))):
        c = c0 + out * dx + t * dy + UP * 0.0042
        v, f, u = I.box(c, (0.06 if i % 2 else 0.012, 0.012 if i % 2 else 0.06, 0.0008))
        rings.append((v, f, u))
    I.part(f"target{k}_marks", I.merge(*rings), "target_black", link_ob, smooth=0, bev=0)
    posts = []
    for lc in lobes:
        posts.append(I.cylinder(lc, lc + UP * 0.14, 0.007, 16))
        posts.append(G.lathe_geo([(1e-4, 0.14), (0.045, 0.14), (0.045, 0.152), (1e-4, 0.152)], 32,
                                 center=(lc.x, lc.y, lc.z)))
    I.part(f"target{k}_posts", I.merge(*posts), "aluminium_machined", link_ob, bev=0.0008)
    I.part(f"target{k}_bracket", I.box(c0 - out * 0.02 - UP * 0.03, (0.06, 0.08, 0.06), (out, t, UP)),
           "aluminium_machined", link_ob, bev=0.001)


def _hull2d(pts):
    pts = sorted(set((round(x, 5), round(y, 5)) for x, y in pts))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def trunnion(link_ob, k, phi, z, r_surf, length=0.24):
    out = I.radial(phi)
    a = out * (r_surf - 0.02) + UP * z
    b = out * (r_surf + length) + UP * z
    I.part(f"trunnion{k}_shaft", I.cylinder(a, b, 0.032, 32), "aluminium", link_ob, bev=0.001)
    I.part(f"trunnion{k}_collar", I.cylinder(a + out * 0.03, a + out * 0.07, 0.048, 32), "aluminium_machined", link_ob,
           bev=0.001)
    I.part(f"trunnion{k}_ball", G.lathe_geo([(1e-4, -0.055), (0.03, -0.047), (0.052, -0.02), (0.055, 0.0),
                                             (0.052, 0.02), (0.03, 0.047), (1e-4, 0.055)], 32, center=b + out * 0.03,
                                            axis_frame=[[I.tangent(phi).x, 0, out.x], [I.tangent(phi).y, 0, out.y],
                                                        [0, 1, 0]]), "chrome", link_ob, smooth=60, bev=0)


def connector_panel(link_ob, phi, z, r_surf):
    out, t = I.radial(phi), I.tangent(phi)
    c = out * (r_surf + 0.005) + UP * z
    I.part("connector_panel", I.box(c, (0.30, 0.012, 0.22), (t, out, UP)), "mli", link_ob, bev=0.004)
    geos = []
    for i, (dt, dz, rr) in enumerate(((-0.08, 0.05, 0.018), (0.0, 0.05, 0.022), (0.08, 0.05, 0.018),
                                     (-0.05, -0.05, 0.016), (0.05, -0.05, 0.016))):
        p = c + t * dt + UP * dz + out * 0.006
        geos.append(G.lathe_geo([(1e-4, 0.0), (rr + 0.006, 0.0), (rr + 0.006, 0.008), (rr, 0.01), (rr, 0.045),
                                 (rr * 0.8, 0.05), (1e-4, 0.05)], 24, center=(p.x, p.y, p.z),
                                axis_frame=[[t.x, 0, out.x], [t.y, 0, out.y], [t.z, 1, out.z]]))
    I.part("connectors", I.merge(*geos), "connector", link_ob, smooth=50, bev=0)


def retroreflector(link_ob, k, phi, z, r):
    out = I.radial(phi)
    c = out * r + UP * z
    I.part(f"reflector{k}_base", I.cylinder(c - UP * 0.02, c, 0.04, 32), "black_anodize", link_ob, bev=0.001)
    I.part(f"reflector{k}_dome", G.lathe_geo([(0.032 * math.cos(math.radians(a)), 0.032 * math.sin(math.radians(a)))
                                             for a in range(0, 91, 10)] + [(1e-4, 0.032)], 32, center=(c.x, c.y, c.z)),
           "retroreflector", link_ob, smooth=80, bev=0)


def build_body(cfg, base):
    z_h = P.IDA_HCS_Z
    ri = P.TUNNEL_ID / 2
    # MAPAS (modified APAS-89) interface ring at the PMA plane, with its seal and eight guide blades
    mapas = [(0.627, 0.0), (0.80, 0.0), (0.80, 0.045), (0.78, 0.06), (0.66, 0.06), (0.627, 0.035)]
    ob = I.part("mapas_ring", I.revolve(mapas, segs=256), "aluminium_machined", base, bev=0.001)
    I.boolean(ob, [I.cylinder(0.765 * I.radial(math.tau * (k + 0.5) / 72) - UP * 0.01,
                              0.765 * I.radial(math.tau * (k + 0.5) / 72) + UP * 0.02, 0.004, 10) for k in range(72)])
    I.part("mapas_seal", I.revolve([(0.700, -0.002), (0.708, -0.002), (0.708, 0.0005), (0.700, 0.0005)], segs=256),
           "seal_silicone", base, bev=0)
    for k in range(P.MAPAS_GUIDES):
        phi = math.tau * (k + 0.5) / P.MAPAS_GUIDES
        out, t = I.radial(phi), I.tangent(phi)
        top = 0.66 * out + UP * 0.03
        H = P.MAPAS_GUIDE_H
        poly = [(-0.045, 0.0), (0.045, 0.0), (0.012, -H), (-0.012, -H)]
        geo = I.prism(poly, top, t, UP, out, 0.035)
        I.part(f"mapas_guide_{k + 1}", geo, "hard_anodize", base, smooth=25, bev=0.002)
    # EIPS shell, adapter ring and inner tunnel (mostly under blankets)
    I.tunnel("eips_tunnel", base, 0.03, z_h - P.HCS_RING_H, "aluminium_machined")
    I.part("eips_shell", I.revolve([(0.775, 0.06), (0.79, 0.06), (0.79, 0.56), (0.775, 0.56)], segs=256),
           "aluminium", base, bev=0)
    I.part("ear_ring", I.revolve([(ri + 0.004, 0.555), (0.80, 0.555), (0.80, 0.578), (0.712, 0.590), (ri + 0.004, 0.590)],
                                 segs=256), "aluminium_machined", base, bev=0.001)
    # blankets: shell, top roll round the HCS flange, bottom roll, bore liner
    shell_blanket("mli_shell", base, 0.835, 0.07, 0.535)
    blanket_ring("mli_top_roll", base, 0.715, 0.865, 0.53, 0.628, puff=0.004, seed=3)
    blanket_ring("mli_bottom_roll", base, 0.79, 0.86, 0.04, 0.10, puff=0.003, seed=5)
    I.part("mli_bore_liner", I.revolve([(ri + 0.0015, 0.04), (ri + 0.006, 0.04), (ri + 0.006, 0.55), (ri + 0.0015, 0.55)],
                                       segs=192), "mli", base, smooth=60, bev=0)
    # Velcro closure tabs and seam tapes on the shell blanket
    tabs = []
    for k in range(16):
        phi = math.tau * k / 16 + 0.05
        out, t = I.radial(phi), I.tangent(phi)
        for z in (0.16, 0.42):
            tabs.append(I.box(out * 0.843 + UP * z, (0.035, 0.004, 0.09), (t, out, UP)))
    I.part("mli_tabs", I.merge(*tabs), "mli", base, bev=0.001)
    # hardware on the outside
    r_s = 0.84
    for k, phi in enumerate(cfg.HANDRAIL_ANGLES):
        handrail(f"handrail_{k + 1}", base, phi, 0.13, 0.47, r_s)
    for k, phi in enumerate(cfg.TARGET_ANGLES):
        target_plate(base, k + 1, phi, 0.46, r_s)
    for k, phi in enumerate(cfg.TRUNNION_ANGLES):
        trunnion(base, k + 1, phi, 0.30, r_s)
    connector_panel(base, cfg.CONNECTOR_ANGLE, 0.32, r_s)
    for k, phi in enumerate(cfg.REFLECTOR_ANGLES):
        retroreflector(base, k + 1, phi, 0.62, 0.79)
    # passive SCS: fixed capture ring + truss petals + strikers, on six brackets inside the tunnel
    I.passive_scs(base, z_h, "truss", "hard_anodize", "hard_anodize")
    for k in range(6):
        phi = math.tau * (k + 0.5) / 6
        out = I.radial(phi)
        c = out * ((P.SCS_RING_OD / 2 + ri) / 2) + UP * (z_h + P.SCS_PASSIVE_Z - P.SCS_RING_H - 0.012)
        I.part(f"scs_bracket_{k + 1}", I.box(c, (0.05, 0.06, 0.03), (out, I.tangent(phi), UP)), "hard_anodize", base,
               bev=0.001)
    # HCS ring (flat IDA sealing land), pins, receptacles, separator/sensor bores, hook housings
    bores = []
    for phi in P.IDA_SEPARATORS + cfg.SENSOR_ANGLES:
        c = P.SEPARATOR_R * I.radial(phi)
        bores.append(I.cylinder(c + UP * (z_h - 0.05), c + UP * (z_h + 0.01), P.SEPARATOR_D / 2 + 0.0008, 24))
    I.hcs_ring("hcs_ring", base, z_h, "hcs", mating="flat", extra_cuts=bores)
    I.hcs_ring_collider("hcs", base, z_h)
    I.guide_pins(base, z_h)
    I.receptacle_colliders(base, z_h)
    I.hook_pocket_parts(base, z_h)


def build(cfg):
    base = G.link("ida_base", (0, 0, 0), mass=cfg.MASSES["ida_base"])
    build_body(cfg, base)
    nds.build_hooks(cfg, base, P.IDA_HCS_Z)
    nds.build_plungers(cfg, base, P.IDA_HCS_Z, P.IDA_SEPARATORS, "separator", P.SEPARATOR_STROKE, P.SEPARATOR_D, 0.0,
                       cfg.MASSES["separator"], cfg.SEPARATOR_SPRING, "separator")
    nds.build_plungers(cfg, base, P.IDA_HCS_Z, cfg.SENSOR_ANGLES, "sensor", 0.004, 0.016, 0.004, cfg.MASSES["sensor"],
                       cfg.SENSOR_SPRING, "sensor")
    G.site("docking_interface", base, (0, 0, P.IDA_HCS_Z), (0, 0, 1), kind="interface",
           description="IDSS HCS mating plane centre (VV-to-IDA plane); +Z out of the port, +X line of symmetry")
    G.site("pma_interface", base, (0, 0, 0), (0, 0, -1), kind="interface",
           description="PMA-to-IDA interface plane (MAPAS), +Z towards the PMA")
