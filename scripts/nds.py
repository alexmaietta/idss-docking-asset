"""NASA Docking System Block 1 (active IDSS docking system) builder.

NDS IDD JSC 65795 Rev H Fig 3.1.1-1: the soft-capture system (SCS) is a narrow capture ring carrying
three guide petals and three mechanical capture latches, driven in six degrees of freedom by six
Moog electromechanical linear actuators (a Stewart platform: struts 1 and 2 on petal 1). The hard-
capture system (HCS) is the machined ring with 12 active/passive hook pairs (two gangs of six), two
guide pins and two receptacles, the dual seal and the push-off separators.

Built in the ready-to-capture pose: ring extended to RING_READY_Z, hooks open, latches armed,
separators retracted (charged). Link frames sit at their joint origins (geo.link).

Links and joints
  nds_base (root)               mount flange, tunnel, HCS ring, pins, receptacles, seals, gearheads
  hook_a_00..11   revolute      active hooks, axis -radial (closing positive), gangs per HOOK_GANGS
  hook_p_00..11   prismatic     passive hooks (Belleville stack), +Z: pulled outward by the mate
  separator_1..3  prismatic     push-off plungers, +Z (extends), spring + lock drive
  sensor_1..3     prismatic     interface "ready to hook" sensors, -Z (compression)
  leg_i_ujoint    revolute      base universal joint, axis 1 (i = 1..6)
  leg_i_body      revolute      base universal joint, axis 2 (actuator housing)
  leg_i_rod       prismatic     actuator stroke along the leg (driven)
  scs_ring        spherical     ring on leg 1's rod end (tree); legs 2..6 close loops (spherical)
  latch_1..3      revolute      capture latch rockers in the petals, axis tangential (retract positive)
"""
import math

from mathutils import Vector

import geo as G
import idss as I
import params as P

UP = Vector((0, 0, 1))


def leg_points(z_hcs, ring_z):
    """Base U-joint centres B_i and ring ball-joint centres T_i (world, ring at height ring_z above HCS).

    Pairs: legs (2k, 2k+1) meet petal k's sides on the ring and run down to the base joints either
    side of the neighbouring strikers, so the six legs form the usual triangulated hexapod."""
    B, T = [], []
    for k, phi in enumerate(P.PETAL_ANGLES):
        for side in (-1, 1):
            a_top = phi + side * P.LEG_TOP_SPLIT
            a_base = phi + side * (math.radians(60.0) - P.LEG_BASE_SPLIT)
            T.append(P.LEG_TOP_R * I.radial(a_top) + Vector((0, 0, z_hcs + ring_z + P.LEG_TOP_DZ)))
            B.append(P.LEG_BASE_R * I.radial(a_base) + Vector((0, 0, z_hcs + P.LEG_BASE_Z)))
    return B, T


def _perp(u):
    a1 = u.cross(UP)
    if a1.length < 1e-6:
        a1 = Vector((1, 0, 0))
    a1.normalize()
    a2 = u.cross(a1).normalized()
    return a1, a2


def build_base(cfg, base, z_h):
    # host-vehicle mounting flange (Ø1249.68 interface) and the tunnel (2219, chromate conversion coat)
    ri = P.TUNNEL_ID / 2
    flange = [(ri, -0.004), (0.735, -0.004), (0.735, 0.012), (ri + 0.04, 0.012), (ri + 0.006, 0.03), (ri, 0.03)]
    ob = I.part("mount_flange", I.revolve(flange, segs=256), "iridite", base, bev=0.0008)
    holes = []
    for k in range(48):
        c = 0.712 * I.radial(math.tau * (k + 0.5) / 48)
        holes.append(I.cylinder(c + Vector((0, 0, -0.01)), c + Vector((0, 0, 0.02)), 0.0042, 10))
    I.boolean(ob, holes)
    for k in range(48):   # bolt heads
        c = 0.712 * I.radial(math.tau * (k + 0.5) / 48)
        I.part(f"flange_bolt_{k:02d}", G.lathe_geo([(1e-4, 0.012), (0.0065, 0.012), (0.0065, 0.017), (0.006, 0.0185),
                                                    (1e-4, 0.0185)], 12, center=(c.x, c.y, 0)), "steel", base,
               smooth=40, bev=0)
    I.tunnel("tunnel", base, -0.110, z_h - P.HCS_RING_H, "iridite")
    # stiffening rings on the tunnel
    for z in (0.08, 0.15):
        I.part(f"tunnel_rib_{int(z * 100)}", I.revolve([(ri + 0.005, z - 0.006), (ri + 0.022, z - 0.006),
                                                       (ri + 0.022, z + 0.006), (ri + 0.005, z + 0.006)], segs=256),
               "iridite", base, bev=0.0006)
    # HCS ring with seal beads, separator and sensor bores
    bores = []
    for phi in P.NDS_SEPARATORS + cfg.SENSOR_ANGLES:
        c = P.SEPARATOR_R * I.radial(phi)
        bores.append(I.cylinder(c + Vector((0, 0, z_h - 0.05)), c + Vector((0, 0, z_h + 0.01)),
                                P.SEPARATOR_D / 2 + 0.0008, 24))
    I.hcs_ring("hcs_ring", base, z_h, "hcs", mating="seal", extra_cuts=bores)
    I.hcs_ring_collider("hcs", base, z_h)
    for d in P.SEAL_D:   # seal beads: soft contact rings
        r = d / 2
        prof = [(r - P.SEAL_W / 2, -0.001), (r + P.SEAL_W / 2, -0.001), (r + P.SEAL_W * 0.3, P.SEAL_H), (r - P.SEAL_W * 0.3, P.SEAL_H)]
        for k in range(24):
            a0, a1 = math.tau * k / 24, math.tau * (k + 1) / 24
            I.collider(f"seal_{int(d * 1000)}_{k:02d}", I.revolve(prof, a0, a1, 4, z_h), base, "convex", "seal",
                       variants=I.EXACT, solref=P.SEAL_SOLREF)
        I.collider(f"seal_{int(d * 1000)}_sdf", I.revolve(prof, segs=720, z0=z_h), base, "sdf", "seal", voxel=0.0005,
                   variants=I.FIELD, solref=P.SEAL_SOLREF)
    I.guide_pins(base, z_h)
    I.receptacle_colliders(base, z_h)
    I.hook_pocket_parts(base, z_h)
    # hook drive train: two gearhead motors and the drive-cable conduit ring under the flange
    conduit = [I.radial(math.tau * k / 96) * 0.676 + Vector((0, 0, z_h - P.HCS_RING_H - 0.075)) for k in range(97)]
    G.pipe("hook_drive_conduit", conduit, 0.009, I.M.get("aluminium_machined"), base, closed=True, sides=12)
    for k, phi in enumerate((math.radians(97.0), math.radians(277.0))):
        c = 0.70 * I.radial(phi) + Vector((0, 0, z_h - 0.11))
        I.part(f"hook_motor_{k + 1}", I.cylinder(c - UP * 0.06, c + UP * 0.035, 0.045, 32), "aluminium_machined",
               base, bev=0.001)
        I.part(f"hook_gearhead_{k + 1}", I.box(c + UP * 0.05, (0.11, 0.09, 0.03), (I.tangent(phi), I.radial(phi), UP)),
               "iridite", base, bev=0.001)
    # actuator base brackets (U-joint yokes) bolted to the tunnel wall
    B, T = leg_points(z_h, P.RING_READY_Z)
    for i, b in enumerate(B):
        out = Vector((b.x, b.y, 0)).normalized()
        a = b + out * (ri - P.LEG_BASE_R + 0.004) / 2
        I.part(f"leg{i + 1}_bracket", I.box(a, (ri - P.LEG_BASE_R + 0.004, 0.07, 0.06),
                                             (out, UP.cross(out), UP)), "aluminium_machined", base, bev=0.001)
    # host-side avionics: actuator controller box and harness stand-ins on the tunnel
    for k, phi in enumerate((math.radians(30.0), math.radians(210.0))):
        c = (ri + 0.06) * I.radial(phi) + Vector((0, 0, 0.11))
        I.part(f"lac_box_{k + 1}", I.box(c, (0.16, 0.09, 0.12), (I.tangent(phi), I.radial(phi), UP)), "black_anodize",
               base, bev=0.0015)


def build_hooks(cfg, base, z_h):
    """12 hook stations. Each active hook swings on its pivot (open -> tangent, IDD Fig 3.2.3.4-1/-2: leading
    edge 37 -> 52.5 mm off the station centre line) and the pivot rides a pull-down carriage that then draws the
    hook's tangent plane down for the joint preload (Fig 3.2.3.4-3: leading edge still at 95 REF; "active hook
    tangent plane positioned for joint preload"). The carriage drive's force limit sets the preload (IDD
    31.3-44.3 kN per hook); its stiffness is the active hook stiffness (49-56 kN/mm). Passive hooks sit on
    Belleville stacks (prismatic springs)."""
    for k, phi in enumerate(P.HOOK_ANGLES):
        c, t, r = I.hook_station_frame(phi)
        pivot = c + t * P.HOOK_PIVOT_T + Vector((0, 0, z_h + P.HOOK_PIVOT_Z))
        gang = 1 if k in P.HOOK_GANGS[0] else 2
        cl = G.link(f"hook_c_{k:02d}", pivot, base, mass=cfg.MASSES["hook_carriage"], joint={
            "type": "prismatic", "axis": [0, 0, -1], "limit": [0.0, P.HOOK_PULL_STROKE], "drive": cfg.HOOK_PULL,
            "role": f"hook_pull{gang}"})
        I.part(f"hook_c_{k:02d}_block", I.box(pivot - UP * 0.012, (0.05, 0.03, 0.03), (t, r, UP)), "steel", cl,
               bev=0.0008)
        hl = G.link(f"hook_a_{k:02d}", pivot, cl, mass=cfg.MASSES["hook_active"], joint={
            "type": "revolute", "axis": list(-r), "limit": [0.0, -P.HOOK_OPEN + P.HOOK_OVERTRAVEL],
            "drive": cfg.HOOK_DRIVE, "role": f"hook_gang{gang}"})
        I.active_hook(f"hook_a_{k:02d}", hl, phi, z_h, angle=P.HOOK_OPEN)
        le = c - t * P.HOOK_OFFSET + Vector((0, 0, z_h + P.PASSIVE_HOOK_BASE_Z))
        pl = G.link(f"hook_p_{k:02d}", le - t * 0.006, base, mass=cfg.MASSES["hook_passive"], joint={
            "type": "prismatic", "axis": [0, 0, 1], "limit": [0.0, 0.004], "drive": cfg.PASSIVE_HOOK_SPRING,
            "role": "hook_passive"})
        I.passive_hook(f"hook_p_{k:02d}", pl, phi, z_h)


def build_plungers(cfg, base, z_h, angles, prefix, stroke, d, extended, mass, drive, role, mat="steel"):
    """Spring plungers on the HCS face: separators (built retracted, extend +Z) or sensors (built
    extended, compress -Z)."""
    for k, phi in enumerate(angles):
        c = P.SEPARATOR_R * I.radial(phi)
        top = z_h - P.STRIKER_ZONE_RECESS + (extended if extended else -0.0005)
        lk = G.link(f"{prefix}_{k + 1}", Vector((c.x, c.y, top)), base, mass=mass, joint={
            "type": "prismatic", "axis": [0, 0, 1] if not extended else [0, 0, -1],
            "limit": [0.0, stroke], "drive": drive, "role": role})
        geo = I.plunger_geo(c, top, d, 0.0, housing_len=0.07)
        ob = I.part(f"{prefix}_{k + 1}", geo, mat, lk, smooth=45, bev=0)
        I.collider(f"{prefix}_{k + 1}", G.lathe_geo([(1e-4, -0.004), (d / 2, -0.004), (d / 2, -0.0015), (d / 2 - 0.0015, 0.0),
                                                      (1e-4, 0.0)], 20, center=(c.x, c.y, top)), lk, "convex", "steel",
                   variants=I.EXACT)
        I.collider(f"{prefix}_{k + 1}_sdf", G.lathe_geo([(1e-4, -0.004), (d / 2, -0.004), (d / 2, -0.0015),
                                                          (d / 2 - 0.0015, 0.0), (1e-4, 0.0)], 20, center=(c.x, c.y, top)),
                   lk, "sdf", "steel", voxel=0.0006, variants=I.FIELD)
        # housing below the ring (on the base)
        I.part(f"{prefix}_{k + 1}_housing", I.cylinder(Vector((c.x, c.y, z_h - P.HCS_RING_H - 0.07)),
                                                        Vector((c.x, c.y, z_h - P.HCS_RING_H + 0.001)), d / 2 + 0.008, 24),
               "aluminium_machined", base, bev=0.0008)


def build_scs(cfg, base, z_h):
    """Stewart platform: six actuators, the capture ring with petals, strikers and latches."""
    ring_z = P.RING_READY_Z
    z_scs = z_h + ring_z
    B, T = leg_points(z_h, ring_z)
    rods = []
    for i, (b, tp) in enumerate(zip(B, T)):
        u = (tp - b).normalized()
        L = (tp - b).length
        a1, a2 = _perp(u)
        uj = G.link(f"leg{i + 1}_ujoint", b, base, mass=cfg.MASSES["leg_ujoint"], joint={
            "type": "revolute", "axis": list(a1), "limit": [-0.6, 0.6], "role": "leg_ujoint"})
        body = G.link(f"leg{i + 1}_body", b, uj, mass=cfg.MASSES["leg_body"], joint={
            "type": "revolute", "axis": list(a2), "limit": [-0.6, 0.6], "role": "leg_ujoint"})
        L_stow = (leg_points(z_h, P.RING_STOWED_Z)[1][i] - b).length
        L_max = (leg_points(z_h, P.RING_MAX_Z)[1][i] - b).length
        rod = G.link(f"leg{i + 1}_rod", tp, body, mass=cfg.MASSES["leg_rod"], joint={
            "type": "prismatic", "axis": list(u), "limit": [L_stow - L - 0.01, L_max - L + 0.01],
            "drive": cfg.LEG_DRIVE, "role": "leg_actuator", "rest_length": L,
            "base": list(b), "top": list(tp)})
        rods.append(rod)
        # visuals: U-joint cross, housing with offset motor and gearbox, chrome rod, rod-end ball
        I.part(f"leg{i + 1}_cross", I.merge(I.cylinder(b - a1 * 0.03, b + a1 * 0.03, 0.009, 16)), "steel", uj, bev=0)
        I.part(f"leg{i + 1}_yoke", I.merge(I.cylinder(b - a2 * 0.032, b + a2 * 0.032, 0.011, 16),
                                            I.box(b + u * 0.03, (0.05, 0.05, 0.03), (a1, a2, u))), "steel", body, bev=0.0006)
        h0, h1 = b + u * 0.035, b + u * P.LEG_BODY_L
        I.part(f"leg{i + 1}_housing", I.cylinder(h0, h1, P.LEG_BODY_D / 2, 32), "iridite", body, bev=0.001)
        I.part(f"leg{i + 1}_endcap", I.cylinder(h1 - u * 0.015, h1 + u * 0.006, P.LEG_BODY_D / 2 + 0.004, 32),
               "aluminium_machined", body, bev=0.0008)
        m0 = b + u * 0.06 + a2 * (P.LEG_BODY_D / 2 + 0.03)
        I.part(f"leg{i + 1}_motor", I.cylinder(m0, m0 + u * 0.16, 0.03, 28), "black_anodize", body, bev=0.0008)
        I.part(f"leg{i + 1}_gearbox", I.box(b + u * 0.055 + a2 * 0.035, (0.08, 0.05, 0.04), (a1, a2, u)),
               "aluminium_machined", body, bev=0.001)
        rod_len = 0.40
        I.part(f"leg{i + 1}_rod", I.cylinder(tp - u * rod_len, tp - u * 0.022, P.LEG_ROD_D / 2, 28), "chrome", rod, bev=0)
        I.part(f"leg{i + 1}_rodend", G.lathe_geo([(1e-4, -0.018), (0.012, -0.016), (0.018, -0.006), (0.018, 0.006),
                                                   (0.012, 0.016), (1e-4, 0.018)], 24, center=tp,
                                                  axis_frame=[[a1.x, a2.x, u.x], [a1.y, a2.y, u.y], [a1.z, a2.z, u.z]]),
               "steel", rod, smooth=60, bev=0)
    ring = G.link("scs_ring", T[0], rods[0], mass=cfg.MASSES["scs_ring"], joint={
        "type": "spherical", "role": "ring_ball"})
    for i in range(1, 6):
        G.loop_joint(f"leg{i + 1}_ball", f"leg{i + 1}_rod", "scs_ring", list(T[i]), [0, 0, 1], jtype="spherical")
    # capture ring, ring frame (leg clevises), petals with latches, strikers
    rob, prof = I.capture_ring("capture_ring", ring, z_scs, "black_anodize")
    I.capture_ring_colliders("capture_ring", ring, z_scs, prof, "anodize")
    frame = [(0.548, -0.062), (0.622, -0.062), (0.622, -0.074), (0.548, -0.074)]
    I.part("ring_frame", I.revolve(frame, segs=256, z0=z_scs), "aluminium_machined", ring, bev=0.0008)
    for i, tp in enumerate(T):
        out = Vector((tp.x, tp.y, 0)).normalized()
        I.part(f"leg{i + 1}_clevis", I.box(tp + out * 0.012 + UP * 0.012, (0.05, 0.05, 0.04), (out, UP.cross(out), UP)),
               "aluminium_machined", ring, bev=0.0008)
    for k, phi in enumerate(P.PETAL_ANGLES):
        I.solid_petal(f"petal_{k + 1}", ring, phi, z_scs, "ptfe_anodize")
        I.petal_collider(f"petal_{k + 1}", ring, phi, z_scs, "ptfe")
        pivot, catch, roller, (s, t, n) = I.latch_geometry(phi, z_scs)
        lk = G.link(f"latch_{k + 1}", pivot, ring, mass=cfg.MASSES["latch"], joint={
            "type": "revolute", "axis": list(t), "limit": [0.0, 0.20], "drive": cfg.LATCH_SPRING, "role": "latch",
            "catch": list(catch), "roller": list(roller), "pivot": list(pivot)})
        I.latch(f"latch_{k + 1}", lk, phi, z_scs)
        # latch motor housing behind the window (inside the petal wedge)
        I.part(f"latch_{k + 1}_actuator", I.box(pivot - n * 0.02 + s * 0.01, (0.05, 0.05, 0.025), (s, t, n)),
               "black_anodize", ring, bev=0.0008)
    # NDS Block 1 carries no passive latch strikers on its ring (NDS capture latch paper, AMS 2018): it only
    # docks actively, so the IDSS striker locations between its petals are left empty.
    return ring


def build(cfg):
    z_h = P.NDS_HCS_Z
    base = G.link("nds_base", (0, 0, 0), mass=cfg.MASSES["nds_base"])
    build_base(cfg, base, z_h)
    build_hooks(cfg, base, z_h)
    build_plungers(cfg, base, z_h, P.NDS_SEPARATORS, "separator", P.SEPARATOR_STROKE, P.SEPARATOR_D, 0.0,
                   cfg.MASSES["separator"], cfg.SEPARATOR_SPRING, "separator")
    build_plungers(cfg, base, z_h, cfg.SENSOR_ANGLES, "sensor", 0.004, 0.016, 0.004, cfg.MASSES["sensor"],
                   cfg.SENSOR_SPRING, "sensor")
    build_scs(cfg, base, z_h)
    # interface sites
    G.site("docking_interface", base, (0, 0, z_h), (0, 0, 1), kind="interface",
           description="IDSS HCS mating plane centre; +Z out of the interface, +X line of symmetry, +Y line of androgyny")
    G.site("host_vehicle_interface", base, (0, 0, 0), (0, 0, -1), kind="interface",
           description="Host-vehicle mating plane (Ø1249.68 bolt interface), +Z into the vehicle")
    G.site("scs_ring_centre", base, (0, 0, z_h + P.RING_READY_Z), (0, 0, 1), kind="reference",
           description="Capture-ring SCS mating plane centre at ready-to-capture (the ring link moves)")
