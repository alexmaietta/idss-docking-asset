"""IDSS docking-interface geometry shared by the passive (IDA) and active (NDS) builders.

Every function builds in the *interface frame* (see params.py: origin on the axis in the HCS mating
plane, +Z out of the interface, +X line of symmetry, +Y line of androgyny) and places it at world
height `z0` (both assets keep the docking axis on world +Z). Visual meshes get metre UVs and bevels;
each contact-relevant part also gets an explicit collider mesh (`collider()`), tagged with the
contact model the USD packager authors for Newton (SDF / convex hull) and its physics material.
"""
import json
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

import geo as G
import materials as M
import params as P

TAU = 2 * math.pi
UP = Vector((0, 0, 1))


def radial(phi):
    return Vector((math.cos(phi), math.sin(phi), 0.0))


def tangent(phi):
    return Vector((-math.sin(phi), math.cos(phi), 0.0))


# ---------------------------------------------------------------- generic mesh builders

def revolve(profile, a0=0.0, a1=TAU, segs=256, z0=0.0, cap=None, half=None):
    """Closed (r, z) profile swept about +Z between angles a0..a1.

    half(z) -> optional angular half-extent override: the sweep then runs about the centre angle
    (a0+a1)/2 with an extent that varies with each profile point's height (petals, strikers).
    Returns (verts, faces, uvs); end caps are added for partial sweeps (cap=True by default).
    """
    full = half is None and abs(a1 - a0 - TAU) < 1e-9
    cap = (not full) if cap is None else cap
    n = len(profile)
    ns = segs if full else max(2, segs)
    arc = [0.0]
    for i in range(1, n + 1):
        (r0, z_0), (r1, z_1) = profile[i - 1], profile[i % n]
        arc.append(arc[-1] + math.hypot(r1 - r0, z_1 - z_0))
    ac = 0.5 * (a0 + a1)
    V, F, U = [], [], []
    cols = ns if full else ns + 1
    for s in range(cols):
        f = s / ns
        for i, (r, z) in enumerate(profile):
            if half is None:
                a = a0 + (a1 - a0) * f
            else:
                h = half(z)
                a = ac - h + 2 * h * f
            V.append(Vector((r * math.cos(a), r * math.sin(a), z0 + z)))
            U.append((r * (a - ac), arc[i]))
    for s in range(ns):
        s1 = (s + 1) % cols if full else s + 1
        for i in range(n):
            j = (i + 1) % n
            F.append((s * n + i, s * n + j, s1 * n + j, s1 * n + i))
    if cap and not full:
        F.append(tuple(range(n))[::-1])
        F.append(tuple(ns * n + i for i in range(n)))
    return V, F, U


def prism(poly, origin, ex, ey, ez, depth, centred=True):
    """Extrude a 2D polygon (list of (x, y)) along ez. World point = origin + x*ex + y*ey + w*ez."""
    o, ex, ey, ez = Vector(origin), Vector(ex), Vector(ey), Vector(ez)
    w0, w1 = (-depth / 2, depth / 2) if centred else (0.0, depth)
    n = len(poly)
    V, F, U = [], [], []
    for w in (w0, w1):
        for x, y in poly:
            V.append(o + ex * x + ey * y + ez * w)
            U.append((x + w * 0.0, y + w))
    F.append(tuple(range(n))[::-1])
    F.append(tuple(n + i for i in range(n)))
    for i in range(n):
        j = (i + 1) % n
        F.append((i, j, n + j, n + i))
    return V, F, U


def cylinder(p0, p1, r, sides=24, r1=None):
    return G.tube_uv(p0, p1, r, sides, r1)


def box(c, size, frame=None):
    """Oriented box: frame = (ex, ey, ez) unit axes (default world)."""
    ex, ey, ez = frame or (Vector((1, 0, 0)), Vector((0, 1, 0)), UP)
    sx, sy, sz = size
    poly = [(-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)]
    return prism(poly, c, ex, ey, ez, sz)


def merge(*geos):
    return G.merge_geo(*geos)


def fix_normals(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()


def smooth_angle(ob, angle_deg=35.0):
    me = ob.data
    me.shade_smooth()
    try:
        me.set_sharp_from_angle(angle=math.radians(angle_deg))
    except AttributeError:
        pass


def bevel(ob, width=0.0008, segments=2, angle_deg=40.0):
    """Machined-edge bevel (applied at export) so hard edges catch highlights like the photos."""
    m = ob.modifiers.new("bevel", "BEVEL")
    m.width = width
    m.segments = segments
    m.limit_method = "ANGLE"
    m.angle_limit = math.radians(angle_deg)
    m.harden_normals = True
    try:
        m.use_clamp_overlap = True
    except AttributeError:
        pass
    return m


def part(name, geo, mat, parent, smooth=35.0, bev=0.0008, variant=None):
    """Visual mesh on a link: fixed normals, auto-smooth by angle, optional machined bevel."""
    v, f, u = geo
    ob = G.add_mesh(name, v, f, M.get(mat) if isinstance(mat, str) else mat, parent, uvs=u, variant=variant)
    fix_normals(ob)
    if smooth:
        smooth_angle(ob, smooth)
    if bev:
        bevel(ob, bev)
    return ob


def apply_modifiers(ob):
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    old = ob.data
    ob.modifiers.clear()
    ob.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)


def boolean(ob, cutters, op="DIFFERENCE", separate=False):
    """Boolean `ob` against a list of (verts, faces, uvs) cutters, joined into one exact-solver pass
    (cutters must not overlap each other) or, with separate=True, one pass per cutter."""
    if not cutters:
        return ob
    if separate:
        for c in cutters:
            boolean(ob, [c], op)
        return ob
    v, f, u = merge(*cutters)
    cut = G.add_mesh(ob.name + "_cut", v, f, None, ob.parent, uvs=u)
    fix_normals(cut)
    # bevel / other modifiers stay after the boolean so they act on the cut edges too
    keep = [(m.name, m.type) for m in ob.modifiers]
    saved = []
    for m in list(ob.modifiers):
        saved.append({k: getattr(m, k) for k in ("width", "segments", "limit_method", "angle_limit", "harden_normals")
                      if m.type == "BEVEL" and hasattr(m, k)})
        ob.modifiers.remove(m)
    b = ob.modifiers.new("bool", "BOOLEAN")
    b.operation = op
    b.solver = "EXACT"
    b.use_self = True   # cutters may overlap (counterbore + chamfer): union them rather than even-odd
    b.object = cut
    apply_modifiers(ob)
    bpy.data.objects.remove(cut)
    for (n, t), s in zip(keep, saved):
        m = ob.modifiers.new(n, t)
        for k, val in s.items():
            setattr(m, k, val)
    return ob


# ---------------------------------------------------------------- colliders

CONTACT_VARIANTS = ("exact", "sdf", "hydro")
EXACT = ("exact",)
FIELD = ("sdf", "hydro")


def collider(name, geo, link_ob, model="sdf", material="anodize", voxel=0.002, hydro=False, margin=None,
             variants=CONTACT_VARIANTS, solref=None):
    """Collision-only mesh on a link. model: "sdf" (NewtonSDFCollisionAPI, sparse SDF at `voxel`
    metres), "convex" (convex hull) or "mesh" (triangle mesh BVH). Hidden from renders; the
    packager writes it with purpose=guide and the physics material binding."""
    v, f, u = geo
    ob = G.add_mesh(f"col_{name}", v, f, None, link_ob, uvs=u)
    fix_normals(ob)
    ob["collider"] = json.dumps({"model": model, "material": material, "voxel": voxel, "hydro": hydro,
                                 "margin": margin, "variants": list(variants), "solref": solref})
    ob.hide_render = True
    ob.display_type = "WIRE"
    return ob


def collider_from(ob, name, model="sdf", material="anodize", voxel=0.002, hydro=False, variants=CONTACT_VARIANTS):
    """Collider that copies an existing visual mesh's geometry (after its modifiers)."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    c = bpy.data.objects.new(f"col_{name}", me)
    bpy.context.scene.collection.objects.link(c)
    c.parent = ob.parent
    c.matrix_world = ob.matrix_world.copy()
    me.materials.clear()
    c["collider"] = json.dumps({"model": model, "material": material, "voxel": voxel, "hydro": hydro,
                                "margin": None, "variants": list(variants)})
    c.hide_render = True
    c.display_type = "WIRE"
    return c


# ---------------------------------------------------------------- HCS ring (androgynous)

HCS_PROFILE = [  # (r, z) below the mating plane, CCW; outer lip thinner (flange for the hook gearheads)
    (P.TUNNEL_ID / 2, -P.HCS_RING_H), (0.690, -P.HCS_RING_H), (0.690, -0.024), (0.7065, -0.024),
    (0.710, -0.0205), (0.710, -0.0025), (0.7075, 0.0), (P.TUNNEL_ID / 2 + 0.0025, 0.0),
    (P.TUNNEL_ID / 2, -0.0025)]


def hook_station_frame(phi):
    """(origin at the station centre on the HCS plane, tangential t, radial r) of hook station phi."""
    return P.HOOK_R * radial(phi), tangent(phi), radial(phi)


def hcs_ring(name, link_ob, z0, mat, seals=True, mating="seal", sensor_zones=True, bolt_holes=96, extra_cuts=()):
    """Machined HCS ring: hook pockets, guide-pin receptacle bores, striker-zone recesses, bolt holes.

    mating: "seal" (two silicone beads, IDSS seal-on-seal) or "flat" (IDA: flat sealing land).
    Returns (visual object, list of pocket boxes for the collider)."""
    ob = part(name, revolve(HCS_PROFILE, segs=384, z0=z0), mat, link_ob, bev=0.0006)
    cuts = []
    for phi in P.HOOK_ANGLES:
        c, t, r = hook_station_frame(phi)
        L, W, D = P.HOOK_POCKET
        cuts.append(box(c + Vector((0, 0, z0 - D / 2 + 0.002)), (L, W, D + 0.004), (t, r, UP)))
    rcpt = []
    for phi, off in P.GUIDE_RECEPTACLES:   # overlapping cutters: cut in their own passes below
        rcpt += receptacle_cutters(P.GUIDE_R * radial(phi) + off * tangent(phi), phi, z0)
    if sensor_zones:   # 1 mm recessed striker zones for the separators and interface sensors
        r0, r1 = (d / 2 for d in P.STRIKER_ZONE_D)
        for phi in P.SENSOR_ZONES:
            prof = [(r0, -P.STRIKER_ZONE_RECESS), (r1, -P.STRIKER_ZONE_RECESS), (r1, 0.004), (r0, 0.004)]
            cuts.append(revolve(prof, phi - P.STRIKER_ZONE_W / 2, phi + P.STRIKER_ZONE_W / 2, 10, z0))
    for k in range(bolt_holes):   # flange bolt circle (scaled from the IDA KSC photos: ~3.75 deg pitch)
        a = TAU * (k + 0.5) / bolt_holes
        if min(abs(math.remainder(a - h, TAU)) for h in P.HOOK_ANGLES) < math.radians(6.5):
            continue
        c = 0.6985 * radial(a)
        cuts.append(cylinder(c + Vector((0, 0, z0 - 0.03)), c + Vector((0, 0, z0 + 0.01)), 0.0033, 10))
    boolean(ob, cuts + list(extra_cuts))
    boolean(ob, rcpt, separate=True)
    if mating == "seal" and seals:
        for d in P.SEAL_D:
            r = d / 2
            prof = [(r - P.SEAL_W / 2, -0.004), (r + P.SEAL_W / 2, -0.004), (r + P.SEAL_W / 2, 0.0),
                    (r + P.SEAL_W * 0.32, P.SEAL_H * 0.82), (r, P.SEAL_H), (r - P.SEAL_W * 0.32, P.SEAL_H * 0.82),
                    (r - P.SEAL_W / 2, 0.0)]
            part(f"{name}_seal_{int(d * 1000)}", revolve(prof, segs=384, z0=z0), "seal_silicone", link_ob,
                 smooth=60.0, bev=0)
    elif mating == "flat":   # polished seal land, flush (IDA's metal sealing surface)
        r0, r1 = P.SEAL_D[0] / 2 - 0.008, P.SEAL_D[1] / 2 + 0.008
        prof = [(r0, -0.0003), (r1, -0.0003), (r1, 0.00005), (r0, 0.00005)]
        part(f"{name}_seal_land", revolve(prof, segs=384, z0=z0), "seal_land", link_ob, smooth=0, bev=0)
    return ob


def hcs_ring_collider(name, link_ob, z0, material="anodize"):
    """Contact model of the HCS mating face: the sealing land (tunnel bore to inside the hook band, under
    both seal beads) as thin convex sectors. The band outside it (hook pockets, guide pins and receptacles,
    separators and sensors on Ø1375) is left to those parts' own colliders, so the pins can enter their
    receptacles and the plungers reach the mating face."""
    out = []
    lands = (("land", P.TUNNEL_ID / 2 + 0.0005, P.HOOK_R - P.HOOK_POCKET[1] / 2 - 0.002, 48),)
    for tag, r0, r1, n in lands:
        prof = [(r0, -0.012), (r1, -0.012), (r1, 0.0), (r0, 0.0)]
        for k in range(n):
            a0, a1 = TAU * k / n, TAU * (k + 1) / n
            out.append(collider(f"{name}_{tag}{k:02d}", revolve(prof, a0, a1, 2, z0), link_ob, "convex", material,
                                variants=EXACT))
        out.append(collider(f"{name}_{tag}_sdf", revolve(prof, segs=720, z0=z0), link_ob, "sdf", material, voxel=0.001,
                            variants=FIELD))
    return out


def _obround(L, W, n=8):
    r = W / 2
    pts = []
    for cx, a0 in ((L / 2 - r, -math.pi / 2), (-(L / 2 - r), math.pi / 2)):
        for i in range(n + 1):
            a = a0 + math.pi * i / n
            pts.append((cx + r * math.cos(a), r * math.sin(a)))
    return pts


def tunnel(name, link_ob, z_bot, z_top, mat, r_in=None, wall=None, segs=256):
    r_in = r_in or P.TUNNEL_ID / 2
    wall = wall or P.TUNNEL_WALL
    prof = [(r_in, z_bot), (r_in + wall, z_bot), (r_in + wall, z_top), (r_in, z_top)]
    return part(name, revolve(prof, segs=segs), mat, link_ob, bev=0)


# ---------------------------------------------------------------- guide pins / separators / sensors

def guide_pins(link_ob, z0, mat="inconel"):
    obs = []
    h_cone = P.GUIDE_PIN_H - P.GUIDE_PIN_CYL_H
    for k, (phi, off) in enumerate(P.GUIDE_PINS):
        c = P.GUIDE_R * radial(phi) + off * tangent(phi)
        r = P.GUIDE_PIN_D / 2
        prof = [(0.0, -0.012), (0.019, -0.012), (0.019, -0.0005), (r + 0.002, 0.0), (r, 0.002),
                (r, P.GUIDE_PIN_CYL_H)]
        for i in range(1, 7):   # 90 deg cone to the SR3 tip
            t = i / 6
            prof.append((r - (r - P.GUIDE_PIN_TIP_R * 0.7) * t, P.GUIDE_PIN_CYL_H + (h_cone - P.GUIDE_PIN_TIP_R * 0.3) * t))
        prof += [(P.GUIDE_PIN_TIP_R * 0.5, P.GUIDE_PIN_H - 0.0004), (0.0, P.GUIDE_PIN_H)]
        geo = G.lathe_geo([(max(a, 1e-4), b) for a, b in prof], 40, center=(c.x, c.y, z0))
        ob = part(f"guide_pin_{k + 1}", geo, mat, link_ob, smooth=50.0, bev=0)
        obs.append(ob)
        collider(f"guide_pin_{k + 1}", G.lathe_geo([(1e-4, 0.0)] + [(max(a, 1e-4), b) for a, b in prof[4:]] + [(1e-4, P.GUIDE_PIN_H)],
                                                    24, center=(c.x, c.y, z0)), link_ob, "convex", "steel", variants=EXACT)
        collider(f"guide_pin_{k + 1}_sdf", geo, link_ob, "sdf", "steel", voxel=0.0008, variants=FIELD)
    return obs


def receptacle_cutters(c, phi, z0):
    """Guide-pin receptacle: Ø39 counterbore recessed 1 mm, a 90 deg lead-in chamfer and the 30.2 x 25.2
    obround bore, 12 deep (IDD Fig 3.2.3.3-3/-4)."""
    slot = _obround(P.GUIDE_RECEPT_SLOT[0], P.GUIDE_RECEPT_SLOT[1], 10)
    top = z0 - P.GUIDE_RECEPT_RECESS
    lead = (P.GUIDE_RECEPT_CBORE_D - P.GUIDE_RECEPT_SLOT[1]) / 2   # chamfer depth (45 deg each side)
    t, r = tangent(phi), radial(phi)
    v, f, u = [], [], []
    rings = [(top + 0.004, 1.0 + 2 * (lead + 0.004) / P.GUIDE_RECEPT_SLOT[1]),
             (top, 1.0 + 2 * lead / P.GUIDE_RECEPT_SLOT[1]), (top - lead, 1.0),
             (top - P.GUIDE_RECEPT_CLEAR, 1.0)]
    n = len(slot)
    cx = sum(x for x, _ in slot) / n
    for z, k in rings:
        for x, y in slot:
            v.append(c + t * (cx + (x - cx) * k) + r * (y * k) + UP * z)
            u.append((x, y))
    for j in range(len(rings) - 1):
        for i in range(n):
            i1 = (i + 1) % n
            f.append((j * n + i, j * n + i1, (j + 1) * n + i1, (j + 1) * n + i))
    f.append(tuple(range(n))[::-1])
    f.append(tuple((len(rings) - 1) * n + i for i in range(n)))
    # counterbore floor 0.3 mm below the chamfer top: coplanar cutter faces leave sliver caps in the boolean
    return [(v, f, u), cylinder(c + UP * (top - 0.0003), c + UP * (z0 + 0.01), P.GUIDE_RECEPT_CBORE_D / 2, 40)]


def receptacle_colliders(link_ob, z0):
    """Guide-pin receptacle walls: a block with the chamfered obround bore - exact triangle mesh for the
    exact contact variant, a 0.4 mm SDF for the SDF / hydroelastic variants."""
    out = []
    for k, (phi, off) in enumerate(P.GUIDE_RECEPTACLES):
        c = P.GUIDE_R * radial(phi) + off * tangent(phi)
        blk = box(c + Vector((0, 0, z0 - 0.028)), (0.07, 0.045, 0.056), (tangent(phi), radial(phi), UP))
        ob = G.add_mesh(f"tmp_rcpt{k}", *blk[:2], None, link_ob, uvs=blk[2])
        fix_normals(ob)
        boolean(ob, receptacle_cutters(c, phi, z0), separate=True)
        out.append(collider_from(ob, f"guide_receptacle_{k + 1}", "mesh", "steel", variants=EXACT))
        out.append(collider_from(ob, f"guide_receptacle_{k + 1}_sdf", "sdf", "steel", voxel=0.0004, variants=FIELD))
        bpy.data.objects.remove(ob)
    return out


def plunger_geo(c, z0, d, out_len, housing_d=None, housing_len=0.06):
    """Spring plunger (separator / sensor): head flush with the recessed zone, rod into the housing."""
    r = d / 2
    head = G.lathe_geo([(1e-4, -housing_len), (r * 0.8, -housing_len), (r * 0.8, -0.004), (r, -0.0035),
                        (r, out_len - 0.002), (r * 0.85, out_len), (1e-4, out_len)], 32, center=(c.x, c.y, z0))
    return head


# ---------------------------------------------------------------- hooks

def active_hook(name, link_ob, phi, z0, mat="titanium_dfl", angle=0.0):
    """Active hook rotated `angle` (closing positive, about -radial) from its tangent-horizontal pose.
    Link origin must be at the pivot."""
    c, t, r = hook_station_frame(phi)
    pivot = c + t * P.HOOK_PIVOT_T + Vector((0, 0, z0 + P.HOOK_PIVOT_Z))
    poly = [(x * P.MM, y * P.MM) for x, y in P.ACTIVE_HOOK_PROFILE]
    v, f, u = prism(poly, pivot, t, UP, r, P.HOOK_W)
    rot = Matrix.Rotation(angle, 3, -r)
    geo = ([pivot + rot @ (p - pivot) for p in v], f, u)
    ob = part(name, geo, mat, link_ob, smooth=30.0, bev=0.0005)
    # pivot boss / shaft
    part(name + "_shaft", cylinder(pivot - r * 0.014, pivot + r * 0.014, 0.0075, 20), "steel", link_ob, bev=0)
    col = collider(name, geo, link_ob, "sdf", "titanium", voxel=0.0006)
    return ob, col, pivot


def passive_hook(name, link_ob, phi, z0, mat="titanium_dfl"):
    """Passive hook at its seated (zero-load) position. Link origin at the base centre."""
    c, t, r = hook_station_frame(phi)
    le = c - t * P.HOOK_OFFSET + Vector((0, 0, z0 + P.PASSIVE_HOOK_BASE_Z))
    poly = [(x * P.MM, y * P.MM) for x, y in P.PASSIVE_HOOK_PROFILE]
    geo = prism(poly, le, t, UP, r, P.HOOK_W)
    ob = part(name, geo, mat, link_ob, smooth=30.0, bev=0.0005)
    # Belleville stack stem under the hook (visual)
    stem = le - t * 0.006
    part(name + "_stem", cylinder(stem - UP * 0.016, stem, 0.006, 16), "steel", link_ob, bev=0)
    col = collider(name, geo, link_ob, "sdf", "titanium", voxel=0.0006)
    return ob, col, le - t * 0.006


def hook_pocket_parts(link_ob, z0, mat_box="aluminium_machined"):
    """Hook housing blocks hanging under the flange at every station (gearhead / spring-pack housings)."""
    for k, phi in enumerate(P.HOOK_ANGLES):
        c, t, r = hook_station_frame(phi)
        geo = box(c + Vector((0, 0, z0 - P.HCS_RING_H - 0.032)), (0.17, 0.05, 0.064), (t, r, UP))
        part(f"hook_housing_{k:02d}", geo, mat_box, link_ob, bev=0.001)


# ---------------------------------------------------------------- SCS capture ring, petals, strikers

def capture_ring(name, link_ob, z_scs, mat):
    """Solid capture ring (visual + SDF collider)."""
    prof = [(P.SCS_RING_ID / 2, -P.SCS_RING_H), (P.SCS_RING_OD / 2, -P.SCS_RING_H),
            (P.SCS_RING_OD / 2, -(P.SCS_RING_OD - P.SCS_FLAT_OD) / 2 - 0.0015),
            (P.SCS_FLAT_OD / 2 + 0.0012, -0.0004), (P.SCS_FLAT_OD / 2 - 0.0005, 0.0),
            (P.SCS_FLAT_ID / 2 + 0.0005, 0.0), (P.SCS_FLAT_ID / 2 - 0.0012, -0.0004),
            (P.SCS_RING_ID / 2, -(P.SCS_FLAT_ID - P.SCS_RING_ID) / 2 - 0.0015)]
    geo = revolve(prof, segs=384, z0=z_scs)
    ob = part(name, geo, mat, link_ob, bev=0.0008)
    # bolt heads on the outer wall (petal and actuator brackets)
    return ob, prof


def capture_ring_colliders(name, link_ob, z_scs, prof, material, crest_pads=True):
    """Ring contact model, exact to well under the 0.7 mm petal-to-chamfer clearance:
    - 36 thin convex crest pads on the active ring only (the SCS mating flat from Ø1208 to Ø1219; 10 deg
      sectors keep the inner chord outside the mating petals); they seat on the passive ring's exact mesh;
    - the inner and outer 45 deg chamfers as one triangle-mesh collider (BVH, non-convex, exact).
    A convex sector hull would bulge inside the inner chamfer (5 mm for 15 deg sectors), and an SDF at
    practical voxel sizes rounds it by about a voxel - comparable to the 1 mm latch engagement."""
    out = []
    fid, fod = P.SCS_FLAT_ID / 2, P.SCS_FLAT_OD / 2
    pad = [(fid + 0.003, -0.0015), (fod - 0.0006, -0.0015), (fod - 0.0006, 0.0), (fid + 0.003, 0.0)]
    n = 36 if crest_pads else 0   # one ring carries the pads; they seat on the other ring's exact mesh
    for k in range(n):
        a0, a1 = TAU * k / n, TAU * (k + 1) / n
        out.append(collider(f"{name}_crest{k:02d}", revolve(pad, a0, a1, 2, z_scs), link_ob, "convex", material))
    for c in out:
        c["collider"] = c["collider"].replace('"variants": ["exact", "sdf", "hydro"]', '"variants": ["exact"]')
    out.append(collider(f"{name}_chamfers", revolve(prof, segs=720, z0=z_scs), link_ob, "mesh", material, variants=EXACT))
    out.append(collider(f"{name}_sdf", revolve(prof, segs=720, z0=z_scs), link_ob, "sdf", material, voxel=0.0008,
                        variants=FIELD))
    return out


def petal_half(z):
    """Angular half-extent of a petal at height z above the SCS plane (straight 32.5 deg edges)."""
    h = min(max(z, 0.0), P.PETAL_H)
    base = P.PETAL_BASE_HALF - 0.001 / P.PETAL_BASE_R    # 1 mm edge relief near the base (View E-E)
    return base + (P.PETAL_TIP_HALF - base) * h / P.PETAL_H


def petal_profile():
    """Petal centre-line section (r, z rel. SCS plane): 45 deg contact face, flat tip, back web, foot."""
    tip_t = 0.004
    return [(0.580, -0.030), (0.5795, -0.0205), (P.PETAL_BASE_R - 0.0005, -0.0002), (P.PETAL_BASE_R, 0.0),
            (P.PETAL_TIP_R + 0.0015, P.PETAL_H - 0.0015), (P.PETAL_TIP_R - 0.0005, P.PETAL_H),
            (P.PETAL_TIP_R - tip_t - 0.002, P.PETAL_H), (P.PETAL_ROOT_R, 0.0), (P.PETAL_ROOT_R, -0.030)]


def petal_solid(phi, z_scs, segs=24):
    return revolve(petal_profile(), phi - 0.1, phi + 0.1, segs, z_scs, half=petal_half)


def petal_slant_point(phi, z_scs, slant, depth=0.0):
    """World point on the petal contact face `slant` metres up from the base, `depth` under the face."""
    h = slant / math.sqrt(2)
    r = P.PETAL_BASE_R - h
    n = (radial(phi) + UP).normalized()    # outward face normal (away from the axis, towards +Z)
    return r * radial(phi) + Vector((0, 0, z_scs + h)) - n * depth


def petal_frame(phi):
    """(s, t, n): s up the slant (towards the tip), t tangential, n outward face normal."""
    s = (-radial(phi) + UP).normalized()
    n = (radial(phi) + UP).normalized()
    return s, tangent(phi), n


def solid_petal(name, link_ob, phi, z_scs, mat, latch=True):
    """NDS petal: closed wedge (7075, PTFE hard anodize) with the capture-latch window."""
    ob = part(name, petal_solid(phi, z_scs), mat, link_ob, smooth=20.0, bev=0.0012)
    if latch:
        s, t, n = petal_frame(phi)
        c = petal_slant_point(phi, z_scs, P.LATCH_SLANT + 0.025, 0.0)
        cut = box(c, (P.LATCH_WINDOW[0], P.LATCH_WINDOW[1], 0.05), (s, t, n))
        boolean(ob, [cut])
    return ob


def truss_petal(name, link_ob, phi, z_scs, mat):
    """IDA petal: machined truss frame of the same envelope (KSC photos): side rails, a centre spine,
    the base bar and the back web, with three triangular lightening windows."""
    ob = part(name, petal_solid(phi, z_scs, 32), mat, link_ob, smooth=20.0, bev=0.0012)
    s, t, n = petal_frame(phi)
    w = P.PETAL_TRUSS_W
    cuts = []
    # face windows either side of the spine: triangles in the slant/tangent plane, cut through the wedge
    L = P.petal_slant(P.PETAL_H)
    for side in (-1, 1):
        tri = []
        for sl, frac in ((0.05, 0.94), (0.05, 0.10), (L - 0.07, 0.30)):
            hh = sl / math.sqrt(2)
            half_w = (P.PETAL_BASE_R - hh) * petal_half(hh)
            tri.append((sl, side * (w / 2 + (half_w - w * 1.5) * (frac if side > 0 else frac))))
        tri = [(a, b) for a, b in tri]
        if side < 0:
            tri = tri[::-1]
        base = petal_slant_point(phi, z_scs, 0.0)
        cuts.append(prism(tri, base, s, t, n, 0.30))
    boolean(ob, cuts)
    return ob


def petal_collider(name, link_ob, phi, z_scs, material):
    """Convex wedge of the contact face and back web, without the mounting foot below the SCS plane
    (the foot never touches the mating half; its full-width hull would catch the opposite petals)."""
    prof = [(P.PETAL_ROOT_R + 0.004, 0.0005), (P.PETAL_BASE_R - 0.0005, 0.0005), (P.PETAL_BASE_R - 0.0003, 0.001),
            (P.PETAL_TIP_R + 0.0015, P.PETAL_H - 0.0015), (P.PETAL_TIP_R - 0.0005, P.PETAL_H),
            (P.PETAL_TIP_R - 0.006, P.PETAL_H), (P.PETAL_ROOT_R + 0.004, 0.0)]
    geo = revolve(prof, phi - 0.1, phi + 0.1, 16, z_scs, half=petal_half)
    collider(name + "_sdf", geo, link_ob, "sdf", material, voxel=0.0012, variants=FIELD)
    return collider(name, geo, link_ob, "convex", material, variants=EXACT)


def striker_profile():
    """Mechanical latch striker section (r, z rel. SCS plane), IDD Fig 3.2.2.4-1 / 3.2.2.4.1.1-1:
    the 45 deg upper planar surface continues the ring's inner chamfer from point X (Ø1202) for 114 mm,
    then a 2 mm catch step and the 35 mm lower planar surface."""
    rx = P.SCS_FLAT_ID / 2
    k = 1 / math.sqrt(2)
    s_b = P.STRIKER_X_SLANT
    B = (rx - s_b * k, -s_b * k)
    Bp = (B[0] + P.STRIKER_STEP * k, B[1] - P.STRIKER_STEP * k)
    Lw = P.STRIKER_LOWER_W + 0.004
    L = (Bp[0] - Lw * k, Bp[1] - Lw * k)
    r_wall = P.SCS_RING_ID / 2
    z_wall = -(rx - r_wall)
    return [(r_wall, -0.125), (r_wall, z_wall), (B[0] + 0.0002, B[1] + 0.0002), B, Bp, L, (L[0], -0.125)]


def striker_half(r_out=None):
    """Top-view trapezoid (IDD Fig 3.2.2.4-3): 232 wide at the ring, 70 straight front edge."""
    def half(z):
        prof_r = None  # half-extent by height: wide at the top (ring), narrow at the bottom (front edge)
        f = min(max(-z / 0.125, 0.0), 1.0)
        w = P.STRIKER_WIDTH / 2 * (1 - f) + P.STRIKER_FRONT / 2 * f
        r = P.SCS_RING_ID / 2 - 0.08 * f
        return w / r
    return half


def striker(name, link_ob, phi, z_scs, mat="a286"):
    geo = revolve(striker_profile(), phi - 0.1, phi + 0.1, 20, z_scs, half=striker_half())
    ob = part(name, geo, mat, link_ob, smooth=25.0, bev=0.0006)
    # Contact model: the upper (X -> B) and lower (past the 2 mm catch step) planar bands as exact convex
    # sectors (8 x 3 deg each: < 0.2 mm chord sag on the cone), so the step edge is sharp; an SDF at
    # sub-millimetre voxels would round it off.
    prof = striker_profile()
    r_wall, z_wall = prof[1]
    B, Bp, L = prof[3], prof[4], prof[5]
    k = 1 / math.sqrt(2)
    upper = [(r_wall, z_wall), B, (B[0] + 0.012 * k, B[1] - 0.012 * k), (r_wall, z_wall - 0.012 / k)]
    lower = [Bp, L, (L[0] + 0.012 * k, L[1] - 0.012 * k), (Bp[0] + 0.012 * k, Bp[1] - 0.012 * k)]
    half = striker_half()
    cols = []
    n = 8
    for band, poly in (("upper", upper), ("lower", lower)):
        zc = sum(z for _, z in poly) / len(poly)
        h = half(zc)
        for i in range(n):
            a0, a1 = phi - h + 2 * h * i / n, phi - h + 2 * h * (i + 1) / n
            cols.append(collider(f"{name}_{band}{i}", revolve(poly, a0, a1, 2, z_scs), link_ob, "convex", "steel",
                                 variants=EXACT))
    cols.append(collider(name + "_sdf", geo, link_ob, "sdf", "steel", voxel=0.0005, variants=FIELD))
    return ob, cols


# ---------------------------------------------------------------- capture latch (active petals)

def latch_geometry(phi, z_scs):
    """Pivot, catch-face point, roller centre and frame of the capture latch in petal `phi`.

    IDD Fig 3.2.2.4-4: the "bottom of latch" (its catch face, square to the slant) sits 115 mm up the
    petal slant from the base, 1 mm past the striker's catch step (114 mm from point X) when the capture
    rings are seated. The Ø10 roller is the lead-in on the tip side of the tooth: during insertion the
    striker's point X reaches the roller first and rolls the tooth in; once the step passes the catch
    face the spring swings the tooth out onto the striker's lower surface. A pull-out load then pushes
    the flat catch face straight up the slant towards the rocker pivot (further up the petal), so the
    latch is self-locking. Returns (pivot, catch, roller, (s, t, n)); catch = catch-face point on the
    tooth's outer surface."""
    s, t, n = petal_frame(phi)
    rr = P.LATCH_ROLLER_D / 2
    catch = petal_slant_point(phi, z_scs, P.LATCH_SLANT, 0.0) + n * P.LATCH_PROTRUSION
    roller = petal_slant_point(phi, z_scs, P.LATCH_SLANT + P.LATCH_TOOTH, 0.0) + n * (P.LATCH_PROTRUSION - rr)
    pivot = petal_slant_point(phi, z_scs, P.LATCH_SLANT + P.LATCH_ARM, 0.0)   # pin in the window cheeks, at the face
    return pivot, catch, roller, (s, t, n)


def latch(name, link_ob, phi, z_scs, mat="steel"):
    """Spring-loaded capture latch: rocker arm from the pivot to a tooth with a flat catch face and a
    roller lead-in. The collider is the tooth + roller (one convex hull)."""
    pivot, catch, roller, (s, t, n) = latch_geometry(phi, z_scs)
    rr = P.LATCH_ROLLER_D / 2
    o = petal_slant_point(phi, z_scs, P.LATCH_SLANT, 0.0)   # catch-face foot on the petal face

    def pt(ds, dn):
        return o + s * ds + n * dn
    tooth2d = [(0.0, -0.010), (0.0, P.LATCH_PROTRUSION)]       # (slant from the catch face, normal)
    for i in range(7):   # roller lead-in arc, from the top round to the petal face
        a = math.pi / 2 - (math.pi / 2) * i / 6
        tooth2d.append((P.LATCH_TOOTH + rr * math.cos(a), P.LATCH_PROTRUSION - rr + rr * math.sin(a)))
    tooth2d.append((P.LATCH_TOOTH + rr, -0.010))
    tooth = prism([(x, y) for x, y in tooth2d], o, s, n, t, P.LATCH_WIDTH * 0.8)
    pv = pivot - o
    arm2d = [(0.004, -0.010), (pv.dot(s) + 0.010, pv.dot(n) - 0.008), (pv.dot(s) + 0.012, pv.dot(n) + 0.008),
             (P.LATCH_TOOTH, -0.004)]
    arm = prism(arm2d, o, s, n, t, P.LATCH_WIDTH)
    ob = part(name, merge(tooth, arm), "latch_body", link_ob, smooth=30.0, bev=0.0005)
    part(name + "_roller", cylinder(roller - t * 0.016, roller + t * 0.016, rr, 28), mat, link_ob, smooth=60.0, bev=0)
    part(name + "_pin", cylinder(pivot - t * 0.024, pivot + t * 0.024, 0.005, 16), "steel", link_ob, bev=0)
    tg = prism(tooth2d, o, s, n, t, P.LATCH_WIDTH * 0.8)
    col = collider(name + "_tooth", tg, link_ob, "convex", "steel", variants=EXACT)
    collider(name + "_tooth_sdf", tg, link_ob, "sdf", "steel", voxel=0.0004, variants=FIELD)
    return ob, col


# ---------------------------------------------------------------- shared assembly

def passive_scs(link_ob, z_hcs, petal_style, ring_mat, petal_mat, colliders=True):
    """Capture ring + 3 petals + 3 strikers fixed at the passive (retracted) position."""
    z = z_hcs + P.SCS_PASSIVE_Z
    ring, prof = capture_ring("capture_ring", link_ob, z, ring_mat)
    for k, phi in enumerate(P.PETAL_ANGLES):
        if petal_style == "truss":
            truss_petal(f"petal_{k + 1}", link_ob, phi, z, petal_mat)
        else:
            solid_petal(f"petal_{k + 1}", link_ob, phi, z, petal_mat, latch=False)
        if colliders:
            petal_collider(f"petal_{k + 1}", link_ob, phi, z, "anodize")
    for k, phi in enumerate(P.STRIKER_ANGLES):
        striker(f"striker_{k + 1}", link_ob, phi, z)
    if colliders:
        capture_ring_colliders("capture_ring", link_ob, z, prof, "anodize", crest_pads=False)
    return ring
