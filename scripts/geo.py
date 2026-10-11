"""Blender mesh-building helpers shared by every asset builder (world coords, rest pose, Z up, metres).

Adapted from ~/code/lrv/scripts/geo.py. Additions: lathe/capsule/prism primitives with metre UVs,
blanket "puff" displacement, and registries for sites, thrusters and sensor cameras, which the
exporter writes to asset.json for the USD packager.
"""
import json
import math

import bpy
from mathutils import Vector

TAU = 2 * math.pi

# ---------------------------------------------------------------- scene helpers

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    return sc


def link(name, loc, parent=None, joint=None, mass=None):
    """Link frame at its joint origin. `joint` describes the joint to `parent`:
    {"type": revolute|prismatic|fixed, "axis": (x,y,z), "limit": (lo, hi) in rad or m,
     "drive": {...}, "role": str} — stored as JSON for the exporters."""
    e = bpy.data.objects.new(f"link:{name}", None)
    e["link"] = name
    e["joint"] = json.dumps(joint or {"type": "fixed"})
    if mass is not None:
        e["mass"] = float(mass)  # kg; links without one fall back to params.MASS
    e.empty_display_type = "ARROWS"
    e.empty_display_size = 0.12
    bpy.context.scene.collection.objects.link(e)
    e.location = Vector(loc) - (parent.matrix_world.translation if parent else Vector())
    if parent:
        e.parent = parent
    bpy.context.view_layer.update()
    return e


def loop_joint(name, parent, child, anchor, axis, jtype="revolute", **extra):
    """Loop-closing joint (excluded from the articulation tree); anchor in world coords."""
    sc = bpy.context.scene
    loops = json.loads(sc.get("loops", "[]"))
    loops.append({"name": name, "type": jtype, "parent": parent, "child": child,
                  "anchor": list(anchor), "axis": list(axis), **extra})
    sc["loops"] = json.dumps(loops)


def add_mesh(name, verts, faces, mat, parent, uvs=None, smooth=False, variant=None):
    """verts in world coordinates (rest pose); stored relative to the parent link.

    variant: optional "Set:option[,option...]" tag (e.g. "Mission:a17_repair", "LCRU:radiator_open",
    "Mission:a15,a17"); the USD packager then places this part only in those variants' payloads
    instead of the base asset (so it is absent from every other option of that set).
    """
    origin = parent.matrix_world.translation
    me = bpy.data.meshes.new(name)
    me.from_pydata([Vector(v) - origin for v in verts], [], faces)
    if uvs is not None:
        uv = me.uv_layers.new(name="UVMap")
        for poly in me.polygons:
            for li in poly.loop_indices:
                uv.data[li].uv = uvs[me.loops[li].vertex_index]
    me.validate()
    if smooth:
        me.shade_smooth()
    if mat:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    if variant:
        ob["variant"] = variant
    return ob


def tag_variant(ob, variant):
    """Tag an existing object as belonging to a variant ("Set:option"); see add_mesh."""
    ob["variant"] = variant
    return ob


def _frame(d):
    d = d.normalized()
    up = Vector((0, 0, 1)) if abs(d.z) < 0.9 else Vector((1, 0, 0))
    a = d.cross(up).normalized()
    b = d.cross(a).normalized()
    return a, b


def tube_geo(p0, p1, r, sides=12, caps=True, verts=None, faces=None):
    verts = [] if verts is None else verts
    faces = [] if faces is None else faces
    p0, p1 = Vector(p0), Vector(p1)
    a, b = _frame(p1 - p0)
    base = len(verts)
    for p in (p0, p1):
        for i in range(sides):
            t = TAU * i / sides
            verts.append(p + r * (math.cos(t) * a + math.sin(t) * b))
    for i in range(sides):
        j = (i + 1) % sides
        faces.append((base + i, base + j, base + sides + j, base + sides + i))
    if caps:
        faces.append(tuple(base + i for i in reversed(range(sides))))
        faces.append(tuple(base + sides + i for i in range(sides)))
    return verts, faces


def tubes(name, segments, r, mat, parent, sides=12):
    v, f = [], []
    for p0, p1 in segments:
        tube_geo(p0, p1, r, sides, True, v, f)
    return add_mesh(name, v, f, mat, parent, smooth=True)


def box_geo(c, size, verts=None, faces=None):
    verts = [] if verts is None else verts
    faces = [] if faces is None else faces
    c, (sx, sy, sz) = Vector(c), size
    base = len(verts)
    for z in (-1, 1):
        for y in (-1, 1):
            for x in (-1, 1):
                verts.append(c + Vector((x * sx / 2, y * sy / 2, z * sz / 2)))
    for q in ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)):
        faces.append(tuple(base + i for i in q))
    return verts, faces


def box(name, c, size, mat, parent):
    v, f = box_geo(c, size)
    return add_mesh(name, v, f, mat, parent)


def revolve_geo(profile, segments, xform, flip=False, u_scale=None):
    """Revolve a (radius, axial) profile about the local Y axis.

    xform maps local wheel coordinates to world. UVs are in metres: u around the
    circumference (at u_scale radius), v along the profile arc length.
    """
    verts, faces, uvs = [], [], []
    n = len(profile)
    arc = [0.0]
    for i in range(1, n):
        (r0, a0), (r1, a1) = profile[i - 1], profile[i]
        arc.append(arc[-1] + math.hypot(r1 - r0, a1 - a0))
    ru = u_scale or max(r for r, _ in profile)
    for s in range(segments + 1):
        t = TAU * s / segments
        for i, (r, a) in enumerate(profile):
            verts.append(xform @ Vector((r * math.sin(t), a, r * math.cos(t))))
            uvs.append((t * ru, arc[i]))
    for s in range(segments):
        for i in range(n - 1):
            q = (s * n + i, s * n + i + 1, (s + 1) * n + i + 1, (s + 1) * n + i)
            faces.append(q[::-1] if flip else q)
    return verts, faces, uvs


def box_beam_geo(p0, p1, w, h, verts=None, faces=None):
    """Rectangular box-section member from p0 to p1: width w horizontal, height h vertical."""
    verts = [] if verts is None else verts
    faces = [] if faces is None else faces
    p0, p1 = Vector(p0), Vector(p1)
    d = (p1 - p0).normalized()
    side = d.cross(Vector((0, 0, 1)))
    side = side.normalized() if side.length > 1e-6 else Vector((0, 1, 0))
    up = side.cross(d).normalized()
    base = len(verts)
    for p in (p0, p1):
        for sw, sh in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append(p + side * (sw * w / 2) + up * (sh * h / 2))
    for i in range(4):
        j = (i + 1) % 4
        faces.append((base + i, base + j, base + 4 + j, base + 4 + i))
    faces.append((base + 3, base + 2, base + 1, base + 0))
    faces.append((base + 4, base + 5, base + 6, base + 7))
    return verts, faces


def pipe(name, points, r, mat, parent, closed=False, sides=10, smooth=True):
    """Continuous swept tube through world-space points (rolled edges, cables, hoses)."""
    origin = parent.matrix_world.translation
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = r
    cu.bevel_resolution = max(1, sides // 4)
    cu.use_fill_caps = not closed
    sp = cu.splines.new("POLY")
    sp.points.add(len(points) - 1)
    for p, q in zip(sp.points, points):
        p.co = (*(Vector(q) - origin), 1.0)
    sp.use_cyclic_u = closed
    tmp = bpy.data.objects.new(name + "_curve", cu)
    bpy.context.scene.collection.objects.link(tmp)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    bpy.data.objects.remove(tmp)
    bpy.data.curves.remove(cu)
    me.name = name
    if smooth:
        me.shade_smooth()
    if mat:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    return ob


def ensure_uvs(ob, scale=1.0):
    """Give a mesh without UVs a box projection in object-local metres (seamless enough for tileable sets)."""
    me = ob.data
    if me.uv_layers:
        return
    uv = me.uv_layers.new(name="UVMap")
    co = [v.co for v in me.vertices]
    for poly in me.polygons:
        n = poly.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        a, b = [(1, 2), (0, 2), (0, 1)][ax]
        flip = -1.0 if n[ax] < 0 else 1.0
        for li in poly.loop_indices:
            p = co[me.loops[li].vertex_index]
            uv.data[li].uv = (p[a] * scale * (flip if ax != 2 else 1.0), p[b] * scale)


def rivet_geo(p, n, r, h=None, verts=None, faces=None, sides=10):
    """Domed (button-head) rivet at world point p, facing direction n."""
    verts = [] if verts is None else verts
    faces = [] if faces is None else faces
    h = h if h is not None else r * 0.55
    n = Vector(n).normalized()
    a, b = _frame(n)
    base = len(verts)
    rings = [(r, 0.0), (r * 0.92, h * 0.45), (r * 0.7, h * 0.8), (r * 0.38, h * 0.97)]
    for rr, z in rings:
        for i in range(sides):
            t = TAU * i / sides
            verts.append(Vector(p) + n * z + rr * (math.cos(t) * a + math.sin(t) * b))
    verts.append(Vector(p) + n * h)
    top = len(verts) - 1
    for k in range(len(rings) - 1):
        for i in range(sides):
            j = (i + 1) % sides
            faces.append((base + k * sides + i, base + k * sides + j, base + (k + 1) * sides + j, base + (k + 1) * sides + i))
    last = base + (len(rings) - 1) * sides
    for i in range(sides):
        faces.append((last + i, last + (i + 1) % sides, top))
    return verts, faces


def sensor_camera(name, parent, pos, forward, up, spec):
    """Optical sensor on a link: a Blender camera (for renders) + a record the USD packager turns into
    a UsdGeom.Camera under the link, so it follows every joint upstream of it.

    spec: {"focal_mm": f, "focal_range_mm": (lo, hi), "aperture_mm": (h, v), "fstop": n,
           "fstop_range": (lo, hi), "focus_m": d, "clip_m": (near, far), "source": str, ...}
    """
    fwd, upv = Vector(forward).normalized(), Vector(up).normalized()
    right = fwd.cross(upv).normalized()
    upv = right.cross(fwd).normalized()
    cam = bpy.data.cameras.new(name)
    cam.lens = spec["focal_mm"]
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width, cam.sensor_height = spec["aperture_mm"]
    cam.clip_start, cam.clip_end = spec.get("clip_m", (0.05, 5000.0))
    ob = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(ob)
    # Blender/USD cameras look down -Z with +Y up
    from mathutils import Matrix
    rot = Matrix((right, upv, -fwd)).transposed().to_4x4()
    ob.matrix_world = Matrix.Translation(Vector(pos)) @ rot
    mw = ob.matrix_world.copy()
    ob.parent = parent
    ob.matrix_parent_inverse = parent.matrix_world.inverted()
    ob.matrix_world = mw
    sc = bpy.context.scene
    cams = json.loads(sc.get("cameras", "[]"))
    cams.append({"name": name, "link": parent["link"], "pos": list(pos), "forward": list(fwd), "up": list(upv),
                 **{k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}})
    sc["cameras"] = json.dumps(cams)
    return ob


# ---------------------------------------------------------------- primitives with metre UVs

def lathe_geo(profile, segments, center=(0, 0, 0), axis_frame=None, phase=0.0, cap_ends=False, flip=False):
    """Revolve a (radius, z) profile about the local Z axis.

    UVs are in metres: u = arc length around the circumference at the profile's max radius,
    v = arc length along the profile. axis_frame is an optional 3x3 (columns = local x, y, z in
    world); center is the world position of the local origin.
    """
    from mathutils import Matrix
    c = Vector(center)
    M = Matrix.Identity(3) if axis_frame is None else Matrix(axis_frame)
    verts, faces, uvs = [], [], []
    n = len(profile)
    arc = [0.0]
    for i in range(1, n):
        (r0, z0), (r1, z1) = profile[i - 1], profile[i]
        arc.append(arc[-1] + math.hypot(r1 - r0, z1 - z0))
    ru = max(r for r, _ in profile)
    for s in range(segments + 1):
        t = phase + TAU * s / segments
        for i, (r, z) in enumerate(profile):
            verts.append(c + M @ Vector((r * math.cos(t), r * math.sin(t), z)))
            uvs.append((TAU * s / segments * ru, arc[i]))
    for s in range(segments):
        for i in range(n - 1):
            q = (s * n + i, (s + 1) * n + i, (s + 1) * n + i + 1, s * n + i + 1)
            faces.append(q[::-1] if flip else q)
    return verts, faces, uvs


def add_mesh_uv(name, geo, mat, parent, smooth=True, variant=None):
    """add_mesh for (verts, faces, uvs) tuples; vertices are de-duplicated per UV seam by Blender loops."""
    v, f, uv = geo
    return add_mesh(name, v, f, mat, parent, uvs=uv, smooth=smooth, variant=variant)


def merge_geo(*geos):
    V, F, U = [], [], []
    for v, f, u in geos:
        o = len(V)
        V += list(v)
        U += list(u)
        F += [tuple(i + o for i in q) for q in f]
    return V, F, U


def capsule_profile(radius, length, dome=1.0, rings=10):
    """(r, z) profile of a capsule tank: cylinder of `length` between two elliptic domes of height
    dome*radius, starting at z=0 (bottom pole). Total height = length + 2*dome*radius."""
    prof = []
    h = dome * radius
    for i in range(rings + 1):
        a = -math.pi / 2 + (math.pi / 2) * i / rings
        prof.append((max(radius * math.cos(a), 1e-4), h + h * math.sin(a)))
    for i in range(1, rings + 1):
        a = (math.pi / 2) * i / rings
        prof.append((max(radius * math.cos(a), 1e-4), h + length + h * math.sin(a)))
    prof[0] = (1e-4, prof[0][1])
    prof[-1] = (1e-4, prof[-1][1])
    return prof


def tube_uv(p0, p1, r, sides=16, r1=None, caps=True):
    """Straight (optionally tapered) tube from p0 to p1 with metre UVs (u around, v along)."""
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    a, b = _frame(d)
    r1 = r if r1 is None else r1
    prof = ([(1e-4, 0.0)] if caps else []) + [(r, 0.0), (r1, d.length)] + ([(1e-4, d.length)] if caps else [])
    frame = [[a.x, b.x, d.normalized().x], [a.y, b.y, d.normalized().y], [a.z, b.z, d.normalized().z]]
    return lathe_geo(prof, sides, p0, frame)


def prism_geo(sides, r_flat, z0, z1, phase=None, center=(0, 0), top=True, bottom=True):
    """Vertical regular prism (octagonal deck ring etc.); r_flat = apothem. Metre UVs on the walls,
    planar UVs on the caps. Flat faces (not vertices) sit at angles phase + k*2pi/sides."""
    phase = (math.pi / sides) if phase is None else phase
    rv = r_flat / math.cos(math.pi / sides)
    cx, cy = center
    corners = [(cx + rv * math.cos(phase + TAU * (k + 0.5) / sides - math.pi / sides),
                cy + rv * math.sin(phase + TAU * (k + 0.5) / sides - math.pi / sides)) for k in range(sides)]
    V, F, U = [], [], []
    edge = 2 * rv * math.sin(math.pi / sides)
    for k in range(sides):
        (x0, y0), (x1, y1) = corners[k], corners[(k + 1) % sides]
        o = len(V)
        V += [(x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1)]
        U += [(k * edge, 0), ((k + 1) * edge, 0), ((k + 1) * edge, z1 - z0), (k * edge, z1 - z0)]
        F.append((o, o + 1, o + 2, o + 3))
    for z, on, rev in ((z1, top, False), (z0, bottom, True)):
        if not on:
            continue
        o = len(V)
        V += [(x, y, z) for x, y in corners]
        U += [(x, y) for x, y in corners]
        f = tuple(range(o, o + sides))
        F.append(f[::-1] if rev else f)
    return V, F, U


def box_uv(c, size):
    """Axis-aligned box with per-face metre UVs."""
    cx, cy, cz = c
    sx, sy, sz = (s / 2 for s in size)
    V, F, U = [], [], []
    faces = [  # (normal axis, sign) -> 4 corners
        [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], [(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)],
        [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)], [(1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)],
        [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)], [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)]]
    for quad in faces:
        o = len(V)
        for qx, qy, qz in quad:
            p = (cx + qx * sx, cy + qy * sy, cz + qz * sz)
            V.append(p)
        # project on the two axes with the largest spread
        pts = V[o:o + 4]
        spread = [max(p[i] for p in pts) - min(p[i] for p in pts) for i in range(3)]
        a, b = sorted(range(3), key=lambda i: -spread[i])[:2]
        U += [(p[a], p[b]) for p in pts]
        F.append(tuple(range(o, o + 4)))
    return V, F, U


def puff(geo, amp, scale, seed=0, axis_weight=None):
    """Displace vertices along their outward (radial) direction with smooth 3D value noise, giving a
    hand-installed MLI blanket look. axis_weight(v) can damp the effect (e.g. to pin seams)."""
    import random
    v, f, u = geo
    rnd = random.Random(seed)
    waves = [(Vector((rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1))).normalized() * rnd.uniform(0.6, 1.6) / scale,
              rnd.uniform(0, TAU)) for _ in range(9)]
    out = []
    for p in v:
        p = Vector(p)
        radial = Vector((p.x, p.y, 0))
        nrm = radial.normalized() if radial.length > 1e-6 else Vector((0, 0, 1))
        s = sum(math.sin(k.dot(p) + ph) for k, ph in waves) / 3.0
        w = axis_weight(p) if axis_weight else 1.0
        out.append(p + nrm * amp * s * w)
    return out, f, u


# ---------------------------------------------------------------- sites, thrusters, cameras

def _registry(key):
    return json.loads(bpy.context.scene.get(key, "[]"))


def _register(key, rec):
    sc = bpy.context.scene
    lst = _registry(key)
    lst.append(rec)
    sc[key] = json.dumps(lst)


def _quat_from_z(direction, up_hint=(0, 0, 1)):
    """Quaternion (w, x, y, z) whose local +Z axis points along `direction`."""
    d = Vector(direction).normalized()
    return tuple(Vector((0, 0, 1)).rotation_difference(d))


def site(name, link_ob, pos, z_axis=(0, 0, 1), kind="interface", **attrs):
    """Named frame on a link (becomes a childless Xform -> IsaacSiteAPI site in USD).

    pos is in world coordinates at the rest pose; the frame's local +Z axis points along z_axis.
    Extra attrs are written as custom USD attributes in the site's namespace (e.g. thruster:*).
    """
    e = bpy.data.objects.new(f"site:{name}", None)
    e.empty_display_type = "SINGLE_ARROW"
    e.empty_display_size = 0.25
    bpy.context.scene.collection.objects.link(e)
    q = _quat_from_z(z_axis)
    from mathutils import Matrix, Quaternion
    e.matrix_world = Matrix.Translation(Vector(pos)) @ Quaternion(q).to_matrix().to_4x4()
    mw = e.matrix_world.copy()
    e.parent = link_ob
    e.matrix_parent_inverse = link_ob.matrix_world.inverted()
    e.matrix_world = mw
    _register("sites", {"name": name, "link": link_ob["link"], "pos": list(pos), "quat": list(q),
                        "kind": kind, "attrs": attrs})
    return e


def thruster(name, link_ob, exit_pos, flow_dir, *, kind, thrust_n, isp_s, exit_radius, throat_radius=None,
             propellant, cluster=None, min_thrust_n=None, pulse=None, notes=None, confidence="estimated"):
    """Thruster plume site: origin at the nozzle-exit-plane centre, +Z along the exhaust flow
    (opposite to the thrust force on the vehicle). Feeds plume / impingement visualisation."""
    g0 = 9.80665
    attrs = {"thruster:kind": kind, "thruster:propellant": propellant, "thruster:thrustVac": thrust_n,
             "thruster:ispVac": isp_s, "thruster:massFlow": thrust_n / (isp_s * g0), "thruster:exitRadius": exit_radius,
             "thruster:confidence": confidence}
    if throat_radius:
        attrs["thruster:throatRadius"] = throat_radius
        attrs["thruster:areaRatio"] = (exit_radius / throat_radius) ** 2
    if min_thrust_n is not None:
        attrs["thruster:thrustMin"] = min_thrust_n
    if cluster:
        attrs["thruster:cluster"] = cluster
    if pulse:
        attrs["thruster:minImpulseBit"] = pulse
    if notes:
        attrs["thruster:notes"] = notes
    return site(name, link_ob, exit_pos, flow_dir, kind="thruster", **attrs)
