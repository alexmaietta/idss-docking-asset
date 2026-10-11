"""Export the IDSS assets as URDF and MJCF (MuJoCo XML), plus an MJCF docking scene with both halves.

Run: $PY scripts/export_robots.py --asset nds|ida [--format urdf,mjcf] [--tex-size 512]
     $PY scripts/export_robots.py --scene          (build/mjcf/idss_docking_scene.xml from both assets)

Inputs: the Blender export of the asset (build/<id>/export/full: visuals.usdc with the per-link visual and
collider meshes, asset.json with links, joints, loops and collider specs) and configs/<id>.py (drives,
springs, masses) - the same sources as the USD packager, so every format describes the same mechanism.

Outputs:
  build/urdf/<id>/<id>.urdf, meshes/*.obj (+ .mtl), collision/*.stl, textures/*.png
  build/mjcf/<id>/<id>.xml, meshes/*.obj, collision/*.stl, textures/*.png
  build/mjcf/idss_docking_scene.xml   (both models: IDA fixed to the world, NDS on a free 13 t host vehicle, zero g)

Modelling notes
- Contact geometry: MuJoCo (and most URDF consumers) collide convex shapes only, so every collider of
  the "exact" contact variant is exported as a convex hull, and the non-convex ones (hook profiles,
  hook profiles, seal beads) are split into convex pieces with CoACD. Two colliders are decomposed exactly
  instead, because CoACD's hull error exceeds their clearances: the capture rings (0.7 mm petal-to-chamfer
  clearance; 16 deg CoACD pieces bulge ~5 mm into the inner chamfer and wedge the petals short of the latch
  steps) as RING_SECTORS thin convex sectors of the exact revolved section, and the guide-pin receptacles
  (0.25 mm radial pin clearance) as one convex wedge per face of the chamfered obround bore, faces tangent to
  the true outline. The IDA and NDS use complementary contype/conaffinity bits, so only IDA-NDS pairs are tested.
- MJCF reference data for controllers: sites <leg>_base (on the base) and <leg>_top (on the capture ring) for
  every Stewart leg plus a <custom> numeric <leg>_rest_length; sites <latch>_catch and <latch>_pivot on each
  capture latch; the asset.json sites (docking_interface: HCS plane centre).
- MJCF keeps the Stewart platform's kinematic loops: legs 2-6 are closed onto the capture ring with
  <equality><connect> (a ball joint is a point constraint). URDF cannot express loops: there the capture
  ring hangs on a virtual 6-DOF chain (x, y, z slides + roll, pitch, yaw hinges) from the base and the
  six actuator legs are open chains; a consumer must enforce the hexapod closure itself.
- Springs (Belleville passive hooks, latch rockers, sensors, separator springs) are joint stiffness +
  springref in MJCF and <dynamics>/comments in URDF; servos (hook gangs, legs, separator locks) are MJCF
  position actuators.
"""
import json
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdShade

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import assetcfg  # noqa: E402
import package as PK  # noqa: E402
import params as P  # noqa: E402

DRIVEN = {"hook_gang1", "hook_gang2", "hook_pull1", "hook_pull2", "leg_actuator", "separator"}
SPRINGS = {"hook_passive", "latch", "sensor"}
CONTACT_BITS = {"ida": (1, 2), "nds": (2, 1)}   # (contype, conaffinity): only IDA <-> NDS pairs collide
RING_SECTORS = 180   # convex sectors per capture ring (chord bulge r (1 - cos(pi / n)): 0.09 mm at 180)


def fmt(v, nd=6):
    if isinstance(v, (int, float, np.floating)):
        v = [v]
    out = []
    for x in v:
        s = f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
        out.append("0" if s in ("-0", "") else s)
    return " ".join(out)


# ---------------------------------------------------------------- geometry from the export

def _mesh_arrays(stage, mesh_path, m):
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    M = np.array(m, dtype=np.float64)   # row-vector convention (Gf)
    pts = pts @ M[:3, :3] + M[3, :3]
    st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    uv = None
    if st and st.HasValue():
        vals = np.array(st.Get(), dtype=np.float64)
        if st.GetInterpolation() == "faceVarying" and len(vals) == len(idx):
            uv = vals
        elif len(vals) == len(pts):
            uv = vals[idx]
    tris, tuv = [], []
    o = 0
    for c in counts:   # fan-triangulate (the export is already triangulated; this is a fallback)
        for k in range(1, c - 1):
            tris.append((o, o + k, o + k + 1))
        o += c
    tris = np.array(tris, dtype=np.int64)
    corners = idx[tris]                      # (F, 3) vertex ids
    V = pts[corners.reshape(-1)]             # unwelded corners keep per-corner UVs
    UV = uv[tris.reshape(-1)] if uv is not None else None
    F = np.arange(len(V)).reshape(-1, 3)
    return V, F, UV


def write_obj(path, V, F, UV=None, mtl=None, material=None):
    with open(path, "w") as f:
        if mtl:
            f.write(f"mtllib {mtl}\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if UV is not None:
            for t in UV:
                f.write(f"vt {t[0]:.5f} {t[1]:.5f}\n")
        if material:
            f.write(f"usemtl {material}\n")
        for a, b, c in F + 1:
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n" if UV is not None else f"f {a} {b} {c}\n")


def material_info(src_stage, name, tex_dir):
    """(base-colour texture path or None, mean linear colour, roughness, metallic) of an exported material."""
    prim = src_stage.GetPrimAtPath(f"/visuals/_materials/{name}")
    tex, rough, metal, color = None, 0.5, 0.0, (0.6, 0.6, 0.6)
    if not prim:
        return tex, color, rough, metal
    for p in Usd.PrimRange(prim):
        if not p.IsA(UsdShade.Shader):
            continue
        sh = UsdShade.Shader(p)
        sid = sh.GetIdAttr().Get()
        if sid == "UsdUVTexture":
            f = os.path.basename(sh.GetInput("file").Get().path)
            if f.endswith("_basecolor.png"):
                tex = os.path.join(tex_dir, f)
        elif sid == "UsdPreviewSurface":
            for key, var in (("roughness", "rough"), ("metallic", "metal")):
                inp = sh.GetInput(key)
                if inp and inp.Get() is not None and not inp.HasConnectedSource():
                    if var == "rough":
                        rough = float(inp.Get())
                    else:
                        metal = float(inp.Get())
            dc = sh.GetInput("diffuseColor")
            if dc and dc.Get() is not None and not dc.HasConnectedSource():
                color = tuple(dc.Get())
    if tex:
        rough_png = tex.replace("_basecolor", "_roughness")
        metal_png = tex.replace("_basecolor", "_metallic")
        from PIL import Image
        if os.path.exists(rough_png):
            rough = float(np.asarray(Image.open(rough_png).convert("L"), dtype=np.float32).mean() / 255)
        if os.path.exists(metal_png):
            metal = float(np.asarray(Image.open(metal_png).convert("L"), dtype=np.float32).mean() / 255)
        img = np.asarray(Image.open(tex).convert("RGB"), dtype=np.float32) / 255
        color = tuple(float(c) for c in (img ** 2.2).mean(axis=(0, 1)))
    return tex, color, rough, metal


def srgb(c):
    return tuple(1.055 * x ** (1 / 2.4) - 0.055 if x > 0.0031308 else 12.92 * x for x in c)


class Exported:
    """The asset's export: links, joints, loops, visual meshes by material, collider pieces."""

    def __init__(self, asset):
        self.asset = asset
        self.cfg = assetcfg.load(asset)
        exp = os.path.join(assetcfg.ROOT, "build", asset, "export", "full")
        with open(os.path.join(exp, "asset.json")) as f:
            g = json.load(f)
        self.links, self.loops, self.sites = g["links"], g.get("loops", []), g["sites"]
        self.colliders = {PK.sanitize(k): v for k, v in g.get("colliders", {}).items()}
        PK.ROOT = f"/{asset}"
        self.src = PK.Source(exp, self.colliders)
        self.root = next(n for n, L in self.links.items() if L["parent"] is None)
        self.masses = PK.mass_props(self.src, self.links)
        self.tex_dir = os.path.join(exp, "textures")

    def origin(self, link):
        return np.array(self.links[link]["origin"])

    def order(self):
        return PK.dfs(self.links, self.root)

    def children(self, link):
        return sorted(n for n, L in self.links.items() if L["parent"] == link)

    def visual_parts(self, link):
        """[(material, V, F, UV)] in link-local coordinates (link frames are unrotated)."""
        out = {}
        for _g, mp, m, mat in self.src.items.get(link, []):
            V, F, UV = _mesh_arrays(self.src.stage, mp, m)
            out.setdefault(mat or "default", []).append((V, F, UV))
        merged = []
        for mat, parts in out.items():
            Vs, Fs, UVs, o = [], [], [], 0
            for V, F, UV in parts:
                Vs.append(V)
                Fs.append(F + o)
                UVs.append(UV if UV is not None else np.zeros((len(V), 2)))
                o += len(V)
            merged.append((mat, np.concatenate(Vs), np.concatenate(Fs), np.concatenate(UVs)))
        return merged

    def collision_pieces(self, link, cache_dir):
        """[(name, hull vertices, material)] convex pieces of the exact-variant colliders."""
        import trimesh
        out = []
        for name, mp, m in self.src.colliders.get(link, []):
            spec = self.colliders.get(name, {})
            if "exact" not in spec.get("variants", ["exact"]):
                continue
            V, F, _ = _mesh_arrays(self.src.stage, mp, m)
            tm = trimesh.Trimesh(V, F, process=True)
            tm.merge_vertices()
            if spec.get("model") == "convex":
                pieces = [tm.convex_hull]
            elif name.endswith("capture_ring_chamfers"):
                pieces = ring_sectors(tm.vertices, -self.origin(link)[:2])
            elif name.startswith("col_guide_receptacle_"):
                k = int(name.rsplit("_", 1)[1]) - 1
                z0 = next(s["pos"][2] for s in self.sites if s["name"] == "docking_interface")
                pieces = receptacle_pieces(*P.GUIDE_RECEPTACLES[k], z0, self.origin(link))
                check_inside(pieces, tm, name)
            else:
                pieces = coacd_cached(tm, os.path.join(cache_dir, f"{name}.npz"))
            for k, p in enumerate(pieces):
                out.append((f"{name}" + (f"_{k:02d}" if len(pieces) > 1 else ""), p, spec.get("material", "anodize")))
        return out


def ring_sectors(V, centre, n=RING_SECTORS):
    """Exact convex decomposition of a revolved convex section (the capture-ring collider mesh, 720 segments) into
    n sectors: each is the hull of the mesh's vertex columns in its angular span, so neighbours share a face."""
    import trimesh
    d = V[:, :2] - centre
    a = np.round(np.mod(np.arctan2(d[:, 1], d[:, 0]), 2 * np.pi), 4) % round(2 * np.pi, 4)   # columns 8.7e-3 apart
    cols = np.unique(a)
    step = len(cols) // n
    assert step * n == len(cols), f"{len(cols)} vertex columns do not split into {n} sectors"
    return [trimesh.Trimesh(V[np.isin(a, cols[np.arange(k * step, k * step + step + 1) % len(cols)])]).convex_hull
            for k in range(n)]


def receptacle_pieces(phi, off, z0, origin, n=16):
    """Guide-pin receptacle walls (idss.receptacle_colliders: the 30.2 x 25.2 obround bore, long axis tangential,
    with a 45 deg lead-in from the 39 mm counterbore 1 mm below the HCS plane, in a 56 mm deep block) as one convex
    wedge per bore face, out to 1.75 x the bore. The bore polygon circumscribes the true outline (never undersize)."""
    import trimesh
    L, W = P.GUIDE_RECEPT_SLOT
    r, h = W / 2, (L - W) / 2
    pts = []
    for cx, a0 in ((h, -math.pi / 2), (-h, math.pi / 2)):
        for i in range(n + 1):
            t = a0 + math.pi * i / n
            pts.append((cx + r * math.cos(t) / math.cos(math.pi / (2 * n)), r * math.sin(t) / math.cos(math.pi / (2 * n))))
    tg, rd = np.array([-math.sin(phi), math.cos(phi), 0.0]), np.array([math.cos(phi), math.sin(phi), 0.0])
    c = P.GUIDE_R * rd + off * tg - np.asarray(origin)
    top, bottom = z0 - P.GUIDE_RECEPT_RECESS, z0 - 0.056
    lead = (P.GUIDE_RECEPT_CBORE_D - W) / 2
    levels = [(top, 1.0 + 2 * lead / W), (top - lead, 1.0), (bottom, 1.0)]   # (z, bore scale)
    out = []
    for i in range(len(pts)):
        p0, p1 = pts[i], pts[(i + 1) % len(pts)]
        v = [(z, k, p) for z, k in levels for p in (p0, p1)] + [(z, 1.75, p) for z in (top, bottom) for p in (p0, p1)]
        out.append(trimesh.Trimesh(np.array([c + tg * p[0] * k + rd * p[1] * k + (0, 0, z) for z, k, p in v])).convex_hull)
    return out


def check_inside(pieces, tm, name, tol=2e-4):
    """The replacement pieces must stay inside the exact collider's bounds (catches a misplaced decomposition)."""
    lo, hi = tm.bounds
    for p in pieces:
        if np.any(p.bounds[0] < lo - tol) or np.any(p.bounds[1] > hi + tol):
            raise ValueError(f"{name}: convex piece {np.round(p.bounds, 4).tolist()} outside {np.round(tm.bounds, 4).tolist()}")


def coacd_cached(tm, path):
    """Convex decomposition (CoACD) of a non-convex collider, cached by vertex hash."""
    import trimesh
    key = f"{len(tm.vertices)}_{float(np.round(tm.vertices.sum(), 4))}"
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        if str(d["key"]) == key:
            return [trimesh.Trimesh(v, f) for v, f in zip(d["verts"], d["faces"])]
    import coacd
    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(tm.vertices, tm.faces), threshold=0.02, max_convex_hull=96,
                            preprocess_resolution=60, resolution=2000, mcts_iterations=60)
    hulls = [trimesh.Trimesh(v, f).convex_hull for v, f in parts]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, key=key, verts=np.array([h.vertices for h in hulls], dtype=object),
             faces=np.array([h.faces for h in hulls], dtype=object))
    return hulls


# ---------------------------------------------------------------- shared mesh output

def write_meshes(ex, out, tex_size):
    """Visual OBJs per link and material (with MTL + textures) and collision STLs; returns the inventory."""
    from PIL import Image
    for d in ("meshes", "collision", "textures"):
        shutil.rmtree(os.path.join(out, d), ignore_errors=True)   # no stale pieces from an earlier decomposition
        os.makedirs(os.path.join(out, d), exist_ok=True)
    inv = {"visual": {}, "collision": {}, "materials": {}}
    for link in ex.order():
        for mat, V, F, UV in ex.visual_parts(link):
            if mat not in inv["materials"]:
                tex, color, rough, metal = material_info(ex.src.stage, mat, ex.tex_dir)
                tname = None
                if tex and os.path.exists(tex):
                    tname = f"{mat}.png"
                    im = Image.open(tex).convert("RGB")
                    im.thumbnail((tex_size, tex_size))
                    im.save(os.path.join(out, "textures", tname))
                inv["materials"][mat] = {"texture": tname, "color": color, "roughness": rough, "metallic": metal}
            fname = f"{link}__{mat}.obj"
            mtl = f"{link}__{mat}.mtl"
            info = inv["materials"][mat]
            with open(os.path.join(out, "meshes", mtl), "w") as f:
                r, g, b = srgb(info["color"])
                f.write(f"newmtl {mat}\nKd {r:.4f} {g:.4f} {b:.4f}\nKs 0.5 0.5 0.5\nNs {max(2.0, (1 - info['roughness']) * 200):.1f}\n")
                if info["texture"]:
                    f.write(f"map_Kd ../textures/{info['texture']}\n")
            write_obj(os.path.join(out, "meshes", fname), V, F, UV, mtl, mat)
            inv["visual"].setdefault(link, []).append((mat, fname))
        for name, hull, pmat in ex.collision_pieces(link, os.path.join(assetcfg.ROOT, "build", ex.asset, "coacd")):
            fname = f"{name}.stl"
            hull.export(os.path.join(out, "collision", fname))
            inv["collision"].setdefault(link, []).append((name, fname, pmat))
    return inv


# ---------------------------------------------------------------- MJCF

def mjcf_model(ex, inv, prefix="", mesh_dir="meshes", free_root=True, root_pos=(0, 0, 0), root_quat=(1, 0, 0, 0),
               extra_mass=None):
    """(asset elements, worldbody body element, equality elements, actuator elements, custom elements) for one asset."""
    cfg = ex.cfg
    ctype, caff = CONTACT_BITS[ex.asset]
    assets, eqs, acts, customs = [], [], [], []
    # reference points for controllers, as sites (asset-frame point, placed on the named link)
    sites = {}
    for s in ex.sites:
        sites.setdefault(s["link"], []).append((s["name"], s["pos"], s.get("quat")))
    ring = next((n for n, L in ex.links.items() if L["joint"].get("type") == "spherical"), None)
    for link, L in ex.links.items():
        j = L["joint"]
        if "catch" in j:   # capture latch: catch-face point and rocker pivot
            sites.setdefault(link, []).extend([(f"{link}_catch", j["catch"], None), (f"{link}_pivot", j["pivot"], None)])
        if "rest_length" in j:   # Stewart leg: universal-joint centre on the base, ball centre on the ring
            sites.setdefault(ex.root, []).append((f"{link}_base", j["base"], None))
            sites.setdefault(ring, []).append((f"{link}_top", j["top"], None))
            customs.append(ET.Element("numeric", name=f"{prefix}{link}_rest_length", data=fmt(j["rest_length"], 7)))
    for mat, info in inv["materials"].items():
        if info["texture"]:
            assets.append(ET.Element("texture", name=f"{prefix}{mat}", type="2d",
                                     file=f"{mesh_dir.replace('meshes', 'textures')}/{info['texture']}"))
        a = {"name": f"{prefix}{mat}", "specular": fmt(0.25 + 0.5 * info["metallic"]),
             "shininess": fmt(max(0.02, 1 - info["roughness"])), "reflectance": fmt(0.15 * info["metallic"])}
        if info["texture"]:
            a["texture"] = f"{prefix}{mat}"
        else:
            a["rgba"] = fmt([*srgb(info["color"]), 1.0])
        assets.append(ET.Element("material", **a))
    for link, parts in inv["visual"].items():
        for mat, fname in parts:
            assets.append(ET.Element("mesh", name=f"{prefix}{fname[:-4]}", file=f"{mesh_dir}/{fname}"))
    for link, parts in inv["collision"].items():
        for name, fname, _pm in parts:
            assets.append(ET.Element("mesh", name=f"{prefix}{name}", file=f"{mesh_dir.replace('meshes', 'collision')}/{fname}"))
    fric = {k: v for k, v in cfg.PHYSICS_MATERIALS.items()}

    def body(link, parent_origin):
        L = ex.links[link]
        b = ET.Element("body", name=f"{prefix}{link}", pos=fmt(ex.origin(link) - parent_origin))
        m, com, inertia, q = ex.masses[link]
        if link == ex.root and extra_mass:
            m, com, inertia, q = extra_mass
        ET.SubElement(b, "inertial", pos=fmt(list(com)), mass=fmt(m), diaginertia=fmt(list(inertia)),
                      quat=fmt([q.GetReal(), *q.GetImaginary()]))
        j = L["joint"]
        role = j.get("role", "")
        d = j.get("drive") or {}
        if L["parent"] is None:
            if free_root:
                ET.SubElement(b, "freejoint", name=f"{prefix}{link}_free")
        elif j["type"] in ("revolute", "prismatic"):
            ja = {"name": f"{prefix}{link}", "type": "hinge" if j["type"] == "revolute" else "slide",
                  "axis": fmt(j["axis"]), "armature": fmt(d.get("armature", cfg.PASSIVE_JOINT["armature"]))}
            if "limit" in j:
                ja["range"] = fmt(j["limit"])
            spring = d.get("spring") or (d if role in SPRINGS else None)
            if spring:
                ja["stiffness"] = fmt(spring["stiffness"])
                ja["springref"] = fmt(spring.get("free_length", spring.get("target", 0.0)))
                ja["damping"] = fmt(d.get("damping", 0.0))
            elif not d:
                ja["damping"] = fmt(cfg.PASSIVE_JOINT["damping"])
            ET.SubElement(b, "joint", **ja)
            if role in DRIVEN and d:
                acts.append(ET.Element("position", name=f"{prefix}{link}_servo", joint=f"{prefix}{link}",
                                       kp=fmt(d["stiffness"]), kv=fmt(d.get("damping", 0.0)),
                                       forcerange=fmt([-d.get("max_force", 1000.0), d.get("max_force", 1000.0)]),
                                       forcelimited="true", ctrlrange=fmt(j.get("limit", [-1, 1])), ctrllimited="true"))
        elif j["type"] == "spherical":
            ET.SubElement(b, "joint", name=f"{prefix}{link}", type="ball", damping=fmt(cfg.PASSIVE_JOINT["damping"]),
                          armature=fmt(cfg.PASSIVE_JOINT["armature"]))
        for mat, fname in inv["visual"].get(link, []):
            ET.SubElement(b, "geom", type="mesh", mesh=f"{prefix}{fname[:-4]}", material=f"{prefix}{mat}",
                          contype="0", conaffinity="0", group="1", density="0")
        for name, fname, pmat in inv["collision"].get(link, []):
            mu = fric.get(pmat, (0.15, 0.12, 0))[1]
            ET.SubElement(b, "geom", type="mesh", mesh=f"{prefix}{name}", contype=str(ctype), conaffinity=str(caff),
                          group="3", density="0", friction=fmt([mu, 0.005, 0.0001]), rgba="0.9 0.3 0.2 0.4",
                          solref="0.002 1", solimp="0.95 0.99 0.0005")
        for name, pos, quat in sites.get(link, []):
            sa = {"name": f"{prefix}{name}", "pos": fmt(np.array(pos) - ex.origin(link), 7), "size": "0.004",
                  "group": "5", "rgba": "1 0.8 0 1"}
            if quat is not None and not np.allclose(quat, (1, 0, 0, 0)):
                sa["quat"] = fmt(quat, 7)
            ET.SubElement(b, "site", **sa)
        for c in ex.children(link):
            b.append(body(c, ex.origin(link)))
        return b

    root = body(ex.root, np.zeros(3) - np.array(root_pos))
    if root_quat is not None:
        root.set("quat", fmt(root_quat))
    for lp in ex.loops:   # Stewart platform: legs 2..6 closed on the capture ring (ball joint = connect)
        anchor = np.array(lp["anchor"]) - ex.origin(lp["parent"])
        eqs.append(ET.Element("connect", name=f"{prefix}{lp['name']}", body1=f"{prefix}{lp['parent']}",
                              body2=f"{prefix}{lp['child']}", anchor=fmt(anchor), solref="0.002 1"))
    return assets, root, eqs, acts, customs


def mjcf_header(name):
    m = ET.Element("mujoco", model=name)
    ET.SubElement(m, "compiler", angle="radian", autolimits="true", inertiafromgeom="false")
    ET.SubElement(m, "option", timestep="0.001", gravity="0 0 0", integrator="implicitfast", cone="elliptic",
                  impratio="10", iterations="60", ls_iterations="50")
    ET.SubElement(ET.SubElement(m, "size"), "dummy") if False else None
    vis = ET.SubElement(m, "visual")
    ET.SubElement(vis, "headlight", ambient="0.35 0.35 0.35", diffuse="0.6 0.6 0.6")
    ET.SubElement(vis, "quality", shadowsize="4096")
    return m


def write_xml(root, path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=False)


def export_mjcf(ex, out, tex_size):
    inv = write_meshes(ex, out, tex_size)
    m = mjcf_header(ex.asset)
    assets, body, eqs, acts, customs = mjcf_model(ex, inv, free_root=(ex.asset == "nds"))
    a = ET.SubElement(m, "asset")
    for e in assets:
        a.append(e)
    wb = ET.SubElement(m, "worldbody")
    ET.SubElement(wb, "light", pos="2 -2 4", dir="-0.5 0.5 -1", diffuse="1 1 1", directional="true")
    wb.append(body)
    if eqs:
        eq = ET.SubElement(m, "equality")
        for e in eqs:
            eq.append(e)
    if acts:
        ac = ET.SubElement(m, "actuator")
        for e in acts:
            ac.append(e)
    if customs:
        cu = ET.SubElement(m, "custom")
        for e in customs:
            cu.append(e)
    path = os.path.join(out, f"{ex.asset}.xml")
    write_xml(m, path)
    return path, inv


def export_scene(out_dir, gap=0.62, offset=0.04, misalign_deg=2.0, closing=0.08):
    """MJCF docking scene: IDA fixed at the origin (axis +Z), NDS mirrored about the line of androgyny on a
    free 13 t host vehicle approaching along -Z (same initial conditions as the Newton example)."""
    os.makedirs(out_dir, exist_ok=True)
    m = mjcf_header("idss_docking")
    a = ET.SubElement(m, "asset")
    wb = ET.SubElement(m, "worldbody")
    ET.SubElement(wb, "light", pos="2 -2 6", dir="-0.3 0.3 -1", diffuse="1 1 1", directional="true")
    eq = ET.SubElement(m, "equality")
    ac = ET.SubElement(m, "actuator")
    cu = ET.SubElement(m, "custom")
    for asset in ("ida", "nds"):
        rel = asset
        ex = Exported(asset)
        inv = json.load(open(os.path.join(out_dir, asset, "inventory.json")))
        if asset == "ida":
            parts = mjcf_model(ex, inv, prefix="ida_", mesh_dir=f"{rel}/meshes", free_root=False)
        else:
            mirror = Gf.Rotation(Gf.Vec3d(0, 1, 0), 180.0)
            tilt = Gf.Rotation(Gf.Vec3d(1, 0, 0), misalign_deg)
            q = (tilt * mirror).GetQuat()
            R = Gf.Matrix3d(q)
            hcs = np.array([offset, 0.0, 0.642 + gap])
            p = hcs - np.array(R.GetTranspose() * Gf.Vec3d(0, 0, 0.2304))
            veh = vehicle_mass(ex)
            parts = mjcf_model(ex, inv, prefix="nds_", mesh_dir=f"{rel}/meshes", free_root=True, root_pos=p,
                               root_quat=(q.GetReal(), *q.GetImaginary()), extra_mass=veh)
        assets, body, eqs, acts, customs = parts
        for e in customs:
            cu.append(e)
        for e in assets:
            a.append(e)
        wb.append(body)
        for e in eqs:
            eq.append(e)
        for e in acts:
            ac.append(e)
    # initial closing velocity on the vehicle (keyframe)
    path = os.path.join(out_dir, "idss_docking_scene.xml")
    write_xml(m, path)
    import mujoco
    model = mujoco.MjModel.from_xml_path(path)
    qvel = np.zeros(model.nv)
    j = model.joint("nds_nds_base_free")
    qvel[j.dofadr[0]:j.dofadr[0] + 3] = [0.0, 0.0, -closing]
    key = ET.SubElement(m, "keyframe")
    data = mujoco.MjData(model)
    ET.SubElement(key, "key", name="approach", qpos=fmt(data.qpos, 7), qvel=fmt(qvel, 7))
    write_xml(m, path)
    return path


def vehicle_mass(ex, mass=13000.0, inertia=(26000.0, 26000.0, 33000.0), com_z=-1.8):
    m0, c0, I0, q0 = ex.masses[ex.root]
    c0 = np.array(list(c0))
    cv = np.array([0, 0, com_z])
    mt = m0 + mass
    com = (m0 * c0 + mass * cv) / mt
    R = np.array(Gf.Matrix3d(Gf.Quatd(q0)).GetTranspose())
    Ib = R @ np.diag(list(I0)) @ R.T

    def shift(mm, d):
        return mm * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    It = Ib + shift(m0, c0 - com) + np.diag(inertia) + shift(mass, cv - com)
    w, v = np.linalg.eigh(It)
    if np.linalg.det(v) < 0:
        v[:, 2] *= -1
    q = Gf.Matrix3d(*v.T.flatten().tolist()).ExtractRotation().GetQuat()
    return mt, Gf.Vec3f(*com), Gf.Vec3f(*w), Gf.Quatf(q)


# ---------------------------------------------------------------- URDF

def export_urdf(ex, out, inv):
    """URDF: the joint tree with URDF joint types; the capture ring on a virtual 6-DOF chain (no loops)."""
    robot = ET.Element("robot", name=ex.asset)
    robot.append(ET.Comment(" Generated by idss scripts/export_robots.py. URDF has no kinematic loops: the NDS capture ring "
                            "rides on a virtual x/y/z/roll/pitch/yaw chain from nds_base, and the six actuator legs are "
                            "open chains (the MJCF and USD versions close the Stewart platform). "))
    for mat, info in inv["materials"].items():
        me = ET.SubElement(robot, "material", name=mat)
        ET.SubElement(me, "color", rgba=fmt([*srgb(info["color"]), 1.0]))
        if info["texture"]:
            ET.SubElement(me, "texture", filename=f"textures/{info['texture']}")

    def add_link(name, link=None):
        le = ET.SubElement(robot, "link", name=name)
        if link is None:
            ine = ET.SubElement(le, "inertial")
            ET.SubElement(ine, "mass", value="0.001")
            ET.SubElement(ine, "inertia", ixx="1e-7", iyy="1e-7", izz="1e-7", ixy="0", ixz="0", iyz="0")
            return le
        m, com, inertia, q = ex.masses[link]
        R = np.array(Gf.Matrix3d(Gf.Quatd(q)).GetTranspose())
        I = R @ np.diag(list(inertia)) @ R.T
        ine = ET.SubElement(le, "inertial")
        ET.SubElement(ine, "origin", xyz=fmt(list(com)), rpy="0 0 0")
        ET.SubElement(ine, "mass", value=fmt(m))
        ET.SubElement(ine, "inertia", ixx=fmt(I[0, 0], 8), iyy=fmt(I[1, 1], 8), izz=fmt(I[2, 2], 8),
                      ixy=fmt(I[0, 1], 8), ixz=fmt(I[0, 2], 8), iyz=fmt(I[1, 2], 8))
        for mat, fname in inv["visual"].get(link, []):
            v = ET.SubElement(le, "visual")
            ET.SubElement(ET.SubElement(v, "geometry"), "mesh", filename=f"meshes/{fname}")
            ET.SubElement(v, "material", name=mat)
        for name, fname, _pm in inv["collision"].get(link, []):
            c = ET.SubElement(le, "collision", name=name)
            ET.SubElement(ET.SubElement(c, "geometry"), "mesh", filename=f"collision/{fname}")
        return le

    def add_joint(name, jtype, parent, child, xyz, axis=None, limit=None, effort=1000.0, velocity=1.0, damping=None,
                  comment=None):
        if comment:
            robot.append(ET.Comment(f" {comment} "))
        j = ET.SubElement(robot, "joint", name=name, type=jtype)
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=child)
        ET.SubElement(j, "origin", xyz=fmt(xyz), rpy="0 0 0")
        if axis is not None:
            ET.SubElement(j, "axis", xyz=fmt(axis))
        if limit is not None:
            ET.SubElement(j, "limit", lower=fmt(limit[0]), upper=fmt(limit[1]), effort=fmt(effort), velocity=fmt(velocity))
        if damping is not None:
            ET.SubElement(j, "dynamics", damping=fmt(damping), friction="0")

    for link in ex.order():
        add_link(link, link)
        L = ex.links[link]
        if L["parent"] is None:
            continue
        j = L["joint"]
        d = j.get("drive") or {}
        role = j.get("role", "")
        xyz = ex.origin(link) - ex.origin(L["parent"])
        note = None
        spring = d.get("spring") or (d if role in SPRINGS else None)
        if spring:
            note = (f"{link}: spring k={spring['stiffness']:g}, reference "
                    f"{spring.get('free_length', spring.get('target', 0.0)):g} ({role})")
        if j["type"] in ("revolute", "prismatic"):
            add_joint(link, j["type"], L["parent"], link, xyz, j["axis"], j.get("limit", [-1, 1]),
                      effort=d.get("max_force", 100.0), velocity=1.0 if j["type"] == "prismatic" else 3.0,
                      damping=d.get("damping", ex.cfg.PASSIVE_JOINT["damping"]), comment=note)
        elif j["type"] == "spherical":   # capture ring: virtual 6-DOF chain from the base
            base, ringp = ex.root, "nds_ring_virtual"
            prev = base
            origin = ex.origin(link)
            chain = [("ring_x", "prismatic", (1, 0, 0), (-0.15, 0.15)), ("ring_y", "prismatic", (0, 1, 0), (-0.15, 0.15)),
                     ("ring_z", "prismatic", (0, 0, 1), (-0.35, 0.15)), ("ring_roll", "revolute", (1, 0, 0), (-0.3, 0.3)),
                     ("ring_pitch", "revolute", (0, 1, 0), (-0.3, 0.3)), ("ring_yaw", "revolute", (0, 0, 1), (-0.3, 0.3))]
            for k, (nm, jt, ax, lim) in enumerate(chain):
                child = link if k == len(chain) - 1 else f"{ringp}_{nm}"
                if child != link:
                    add_link(child)
                add_joint(nm, jt, prev, child, (origin - ex.origin(base)) if k == 0 else (0, 0, 0), ax, lim,
                          comment="capture ring pose (virtual chain replacing the Stewart platform closure)" if k == 0 else None)
                prev = child
        else:
            add_joint(link, "fixed", L["parent"], link, xyz)
    path = os.path.join(out, f"{ex.asset}.urdf")
    write_xml(robot, path)
    return path


# ---------------------------------------------------------------- main

def main():
    argv = sys.argv[1:]
    tex = int(assetcfg.arg(argv, "--tex-size", "512"))
    if "--scene" in argv:
        out = os.path.join(assetcfg.ROOT, "build", "mjcf")
        path = export_scene(out)
        import mujoco
        model = mujoco.MjModel.from_xml_path(path)
        print(f"scene: {model.nbody} bodies, {model.njnt} joints, {model.ngeom} geoms, {model.neq} equalities, "
              f"{model.nu} actuators -> {path}")
        return
    asset = assetcfg.arg(argv, "--asset")
    formats = assetcfg.arg(argv, "--format", "urdf,mjcf").split(",")
    ex = Exported(asset)
    mj_out = os.path.join(assetcfg.ROOT, "build", "mjcf", asset)
    path, inv = export_mjcf(ex, mj_out, tex)
    with open(os.path.join(mj_out, "inventory.json"), "w") as f:
        json.dump(inv, f)
    import mujoco
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, 100)
    print(f"mjcf {asset}: {model.nbody} bodies, {model.njnt} joints, {model.ngeom} geoms, {model.neq} loop equalities, "
          f"{model.nu} actuators, mass {sum(model.body_mass):.1f} kg; 100 steps ok -> {path}")
    if "urdf" in formats:
        u_out = os.path.join(assetcfg.ROOT, "build", "urdf", asset)
        if os.path.exists(u_out):
            shutil.rmtree(u_out)
        shutil.copytree(mj_out, u_out, ignore=shutil.ignore_patterns("*.xml", "inventory.json"))
        upath = export_urdf(ex, u_out, inv)
        print(f"urdf {asset}: -> {upath}")


if __name__ == "__main__":
    main()
