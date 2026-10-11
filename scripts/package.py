"""Package an exported IDSS asset as an Isaac Sim / SimReady-style USD asset (same layout as ~/code/lrv:
see ~/code/lrv/refs/USD_ASSET_SPEC.md), with Newton contact models for the docking interface.

Run: $PY scripts/package.py --asset nds|ida

Inputs: build/<id>/export/{full,proxy}/ (scripts/export.py). Output: build/<id>/usd/<id>/

<id>/
  <id>.usda                     interface: variant sets Physics / Fidelity / Materials / Contact, SimReady metadata
  README.md                     generated summary (links, joints, colliders, sites)
  payloads/base.usda            link tree under /<id>/Geometry, instanceable visual + collider roots, sites
  payloads/robot.usda           Isaac robot schema (links, joints, sites)
  payloads/geometries.usd       mesh data (crate): visuals, collider meshes and the medium / low visual LODs
  payloads/instances.usda       instance prototypes: mesh (+ collider APIs); no material bindings
  payloads/materials.usda       UsdPreviewSurface materials, textures in payloads/Textures
  payloads/Materials/{full,simple}.usda   looks + bindings (full: textured; simple: flat average colours)
  payloads/physics_materials.usda   friction / restitution + Newton contact stiffness per material
  payloads/Physics/{physics,physx,mujoco}.usda
  payloads/Fidelity/{full,medium,low,proxy,none}.usda

Visual variants (independent of the physics, which never changes):
  Fidelity   full    one mesh per link per material, instanced (as built)
             medium  collapse-decimated to ~35 % of the triangles, one mesh per link per material
             low     ~10 % of the triangles, merged to ONE mesh per rigid link, material GeomSubsets (sim LOD;
                     viewers that take one material per mesh, such as Newton's, show the link's main material)
             proxy   one box per part;  none: no visual geometry (headless simulation)
  Materials  full    textured UsdPreviewSurface;  simple: flat PreviewSurface from the textures' average colour,
             metallic and roughness (no textures);  none: no bindings, meshes show their displayColor

Contact models (one per collider mesh, chosen in the builder):
  sdf     NewtonSDFCollisionAPI: sparse signed-distance field at a target voxel size (non-convex parts:
          hooks, strikers, guide-pin receptacles). Newton collides SDF-vs-SDF / SDF-vs-convex.
  convex  PhysicsMeshCollisionAPI convexHull (+ NewtonMeshCollisionAPI): petals, ring sectors, HCS face
          sectors, seal sectors, pins, plungers, latch rollers.
  mesh    triangle-mesh BVH (none used by default).
Kinematic loops (the Stewart platform's ball joints 2..6) are PhysicsSphericalJoints with
physics:excludeFromArticulation = true; Newton turns them into loop constraints (MuJoCo CONNECT).
"""
import datetime
import json
import math
import os
import re
import shutil
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import assetcfg  # noqa: E402

DEG = 180.0 / math.pi
CREATOR = "idss pipeline (scripts/package.py)"
SPRING_ROLES = {"hook_passive", "latch", "sensor"}            # drive authored as a passive spring
DRIVEN_ROLES = {"hook_gang1", "hook_gang2", "hook_pull1", "hook_pull2", "leg_actuator", "separator"}   # servo actuators
PASSIVE_ROLES = {"leg_ujoint", "ring_ball"}

ROOT = None   # "/<asset_id>", set in main()


def sanitize(name):
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return s if s[0].isalpha() or s[0] == "_" else "_" + s


# ---------------------------------------------------------------- layer helpers

def new_stage(path, default_prim=True, sublayers=()):
    if os.path.exists(path):
        os.remove(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    layer = stage.GetRootLayer()
    layer.customLayerData = {"creator": CREATOR}
    for s in sublayers:
        layer.subLayerPaths.append(s)
    if default_prim:
        layer.defaultPrim = ROOT[1:]
    return stage


def apply_schemas(prim, names):
    """Prepend API schema tokens without needing the schema registered (codeless Isaac/Newton/Mjc)."""
    ops = prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
    items = []
    for lst in (ops.explicitItems, ops.prependedItems, ops.addedItems, ops.appendedItems):
        items += [i for i in lst if i not in items]
    items += [n for n in names if n not in items]
    ops = Sdf.TokenListOp()
    ops.prependedItems = items
    prim.SetMetadata("apiSchemas", ops)


def set_attr(prim, name, vtype, value, uniform=False, custom=False):
    a = prim.CreateAttribute(name, vtype, custom, Sdf.VariabilityUniform if uniform else Sdf.VariabilityVarying)
    a.Set(value)
    return a


def xform_ops(prim, translate=(0, 0, 0), orient=Gf.Quatd(1, 0, 0, 0), scale=(1, 1, 1)):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translate))
    x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(orient))
    x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))


def retarget(layer, old, new, root=None):
    old, new = Sdf.Path(old), Sdf.Path(new)

    def fix(p):
        return p.ReplacePrefix(old, new) if p.HasPrefix(old) else p

    def visit(path):
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.RelationshipSpec):
            items = list(spec.targetPathList.explicitItems) or list(spec.targetPathList.prependedItems)
            if items:
                spec.targetPathList.ClearEdits()
                spec.targetPathList.explicitItems = [fix(p) for p in items]
        elif isinstance(spec, Sdf.AttributeSpec):
            items = list(spec.connectionPathList.explicitItems) or list(spec.connectionPathList.prependedItems)
            if items:
                spec.connectionPathList.ClearEdits()
                spec.connectionPathList.explicitItems = [fix(p) for p in items]

    layer.Traverse(Sdf.Path(root) if root else Sdf.Path.absoluteRootPath, visit)


# ---------------------------------------------------------------- inputs

class Source:
    """Blender export of one LOD: per-link visual meshes and collider meshes with their local transforms."""

    def __init__(self, d, collider_names=()):
        self.dir = d
        self.layer = Sdf.Layer.FindOrOpen(os.path.join(d, "visuals.usdc"))
        self.stage = Usd.Stage.Open(self.layer)
        self.items = {}      # link -> [(geom_name, mesh_path, local matrix, material name)]
        self.colliders = {}  # link -> [(collider name, mesh_path, local matrix)]
        cset = {sanitize(n) for n in collider_names}
        for link_prim in self.stage.GetPrimAtPath("/visuals").GetChildren():
            if not link_prim.GetName().startswith("L_"):
                continue
            link = link_prim.GetName()[2:]
            for obj in link_prim.GetChildren():
                mesh = next((c for c in Usd.PrimRange(obj) if c.IsA(UsdGeom.Mesh)), None)
                if mesh is None:
                    continue
                m = UsdGeom.Xformable(obj).GetLocalTransformation()
                if mesh != obj:
                    m = UsdGeom.Xformable(mesh).GetLocalTransformation() * m
                name = sanitize(obj.GetName())
                if name in cset or name.startswith("col_"):
                    self.colliders.setdefault(link, []).append((name, mesh.GetPath(), m))
                    continue
                tgt = UsdShade.MaterialBindingAPI(mesh).GetDirectBindingRel().GetTargets()
                self.items.setdefault(link, []).append((name, mesh.GetPath(), m, tgt[0].name if tgt else None))

    def mesh_points(self, mesh_path, m):
        p = UsdGeom.Mesh(self.stage.GetPrimAtPath(mesh_path)).GetPointsAttr().Get()
        return np.array([list(m.Transform(Gf.Vec3d(*q))) for q in p]) if p else np.zeros((0, 3))

    def points(self, link, colliders=True):
        pts = []
        src = self.colliders.get(link, []) if colliders and self.colliders.get(link) else \
            [(g, mp, m) for g, mp, m, _ in self.items.get(link, [])]
        for _g, mp, m in src:
            pts.append(self.mesh_points(mp, m))
        return np.concatenate(pts) if pts else np.zeros((0, 3))


def visual_stats(src):
    nmesh = npts = ntri = 0
    for items in src.items.values():
        for _g, mp, _m, _mat in items:
            mesh = UsdGeom.Mesh(src.stage.GetPrimAtPath(mp))
            nmesh += 1
            npts += len(mesh.GetPointsAttr().Get() or [])
            ntri += sum(c - 2 for c in (mesh.GetFaceVertexCountsAttr().Get() or []))
    return (nmesh, npts, ntri)


def link_paths(links, root_link):
    paths = {}

    def path(name):
        if name not in paths:
            parent = links[name]["parent"]
            paths[name] = f"{ROOT}/Geometry/{name}" if parent is None else f"{path(parent)}/{name}"
        return paths[name]

    for n in links:
        path(n)
    return paths


def dfs(links, root):
    kids = {}
    for n, L in links.items():
        if L["parent"]:
            kids.setdefault(L["parent"], []).append(n)
    order = []

    def walk(n):
        order.append(n)
        for c in sorted(kids.get(n, [])):
            walk(c)

    walk(root)
    return order


def is_open(mesh):
    counts, idx = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
    if not counts or not idx:
        return False
    edges, o = {}, 0
    for c in counts:
        f = idx[o:o + c]
        for i in range(c):
            a, b = f[i], f[(i + 1) % c]
            k = (a, b) if a < b else (b, a)
            edges[k] = edges.get(k, 0) + 1
        o += c
    return any(v == 1 for v in edges.values())


# ---------------------------------------------------------------- payload layers

def _set_display(mesh, rgb, uniform_count=None):
    pv = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    for name in ("displayColor", "displayOpacity"):
        if pv.HasPrimvar(name):
            pv.RemovePrimvar(name)
    if uniform_count is None:
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*rgb)])
    else:
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set([Gf.Vec3f(*c) for c in rgb])


def merge_link(src, items, looks):
    """One link's visual meshes merged into a single triangle mesh (link frame) with a GeomSubset per material.
    The material with the most faces comes first, so viewers that read one colour per mesh show it."""
    parts = []
    for _g, mp, m, mat in items:
        mesh = UsdGeom.Mesh(src.stage.GetPrimAtPath(mp))
        pts = mesh.GetPointsAttr().Get() or []
        counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
        idx = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        if not pts or not counts:
            continue
        p = np.array([list(m.Transform(Gf.Vec3d(*q))) for q in pts], dtype=np.float64)
        nrm = mesh.GetNormalsAttr().Get()
        if nrm is not None and len(nrm) == len(idx):
            n = np.array([list(m.TransformDir(Gf.Vec3d(*q))) for q in nrm], dtype=np.float64)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        else:   # flat normals
            n = np.zeros((len(idx), 3))
            o = 0
            for c in counts:
                f = p[idx[o:o + c]]
                fn = np.cross(f[1] - f[0], f[2] - f[0])
                n[o:o + c] = fn / max(np.linalg.norm(fn), 1e-12)
                o += c
        st_pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("st")
        st = np.zeros((len(idx), 2))
        if st_pv and st_pv.GetInterpolation() == UsdGeom.Tokens.faceVarying:
            vals = st_pv.ComputeFlattened()
            if vals is not None and len(vals) == len(idx):
                st = np.array(vals, dtype=np.float64)
        parts.append({"mat": mat, "p": p, "counts": counts, "idx": idx, "n": n, "st": st, "open": is_open(mesh)})
    nf = {}
    for q in parts:
        nf[q["mat"]] = nf.get(q["mat"], 0) + len(q["counts"])
    parts.sort(key=lambda q: -nf[q["mat"]])
    P, C, I, N, ST, face_mat = [], [], [], [], [], []
    off = 0
    for q in parts:
        P.append(q["p"])
        C += q["counts"]
        I += [i + off for i in q["idx"]]
        N.append(q["n"])
        ST.append(q["st"])
        face_mat += [q["mat"]] * len(q["counts"])
        off += len(q["p"])
    return {"points": np.concatenate(P), "counts": C, "indices": I, "normals": np.concatenate(N),
            "st": np.concatenate(ST), "face_mat": face_mat, "open": any(q["open"] for q in parts)}


def write_lod_meshes(layer_path, lods, looks):
    """Medium (per item, as exported) and low (merged per link) visual LODs, appended to geometries.usd."""
    stage = Usd.Stage.Open(layer_path)
    layer = stage.GetRootLayer()
    stats = {}
    for lod, lsrc in lods.items():
        root = f"/Geometries/lod_{lod}"
        UsdGeom.Scope.Define(stage, root)
        nmesh = npts = ntri = 0
        if lod == "medium":
            for items in lsrc.items.values():
                for g, mp, _m, mat in items:
                    UsdGeom.Xform.Define(stage, f"{root}/{g}")
                    Sdf.CopySpec(lsrc.layer, mp, layer, Sdf.Path(f"{root}/{g}/{g}"))
                    spec = layer.GetPrimAtPath(f"{root}/{g}/{g}")
                    for rel in [r for r in spec.relationships if r.name.startswith("material:binding")]:
                        spec.RemoveProperty(rel)
                    if "apiSchemas" in spec.ListInfoKeys():
                        spec.ClearInfo("apiSchemas")
                    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{root}/{g}/{g}"))
                    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
                    if is_open(mesh):
                        mesh.CreateDoubleSidedAttr().Set(True)
                    _set_display(mesh, looks.get(mat, {}).get("diffuse", (0.6, 0.6, 0.6)))
                    counts = mesh.GetFaceVertexCountsAttr().Get()
                    nmesh, npts, ntri = nmesh + 1, npts + len(mesh.GetPointsAttr().Get()), ntri + sum(c - 2 for c in counts)
        else:
            for link, items in lsrc.items.items():
                mg = merge_link(lsrc, items, looks)
                UsdGeom.Xform.Define(stage, f"{root}/{link}")
                mesh = UsdGeom.Mesh.Define(stage, f"{root}/{link}/{link}")
                mesh.CreatePointsAttr([Gf.Vec3f(*q) for q in mg["points"]])
                mesh.CreateFaceVertexCountsAttr(mg["counts"])
                mesh.CreateFaceVertexIndicesAttr(mg["indices"])
                mesh.CreateNormalsAttr([Gf.Vec3f(*q) for q in mg["normals"]])
                mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
                UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
                    [Gf.Vec2f(*q) for q in mg["st"]])
                mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
                if mg["open"]:
                    mesh.CreateDoubleSidedAttr().Set(True)
                lo, hi = mg["points"].min(axis=0), mg["points"].max(axis=0)
                mesh.CreateExtentAttr([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])
                _set_display(mesh, [looks.get(fm, {}).get("diffuse", (0.6, 0.6, 0.6)) for fm in mg["face_mat"]],
                             uniform_count=len(mg["counts"]))
                bind = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
                for mat in dict.fromkeys(mg["face_mat"]):
                    faces = [i for i, fm in enumerate(mg["face_mat"]) if fm == mat]
                    bind.CreateMaterialBindSubset(sanitize(mat or "unbound"), Vt.IntArray(faces), UsdGeom.Tokens.face)
                nmesh, npts, ntri = nmesh + 1, npts + len(mg["points"]), ntri + sum(c - 2 for c in mg["counts"])
        stats[lod] = (nmesh, npts, ntri)
    layer.Save()
    return stats


def write_geometries(src, path, looks=None):
    stage = new_stage(path, default_prim=False)
    layer = stage.GetRootLayer()
    UsdGeom.Scope.Define(stage, "/Geometries")
    every = [(g, mp) for items in src.items.values() for g, mp, _m, _mat in items] + \
            [(g, mp) for cols in src.colliders.values() for g, mp, _m in cols]
    for g, mesh_path in every:
        UsdGeom.Xform.Define(stage, f"/Geometries/{g}")
        Sdf.CopySpec(src.layer, mesh_path, layer, Sdf.Path(f"/Geometries/{g}/{g}"))
        spec = layer.GetPrimAtPath(f"/Geometries/{g}/{g}")
        for rel in [r for r in spec.relationships if r.name.startswith("material:binding")]:
            spec.RemoveProperty(rel)
        if "apiSchemas" in spec.ListInfoKeys():
            spec.ClearInfo("apiSchemas")
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/Geometries/{g}/{g}"))
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        if not g.startswith("col_") and is_open(mesh):
            mesh.CreateDoubleSidedAttr().Set(True)
    if looks is not None:   # Materials=none shows each visual mesh in its material's average colour
        for items in src.items.values():
            for g, _mp, _m, mat in items:
                _set_display(UsdGeom.Mesh(stage.GetPrimAtPath(f"/Geometries/{g}/{g}")),
                             looks.get(mat, {}).get("diffuse", (0.6, 0.6, 0.6)))
    layer.Save()


def write_materials(src, path, tex_out):
    stage = new_stage(path, default_prim=False)
    layer = stage.GetRootLayer()
    UsdGeom.Scope.Define(stage, "/Materials")
    names = [c.name for c in src.layer.GetPrimAtPath("/visuals/_materials").nameChildren]
    for n in names:
        Sdf.CopySpec(src.layer, f"/visuals/_materials/{n}", layer, Sdf.Path(f"/Materials/{n}"))
    retarget(layer, "/visuals/_materials", "/Materials")
    os.makedirs(tex_out, exist_ok=True)
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        sh = UsdShade.Shader(prim)
        sid = sh.GetIdAttr().Get()
        if sid == "UsdPrimvarReader_float2":
            sh.GetInput("varname").Set("st")
        elif sid == "UsdUVTexture":
            f = sh.GetInput("file")
            fname = os.path.basename(f.Get().path)
            src_file = os.path.join(src.dir, "textures", fname)
            if os.path.exists(src_file):
                shutil.copy(src_file, os.path.join(tex_out, fname))
            f.Set(Sdf.AssetPath(f"./Textures/{fname}"))
            st = sh.GetInput("st")
            conn = st.GetConnectedSource() if st else None
            if conn and UsdShade.Shader(conn[0].GetPrim()).GetIdAttr().Get() == "UsdTransform2d":
                up = UsdShade.Shader(conn[0].GetPrim()).GetInput("in").GetConnectedSource()
                if up:
                    st.ConnectToSource(up[0], up[1])
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdShade.Shader) and UsdShade.Shader(prim).GetIdAttr().Get() == "UsdTransform2d":
            stage.RemovePrim(prim.GetPath())
    layer.Save()
    return names


def _texture_mean(path, srgb=False, channel=None):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    if srgb:
        a = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    m = a.reshape(-1, 3).mean(axis=0)
    return float(m[0]) if channel == "r" else tuple(float(x) for x in m)


def material_looks(materials_path):
    """Per material: the flat look (linear diffuse colour, metallic, roughness, opacity) its textures average to."""
    stage = Usd.Stage.Open(materials_path)
    looks = {}
    for mat in [p for p in stage.Traverse() if p.IsA(UsdShade.Material)]:
        surf = next((UsdShade.Shader(p) for p in Usd.PrimRange(mat) if p.IsA(UsdShade.Shader)
                     and UsdShade.Shader(p).GetIdAttr().Get() == "UsdPreviewSurface"), None)
        look = {"diffuse": (0.6, 0.6, 0.6), "metallic": 0.0, "roughness": 0.5, "opacity": 1.0}
        if surf is not None:
            for key, name in (("diffuse", "diffuseColor"), ("metallic", "metallic"), ("roughness", "roughness"),
                              ("opacity", "opacity")):
                inp = surf.GetInput(name)
                if not inp:
                    continue
                if inp.HasConnectedSource():
                    src, out, _ = inp.GetConnectedSource()
                    tex = UsdShade.Shader(src.GetPrim())
                    f = tex.GetInput("file").Get() if tex.GetInput("file") else None
                    if f is not None and f.resolvedPath and os.path.exists(f.resolvedPath):
                        srgb = tex.GetInput("sourceColorSpace").Get() == "sRGB" if tex.GetInput("sourceColorSpace") else False
                        look[key] = _texture_mean(f.resolvedPath, srgb, None if key == "diffuse" else "r")
                elif inp.Get() is not None:
                    v = inp.Get()
                    look[key] = tuple(float(x) for x in v) if key == "diffuse" else float(v)
        looks[mat.GetName()] = look
    return looks


def write_physics_materials(path, cfg):
    stage = new_stage(path, default_prim=False)
    UsdGeom.Scope.Define(stage, "/PhysicsMaterials")
    contact = getattr(cfg, "CONTACT", {})
    for name, (mu_s, mu_d, e) in cfg.PHYSICS_MATERIALS.items():
        m = UsdShade.Material.Define(stage, f"/PhysicsMaterials/{name}")
        api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        api.CreateStaticFrictionAttr(mu_s)
        api.CreateDynamicFrictionAttr(mu_d)
        api.CreateRestitutionAttr(e)
        apply_schemas(m.GetPrim(), ["NewtonMaterialAPI"])
        set_attr(m.GetPrim(), "newton:torsionalFriction", Sdf.ValueTypeNames.Float, 0.002)
        set_attr(m.GetPrim(), "newton:rollingFriction", Sdf.ValueTypeNames.Float, 0.0001)
        ke = contact.get("ke_" + name, contact.get("ke"))
        kd = contact.get("kd_" + name, contact.get("kd"))
        if ke:
            set_attr(m.GetPrim(), "newton:contactStiffness", Sdf.ValueTypeNames.Float, float(ke))
        if kd:
            set_attr(m.GetPrim(), "newton:contactDamping", Sdf.ValueTypeNames.Float, float(kd))
    stage.GetRootLayer().Save()


def write_instances(src, path, colliders, cfg):
    t = Sdf.ValueTypeNames
    stage = new_stage(path, default_prim=False)
    UsdGeom.Scope.Define(stage, "/Instances")
    contact = getattr(cfg, "CONTACT", {})
    for items in src.items.values():
        for g, _mp, _m, mat in items:
            inst = UsdGeom.Xform.Define(stage, f"/Instances/{g}").GetPrim()
            inst.GetReferences().AddReference("./geometries.usd", f"/Geometries/{g}")
            # the look is bound on the instance prim by the Materials variant (bindings inside a prototype could
            # not be switched)
    for cols in src.colliders.values():
        for name, _mp, _m in cols:
            spec = colliders.get(name, {"model": "convex", "material": "anodize"})
            inst = UsdGeom.Xform.Define(stage, f"/Instances/{name}").GetPrim()
            inst.GetReferences().AddReference("./geometries.usd", f"/Geometries/{name}")
            mesh = stage.OverridePrim(f"/Instances/{name}/{name}")
            UsdPhysics.CollisionAPI.Apply(mesh)
            apis = ["NewtonCollisionAPI"]
            if spec["model"] == "sdf":
                apis.append("NewtonSDFCollisionAPI")
                apply_schemas(mesh, apis)
                UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(UsdPhysics.Tokens.none)
                set_attr(mesh, "newton:sdfTargetVoxelSize", t.Float, float(spec.get("voxel") or 0.002), uniform=True)
                band = max(4 * float(spec.get("voxel") or 0.002), 0.004)
                set_attr(mesh, "newton:sdfNarrowBandInner", t.Float, -band, uniform=True)
                set_attr(mesh, "newton:sdfNarrowBandOuter", t.Float, band, uniform=True)
                if spec.get("hydro"):
                    set_attr(mesh, "newton:hydroelasticEnabled", t.Bool, True)
            else:
                apis.append("NewtonMeshCollisionAPI")
                apply_schemas(mesh, apis)
                approx = UsdPhysics.Tokens.convexHull if spec["model"] == "convex" else UsdPhysics.Tokens.none
                UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(approx)
            set_attr(mesh, "newton:contactMargin", t.Float, float(spec.get("margin") or contact.get("margin", 0.0)))
            if spec.get("solref"):   # explicit MuJoCo contact impedance (time constant, damping ratio)
                set_attr(mesh, "mjc:solref", t.Float2, Gf.Vec2f(*spec["solref"]))
            set_attr(mesh, "newton:contactGap", t.Float, float(contact.get("gap", 0.004)))
            UsdGeom.Imageable(mesh).CreatePurposeAttr(UsdGeom.Tokens.guide)
            pmat = spec.get("material", "anodize")
            UsdGeom.Scope.Define(stage, f"/Instances/{name}/PhysicsMaterials")
            pm = stage.DefinePrim(f"/Instances/{name}/PhysicsMaterials/{pmat}", "Material")
            pm.GetReferences().AddReference("./physics_materials.usda", f"/PhysicsMaterials/{pmat}")
            UsdShade.MaterialBindingAPI.Apply(mesh).Bind(UsdShade.Material(pm), UsdShade.Tokens.weakerThanDescendants,
                                                       "physics")
            if spec["model"] == "sdf" and "hydro" in spec.get("variants", ["hydro"]):
                # hydroelastic twin of the prototype (selected by the Contact=hydro variant)
                h = stage.DefinePrim(f"/Instances/{name}_hydro", "Xform")
                h.GetReferences().AddInternalReference(f"/Instances/{name}")
                hm = stage.OverridePrim(f"/Instances/{name}_hydro/{name}")
                set_attr(hm, "newton:hydroelasticEnabled", t.Bool, True)
                set_attr(hm, "newton:hydroelasticStiffness", t.Float, float(contact.get("kh", 1.0e10)))
    stage.GetRootLayer().Save()


def _site_xform(site, links):
    o = Gf.Vec3d(*links[site["link"]]["origin"])
    w, x, y, z = site["quat"]
    return Gf.Vec3d(*site["pos"]) - o, Gf.Quatd(w, x, y, z)


def write_base(src, path, links, paths, sites, cfg):
    t = Sdf.ValueTypeNames
    stage = new_stage(path, sublayers=["./robot.usda"])
    root = UsdGeom.Xform.Define(stage, ROOT).GetPrim()
    root.SetMetadata("kind", "component")
    root.SetMetadata("assetInfo", {"name": ROOT[1:], "version": cfg.VERSION, "identifier": Sdf.AssetPath(f"./{ROOT[1:]}.usda")})
    root.SetMetadata("documentation", cfg.DOC)
    apply_schemas(root, ["GeomModelAPI", "SemanticsLabelsAPI:class"])
    set_attr(root, "semantics:labels:class", t.TokenArray, cfg.SEMANTIC_CLASS)
    UsdGeom.Scope.Define(stage, f"{ROOT}/Geometry")
    UsdGeom.Scope.Define(stage, f"{ROOT}/Physics")
    UsdGeom.Scope.Define(stage, f"{ROOT}/Materials")
    for name in paths:
        L = links[name]
        po = Gf.Vec3d(*links[L["parent"]]["origin"]) if L["parent"] else Gf.Vec3d(0, 0, 0)
        prim = UsdGeom.Xform.Define(stage, paths[name]).GetPrim()
        xform_ops(prim, Gf.Vec3d(*L["origin"]) - po)
        for g, _mp, m, _mat in src.items.get(name, []):
            inst = stage.DefinePrim(f"{paths[name]}/{g}", "Xform")
            inst.GetReferences().AddReference("./instances.usda", f"/Instances/{g}")
            inst.SetInstanceable(True)
            tf = Gf.Transform(m)
            xform_ops(inst, tf.GetTranslation(), tf.GetRotation().GetQuat(), tf.GetScale())
        if src.colliders.get(name):
            holder = UsdGeom.Scope.Define(stage, f"{paths[name]}/colliders").GetPrim()
            UsdGeom.Imageable(holder).CreatePurposeAttr(UsdGeom.Tokens.guide)
            for cname, _mp, m in src.colliders[name]:
                inst = stage.DefinePrim(f"{paths[name]}/colliders/{cname}", "Xform")
                inst.GetReferences().AddReference("./instances.usda", f"/Instances/{cname}")
                inst.SetInstanceable(True)
                tf = Gf.Transform(m)
                xform_ops(inst, tf.GetTranslation(), tf.GetRotation().GetQuat(), tf.GetScale())
    for s in sites:   # childless Xforms -> IsaacSiteAPI / NewtonSiteAPI
        prim = UsdGeom.Xform.Define(stage, f"{paths[s['link']]}/{s['name']}").GetPrim()
        pos, q = _site_xform(s, links)
        xform_ops(prim, pos, q)
        set_attr(prim, "site:kind", t.Token, s["kind"], custom=True)
        for k, v in s["attrs"].items():
            key = k if ":" in k else f"site:{k}"
            vt = t.Double if isinstance(v, float) else t.Int if isinstance(v, int) and not isinstance(v, bool) else t.String
            set_attr(prim, key, vt, v, custom=True)
    stage.GetRootLayer().Save()


def write_robot(path, links, paths, sites, joints, loops, root_link, cfg):
    t = Sdf.ValueTypeNames
    stage = new_stage(path)
    root = stage.OverridePrim(ROOT)
    apply_schemas(root, ["IsaacRobotAPI"])
    set_attr(root, "isaac:robotType", t.Token, cfg.ROBOT_TYPE)
    set_attr(root, "isaac:description", t.String, cfg.TITLE)
    set_attr(root, "isaac:namespace", t.String, ROOT[1:])
    set_attr(root, "isaac:license", t.Token, "Other")
    set_attr(root, "isaac:version", t.String, cfg.VERSION)
    set_attr(root, "isaac:source", t.String, CREATOR)
    by_link = {}
    for s in sites:
        by_link.setdefault(s["link"], []).append(f"{paths[s['link']]}/{s['name']}")
    order = []
    for n in dfs(links, root_link):
        order.append(paths[n])
        order += by_link.get(n, [])
    rl = root.CreateRelationship("isaac:physics:robotLinks", False)
    rj = root.CreateRelationship("isaac:physics:robotJoints", False)
    for rel, targets in ((rl, order), (rj, [f"{ROOT}/Physics/{j}" for j in list(joints) + [lp["name"] for lp in loops]])):
        for tgt in targets:
            rel.AddTarget(Sdf.Path(tgt))
    for n in links:
        apply_schemas(stage.OverridePrim(paths[n]), ["IsaacLinkAPI"])
    for s in sites:
        sp = stage.OverridePrim(f"{paths[s['link']]}/{s['name']}")
        apply_schemas(sp, ["IsaacSiteAPI", "NewtonSiteAPI"])
        set_attr(sp, "isaac:Description", t.String, s["attrs"].get("description", s["kind"]))
        set_attr(sp, "isaac:forwardAxis", t.Token, "Z", uniform=True)
    for j, spec in joints.items():
        jp = stage.OverridePrim(f"{ROOT}/Physics/{j}")
        apply_schemas(jp, ["IsaacJointAPI"])
        set_attr(jp, "isaac:actuator", t.BoolArray, [spec.get("role") in DRIVEN_ROLES])
    stage.GetRootLayer().Save()


def _inertia_from_points(pts, mass):
    """Mass properties of the convex hull of `pts` (uniform density) scaled to `mass`:
    (com, principal moments, principal-axes quaternion (w, x, y, z))."""
    import trimesh
    if len(pts) < 4:
        return Gf.Vec3f(0, 0, 0), Gf.Vec3f(1e-4, 1e-4, 1e-4), Gf.Quatf(1, 0, 0, 0)
    try:
        hull = trimesh.convex.convex_hull(pts)
        vol = hull.volume
        if vol < 1e-12:
            raise ValueError
        com = hull.center_mass
        I = hull.moment_inertia * (mass / vol)       # moment_inertia is for unit density about com
    except Exception:
        lo, hi = pts.min(0), pts.max(0)
        com = (lo + hi) / 2
        s = np.maximum(hi - lo, 0.01)
        I = np.diag([mass / 12 * (s[1] ** 2 + s[2] ** 2), mass / 12 * (s[0] ** 2 + s[2] ** 2),
                     mass / 12 * (s[0] ** 2 + s[1] ** 2)])
    w, v = np.linalg.eigh(I)
    if np.linalg.det(v) < 0:
        v[:, 2] *= -1
    w = np.maximum(w, 1e-6)
    q = Gf.Matrix3d(*v.T.flatten().tolist()).ExtractRotation().GetQuat()
    return Gf.Vec3f(*com), Gf.Vec3f(*w), Gf.Quatf(q)


def mass_props(src, links):
    out = {}
    for name, L in links.items():
        m = float(L.get("mass") or 1.0)
        pts = src.points(name, colliders=False)
        if len(pts) == 0:
            pts = src.points(name, colliders=True)
        if len(pts) > 20000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]
        out[name] = (m, *_inertia_from_points(pts, m))
    return out


def _joint_frame(axis):
    return Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), Gf.Vec3d(*axis)).GetQuat())


def write_physics(path, links, paths, masses, loops, root_link, cfg):
    t = Sdf.ValueTypeNames
    stage = new_stage(path)
    for name in dfs(links, root_link):
        prim = stage.OverridePrim(paths[name])
        apis = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        if name == root_link:
            apis += ["PhysicsArticulationRootAPI", "NewtonArticulationRootAPI"]
        apply_schemas(prim, apis)
        m, com, inertia, q = masses[name]
        set_attr(prim, "physics:mass", t.Float, m)
        set_attr(prim, "physics:centerOfMass", t.Point3f, com)
        set_attr(prim, "physics:diagonalInertia", t.Float3, inertia)
        set_attr(prim, "physics:principalAxes", t.Quatf, q)
        if name == root_link:
            set_attr(prim, "newton:selfCollisionEnabled", t.Bool, False)
    stage.OverridePrim(f"{ROOT}/Physics")
    joints = {}
    passive = getattr(cfg, "PASSIVE_JOINT", {})
    for name in dfs(links, root_link)[1:]:
        L = links[name]
        spec = L["joint"]
        jt = spec["type"]
        kind = {"revolute": "PhysicsRevoluteJoint", "prismatic": "PhysicsPrismaticJoint", "fixed": "PhysicsFixedJoint",
                "spherical": "PhysicsSphericalJoint"}[jt]
        prim = stage.DefinePrim(f"{ROOT}/Physics/{name}", kind)
        prim.CreateRelationship("physics:body0", False).SetTargets([Sdf.Path(paths[L["parent"]])])
        prim.CreateRelationship("physics:body1", False).SetTargets([Sdf.Path(paths[name])])
        po, co = Gf.Vec3d(*links[L["parent"]]["origin"]), Gf.Vec3d(*L["origin"])
        rot = _joint_frame(spec.get("axis", (1, 0, 0))) if jt in ("revolute", "prismatic") else Gf.Quatf(1, 0, 0, 0)
        set_attr(prim, "physics:localPos0", t.Point3f, Gf.Vec3f(co - po))
        set_attr(prim, "physics:localPos1", t.Point3f, Gf.Vec3f(0, 0, 0))
        set_attr(prim, "physics:localRot0", t.Quatf, rot)
        set_attr(prim, "physics:localRot1", t.Quatf, rot)
        role = spec.get("role", "")
        set_attr(prim, "idss:role", t.String, role, custom=True)
        joints[name] = spec
        if jt in ("fixed",):
            continue
        apply_schemas(prim, ["NewtonJointAPI"])
        d = spec.get("drive") or {}
        set_attr(prim, "newton:armature", t.Float, float(d.get("armature", passive.get("armature", 0.0))))
        set_attr(prim, "newton:damping", t.Float, float(passive.get("damping", 0.0)) if not d else 0.0)
        if jt == "spherical":
            continue
        set_attr(prim, "physics:axis", t.Token, "X", uniform=True)
        angular = jt == "revolute"
        sc = DEG if angular else 1.0
        if "limit" in spec:
            lo, hi = spec["limit"]
            set_attr(prim, "physics:lowerLimit", t.Float, lo * sc)
            set_attr(prim, "physics:upperLimit", t.Float, hi * sc)
        tok = "angular" if angular else "linear"
        apply_schemas(prim, [f"PhysicsJointStateAPI:{tok}"])
        set_attr(prim, f"state:{tok}:physics:position", t.Float, 0.0)
        set_attr(prim, f"state:{tok}:physics:velocity", t.Float, 0.0)
        for k in ("rest_length", "base", "top", "catch", "roller", "pivot"):   # mechanism geometry for controllers
            if k in spec:   # (asset frame at the build pose)
                v = spec[k]
                set_attr(prim, f"idss:{k.replace('_l', 'L')}", t.Float3 if isinstance(v, list) else t.Float,
                         Gf.Vec3f(*v) if isinstance(v, list) else float(v), custom=True)
        if "spring" in d:
            set_attr(prim, "idss:springStiffness", t.Float, float(d["spring"]["stiffness"]), custom=True)
            set_attr(prim, "idss:springFreeLength", t.Float, float(d["spring"]["free_length"]), custom=True)
        if not d:
            continue
        spring = d.get("spring") or (d if role in SPRING_ROLES else None)
        if spring:   # MuJoCo passive spring: the reference may lie beyond the hard stop (preload)
            apply_schemas(prim, ["MjcJointAPI"])
            set_attr(prim, "mjc:stiffness", t.Float, float(spring["stiffness"]))
            set_attr(prim, "mjc:springref", t.Float, float(spring.get("free_length", spring.get("target", 0.0))))
            set_attr(prim, "newton:damping", t.Float, float(d.get("damping", 0.0)))
            if role in SPRING_ROLES:
                continue
        apply_schemas(prim, [f"PhysicsDriveAPI:{tok}"])
        base = f"drive:{tok}:physics:"
        set_attr(prim, base + "type", t.Token, "force", uniform=True)
        set_attr(prim, base + "stiffness", t.Float, d.get("stiffness", 0.0) / sc)
        set_attr(prim, base + "damping", t.Float, d.get("damping", 0.0) / sc)
        set_attr(prim, base + "maxForce", t.Float, d.get("max_force", 1000.0))
        set_attr(prim, base + "targetPosition", t.Float, d.get("target", 0.0) * sc)
        set_attr(prim, base + "targetVelocity", t.Float, 0.0)
    for lp in loops:   # Stewart-platform ball joints 2..6: loop closures outside the articulation tree
        prim = stage.DefinePrim(f"{ROOT}/Physics/{lp['name']}", "PhysicsSphericalJoint" if lp["type"] == "spherical"
                                else "PhysicsRevoluteJoint")
        prim.CreateRelationship("physics:body0", False).SetTargets([Sdf.Path(paths[lp["parent"]])])
        prim.CreateRelationship("physics:body1", False).SetTargets([Sdf.Path(paths[lp["child"]])])
        a = Gf.Vec3d(*lp["anchor"])
        set_attr(prim, "physics:localPos0", t.Point3f, Gf.Vec3f(a - Gf.Vec3d(*links[lp["parent"]]["origin"])))
        set_attr(prim, "physics:localPos1", t.Point3f, Gf.Vec3f(a - Gf.Vec3d(*links[lp["child"]]["origin"])))
        set_attr(prim, "physics:localRot0", t.Quatf, Gf.Quatf(1, 0, 0, 0))
        set_attr(prim, "physics:localRot1", t.Quatf, Gf.Quatf(1, 0, 0, 0))
        set_attr(prim, "physics:excludeFromArticulation", t.Bool, True, uniform=True)
        set_attr(prim, "idss:role", t.String, "loop_ball", custom=True)
    stage.GetRootLayer().Save()
    return joints


def write_physx(path, joints, paths, root_link):
    t = Sdf.ValueTypeNames
    stage = new_stage(path, sublayers=["./physics.usda"])
    ch = stage.OverridePrim(paths[root_link])
    apply_schemas(ch, ["PhysxArticulationAPI"])
    set_attr(ch, "physxArticulation:enabledSelfCollisions", t.Bool, False)
    set_attr(ch, "physxArticulation:solverPositionIterationCount", t.Int, 64)
    set_attr(ch, "physxArticulation:solverVelocityIterationCount", t.Int, 4)
    for j, spec in joints.items():
        if spec["type"] in ("fixed", "spherical"):
            continue
        jp = stage.OverridePrim(f"{ROOT}/Physics/{j}")
        apply_schemas(jp, ["PhysxJointAPI"])
        d = spec.get("drive") or {}
        if spec.get("role") in SPRING_ROLES and d:   # PhysX has no passive joint spring: use a drive
            tok = "angular" if spec["type"] == "revolute" else "linear"
            sc = DEG if tok == "angular" else 1.0
            apply_schemas(jp, [f"PhysicsDriveAPI:{tok}"])
            base = f"drive:{tok}:physics:"
            set_attr(jp, base + "type", t.Token, "force", uniform=True)
            set_attr(jp, base + "stiffness", t.Float, d["stiffness"] / sc)
            set_attr(jp, base + "damping", t.Float, d.get("damping", 0.0) / sc)
            set_attr(jp, base + "maxForce", t.Float, d.get("max_force", 1000.0))
            set_attr(jp, base + "targetPosition", t.Float, d.get("target", 0.0) * sc)
        set_attr(jp, "physxJoint:maxJointVelocity", t.Float, 1.0 if spec["type"] == "prismatic" else 360.0)
    stage.GetRootLayer().Save()


def write_mujoco(path, joints):
    """MuJoCo layer (also what Isaac Sim's Newton backend selects): position actuators for the servos
    (hook gangs, Stewart legs, separator locks); springs keep their PhysicsDriveAPI."""
    t = Sdf.ValueTypeNames
    stage = new_stage(path, sublayers=["./physics.usda"])
    for j, spec in joints.items():
        if spec["type"] in ("fixed", "spherical"):
            continue
        jp = stage.OverridePrim(f"{ROOT}/Physics/{j}")
        apply_schemas(jp, ["MjcJointAPI"])
        d = spec.get("drive")
        set_attr(jp, "mjc:armature", t.Double, float((d or {}).get("armature", 0.01)), uniform=True)
        if not d or spec.get("role") not in DRIVEN_ROLES:
            continue
        tok = "angular" if spec["type"] == "revolute" else "linear"
        jp.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(
            deletedItems=[f"PhysicsDriveAPI:{tok}", f"PhysicsJointStateAPI:{tok}"], prependedItems=["MjcJointAPI"]))
        act = stage.DefinePrim(f"{ROOT}/Physics/{j}_actuator", "MjcActuator")
        act.CreateRelationship("mjc:target", False).SetTargets([jp.GetPath()])
        f = d.get("max_force", 1000.0)
        set_attr(act, "mjc:forceRange:min", t.Double, -f, uniform=True)
        set_attr(act, "mjc:forceRange:max", t.Double, f, uniform=True)
        set_attr(act, "mjc:forceLimited", t.Token, "true", uniform=True)
        kp, kd = d.get("stiffness", 0.0), d.get("damping", 0.0)
        gain, bias = ([kd], [0.0, 0.0, -kd]) if kp == 0.0 else ([kp], [0.0, -kp, -kd])
        set_attr(act, "mjc:gainPrm", t.DoubleArray, gain + [0.0] * (10 - len(gain)), uniform=True)
        set_attr(act, "mjc:biasType", t.Token, "affine", uniform=True)
        set_attr(act, "mjc:biasPrm", t.DoubleArray, bias + [0.0] * (10 - len(bias)), uniform=True)
    stage.GetRootLayer().Save()


def write_contact(out_dir, src, colliders, paths):
    """Contact variant set: exact (convex hulls + triangle meshes for the sub-millimetre fits, SDF hooks),
    sdf (every collider a sparse SDF) and hydro (the SDF set with hydroelastic pressure-field contact)."""
    for variant in ("exact", "sdf", "hydro"):
        st = new_stage(os.path.join(out_dir, f"{variant}.usda"))
        st.OverridePrim(ROOT)
        for link, cols in src.colliders.items():
            for name, _mp, m in cols:
                spec = colliders.get(name, {})
                inst = f"{paths[link]}/colliders/{name}"
                if variant not in spec.get("variants", ["exact", "sdf", "hydro"]):
                    st.OverridePrim(inst).SetActive(False)
                elif variant == "hydro" and spec.get("model") == "sdf":
                    st.OverridePrim(inst).SetActive(False)
                    h = st.DefinePrim(f"{inst}_hydro", "Xform")
                    h.GetReferences().AddReference("../instances.usda", f"/Instances/{name}_hydro")
                    h.SetInstanceable(True)
                    tf = Gf.Transform(m)
                    xform_ops(h, tf.GetTranslation(), tf.GetRotation().GetQuat(), tf.GetScale())
        st.GetRootLayer().Save()


def write_material_variants(out_dir, src, lods, paths, looks):
    """Materials/full.usda (textured looks) and Materials/simple.usda (flat looks), each binding the full visual
    instances, the medium instances and the low LOD's material subsets. Materials=none authors nothing."""
    t = Sdf.ValueTypeNames
    targets = []   # (prim path, material)
    for link, items in src.items.items():
        targets += [(f"{paths[link]}/{g}", mat) for g, _mp, _m, mat in items if mat]
    if "medium" in lods:
        for link, items in lods["medium"].items.items():
            targets += [(f"{paths[link]}/lod_medium/{g}", mat) for g, _mp, _m, mat in items if mat]
    if "low" in lods:
        for link, items in lods["low"].items.items():
            for mat in dict.fromkeys(m for *_x, m in items):
                if mat:
                    targets.append((f"{paths[link]}/lod_low/{link}/{link}/{sanitize(mat)}", mat))
    for variant in ("full", "simple"):
        st = new_stage(os.path.join(out_dir, f"{variant}.usda"))
        st.OverridePrim(ROOT)
        UsdGeom.Scope.Define(st, f"{ROOT}/Looks")
        for mat, look in looks.items():
            mp = st.DefinePrim(f"{ROOT}/Looks/{mat}", "Material")
            if variant == "full":
                mp.GetReferences().AddReference("../materials.usda", f"/Materials/{mat}")
                continue
            sh = UsdShade.Shader.Define(st, f"{ROOT}/Looks/{mat}/PreviewSurface")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", t.Color3f).Set(Gf.Vec3f(*look["diffuse"]))
            sh.CreateInput("metallic", t.Float).Set(float(look["metallic"]))
            sh.CreateInput("roughness", t.Float).Set(float(look["roughness"]))
            sh.CreateInput("opacity", t.Float).Set(float(look["opacity"]))
            UsdShade.Material(mp).CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        for path, mat in targets:
            prim = st.OverridePrim(path)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(st.GetPrimAtPath(f"{ROOT}/Looks/{mat}")))
        st.GetRootLayer().Save()


def write_fidelity(out_dir, src, proxy, paths, lods=None):
    full = new_stage(os.path.join(out_dir, "full.usda"))
    full.OverridePrim(ROOT)
    full.GetRootLayer().Save()

    def hide_full(st):
        st.OverridePrim(ROOT)
        for link, items in src.items.items():
            for g, *_ in items:
                st.OverridePrim(f"{paths[link]}/{g}").SetActive(False)

    none = new_stage(os.path.join(out_dir, "none.usda"))   # no visual geometry (headless simulation)
    hide_full(none)
    none.GetRootLayer().Save()
    for lod, lsrc in (lods or {}).items():
        st = new_stage(os.path.join(out_dir, f"{lod}.usda"))
        hide_full(st)
        for link, items in lsrc.items.items():
            holder = f"{paths[link]}/lod_{lod}"
            UsdGeom.Xform.Define(st, holder)
            if lod == "medium":   # instanced like the full LOD, one prototype per link and material
                for g, _mp, m, _mat in items:
                    inst = st.DefinePrim(f"{holder}/{g}", "Xform")
                    inst.GetReferences().AddReference("../geometries.usd", f"/Geometries/lod_medium/{g}")
                    inst.SetInstanceable(True)
                    tf = Gf.Transform(m)
                    xform_ops(inst, tf.GetTranslation(), tf.GetRotation().GetQuat(), tf.GetScale())
            else:                 # one merged mesh per link (already in the link frame); its subsets are bound
                prim = st.DefinePrim(f"{holder}/{link}", "Xform")
                prim.GetReferences().AddReference("../geometries.usd", f"/Geometries/lod_{lod}/{link}")
        st.GetRootLayer().Save()
    st = new_stage(os.path.join(out_dir, "proxy.usda"))
    layer = st.GetRootLayer()
    st.OverridePrim(ROOT)
    for link, items in src.items.items():
        for g, *_ in items:
            st.OverridePrim(f"{paths[link]}/{g}").SetActive(False)
    UsdGeom.Scope.Define(st, f"{ROOT}/ProxyLooks")
    for m in proxy.layer.GetPrimAtPath("/visuals/_materials").nameChildren:
        Sdf.CopySpec(proxy.layer, m.path, layer, Sdf.Path(f"{ROOT}/ProxyLooks/{m.name}"))
    for link, items in proxy.items.items():
        holder = f"{paths[link]}/proxy_visuals"
        UsdGeom.Xform.Define(st, holder)
        for g, mesh_path, m, _mat in items:
            Sdf.CopySpec(proxy.layer, mesh_path, layer, Sdf.Path(f"{holder}/{g}"))
            tf = Gf.Transform(m)
            xform_ops(st.GetPrimAtPath(f"{holder}/{g}"), tf.GetTranslation(), tf.GetRotation().GetQuat(), tf.GetScale())
    retarget(layer, "/visuals/_materials", f"{ROOT}/ProxyLooks")
    layer.Save()


def write_interface(path, cfg, links, extents, default_physics="physics"):
    stage = new_stage(path)
    layer = stage.GetRootLayer()
    root = stage.DefinePrim(ROOT, "Xform")
    root.GetReferences().AddReference("./payloads/base.usda")
    sets = [
        ("Physics", {"none": None, "physics": "Physics/physics.usda", "physx": "Physics/physx.usda",
                     "mujoco": "Physics/mujoco.usda"}, default_physics),
        ("Fidelity", {"full": "Fidelity/full.usda", "medium": "Fidelity/medium.usda", "low": "Fidelity/low.usda",
                      "proxy": "Fidelity/proxy.usda", "none": "Fidelity/none.usda"}, "full"),
        ("Materials", {"full": "Materials/full.usda", "simple": "Materials/simple.usda", "none": None}, "full"),
        ("Contact", {"exact": "Contact/exact.usda", "sdf": "Contact/sdf.usda", "hydro": "Contact/hydro.usda"}, "exact"),
    ]
    for set_name, options, default in sets:
        vs = root.GetVariantSets().AddVariantSet(set_name)
        for opt, payload in options.items():
            vs.AddVariant(opt)
            if payload and os.path.exists(os.path.join(os.path.dirname(path), "payloads", payload)):
                vs.SetVariantSelection(opt)
                with vs.GetVariantEditContext():
                    root.GetPayloads().AddPayload(f"./payloads/{payload}")
        vs.SetVariantSelection(default)
    today = datetime.date.today().isoformat()
    meta = {"asset_name": ROOT[1:], "asset_type": "robot", "category": "spacecraft_mechanism",
            "asset_license": "CC-BY-4.0", "asset_version": cfg.VERSION, "usd_version": str(Usd.GetVersion()),
            "source_file": f"build/{ROOT[1:]}/{ROOT[1:]}.blend (scripts/build.py)", "usd_date_generated": today,
            "rigid_body_count": len(links), "asset_extents": Gf.Vec3f(*extents),
            "mass": float(sum((L.get("mass") or 1.0) for L in links.values()))}
    layer.customLayerData = {"creator": CREATOR, "asset_name": ROOT[1:], "asset_type": "robot",
                             "asset_version": cfg.VERSION, "usd_date_generated": today, "SimReady_Metadata": meta}
    layer.Save()


def finish_extents(asset_dir, asset):
    stage = Usd.Stage.Open(os.path.join(asset_dir, f"{asset}.usda"))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), UsdGeom.Imageable.GetOrderedPurposeTokens())
    hint = UsdGeom.ModelAPI(stage.GetPrimAtPath(ROOT)).ComputeExtentsHint(cache)
    vis = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = vis.ComputeWorldBound(stage.GetPrimAtPath(ROOT)).ComputeAlignedRange()
    base = Usd.Stage.Open(os.path.join(asset_dir, "payloads", "base.usda"))
    UsdGeom.ModelAPI(base.GetPrimAtPath(ROOT)).SetExtentsHint(hint)
    base.GetRootLayer().Save()
    return tuple(rng.GetSize())


def write_readme(path, cfg, links, joints, loops, colliders, sites, masses, lod_stats=None):
    lines = [f"# {cfg.TITLE}", "", cfg.DOC, "", f"Version {cfg.VERSION}. Generated by `scripts/package.py`.", "",
             "## Links", "", "| Link | Parent | Joint | Role | Mass (kg) |", "|---|---|---|---|---|"]
    for n, L in links.items():
        j = L["joint"]
        lines.append(f"| `{n}` | `{L['parent']}` | {j['type'] if L['parent'] else 'root (free)'} | {j.get('role', '')} | "
                     f"{masses[n][0]:.2f} |")
    if loops:
        lines += ["", "## Loop joints (excludeFromArticulation)", ""]
        lines += [f"- `{lp['name']}`: {lp['type']} `{lp['parent']}` -> `{lp['child']}`" for lp in loops]
    if lod_stats:
        lines += ["", "## Visual LODs (Fidelity variant)", "", "| Fidelity | Meshes | Points | Triangles |", "|---|---|---|---|"]
        lines += [f"| {k} | {v[0]} | {v[1]} | {v[2]} |" for k, v in lod_stats.items()]
    kinds = {}
    for c in colliders.values():
        kinds[c["model"]] = kinds.get(c["model"], 0) + 1
    lines += ["", "## Colliders", "", ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())), "",
              "## Sites", ""]
    lines += [f"- `{s['name']}` on `{s['link']}`: {s['attrs'].get('description', '')}" for s in sites]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    global ROOT
    argv = sys.argv[1:]
    asset = assetcfg.arg(argv, "--asset")
    cfg = assetcfg.load(asset)
    ROOT = f"/{asset}"
    exp = assetcfg.build_dir(asset, "export")
    out = assetcfg.arg(argv, "--out", os.path.join(assetcfg.ROOT, "build", asset, "usd", asset))
    if os.path.exists(out):
        shutil.rmtree(out)
    pay = os.path.join(out, "payloads")
    os.makedirs(pay)
    with open(os.path.join(exp, "full", "asset.json")) as f:
        graph = json.load(f)
    colliders = {sanitize(k): v for k, v in graph.get("colliders", {}).items()}
    src = Source(os.path.join(exp, "full"), colliders)
    proxy = Source(os.path.join(exp, "proxy")) if os.path.exists(os.path.join(exp, "proxy", "visuals.usdc")) else None
    lods = {lod: Source(os.path.join(exp, lod)) for lod in ("medium", "low")
            if os.path.exists(os.path.join(exp, lod, "visuals.usdc"))}
    links, sites, loops = graph["links"], graph["sites"], graph.get("loops", [])
    root_link = next(n for n, L in links.items() if L["parent"] is None)
    paths = link_paths(links, root_link)

    write_materials(src, os.path.join(pay, "materials.usda"), os.path.join(pay, "Textures"))
    looks = material_looks(os.path.join(pay, "materials.usda"))
    write_geometries(src, os.path.join(pay, "geometries.usd"), looks)
    lod_stats = {"full": visual_stats(src)}
    lod_stats.update(write_lod_meshes(os.path.join(pay, "geometries.usd"), lods, looks))
    write_physics_materials(os.path.join(pay, "physics_materials.usda"), cfg)
    write_instances(src, os.path.join(pay, "instances.usda"), colliders, cfg)
    masses = mass_props(src, links)
    joints = write_physics(os.path.join(pay, "Physics", "physics.usda"), links, paths, masses, loops, root_link, cfg)
    write_physx(os.path.join(pay, "Physics", "physx.usda"), joints, paths, root_link)
    write_mujoco(os.path.join(pay, "Physics", "mujoco.usda"), joints)
    write_robot(os.path.join(pay, "robot.usda"), links, paths, sites, joints, loops, root_link, cfg)
    write_base(src, os.path.join(pay, "base.usda"), links, paths, sites, cfg)
    if proxy:
        write_fidelity(os.path.join(pay, "Fidelity"), src, proxy, paths, lods)
    write_material_variants(os.path.join(pay, "Materials"), src, lods, paths, looks)
    write_contact(os.path.join(pay, "Contact"), src, colliders, paths)
    write_interface(os.path.join(out, f"{asset}.usda"), cfg, links, (0, 0, 0))
    extents = finish_extents(out, asset)
    write_interface(os.path.join(out, f"{asset}.usda"), cfg, links, extents)
    write_readme(os.path.join(out, "README.md"), cfg, links, joints, loops, colliders, sites, masses, lod_stats)
    ncol = sum(len(v) for v in src.colliders.values())
    print(f"packaged {asset}: {len(links)} links, {len(joints)} joints + {len(loops)} loops, {len(sites)} sites, "
          f"{ncol} colliders, {sum(m[0] for m in masses.values()):.1f} kg, extents "
          f"{tuple(round(e, 3) for e in extents)} -> {out}")


if __name__ == "__main__":
    main()
