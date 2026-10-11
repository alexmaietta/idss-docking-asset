"""Export a built .blend: per-link visual meshes (USD) plus the link/joint/site graph (JSON).

Run: blender -b build/<id>/<id>.blend -P scripts/export.py -- --asset <id> [--lod full|proxy]

Writes build/<id>/export/<lod>/:
  visuals.usdc   one prim per link (/visuals/L_<link>), meshes in link-local coordinates, UsdPreviewSurface
  textures/      textures referenced by the materials
  asset.json     links (parent, world origin, joint, mass), sites, thrusters, cameras
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import assetcfg  # noqa: E402


def _avg_colour(mat):
    if mat is None or not mat.node_tree:
        return (0.6, 0.6, 0.6)
    bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return tuple(mat.diffuse_color)[:3]
    s = bsdf.inputs["Base Color"]
    if not s.is_linked:
        return tuple(s.default_value)[:3]
    node = s.links[0].from_node
    if node.type == "TEX_IMAGE" and node.image:
        px = node.image.pixels[:]
        n = len(px) // 4
        step = max(1, n // 4096)
        idx = range(0, n, step)
        return tuple(sum(px[i * 4 + c] for i in idx) / len(idx) for c in range(3))
    return tuple(mat.diffuse_color)[:3]


def bake_uv_tiling():
    """Fold each material's Mapping-node scale into the mesh UVs, then reset the mapping, so viewers
    that ignore UsdTransform2d (e.g. Newton's GL viewer) still get real-world texture scale."""
    scale = {}
    for m in bpy.data.materials:
        if not m.node_tree:
            continue
        mp = next((n for n in m.node_tree.nodes if n.type == "MAPPING"), None)
        if mp is not None:
            s = tuple(mp.inputs["Scale"].default_value)
            if abs(s[0] - 1) > 1e-6 or abs(s[1] - 1) > 1e-6:
                scale[m.name] = (s[0], s[1])
    for ob in bpy.data.objects:
        if ob.type != "MESH" or not ob.data.uv_layers:
            continue
        me = ob.data
        if me.users > 1:
            ob.data = me = me.copy()
        mats = [ms.material for ms in ob.material_slots]
        uv = me.uv_layers.active.data
        for poly in me.polygons:
            m = mats[poly.material_index] if poly.material_index < len(mats) else None
            if m is None or m.name not in scale:
                continue
            sx, sy = scale[m.name]
            for li in poly.loop_indices:
                u, v = uv[li].uv
                uv[li].uv = (u * sx, v * sy)
    for name in scale:
        mp = next(n for n in bpy.data.materials[name].node_tree.nodes if n.type == "MAPPING")
        mp.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    return len(scale)


def proxy():
    """One box per part in its average colour (Fidelity=proxy)."""
    flat = {}
    for ob in [o for o in bpy.data.objects if o.type == "MESH"]:
        pts = [Vector(c) for c in ob.bound_box]
        mat = ob.data.materials[0] if ob.data.materials else None
        col = _avg_colour(mat)
        key = tuple(round(c, 2) for c in col)
        if key not in flat:
            m = bpy.data.materials.new(f"proxy_{len(flat)}")
            m.use_nodes = True
            b = m.node_tree.nodes["Principled BSDF"]
            b.inputs["Base Color"].default_value = (*col, 1.0)
            b.inputs["Roughness"].default_value = 0.6
            flat[key] = m
        lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        me = bpy.data.meshes.new(ob.name + "_proxy")
        v = [(x, y, z) for z in (lo.z, hi.z) for y in (lo.y, hi.y) for x in (lo.x, hi.x)]
        me.from_pydata(v, [], [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)])
        me.materials.append(flat[key])
        ob.modifiers.clear()
        ob.data = me


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    asset = assetcfg.arg(argv, "--asset", bpy.context.scene.get("asset_id"))
    lod = assetcfg.arg(argv, "--lod", "full")
    outdir = assetcfg.build_dir(asset, "export", lod)
    sc = bpy.context.scene

    empties = [o for o in bpy.data.objects if o.type == "EMPTY" and "link" in o]
    links = {}
    for e in empties:
        links[e["link"]] = {"parent": e.parent["link"] if e.parent else None, "origin": list(e.matrix_world.translation),
                            "joint": json.loads(e["joint"]), "mass": e.get("mass")}
    graph = {"asset": asset, "links": links, "sites": json.loads(sc.get("sites", "[]")),
             "cameras": json.loads(sc.get("cameras", "[]")), "loops": json.loads(sc.get("loops", "[]")),
             "colliders": {o.name: json.loads(o["collider"]) for o in bpy.data.objects if "collider" in o}}
    with open(os.path.join(outdir, "asset.json"), "w") as f:
        json.dump(graph, f, indent=1)

    for ob in bpy.data.objects:   # colliders are hidden from renders in the .blend; export them anyway
        if "collider" in ob:
            ob.hide_render = False
            ob.hide_viewport = False
    # drop site empties and cameras (re-created by the packager); flatten links to the origin
    for ob in [o for o in bpy.data.objects if o.name.startswith("site:") or o.type == "CAMERA"]:
        bpy.data.objects.remove(ob)
    for e in empties:
        e.parent = None
    for e in empties:
        e.location = (0, 0, 0)
        e.rotation_euler = (0, 0, 0)
        e.name = f"L_{e['link']}"
    bpy.context.view_layer.update()
    if lod == "proxy":
        for ob in [o for o in bpy.data.objects if "collider" in o]:   # colliders come from the full export
            bpy.data.objects.remove(ob)
        proxy()
    print(f"baked UV tiling for {bake_uv_tiling()} materials")
    bpy.ops.wm.usd_export(
        filepath=os.path.join(outdir, "visuals.usdc"), root_prim_path="/visuals", export_materials=True,
        generate_preview_surface=True, export_textures_mode="NEW", overwrite_textures=True, relative_paths=True,
        export_lights=False, export_cameras=False, export_custom_properties=False, evaluation_mode="RENDER",
        convert_scene_units="METERS", triangulate_meshes=True)
    print(f"exported {asset} ({lod}): {len(links)} links, {len(graph['sites'])} sites -> {outdir}")


main()
