"""Build an asset in Blender.

Run: blender -b --factory-startup -P scripts/build.py -- --asset nds|ida [--out build/<id>/<id>.blend] [--no-merge]

The asset's config (configs/<id>.py) names its builder module (scripts/<FAMILY>.py, a build(cfg)
function). After building: meshes without UVs get a metre box projection, modifiers (machined-edge
bevels) are applied, and every set of visual meshes fixed to the same link with the same material is
joined into one object '<link>__<material>' (moving parts are separate links, so nothing that
articulates is merged). Collider meshes (objects with a "collider" property) are kept as they are.
"""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy  # noqa: E402

import assetcfg  # noqa: E402
import geo as G  # noqa: E402
import materials as M  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
asset = assetcfg.arg(argv, "--asset")
cfg = assetcfg.load(asset)
G.reset()
M.configure(assetcfg.build_dir(asset, "textures"), cfg.MATERIALS)
family = importlib.import_module(cfg.FAMILY)
family.build(cfg)
sc = bpy.context.scene
sc["asset_id"] = asset


def is_collider(ob):
    return "collider" in ob


def apply_all_modifiers():
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    n = 0
    for ob in bpy.data.objects:
        if ob.type != "MESH" or not ob.modifiers:
            continue
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
        ob.modifiers.clear()
        ob.data = me
        n += 1
    return n


def merge_rigid_parts():
    groups = {}
    for ob in bpy.data.objects:
        if ob.type != "MESH" or ob.parent is None or "link" not in ob.parent or is_collider(ob):
            continue
        mat = ob.data.materials[0].name if ob.data.materials else "none"
        groups.setdefault((ob.parent["link"], mat, ob.get("variant", "")), []).append(ob)
    for (link, mat, variant), obs in groups.items():
        target = obs[0]
        if len(obs) > 1:
            for o in obs:
                if o.data.users > 1:
                    o.data = o.data.copy()
            with bpy.context.temp_override(active_object=target, selected_editable_objects=obs, object=target):
                bpy.ops.object.join()
        tag = f"__{variant.replace(':', '_').replace(',', '_')}" if variant else ""
        target.name = target.data.name = f"{link}__{mat}{tag}"
    return len(groups)


for ob in bpy.data.objects:
    if ob.type == "MESH" and not is_collider(ob):
        G.ensure_uvs(ob)
print(f"applied modifiers on {apply_all_modifiers()} meshes")
if "--no-merge" not in argv:
    print(f"merged rigid parts into {merge_rigid_parts()} meshes")
out = assetcfg.arg(argv, "--out", os.path.join(assetcfg.build_dir(asset), f"{asset}.blend"))
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out))
nm = sum(1 for o in bpy.data.objects if o.type == "MESH" and not is_collider(o))
nc = sum(1 for o in bpy.data.objects if is_collider(o))
nl = sum(1 for o in bpy.data.objects if "link" in o)
print(f"built {asset}: {nm} visual meshes, {nc} colliders, {nl} links, {len(G._registry('sites'))} sites -> {out}")
