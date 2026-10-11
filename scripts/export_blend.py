"""Self-contained Blender asset: pack every texture into the built .blend and save it to assets/.

Run: blender -b build/<id>/<id>.blend -P scripts/export_blend.py -- --out assets/blender/<id>.blend

Collider meshes (objects named col_*) stay in the file, hidden, so the contact model can be inspected.
"""
import os
import sys

import bpy


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = os.path.abspath(argv[argv.index("--out") + 1])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ob in bpy.data.objects:
        if "collider" in ob:
            ob.hide_viewport = ob.hide_render = True
            ob.hide_set(True)
    bpy.ops.file.pack_all()
    bpy.context.preferences.filepaths.save_version = 0   # no .blend1 backup next to the asset
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True, copy=True)
    print(f"saved {out}: {len(bpy.data.objects)} objects, {len(bpy.data.images)} packed images")


if __name__ == "__main__":
    main()
