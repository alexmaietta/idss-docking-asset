# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Render IDSS Stills (OVRTX)
#
# The README gallery stills and the USD asset thumbnails, path traced by
# OVRTX in its production PathTracing mode (see rtx.py):
#   - space shots of the docking, posed from a recorded run (record_docking.py)
#     under the orbital HDRI;
#   - studio shots of one asset at rest under a neutral studio rig;
#   - --thumb <asset>: the 256x256 thumbnail the USD asset ships in
#     .thumbs/256x256/<asset>.usda.png.
# Each shot renders in its own process (the camera FOV is fixed per stage).
#
# Command (from the idss repo root):
#   python examples/newton/utils/render_stills_rtx.py [--shots all] [--spp 256] [--out docs/images]
#   python examples/newton/utils/render_stills_rtx.py --thumb nds --assets build/nds/usd
#
###########################################################################

import argparse
import os
import subprocess
import sys
import time

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))   # examples/newton

import rtx  # noqa: E402

# name -> (rig, assets, pose, camera position, target, focal length [mm], resolution)
# pose: ("phase", name, offset_s) in the recorded run, or ("rest",) for one asset on its own
SHOTS = {
    "approach": ("space", ("ida", "nds"), ("phase", "contact", -0.6), (3.1, -3.9, 1.6), (0.0, 0.0, 0.95), 38,
                 (1600, 900)),
    "capture": ("space_detail", ("ida", "nds"), ("phase", "captured", 0.4), (1.25, -0.9, 0.95), (0.25, 0.0, 0.9), 35,
                (1600, 900)),
    "retract": ("space", ("ida", "nds"), ("phase", "retract", 6.0), (3.6, -3.0, 2.0), (0.0, 0.0, 0.6), 34, (1600, 900)),
    "docked": ("space", ("ida", "nds"), ("phase", "docked", 1.0), (3.1, -3.9, 1.5), (0.0, 0.0, 0.75), 38, (1600, 900)),
    "ida_studio": ("studio", ("ida",), ("rest",), (2.6, -2.2, 2.2), (0.0, 0.0, 0.40), 40, (1600, 900)),
    "nds_studio": ("studio", ("nds",), ("rest",), (1.9, -1.6, 1.0), (0.0, 0.0, 0.42), 40, (1600, 900)),
    "stewart_platform": ("studio", ("nds",), ("rest",), (1.05, -0.65, 0.45), (0.0, 0.0, 0.38), 28, (1600, 900)),
    "latch_closeup": ("studio", ("nds",), ("rest",), (0.98, 0.30, 1.05), (0.49, 0.0, 0.67), 60, (1600, 900)),
    "hook_closeup": ("studio", ("ida",), ("rest",), (0.95, 0.30, 0.79), (0.64, 0.172, 0.645), 85, (1600, 900)),
    "petals_closeup": ("studio", ("ida",), ("rest",), (1.0, -0.55, 1.15), (0.4, -0.05, 0.72), 45, (1600, 900)),
}
THUMB = {"ida": "ida_studio", "nds": "nds_studio"}   # thumbnails reuse the studio camera, square


def asset_model(name, assets):
    """One asset on its own, fixed, at its rest pose (the joints' default positions)."""
    import newton
    from pxr import Usd

    st = Usd.Stage.Open(os.path.join(assets, name, f"{name}.usda"))
    root = st.GetDefaultPrim()
    for vset, sel in (("Physics", "physics"), ("Fidelity", "full"), ("Contact", "exact")):
        root.GetVariantSet(vset).SetVariantSelection(sel)
    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_usd(st, floating=False, collapse_fixed_joints=False, enable_self_collisions=False, hide_collision_shapes=True)
    model = b.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return model, state


def render_shot(a, name, out_path, res=None):
    rig, assets, pose, pos, target, lens, shot_res = SHOTS[name]
    w, h = res or shot_res
    viewer = rtx.ProductionRTX(w, h, rig=rig, spp=a.spp, spp_per_step=a.spp_per_step, fov=rtx.lens_fov(lens, w / h))
    t0 = time.perf_counter()
    if pose[0] == "rest":
        model, state = asset_model(assets[0], a.assets)
        viewer.set_model(model)
        t = 0.0
    else:
        import example_idss_docking as E
        sim = rtx.load_recording(a.sim)
        ex = E.Example(viewer, E.Example.create_parser().parse_args(["--viewer", "null", "--assets", a.assets]))
        assert list(sim["labels"]) == list(ex.model.body_label), "recording does not match the model"
        i = rtx.sim_index(sim, pose[1], pose[2])
        ex.state_0.body_q.assign(sim["body_q"][i])
        state, t = ex.state_0, float(sim["time"][i])
    rtx.look(viewer, pos, target)
    img = Image.fromarray(viewer.render_state(state, t))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path, quality=92) if out_path.endswith(".jpg") else img.save(out_path)
    print(f"{name}: {w}x{h}, {a.spp} spp -> {out_path} ({time.perf_counter() - t0:.1f} s)", flush=True)
    viewer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="all", help=f"Comma-separated shots, or all: {', '.join(SHOTS)}.")
    ap.add_argument("--out", default=os.path.join(rtx.REPO, "docs", "images"))
    ap.add_argument("--sim", default=os.path.join(rtx.REPO, "build", "sim", "docking.npz"))
    ap.add_argument("--assets", default=os.environ.get("IDSS_ASSETS", os.path.join(rtx.REPO, "assets")),
                    help="Folder holding <asset>/<asset>.usda.")
    ap.add_argument("--spp", type=int, default=256, help="Path-traced samples per pixel.")
    ap.add_argument("--spp-per-step", type=int, default=32)
    ap.add_argument("--thumb", choices=list(THUMB), default=None,
                    help="Render this asset's 256x256 thumbnail into <assets>/<asset>/.thumbs/256x256/.")
    a = ap.parse_args()
    if a.thumb:
        render_shot(a, THUMB[a.thumb], os.path.join(a.assets, a.thumb, ".thumbs", "256x256", f"{a.thumb}.usda.png"),
                    res=(256, 256))
        return
    shots = list(SHOTS) if a.shots == "all" else a.shots.split(",")
    if len(shots) == 1:
        render_shot(a, shots[0], os.path.join(a.out, f"{shots[0]}.jpg"))
    else:
        argv = list(sys.argv[1:])
        if "--shots" in argv:
            del argv[argv.index("--shots"):argv.index("--shots") + 2]
        for name in shots:   # one process per shot
            subprocess.run([sys.executable, os.path.abspath(__file__), *argv, "--shots", name], check=True)


if __name__ == "__main__":
    main()
