# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Render IDSS Docking (OVRTX)
#
# Replays a recorded docking run (record_docking.py) in the headless Newton
# RTX viewer with OVRTX in its production path-tracing mode (PathTracing,
# OptiX denoiser, accumulated to a fixed sample count per frame) under the
# orbital HDRI (see rtx.py), from four cameras:
#   hero     the overall docking, a slow orbit that pushes in on the capture
#            latch, the interface and a hook station as the phases get going
#   hero_xray  the same camera path with the outer structure x-rayed; the hero
#            GIF crossfades the two by the phase's x-ray amount (rtx.hero_path)
#   wide     three-quarter view of the vehicle closing on the IDA
#   capture  petal / capture latch / striker interface, outer structure x-rayed
#   hooks    grazing view across a hook station, outer structure x-rayed
# The x-ray views make the fixed IDA and NDS structure (and the vehicle and
# PMA dressing) see-through, so the latches, guide pins and hooks stay visible
# while they work. Frames and frames.json land in the layout scripts/make_gif.py
# composites.
#
# Needs `pip install ovrtx ovstage`. Each view renders in its own process.
#
# Command (from the idss repo root):
#   python examples/newton/utils/render_docking_rtx.py [--view all] [--frames 150] [--spp 64] [--out build/anim_rtx]
#   python scripts/make_gif.py --frames build/anim_rtx --single hero --out docs/images/docking.gif
#   python scripts/make_gif.py --frames build/anim_rtx --plain --out docs/images/docking_xray.gif
#
###########################################################################

import argparse
import json
import os
import subprocess
import sys
import time

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))   # examples/newton

import rtx  # noqa: E402

import example_idss_docking as E  # noqa: E402

# camera (position, target, focal length [mm]; or None for the hero path at the viewer's 45 deg FOV), resolution,
# opacity of the x-rayed structure (1: solid) and lighting rig
VIEWS = {
    "hero": (None, (1280, 720), 1.0, "space"),
    "hero_xray": (None, (1280, 720), 0.15, "space_detail"),
    "wide": (((3.5, -2.9, 1.3), (0.0, 0.0, 0.85), 34), (640, 480), 1.0, "space"),
    "capture": (((-0.95, 0.42, 0.98), (-0.52, 0.0, 0.70), 32), (640, 480), 0.22, "space_detail"),
    "hooks": (((0.94, 0.10, 0.655), (0.60, 0.20, 0.650), 50), (640, 480), 0.12, "space_detail"),
}


def render_view(a, view):
    sim = rtx.load_recording(a.sim)
    cam, (w, h), opacity, rig = VIEWS[view]
    viewer = rtx.ProductionRTX(w, h, rig=rig, spp=a.spp, spp_per_step=a.spp_per_step, hdri_intensity=a.hdri_intensity,
                               light_scale=a.light_scale, shape_opacity=rtx.ghost_opacity(opacity),
                               fov=rtx.lens_fov(cam[2], w / h) if cam else None)
    ex = E.Example(viewer, E.Example.create_parser().parse_args(["--viewer", "null"]))
    assert list(sim["labels"]) == list(ex.model.body_label), "recording does not match the model"
    out = os.path.join(a.out, view)
    os.makedirs(out, exist_ok=True)
    idx = rtx.frame_times(sim, a.frames)
    path = rtx.hero_path(sim, idx, a.fps)
    todo = a.only if a.only is not None else range(len(idx))
    t0 = time.perf_counter()
    for n, k in enumerate(todo):
        i = idx[min(k, len(idx) - 1)]
        png = os.path.join(out, f"{k:04d}.png")
        if os.path.exists(png) and not a.force:
            continue
        rtx.look(viewer, *(cam[:2] if cam else path[min(k, len(idx) - 1)][:2]))
        ex.state_0.body_q.assign(sim["body_q"][i])
        Image.fromarray(viewer.render_state(ex.state_0, float(sim["time"][i]))).save(png)
        if n == 0 or (n + 1) % 10 == 0:
            print(f"{view}: frame {n + 1}/{len(todo)} t={sim['time'][i]:.2f} s {sim['phase'][i]}  "
                  f"({(time.perf_counter() - t0) / (n + 1):.2f} s/frame)", flush=True)
    viewer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=os.path.join(rtx.REPO, "build", "sim", "docking.npz"))
    ap.add_argument("--out", default=os.path.join(rtx.REPO, "build", "anim_rtx"))
    ap.add_argument("--view", default="all", choices=["all", *VIEWS])
    ap.add_argument("--frames", type=int, default=150, help="Frames (from 1.5 s before contact to hard docked).")
    ap.add_argument("--only", type=lambda s: [int(x) for x in s.split(",")], default=None,
                    help="Render only these frame numbers (comma separated), e.g. for a quick look.")
    ap.add_argument("--fps", type=int, default=12, help="GIF frame rate the hero camera path is timed for.")
    ap.add_argument("--spp", type=int, default=64, help="Path-traced samples per pixel per frame.")
    ap.add_argument("--spp-per-step", type=int, default=16, help="Samples per renderer step.")
    ap.add_argument("--hdri-intensity", type=float, default=500.0, help="Dome light intensity of the HDRI.")
    ap.add_argument("--light-scale", type=float, default=1.0, help="Scale the extra distant lights.")
    ap.add_argument("--force", action="store_true", help="Re-render frames that already exist.")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "frames.json"), "w") as f:
        sim = rtx.load_recording(a.sim)
        idx = rtx.frame_times(sim, a.frames)
        meta = rtx.frame_meta(sim, idx)
        for m, (_pos, _tgt, ghost) in zip(meta, rtx.hero_path(sim, idx, a.fps)):
            m["xray"] = ghost   # hero crossfade: 0 solid, 1 x-ray
        json.dump(meta, f, indent=1)
    if a.view != "all":
        render_view(a, a.view)
        return
    for view in VIEWS:   # one process per view: the camera FOV and per-shape opacity are fixed per stage
        argv = list(sys.argv[1:])
        if "--view" in argv:
            del argv[argv.index("--view"):argv.index("--view") + 2]
        subprocess.run([sys.executable, os.path.abspath(__file__), *argv, "--view", view], check=True)


if __name__ == "__main__":
    main()
