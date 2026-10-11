# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Record Stewart Platform GIF
#
# Runs one cycle of example_idss_stewart's 6-DOF demo trajectory (ready, down
# to just above stowed, back to ready) in the headless Newton RTX viewer
# (OVRTX path tracing, `pip install ovrtx ovstage`), slowly orbiting the
# camera, and writes an animated GIF of the frames for the README. Use
# --viewer gl for the rasterized OpenGL viewer.
#
# Command (from the idss repo root):
#   python examples/newton/utils/record_stewart_gif.py [--out docs/images/stewart_demo.gif]
#
###########################################################################

import argparse
import math
import os
import sys

import numpy as np
import warp as wp
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # examples/newton

import newton  # noqa: E402
import newton.viewer  # noqa: E402

from example_idss_stewart import DEMO_PERIOD, Example  # noqa: E402


def look(viewer, pos, target):
    d = np.subtract(target, pos)
    d /= np.linalg.norm(d)
    viewer.set_camera(wp.vec3(*pos), math.degrees(math.asin(d[2])), math.degrees(math.atan2(d[1], d[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/images/stewart_demo.gif")
    ap.add_argument("--seconds", type=float, default=DEMO_PERIOD, help="Sim time to record (one demo cycle).")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--viewer", default="rtx", choices=["rtx", "gl"])
    ap.add_argument("--passes", type=int, default=4, help="RTX: path-traced passes averaged per frame (1 spp each).")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    a = ap.parse_args()
    args = Example.create_parser().parse_args(["--viewer", a.viewer, "--headless", "--demo"])
    if a.viewer == "rtx":
        viewer = newton.viewer.ViewerRTX(width=a.width, height=a.height, headless=True, environment="studio",
                                         async_rendering=False)
        grab = viewer._capture_screenshot_pixels
    else:
        viewer = newton.viewer.ViewerGL(headless=True, width=a.width, height=a.height)
        grab = lambda: viewer.get_frame().numpy()  # noqa: E731
    ex = Example(viewer, args)
    every = max(1, round(ex.fps / a.fps))
    frames = []
    target = (0.0, 0.0, 0.42)
    n = int(a.seconds * ex.fps)
    for k in range(n):
        ex.step()
        if k % every:
            continue
        az = math.radians(-35.0 + 60.0 * k / n)   # slow orbit
        pos = (target[0] + 1.55 * math.cos(az), target[1] + 1.55 * math.sin(az), 0.95)
        look(viewer, pos, target)
        acc = 0.0
        for _ in range(a.passes if a.viewer == "rtx" else 1):
            ex.render()
            acc = acc + np.asarray(grab())[..., :3].astype(np.float32)
        frames.append(Image.fromarray((acc / max(1, a.passes if a.viewer == "rtx" else 1)).astype(np.uint8)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=96) for f in frames]
    pal[0].save(a.out, save_all=True, append_images=pal[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
    print(f"wrote {a.out}: {len(frames)} frames, {os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
