# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Record IDSS Docking Video
#
# The docking example the way it looks on screen in the Newton RTX viewer
# (OVRTX real-time path tracing, default lighting, `pip install ovrtx
# ovstage`): replays a recorded run (record_docking.py) from approach to hard
# dock with a slowly orbiting camera. Playback speed changes per docking phase
# so the capture plays slowly and the long alignment and retraction pass
# quickly; the small caption chip names the phase. Writes an MP4 (needs
# ffmpeg) and a GIF for the README.
#
# Command (from the idss repo root):
#   python examples/newton/utils/record_docking_video.py [--sim build/sim/docking.npz] [--out docs/images/newton_docking]
#
###########################################################################

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))   # examples/newton

import newton.viewer  # noqa: E402
import rtx  # noqa: E402

import example_idss_docking as E  # noqa: E402

# sim seconds per video second, per phase
SPEED = {"approach": 2.0, "contact": 0.6, "captured": 0.6, "attenuate": 3.0, "align": 6.0, "retract": 5.0,
         "hook gang 1": 2.0, "hook gang 2": 2.0, "docked": 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=os.path.join(rtx.REPO, "build", "sim", "docking.npz"))
    ap.add_argument("--out", default="docs/images/newton_docking", help="Output path without extension (.mp4, .gif).")
    ap.add_argument("--fps", type=int, default=30, help="Video frame rate.")
    ap.add_argument("--gif-fps", type=int, default=10)
    ap.add_argument("--gif-width", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--passes", type=int, default=6, help="Path-traced passes averaged per frame (1 spp each).")
    ap.add_argument("--environment", default="default", choices=["default", "studio"],
                    help="Viewer lighting preset; default is what the example shows (studio is blue tinted).")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after this many video frames (0: whole run).")
    a = ap.parse_args()
    sim = rtx.load_recording(a.sim)
    viewer = newton.viewer.ViewerRTX(width=a.width, height=a.height, headless=True, environment=a.environment,
                                     async_rendering=False)
    ex = E.Example(viewer, E.Example.create_parser().parse_args(["--viewer", "null"]))
    assert list(sim["labels"]) == list(ex.model.body_label), "recording does not match the model"
    frames_dir = os.path.join(rtx.REPO, "build", "video", os.path.basename(a.out))
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)
    idx = rtx.playback(sim, a.fps, SPEED)
    if a.max_frames:
        idx = idx[:a.max_frames]
    gif, gif_every = [], max(1, round(a.fps / a.gif_fps))
    s = a.height / 720.0
    for n, i in enumerate(idx):
        t, phase = float(sim["time"][i]), str(sim["phase"][i])
        rtx.look(viewer, *rtx.orbit_camera(t))
        ex.state_0.body_q.assign(sim["body_q"][i])
        acc = 0.0
        for _ in range(a.passes):
            viewer.begin_frame(t)
            viewer.log_state(ex.state_0)
            viewer.end_frame()
            acc = acc + np.asarray(viewer._capture_screenshot_pixels())[..., :3].astype(np.float32)
        img = rtx.caption(Image.fromarray((acc / a.passes).astype(np.uint8)), phase, t, s)
        img.save(os.path.join(frames_dir, f"{n:05d}.png"))
        if n % gif_every == 0:
            gw = min(a.gif_width, a.width)
            gif.append(img.resize((gw, round(a.height * gw / a.width)), Image.LANCZOS))
        if (n + 1) % 60 == 0:
            print(f"{n + 1}/{len(idx)} frames, t={t:.1f} s {phase}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pal = [g.convert("P", palette=Image.ADAPTIVE, colors=128) for g in gif]
    pal[0].save(a.out + ".gif", save_all=True, append_images=pal[1:], duration=int(1000 / a.gif_fps), loop=0,
                optimize=True)
    print(f"wrote {a.out}.gif: {len(gif)} frames, {os.path.getsize(a.out + '.gif') / 1e6:.1f} MB")
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i",
                        os.path.join(frames_dir, "%05d.png"), "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20",
                        a.out + ".mp4"], check=True)
        print(f"wrote {a.out}.mp4: {len(idx)} frames ({len(idx) / a.fps:.1f} s)")


if __name__ == "__main__":
    main()
