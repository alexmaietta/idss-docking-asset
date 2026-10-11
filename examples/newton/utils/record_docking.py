# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Record IDSS Docking
#
# Runs example_idss_docking headless and stores every body pose (and the
# docking phase, gap, loads and latch states) once per frame, for offline
# rendering: render_docking_rtx.py, render_stills_rtx.py and record_docking_video.py pose the models from
# these states, so the README stills and the docking GIF show simulated
# configurations.
#
# Command (from the idss repo root):
#   python examples/newton/utils/record_docking.py --out build/sim/docking.npz [example args ...]
#
###########################################################################

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # examples/newton

import newton.examples  # noqa: E402

import example_idss_docking as E  # noqa: E402


def main():
    out = "build/sim/docking.npz"
    if "--out" in sys.argv:
        i = sys.argv.index("--out")
        out = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    t_max = 60.0
    if "--duration" in sys.argv:
        i = sys.argv.index("--duration")
        t_max = float(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    sys.argv[1:1] = ["--viewer", "null"]
    parser = E.Example.create_parser()
    viewer, args = newton.examples.init(parser)
    ex = E.Example(viewer, args)
    labels = list(ex.model.body_label)
    frames, times, phases, gaps, preload, latches = [], [], [], [], [], []
    ring_load, swing, pull, locked, lateral, tilt = [], [], [], [], [], []
    hold = 0.0
    while ex.sim_time < t_max:
        ex.step()
        m = ex.meas
        frames.append(ex.state_0.body_q.numpy().copy())
        times.append(ex.sim_time)
        phases.append(ex.phase)
        gaps.append(m["gap"])
        preload.append(float(np.sum(m["preload"])))
        latches.append(m["latch"].copy())
        ring_load.append(float(m["ring_load"]))
        swing.append(float(max(ex.hook_cmd)))
        pull.append(float(max(ex.pull_cmd)))
        locked.append(int(sum(ex.latched)))
        lateral.append(float(m["lateral"]))
        tilt.append(float(m["tilt"]))
        if ex.phase in ("docked", "separated", "diverged", "capture failed") and not args.undock_after:
            hold += ex.frame_dt
            if hold > 2.0:
                break
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, body_q=np.array(frames, dtype=np.float32), time=np.array(times), phase=np.array(phases),
                        gap=np.array(gaps), preload=np.array(preload), latch=np.array(latches),
                        ring_load=np.array(ring_load), hook_swing=np.array(swing), hook_pull=np.array(pull),
                        latched=np.array(locked), lateral=np.array(lateral), tilt=np.array(tilt),
                        labels=np.array(labels), events=np.array([f"{t} {p}" for t, p in ex.events]))
    print(f"recorded {len(frames)} frames ({times[-1]:.1f} s, final phase {ex.phase}) -> {out}")


if __name__ == "__main__":
    main()
