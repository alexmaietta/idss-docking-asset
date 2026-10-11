###########################################################################
# Example IDSS Stewart Platform (MuJoCo)
#
# Plain-MuJoCo (CPU) version of examples/newton/example_idss_stewart.py: the
# soft-capture system of the NASA Docking System Block 1 on its own. The NDS
# MJCF is loaded with its base bolted to a test stand; the capture ring rides
# on six electromechanical linear actuators (universal joint, actuator, ball
# joint; five of the ball joints are <connect> loop equalities), position-
# servoed from a closed-form inverse-kinematics solution of the commanded
# ring pose. Commands outside the actuators' stroke are clamped.
#
# Keys: 7 ready, 8 stow, 9 demo trajectory on/off. Dragging an actuator
# slider in MuJoCo's Control panel switches to manual actuator control (any
# key above switches back). Double-click the ring and Ctrl+right-drag it to
# feel its servo stiffness.
#
# Command (from the idss repo root):
#   python examples/mujoco/example_idss_stewart.py
#   python examples/mujoco/example_idss_stewart.py --gravity earth --demo
#   python examples/mujoco/example_idss_stewart.py --demo --headless --seconds 16
#
###########################################################################

import argparse
import math
import os
import time

import mujoco
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MJCF = os.environ.get("IDSS_MJCF", next((p for p in (os.path.join(REPO, "assets", "mjcf"), os.path.join(REPO, "build", "mjcf"))
                                         if os.path.isdir(p)), os.path.join(REPO, "assets", "mjcf")))

RING_READY_Z = 0.33  # ring SCS plane above the HCS plane at ready-to-capture [m]
RING_STOWED_Z = -0.006  # stowed (passive-mode) ring position [m]
DEMO_LOW = -0.30  # lowest point of the demo trajectory, 36 mm above stowed [m]
DEMO_PERIOD = 16.0  # demo cycle, ready -> near stowed -> ready [s]
GRAVITY = {"zero": 0.0, "moon": 1.62, "earth": 9.81}
KEY_READY, KEY_STOW, KEY_DEMO = ord("7"), ord("8"), ord("9")


def rotation(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return np.array([[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                     [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                     [-sy, cy * sx, cy * cx]])


def hexapod_geometry(m, prefix=""):
    """Leg base points B, top points T, rest lengths L0 and the ring reference centre in the NDS base frame, from the
    exported sites (<leg>_base, <leg>_top, scs_ring_centre; reference pose = ready-to-capture) and the
    <leg>_rest_length custom numerics."""
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    base = m.body(f"{prefix}nds_base").id
    p, R = d.xpos[base], d.xmat[base].reshape(3, 3)
    local = lambda name: (d.site(f"{prefix}{name}").xpos - p) @ R   # noqa: E731
    B = np.array([local(f"leg{i}_rod_base") for i in range(1, 7)])
    T = np.array([local(f"leg{i}_rod_top") for i in range(1, 7)])
    L0 = np.array([m.numeric(f"{prefix}leg{i}_rod_rest_length").data[0] for i in range(1, 7)])
    return B, T, L0, local("scs_ring_centre")


def set_servo(m, act, kp=None, kv=None, fmax=None):
    """Retune MuJoCo position actuators: force = kp (ctrl - q) - kv qdot, clipped to +-fmax."""
    if kp is not None:
        m.actuator_gainprm[act, 0], m.actuator_biasprm[act, 1] = kp, -kp
    if kv is not None:
        m.actuator_biasprm[act, 2] = -kv
    if fmax is not None:
        m.actuator_forcerange[act] = (-fmax, fmax)


class Example:
    def __init__(self, args):
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = args.substeps
        spec = mujoco.MjSpec.from_file(os.path.join(args.mjcf, "nds", "nds.xml"))
        spec.delete(spec.joint("nds_base_free"))   # bolt the base to the test stand
        spec.option.gravity = (0.0, 0.0, -GRAVITY[args.gravity])
        spec.option.timestep = self.frame_dt / self.substeps
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(1.5, 1.5, 0.05), pos=(0, 0, -0.12),
                                rgba=(0.3, 0.32, 0.35, 1), contype=0, conaffinity=0)   # test-stand floor
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.m.eq_solref[:, 0] = 2.0 * self.m.opt.timestep   # stiff loop closures
        self.legs = np.array([self.m.actuator(f"leg{i}_rod_servo").id for i in range(1, 7)])
        self.leg_q = np.array([self.m.jnt_qposadr[self.m.joint(f"leg{i}_rod").id] for i in range(1, 7)])
        self.stroke = self.m.actuator_ctrlrange[self.legs].copy()
        set_servo(self.m, self.legs, args.leg_ke, args.leg_kd)   # stiff position servos (ball-screw actuators)
        self.B, T, self.L0, self.ring_centre = hexapod_geometry(self.m)
        self.T = T - self.ring_centre
        mujoco.mj_forward(self.m, self.d)

        self.sim_time = 0.0
        self.pose = np.zeros(6)        # commanded ring pose relative to ready-to-capture (m, rad)
        self.goal = None               # smooth move target (pose)
        self.demo = args.demo
        self.demo_t = 0.0
        self.demo_from = np.zeros(6)
        self.last_ok = np.zeros(6)
        self.clamped = False
        self.manual = False            # actuator sliders in the viewer drive the legs directly
        self.ctrl_written = None
        self.peak_err = 0.0
        self._write_targets()

    # -------------------------------------------------------------- kinematics
    def strokes(self, pose):
        """Closed-form IK: actuator strokes (length - rest length) for a ring pose relative to ready."""
        R = rotation(*pose[3:])
        tops = self.ring_centre + self.T @ R.T + pose[:3]
        return np.linalg.norm(tops - self.B, axis=1) - self.L0

    def _write_targets(self):
        lo, hi = self.stroke[:, 0] + 0.002, self.stroke[:, 1] - 0.002
        ok = lambda p: bool(np.all((self.strokes(p) >= lo) & (self.strokes(p) <= hi)))   # noqa: E731
        self.clamped = not ok(self.pose)
        if self.clamped:   # outside the actuators' stroke: stop on the line from the last reachable pose
            a, b = 0.0, 1.0
            for _ in range(12):
                c = 0.5 * (a + b)
                a, b = (c, b) if ok(self.last_ok + c * (self.pose - self.last_ok)) else (a, c)
            self.pose = self.last_ok + a * (self.pose - self.last_ok)
        self.last_ok = self.pose.copy()
        self.d.ctrl[self.legs] = self.strokes(self.pose)
        self.ctrl_written = self.d.ctrl.copy()

    def demo_pose(self, t):
        """6-DOF Lissajous through the ring's working envelope: the ring cycles from ready down to just above stowed
        (period DEMO_PERIOD) while swaying and tilting; the sway fades out on the way down, so the ring arrives near
        the stow level and centred with all six actuators at the same stroke."""
        z = 0.5 * DEMO_LOW * (1.0 - math.cos(2.0 * math.pi * t / DEMO_PERIOD))
        u = np.clip((z - DEMO_LOW) / (0.55 * DEMO_LOW - DEMO_LOW), 0.0, 1.0)
        e = u * u * (3.0 - 2.0 * u)   # smoothstep: 0 below ~75 % of the descent, 1 in the upper half
        return np.array([0.035 * e * math.sin(0.7 * t), 0.035 * e * math.sin(1.3 * t), z,
                         math.radians(5) * e * math.sin(1.1 * t), math.radians(5) * e * math.sin(0.8 * t),
                         math.radians(7) * e * math.sin(0.6 * t)])

    def key(self, keycode):
        if keycode == KEY_READY:
            self.demo, self.manual, self.goal = False, False, np.zeros(6)
        elif keycode == KEY_STOW:
            self.demo, self.manual, self.goal = False, False, np.array([0, 0, RING_STOWED_Z - RING_READY_Z, 0, 0, 0])
        elif keycode == KEY_DEMO:
            self.demo, self.manual, self.goal = not self.demo, False, None
            self.demo_t, self.demo_from = 0.0, self.pose.copy()

    # -------------------------------------------------------------- sim loop
    def step(self):
        if self.ctrl_written is not None and np.any(np.abs(self.d.ctrl - self.ctrl_written) > 1e-9):
            self.manual, self.demo, self.goal = True, False, None   # a viewer slider moved
        if self.manual:
            pass
        elif self.demo:
            w = min(1.0, self.demo_t / 1.5)
            w = w * w * (3.0 - 2.0 * w)
            self.pose = (1.0 - w) * self.demo_from + w * self.demo_pose(self.demo_t)
            self.demo_t += self.frame_dt
        elif self.goal is not None:   # rate-limited move (actuators run at ~25 mm/s)
            dp = self.goal - self.pose
            step = np.array([0.025, 0.025, 0.025, 0.1, 0.1, 0.1]) * self.frame_dt
            self.pose = self.pose + np.clip(dp, -step, step)
            if np.all(np.abs(dp) < 1e-6):
                self.goal = None
        if not self.manual:
            self._write_targets()
        self.ctrl_written = self.d.ctrl.copy()
        mujoco.mj_step(self.m, self.d, nstep=self.substeps)
        self.sim_time += self.frame_dt
        if self.sim_time > 1.0 and not self.manual:
            self.peak_err = max(self.peak_err, self.tracking_error())

    def tracking_error(self):
        return float(np.max(np.abs(self.d.qpos[self.leg_q] - self.d.ctrl[self.legs])))

    def status(self):
        mode = "manual" if self.manual else "demo" if self.demo else "moving" if self.goal is not None else "hold"
        left = "\n".join(["Mode", "Ring z", "Clamped"] + [f"Leg {k + 1}" for k in range(6)])
        f = self.d.actuator_force[self.legs]
        right = "\n".join([mode, f"{(self.pose[2] + RING_READY_Z) * 1000:.0f} mm above HCS", "yes" if self.clamped else "no"]
                          + [f"{q * 1000:6.1f} mm {fk:6.0f} N" for q, fk in zip(self.d.qpos[self.leg_q], f)])
        return left, right

    def report(self, wall):
        f = np.max(np.abs(self.d.actuator_force[self.legs]))
        print(f"stewart: {self.sim_time:.1f} s sim in {wall:.2f} s wall (real-time factor {self.sim_time / wall:.1f}x), "
              f"tracking error final {self.tracking_error() * 1000:.2f} mm, peak (after 1 s) {self.peak_err * 1000:.2f} mm, "
              f"final max load {f:.0f} N")


def main():
    parser = argparse.ArgumentParser(description="IDSS Stewart platform (plain MuJoCo).")
    parser.add_argument("--mjcf", default=MJCF, help="Folder holding nds/nds.xml.")
    parser.add_argument("--gravity", default="zero", choices=list(GRAVITY), help="Test-stand gravity.")
    parser.add_argument("--demo", action="store_true", help="Start with the 6-DOF demo trajectory.")
    parser.add_argument("--leg-ke", type=float, default=3.0e5, help="Actuator servo stiffness [N/m].")
    parser.add_argument("--leg-kd", type=float, default=3.0e3, help="Actuator servo damping [N s/m].")
    parser.add_argument("--substeps", type=int, default=8, help="Physics substeps per 60 Hz frame.")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer and print the results.")
    parser.add_argument("--seconds", type=float, default=16.0, help="Simulated time for --headless [s].")
    args = parser.parse_args()
    ex = Example(args)
    if args.headless:
        t0 = time.perf_counter()
        while ex.sim_time < args.seconds - 1e-9:
            ex.step()
        ex.report(time.perf_counter() - t0)
        return
    import mujoco.viewer

    with mujoco.viewer.launch_passive(ex.m, ex.d, key_callback=ex.key) as v:
        v.cam.lookat[:], v.cam.distance, v.cam.elevation, v.cam.azimuth = (0, 0, 0.35), 2.6, -20, 135
        t0 = time.perf_counter()
        while v.is_running():
            with v.lock():
                ex.step()
                v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPRIGHT, *ex.status()))
            v.sync()
            lag = ex.sim_time - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)
        ex.report(time.perf_counter() - t0)


if __name__ == "__main__":
    main()
