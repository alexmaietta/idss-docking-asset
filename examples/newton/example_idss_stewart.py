# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example IDSS Stewart Platform
#
# The soft-capture system of the NASA Docking System Block 1 on its own: the
# capture ring (with its three guide petals and capture latches) rides on six
# electromechanical linear actuators, a Stewart platform. The NDS is loaded
# from USD with its base bolted to a test stand, and the six legs (universal
# joint, actuator, ball joint; five of the ball joints close kinematic loops)
# are position-servoed from a closed-form inverse-kinematics solution of the
# commanded ring pose.
#
# Drive the ring with the sliders (x, y, z, roll, pitch, yaw), jump to the
# stowed or ready-to-capture poses, or play a 6-DOF demo trajectory. The panel
# shows each actuator's stroke and load; commands outside the actuators' stroke
# are clamped. Drag the ring with the mouse (right button) to feel its servo
# stiffness.
#
# The ring command (sliders, rate-limited moves, demo trajectory, stroke
# clamping and inverse kinematics) runs in Warp kernels inside the frame's
# CUDA graph; GUI inputs are queued as kernel launches and telemetry comes
# back through asynchronous copies, so the host never waits on the GPU.
#
# Command (from the idss repo root):
#   python examples/newton/example_idss_stewart.py
#   python examples/newton/example_idss_stewart.py --gravity earth --demo
#
###########################################################################

import math
import os
from collections import deque
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
import newton.examples

wp.set_module_options({"enable_backward": False})   # controller kernels need no gradients (and compile far faster)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ASSETS = os.environ.get("IDSS_ASSETS", os.path.join(REPO, "assets"))

NDS_HCS_Z = 0.2304  # HCS mating plane above the NDS host-vehicle interface [m]
RING_READY_Z = 0.33  # ring SCS plane above the HCS plane at ready-to-capture [m]
RING_STOWED_Z = -0.006  # stowed (passive-mode) ring position [m]
DEMO_LOW = -0.30  # lowest point of the demo trajectory, 36 mm above stowed [m]
DEMO_PERIOD = 16.0  # demo cycle, ready -> near stowed -> ready [s]
GRAVITY = {"zero": 0.0, "moon": 1.62, "earth": 9.81}

# pose sliders: (label, index in the pose vector, min, max, display scale)
SLIDERS = [("x [mm]", 0, -60.0, 60.0, 1000.0), ("y [mm]", 1, -60.0, 60.0, 1000.0),
           ("z [mm]", 2, -336.0, 140.0, 1000.0), ("roll [deg]", 3, -8.0, 8.0, 180.0 / math.pi),
           ("pitch [deg]", 4, -8.0, 8.0, 180.0 / math.pi), ("yaw [deg]", 5, -10.0, 10.0, 180.0 / math.pi)]


# command state on the device: ints, then float64
S_MODE, S_FRAME = 0, 1
NI = 2
HOLD, MOVE, DEMO = range(3)   # hold the pose, rate-limited move to the goal, demo trajectory
C_POSE, C_GOAL, C_FROM, C_LAST, C_DEMO_T, C_TARGET, C_LEGQ, C_LEGF, C_CLAMPED = 0, 6, 12, 18, 24, 25, 31, 37, 43
NC = 44

D = wp.float64
vec6d = wp.types.vector(6, wp.float64)
DEMO_Z0 = wp.constant(D(DEMO_LOW))
DEMO_W = wp.constant(D(2.0 * math.pi / DEMO_PERIOD))
DEMO_FADE = wp.constant(D(0.55 * DEMO_LOW - DEMO_LOW))   # sway fade-in band above the low point
DEMO_ROT = wp.constant(D(math.radians(5.0)))
DEMO_YAW = wp.constant(D(math.radians(7.0)))


@wp.struct
class Rig:
    """Actuator indices and hexapod geometry."""
    leg_B: wp.array(dtype=wp.vec3d)  # leg base points, NDS frame
    leg_T: wp.array(dtype=wp.vec3d)  # leg top points relative to the ring centre at ready-to-capture
    leg_L0: wp.array(dtype=wp.float64)
    leg_lo: wp.array(dtype=wp.float64)  # usable stroke (joint limits less 2 mm)
    leg_hi: wp.array(dtype=wp.float64)
    leg_q: wp.array(dtype=int)
    leg_d: wp.array(dtype=int)


@wp.func
def strokes(rig: Rig, x: vec6d):
    """Closed-form IK: actuator strokes (length - rest length) for a ring pose (dx, dy, dz, rx, ry, rz)
    relative to ready-to-capture."""
    cx, sx, cy, sy, cz, sz = wp.cos(x[3]), wp.sin(x[3]), wp.cos(x[4]), wp.sin(x[4]), wp.cos(x[5]), wp.sin(x[5])
    R = wp.mat33d(cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
                  sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
                  -sy, cy * sx, cy * cx)
    c = wp.vec3d(x[0], x[1], D(NDS_HCS_Z) + D(RING_READY_Z) + x[2])
    out = vec6d()
    for i in range(6):
        out[i] = wp.length(c + R * rig.leg_T[i] - rig.leg_B[i]) - rig.leg_L0[i]
    return out


@wp.func
def reachable(rig: Rig, x: vec6d):
    s = strokes(rig, x)
    ok = True
    for i in range(6):
        if s[i] < rig.leg_lo[i] or s[i] > rig.leg_hi[i]:
            ok = False
    return ok


@wp.func
def load6(c: wp.array(dtype=wp.float64), k: int):
    return vec6d(c[k], c[k + 1], c[k + 2], c[k + 3], c[k + 4], c[k + 5])


@wp.func
def store6(c: wp.array(dtype=wp.float64), k: int, x: vec6d):
    for i in range(6):
        c[k + i] = x[i]


@wp.func
def demo_pose(t: wp.float64):
    """6-DOF Lissajous through the ring's working envelope: the ring cycles from ready down to just above stowed
    (period DEMO_PERIOD) while swaying and tilting; the sway fades out on the way down, so the ring arrives near
    the stow level and centred with all six actuators at the same stroke."""
    z = D(0.5) * DEMO_Z0 * (D(1.0) - wp.cos(DEMO_W * t))
    u = wp.clamp((z - DEMO_Z0) / DEMO_FADE, D(0.0), D(1.0))
    e = u * u * (D(3.0) - D(2.0) * u)   # smoothstep: 0 below ~75 % of the descent, 1 in the upper half
    a, r = D(0.035) * e, DEMO_ROT * e
    return vec6d(a * wp.sin(D(0.7) * t), a * wp.sin(D(1.3) * t), z, r * wp.sin(D(1.1) * t), r * wp.sin(D(0.8) * t),
                 DEMO_YAW * e * wp.sin(D(0.6) * t))


@wp.kernel
def command_kernel(rig: Rig, frame_dt: wp.float64, ist: wp.array(dtype=int), c: wp.array(dtype=wp.float64),
                   target_q: wp.array(dtype=float)):
    """Ring command for one frame: demo / rate-limited move / hold, clamped to the actuator stroke."""
    mode = ist[S_MODE]
    pose = load6(c, C_POSE)
    if mode == DEMO:
        t = c[C_DEMO_T]
        w = wp.min(D(1.0), t / D(1.5))
        w = w * w * (D(3.0) - D(2.0) * w)
        pose = (D(1.0) - w) * load6(c, C_FROM) + w * demo_pose(t)
        c[C_DEMO_T] = t + frame_dt
    elif mode == MOVE:   # rate-limited move (actuators run at ~25 mm/s)
        d = load6(c, C_GOAL) - pose
        done = True
        for i in range(6):
            r = D(0.1) * frame_dt
            if i < 3:
                r = D(0.025) * frame_dt
            pose[i] = pose[i] + wp.clamp(d[i], -r, r)
            if wp.abs(d[i]) >= D(1.0e-6):
                done = False
        if done:
            ist[S_MODE] = HOLD
    clamped = D(0.0)
    if not reachable(rig, pose):   # outside the actuators' stroke: stop on the line from the last reachable pose
        clamped = D(1.0)
        last = load6(c, C_LAST)
        a, b = D(0.0), D(1.0)
        for _ in range(12):
            m = D(0.5) * (a + b)
            if reachable(rig, last + m * (pose - last)):
                a = m
            else:
                b = m
        pose = last + a * (pose - last)
    c[C_CLAMPED] = clamped
    store6(c, C_POSE, pose)
    store6(c, C_LAST, pose)
    s = strokes(rig, pose)
    store6(c, C_TARGET, s)
    for i in range(6):
        target_q[rig.leg_q[i]] = float(s[i])
    ist[S_FRAME] = ist[S_FRAME] + 1


@wp.kernel
def measure_kernel(rig: Rig, joint_q: wp.array(dtype=float), qfrc: wp.array(dtype=float),
                   c: wp.array(dtype=wp.float64)):
    for i in range(6):
        c[C_LEGQ + i] = D(joint_q[rig.leg_q[i]])
        c[C_LEGF + i] = D(qfrc[rig.leg_d[i]])


@wp.kernel
def set_axis(ist: wp.array(dtype=int), c: wp.array(dtype=wp.float64), i: int, value: wp.float64):
    ist[S_MODE] = HOLD
    c[C_POSE + i] = value


@wp.kernel
def set_goal(ist: wp.array(dtype=int), c: wp.array(dtype=wp.float64), goal: vec6d):
    ist[S_MODE] = MOVE
    store6(c, C_GOAL, goal)


@wp.kernel
def set_demo(ist: wp.array(dtype=int), c: wp.array(dtype=wp.float64), on: int):
    if on != 0:
        ist[S_MODE] = DEMO
        c[C_DEMO_T] = D(0.0)
        for i in range(6):
            c[C_FROM + i] = c[C_POSE + i]
    else:
        ist[S_MODE] = HOLD


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.args = args

        newton.use_coord_layout_targets = True
        from pxr import Usd

        stage = Usd.Stage.Open(os.path.join(args.assets, "nds", "nds.usda"))
        root = stage.GetDefaultPrim()
        root.GetVariantSet("Physics").SetVariantSelection("physics")
        b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -GRAVITY[args.gravity]))
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.add_usd(stage, floating=False, collapse_fixed_joints=False, enable_self_collisions=False,
                  hide_collision_shapes=True)
        b.add_ground_plane(height=-0.12)   # test-stand floor under the mounting flange
        labels = list(b.joint_label)
        self.legs = []
        for i in range(6):
            name = f"/nds/Physics/leg{i + 1}_rod"
            p = stage.GetPrimAtPath(name)
            j = labels.index(name)
            self.legs.append({"q": b.joint_q_start[j], "d": b.joint_qd_start[j],
                              "B": np.array(p.GetAttribute("idss:base").Get()),
                              "T": np.array(p.GetAttribute("idss:top").Get()),
                              "L0": p.GetAttribute("idss:restLength").Get()})
        b.request_state_attributes("mujoco:qfrc_actuator")
        self.model = b.finalize()
        ke = self.model.joint_target_ke.numpy()
        kd = self.model.joint_target_kd.numpy()
        for leg in self.legs:   # stiff position servos (ball-screw actuators)
            ke[leg["d"]], kd[leg["d"]] = args.leg_ke, args.leg_kd
        self.model.joint_target_ke.assign(ke)
        self.model.joint_target_kd.assign(kd)

        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True, solver="newton",
                                                  integrator="implicitfast", iterations=30, ls_iterations=50)
        eq = self.solver.mjw_model.eq_solref.numpy()   # stiff loop closures
        eq[..., 0] = 2.0 * self.sim_dt
        self.solver.mjw_model.eq_solref.assign(eq)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._init_controller()
        self.viewer.set_model(self.model)
        self.capture()
        if args.demo:
            self.set_demo(True)

    # -------------------------------------------------------------- controller setup
    def _init_controller(self):
        lo, hi = self.model.joint_limit_lower.numpy(), self.model.joint_limit_upper.numpy()
        ring_centre = np.array([0.0, 0.0, NDS_HCS_Z + RING_READY_Z])
        rig = Rig()
        rig.leg_B = wp.array([leg["B"] for leg in self.legs], dtype=wp.vec3d)
        rig.leg_T = wp.array([leg["T"] - ring_centre for leg in self.legs], dtype=wp.vec3d)
        rig.leg_L0 = wp.array([leg["L0"] for leg in self.legs], dtype=wp.float64)
        rig.leg_lo = wp.array([lo[leg["d"]] + 0.002 for leg in self.legs], dtype=wp.float64)
        rig.leg_hi = wp.array([hi[leg["d"]] - 0.002 for leg in self.legs], dtype=wp.float64)
        rig.leg_q = wp.array([leg["q"] for leg in self.legs], dtype=int)
        rig.leg_d = wp.array([leg["d"] for leg in self.legs], dtype=int)
        self.rig = rig
        self.ist = wp.zeros(NI, dtype=int)
        self.cst = wp.zeros(NC, dtype=wp.float64)
        qfrc = getattr(getattr(self.state_0, "mujoco", None), "qfrc_actuator", None)
        self.qfrc = qfrc if qfrc is not None else wp.zeros(self.model.joint_dof_count, dtype=float)
        # host side: telemetry snapshots land in a ring of pinned buffers, each tagged with a CUDA event
        pinned = wp.get_device().is_cuda
        self._slots = [SimpleNamespace(ist=wp.empty(NI, dtype=int, device="cpu", pinned=pinned),
                                       cst=wp.empty(NC, dtype=wp.float64, device="cpu", pinned=pinned),
                                       evt=None, busy=False) for _ in range(3)]
        self._pending = deque()
        self._launched = 0
        self._hold = 0   # GUI edits not yet reflected in the snapshots: keep the host's values until this frame
        self.mode = HOLD
        self.pose = np.zeros(6)        # commanded ring pose relative to ready-to-capture (m, rad)
        self.slider = np.zeros(6)      # slider values (display units)
        self.clamped = False
        self.leg_f = np.zeros(6)
        self.legq = np.zeros(6)
        self.target = np.zeros(6)

    # -------------------------------------------------------------- GUI commands (asynchronous, picked up next frame)
    def set_axis(self, i, value):
        wp.launch(set_axis, dim=1, inputs=[self.ist, self.cst, i, value])
        self.mode, self.pose[i], self._hold = HOLD, value, self._launched + 1

    def set_goal(self, goal):
        wp.launch(set_goal, dim=1, inputs=[self.ist, self.cst, vec6d(*goal)])
        self.mode, self._hold = MOVE, self._launched + 1

    def set_demo(self, on):
        wp.launch(set_demo, dim=1, inputs=[self.ist, self.cst, int(on)])
        self.mode, self._hold = (DEMO if on else HOLD), self._launched + 1

    # -------------------------------------------------------------- telemetry (host)
    def _readback(self):
        s = self._slots[self._launched % len(self._slots)]
        while s.busy:
            self._poll(wait=True, limit=1)
        wp.copy(s.ist, self.ist)
        wp.copy(s.cst, self.cst)
        s.evt = wp.record_event() if wp.get_device().is_cuda else None
        s.busy = True
        self._pending.append(s)
        self._launched += 1

    def _poll(self, wait=False, limit=None):
        n = 0
        while self._pending and (limit is None or n < limit):
            s = self._pending[0]
            if s.evt is not None:
                if wait:
                    wp.synchronize_event(s.evt)
                elif not s.evt.is_complete:
                    break
            self._pending.popleft()
            self._consume(s)
            s.busy = False
            n += 1

    def _consume(self, s):
        i, c = s.ist.numpy(), s.cst.numpy()
        self.legq, self.leg_f = c[C_LEGQ:C_LEGQ + 6].copy(), c[C_LEGF:C_LEGF + 6].copy()
        self.target, self.clamped = c[C_TARGET:C_TARGET + 6].copy(), bool(c[C_CLAMPED])
        if int(i[S_FRAME]) >= self._hold:
            self.mode, self.pose = int(i[S_MODE]), c[C_POSE:C_POSE + 6].copy()
            self.slider = np.array([self.pose[k] * sc for _, k, _, _, sc in SLIDERS])

    # -------------------------------------------------------------- sim loop
    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        wp.launch(command_kernel, dim=1, inputs=[self.rig, self.frame_dt, self.ist, self.cst,
                                                 self.control.joint_target_q])
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(measure_kernel, dim=1, inputs=[self.rig, self.state_0.joint_q, self.qfrc, self.cst])

    def step(self):
        self._poll()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self._readback()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_scalar("Ring height [mm]", float(self.pose[2] * 1000.0))
        self.viewer.log_scalar("Max actuator load [N]", float(np.max(np.abs(self.leg_f))))
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Ring pose (from ready)")
        for k, (label, i, lo, hi, sc) in enumerate(SLIDERS):
            changed, v = ui.slider_float(label, float(self.slider[k]), lo, hi)
            if changed:
                self.slider[k] = v
                self.set_axis(i, v / sc)
        if ui.button("Ready"):
            self.set_goal(np.zeros(6))
        if ui.button("Stow"):
            self.set_goal([0.0, 0.0, RING_STOWED_Z - RING_READY_Z, 0.0, 0.0, 0.0])
        changed, on = ui.checkbox("Demo trajectory", self.mode == DEMO)
        if changed:
            self.set_demo(on)
        if self.clamped:
            ui.text("Clamped to actuator stroke")
        ui.separator()
        ui.text("Actuators: stroke mm / load N")
        for k in range(6):
            ui.text(f"{k + 1}: {self.legq[k] * 1000:6.1f} / {self.leg_f[k]:6.0f}")

    def test_final(self):
        self._poll(wait=True)
        err = np.max(np.abs(self.legq - self.target))
        assert err < 0.003, f"actuators not tracking: {err * 1000:.1f} mm"
        print(f"stewart: max tracking error {err * 1000:.2f} mm, max load {np.max(np.abs(self.leg_f)):.0f} N")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--assets", default=ASSETS, help="Folder holding nds/nds.usda.")
        parser.add_argument("--gravity", default="zero", choices=list(GRAVITY), help="Test-stand gravity.")
        parser.add_argument("--demo", action="store_true", help="Start with the 6-DOF demo trajectory.")
        parser.add_argument("--leg-ke", type=float, default=3.0e5, help="Actuator servo stiffness [N/m].")
        parser.add_argument("--leg-kd", type=float, default=3.0e3, help="Actuator servo damping [N s/m].")
        parser.add_argument("--substeps", type=int, default=8, help="Physics substeps per 60 Hz frame.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
