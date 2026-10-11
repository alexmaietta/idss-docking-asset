###########################################################################
# Example IDSS Docking (MuJoCo)
#
# Plain-MuJoCo (CPU) version of examples/newton/example_idss_docking.py: a
# visiting vehicle with an active IDSS docking system (NASA Docking System
# Block 1) docks to the International Docking Adapter. Both halves come from
# the MJCF docking scene (idss_docking_scene.xml): the IDA is fixed to the
# world, the NDS rides on a free-floating 13 t host vehicle in zero g.
# Contact is MuJoCo's own convex collision on the exported hulls (exact
# convex sectors for the capture rings and guide-pin receptacles, CoACD
# pieces elsewhere); the hexapod and capture-latch reference points come from
# the exported sites and custom numerics.
#
# The docking sequence is the Newton example's state machine: approach,
# contact (compliant ring, capture latches snap past the striker steps and
# are braked), captured, attenuate, align, retract to the 4.2 mm
# ready-to-hook gap, hook gang 1 / 2 (swing and preload), docked, and
# optionally undock (hooks open, latches driven in, separators push off).
#
# Keys: U undock, 9 restart. The top-right overlay shows the phase and live
# loads.
#
# Command (from the idss repo root):
#   python examples/mujoco/example_idss_docking.py
#   python examples/mujoco/example_idss_docking.py --offset 0.06 --misalign 3 --closing-speed 0.1
#   python examples/mujoco/example_idss_docking.py --headless --seconds 60 --undock-after 3
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

# IDSS interface (IDSS IDD Rev F, NDS IDD Rev H; see the idss repo scripts/params.py)
IDA_HCS_Z = 0.642  # HCS mating plane above the IDA's PMA interface [m]
NDS_HCS_Z = 0.2304  # HCS mating plane above the NDS host-vehicle interface [m]
SCS_PASSIVE_Z = -0.006  # passive capture-ring plane below the HCS plane [m]
RING_READY_Z = 0.33  # NDS capture ring extension at ready-to-capture [m]
READY_TO_HOOK = 0.0042  # HCS gap at which the hooks are swung (IDD Fig 3.2.3.4-2) [m]
HOOK_CLOSE = math.radians(28.0)  # hook swing open -> tangent position [rad]
HOOK_PULL = 0.008  # pull-down carriage command [m]; the carriage force limit (40 kN) sets the preload
PASSIVE_HOOK_K = 8.0e7  # Belleville stack stiffness [N/m] (IDD: 69-94 kN/mm)
PASSIVE_HOOK_PRE = 1.0e-4  # Belleville preload compression [m]
STRIKER_X_R = 0.601  # point X: striker 45 deg surface meets the SCS plane on the 1202 mm circle [m]
STRIKER_STEP_S = 0.114  # catch step, along the 45 deg surface from point X [m] (IDD Fig 3.2.2.4-1)

PHASES = ["approach", "contact", "capture failed", "captured", "attenuate", "align", "retract", "hook gang 1", "hook gang 2",
          "docked", "undock", "separated"]
KEY_UNDOCK, KEY_RESTART = ord("U"), ord("9")


def quat_axis_angle(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, dtype=float), angle)
    return q


def quat_mul(a, b):
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, a, b)
    return q


def rot_matrix(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def rotation(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return np.array([[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                     [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                     [-sy, cy * sx, cy * cx]])


def revolved_mesh(spec, name, rings, n=48, top_cap=False):
    """Visual-only surface of revolution from (r, z) rings (outward-facing triangles)."""
    pts, tris = [], []
    for r, z in rings:
        pts += [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z) for i in range(n)]
    for k in range(len(rings) - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            tris += [a, b, a + n, b, b + n, a + n]
    if top_cap:   # annular top deck (the NDS sits in its centre)
        pts.append((0.0, 0.0, rings[0][1]))
        tris += [t for i in range(n) for t in (len(pts) - 1, i, (i + 1) % n)]
    spec.add_mesh(name=name, uservert=np.array(pts).flatten(), userface=tris, inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)


class Example:
    def __init__(self, args):
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = args.substeps
        spec = mujoco.MjSpec.from_file(os.path.join(args.mjcf, "idss_docking_scene.xml"))
        spec.option.timestep = self.frame_dt / self.substeps
        spec.option.iterations, spec.option.ls_iterations = args.iterations, args.ls_iterations
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if args.cone == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
        # visual-only context: PMA stub under the IDA, crew-module forebody under the NDS host plane
        revolved_mesh(spec, "pma_stub", [(0.80, 0.0), (0.86, -0.08), (1.05, -0.9), (1.18, -1.5), (1.18, -1.6)])
        revolved_mesh(spec, "visiting_vehicle", [(0.82, 0.0), (0.87, -0.06), (2.18, -2.15), (2.25, -2.28), (2.2, -2.4),
                                                 (1.35, -2.62), (0.01, -2.7)], top_cap=True)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="pma_stub", rgba=(0.72, 0.72, 0.70, 1),
                                contype=0, conaffinity=0, density=0)
        spec.body("nds_nds_base").add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="visiting_vehicle", density=0,
                                           rgba=(0.86, 0.86, 0.84, 1), contype=0, conaffinity=0)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.m.eq_solref[:, 0] = 2.0 * self.m.opt.timestep   # stiff loop closures (Stewart platform ball joints)
        self._index()
        self.restart()

    # -------------------------------------------------------------- model lookup
    def _index(self):
        m = self.m
        act = lambda name: m.actuator(name).id   # noqa: E731
        qadr = lambda name: m.jnt_qposadr[m.joint(name).id]   # noqa: E731
        self.free_q = qadr("nds_nds_base_free")
        self.free_d = m.jnt_dofadr[m.joint("nds_nds_base_free").id]
        self.nds_base = m.body("nds_nds_base").id
        self.legs = np.array([act(f"nds_leg{i}_rod_servo") for i in range(1, 7)])
        self.leg_q = np.array([qadr(f"nds_leg{i}_rod") for i in range(1, 7)])
        self.leg_d = np.array([m.jnt_dofadr[m.joint(f"nds_leg{i}_rod").id] for i in range(1, 7)])
        self.latch_j = np.array([m.joint(f"nds_latch_{k}").id for k in (1, 2, 3)])
        self.latch_q = m.jnt_qposadr[self.latch_j]
        self.latch_catch = np.array([m.site(f"nds_latch_{k}_catch").id for k in (1, 2, 3)])
        gangs = ([0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11])
        self.hooks = [np.array([act(f"nds_hook_a_{k:02d}_servo") for k in g]) for g in gangs]
        self.hook_pulls = [np.array([act(f"nds_hook_c_{k:02d}_servo") for k in g]) for g in gangs]
        self.ida_passive = np.array([qadr(f"ida_hook_p_{k:02d}") for k in range(12)])
        self.nds_separators = np.array([act(f"nds_separator_{k}_servo") for k in (1, 2, 3)])
        self.default = {"range": m.jnt_range.copy(), "solref": m.jnt_solref.copy(), "gain": m.actuator_gainprm.copy(),
                        "bias": m.actuator_biasprm.copy(), "force": m.actuator_forcerange.copy()}
        # hexapod geometry from the exported reference sites (ring at ready-to-capture), in the NDS frame
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        p, R = d.xpos[self.nds_base], d.xmat[self.nds_base].reshape(3, 3)
        local = lambda name: (d.site(name).xpos - p) @ R   # noqa: E731
        self.B = np.array([local(f"nds_leg{i}_rod_base") for i in range(1, 7)])
        self.ring_centre = local("nds_scs_ring_centre")
        self.T = np.array([local(f"nds_leg{i}_rod_top") for i in range(1, 7)]) - self.ring_centre
        self.L0 = np.array([m.numeric(f"nds_leg{i}_rod_rest_length").data[0] for i in range(1, 7)])

    def restart(self):
        a, m, d = self.args, self.m, self.d
        for k in ("range", "solref"):
            getattr(m, f"jnt_{k}")[:] = self.default[k]
        m.actuator_gainprm[:], m.actuator_biasprm[:] = self.default["gain"], self.default["bias"]
        m.actuator_forcerange[:] = self.default["force"]
        mujoco.mj_resetData(m, d)
        # visiting vehicle: NDS mirrored about the line of androgyny (+Y), approaching from +Z
        mirror = quat_axis_angle((0, 1, 0), math.pi)
        tilt = quat_axis_angle((1, 0, 0) if a.misalign_axis == "x" else (0, 1, 0), math.radians(a.misalign))
        q0 = quat_mul(tilt, mirror)
        hcs = np.array([a.offset, 0.0, IDA_HCS_Z + a.gap])
        d.qpos[self.free_q:self.free_q + 3] = hcs - rot_matrix(q0) @ (0, 0, NDS_HCS_Z)
        d.qpos[self.free_q + 3:self.free_q + 7] = q0
        # initial relative motion (IDSS IDD 3.3 capture conditions); free-joint angular velocity is in the body frame
        d.qvel[self.free_d:self.free_d + 3] = [a.lateral_speed, 0.0, -a.closing_speed]
        d.qvel[self.free_d + 3:self.free_d + 6] = rot_matrix(q0).T @ (0.0, math.radians(a.rate), 0.0)
        mujoco.mj_forward(m, d)
        self.sim_time = 0.0
        self.wall0 = time.perf_counter()
        self.phase = "approach"
        self.phase_t = 0.0
        self.latched = [False, False, False]
        self.latch_seen = [False, False, False]
        self.ring_cmd = np.zeros(6)  # ring pose command relative to ready: (dx, dy, dz, rx, ry, rz)
        self.hook_cmd = [0.0, 0.0]
        self.pull_cmd = [0.0, 0.0]
        self.events = []
        self.peak = {"ring_load": 0.0}
        self.meas = None
        self._released = False
        self._set_leg_gains(a.leg_ke, a.leg_kd)
        self._write_targets()

    # -------------------------------------------------------------- hexapod kinematics and drives
    def leg_lengths(self, cmd):
        """Closed-form Stewart-platform IK: leg strokes (length - rest length) for a ring pose
        (dx, dy, dz, rx, ry, rz) relative to ready-to-capture, in the NDS frame."""
        R = rotation(cmd[3], cmd[4], cmd[5])
        tops = self.ring_centre + self.T @ R.T + np.asarray(cmd[:3])
        return np.linalg.norm(tops - self.B, axis=1) - self.L0

    def ring_pose_from_legs(self):
        """Hexapod forward kinematics (Gauss-Newton on the IK, warm-started from the last command)."""
        x = self.ring_cmd.copy()
        target = self.d.qpos[self.leg_q]
        eye = np.eye(6) * 1e-6
        for _ in range(3):
            f = self.leg_lengths(x) - target
            J = np.stack([(self.leg_lengths(x + eye[k]) - f - target) / 1e-6 for k in range(6)], axis=1)
            x = x - np.linalg.solve(J, f)
        return x

    def _set_servo(self, act, kp, kv):
        self.m.actuator_gainprm[act, 0], self.m.actuator_biasprm[act, 1], self.m.actuator_biasprm[act, 2] = kp, -kp, -kv

    def _set_leg_gains(self, ke, kd):
        # the servo damping goes on the leg joints, not the actuators: MuJoCo integrates joint damping implicitly,
        # but a force-clamped actuator damper turns explicit when it saturates, and at the align/retract damping
        # (4.5e5 N s/m on ~2 kg of armature) that blows the hexapod up under the hook preload
        self._set_servo(self.legs, ke, 0.0)
        self.m.dof_damping[self.leg_d] = kd

    def _set_leg_force(self, f):
        self.m.actuator_forcerange[self.legs] = (-f, f)

    def _write_targets(self):
        self.d.ctrl[self.legs] = self.leg_lengths(self.ring_cmd)
        for g in range(2):
            self.d.ctrl[self.hooks[g]] = self.hook_cmd[g]
            self.d.ctrl[self.hook_pulls[g]] = self.pull_cmd[g]

    # -------------------------------------------------------------- measurements
    def measure(self):
        d = self.d
        pn, Rn = d.xpos[self.nds_base], d.xmat[self.nds_base].reshape(3, 3)
        hcs_nds = pn + Rn[:, 2] * NDS_HCS_Z
        gap = hcs_nds[2] - IDA_HCS_Z
        lateral = math.hypot(hcs_nds[0], hcs_nds[1])
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, -Rn[2, 2]))))
        leg_f = d.actuator_force[self.legs].copy()
        preload = PASSIVE_HOOK_K * (d.qpos[self.ida_passive] + PASSIVE_HOOK_PRE)
        # capture sensor: each latch's catch face, as a position along the IDA striker's 45 deg surface measured
        # from point X (the IDA is fixed at the world origin, so its frame is the world frame)
        slant = []
        for p in d.site_xpos[self.latch_catch]:
            slant.append((STRIKER_X_R - math.hypot(p[0], p[1]) + (IDA_HCS_Z + SCS_PASSIVE_Z) - p[2]) / math.sqrt(2.0))
        return {"gap": gap, "lateral": lateral, "tilt": tilt, "leg_f": leg_f, "ring_load": float(np.sum(leg_f)),
                "preload": preload, "latch": d.qpos[self.latch_q].copy(), "latch_slant": np.array(slant),
                "contacts": int(d.ncon)}

    # -------------------------------------------------------------- docking sequence (IDSS / NDS ConOps)
    def _goto(self, phase):
        self.events.append((round(self.sim_time, 2), phase))
        print(f"[{self.sim_time:7.2f} s] {phase}  (wall {time.perf_counter() - self.wall0:.1f} s)")
        self.phase, self.phase_t = phase, 0.0

    def sequence(self):
        a, mdl = self.args, self.m
        m = self.meas = self.measure()
        self.phase_t += self.frame_dt
        self.peak["ring_load"] = max(self.peak["ring_load"], float(np.max(np.abs(m["leg_f"]))))
        ph = self.phase
        # capture sensing: each spring-loaded latch roller must be past its striker's catch step; soft capture is
        # declared when all three are past together (ring seated and centred), and only then are the latches
        # locked (NDS capture-latch brakes). A latch locked early would wedge the ring off-centre.
        past = m["latch_slant"] > STRIKER_STEP_S + 0.0003
        if all(self.latched) and ph not in ("undock", "separated"):   # ratchet the pawls as they drop onto the striker
            for k, j in enumerate(self.latch_j):
                mdl.jnt_range[j, 1] = min(mdl.jnt_range[j, 1], m["latch"][k] + 0.001)
        if ph == "contact" and not all(self.latched):
            for k in range(3):
                if past[k] and not self.latch_seen[k]:
                    self.latch_seen[k] = True
                    print(f"[{self.sim_time:7.2f} s] latch {k + 1} past the striker step")
            if past.all():
                self.latched = [True, True, True]
                mdl.jnt_solref[self.latch_j] = (2.0 * mdl.opt.timestep, 1.0)   # latch brake: stiff stop
                mdl.jnt_range[self.latch_j, 1] = m["latch"] + 0.001           # pawl: free to swing out, never back in
                print(f"[{self.sim_time:7.2f} s] latches locked (rockers {np.round(np.degrees(m['latch']), 1)} deg)")
        if ph in ("contact", "attenuate"):
            # compliance mode (NDS load-sensing ring control): the commanded ring pose follows the measured pose
            # (hexapod forward kinematics), so the ring floats laterally and in rotation and aligns itself to the
            # passive ring; during capture an axial offset presses the capture rings together so the latches seat
            x = self.ring_pose_from_legs()
            push = a.capture_push if ph == "contact" else 0.0
            self.ring_cmd = x + np.array([0.0, 0.0, push, 0.0, 0.0, 0.0])
        if ph == "approach" and m["contacts"] > 0:
            self._goto("contact")
        elif ph == "contact" and all(self.latched):
            self._goto("captured")
        elif ph == "contact" and self.phase_t > a.capture_timeout:
            self._goto("capture failed")
        elif ph == "captured":
            # load-limited from here on: the soft-capture interface is rated for ~3.5 kN / 2.8 kN m (IDSS IDD)
            self._set_leg_force(a.leg_force_limit)
            self._set_leg_gains(a.leg_ke * 0.25, a.leg_kd * 4.0)   # attenuation: soft and heavily damped
            self._goto("attenuate")
        elif ph == "attenuate" and self.phase_t > a.attenuate_time:
            self.ring_cmd = self.ring_pose_from_legs()
            self._align_from = self.ring_cmd.copy()
            self._set_leg_force(a.retract_force_limit)   # position-hold servo for alignment and retraction
            self._set_leg_gains(a.align_ke, a.align_kd)
            self._goto("align")
        elif ph == "align":
            # null the lateral and angular ring offsets (this swings the vehicle into line); hold the axial stroke
            s = min(1.0, self.phase_t / a.align_time)
            s = s * s * (3 - 2 * s)
            keep = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            self.ring_cmd = self._align_from * (keep + (1.0 - keep) * (1.0 - s))
            # retract only once the vehicle is actually in line (tilt and lateral offset nulled), or after a timeout
            aligned = m["tilt"] < 0.1 and m["lateral"] < 0.0005
            if self.phase_t > a.align_time + 0.5 and (aligned or self.phase_t > a.align_time + 15.0):
                self._goto("retract")
        elif ph == "retract":
            # pull the vehicle in to the ready-to-hook gap (4.2 mm); slow down while the guide pins seat, then swing
            # the hooks only once the interface is square (a cocked interface lets the swinging hooks pry it apart)
            stop = -(RING_READY_Z + SCS_PASSIVE_Z) + READY_TO_HOOK - 0.0005
            v = a.retract_speed if m["gap"] > a.pin_engage_gap else a.retract_speed_pins
            if self.ring_cmd[2] <= stop + 1e-9 and m["gap"] > READY_TO_HOOK + 0.0003:
                # MuJoCo: the latched ring rides ~2 mm up on the latch teeth under tension (the stop above assumes
                # seated rings), so close the last millimeters on the measured HCS gap, 2 mm/s, at most 5 mm extra
                floor, v = stop - 0.005, 0.002
            else:
                floor = stop
            self.ring_cmd[2] = max(floor, self.ring_cmd[2] - v * self.frame_dt)
            square = m["tilt"] < 0.15 and m["lateral"] < 0.001
            if self.ring_cmd[2] <= stop and m["gap"] < READY_TO_HOOK + 0.0008 and square:
                self._goto("hook gang 1")
        elif ph in ("hook gang 1", "hook gang 2"):
            # the gang swings its six hooks over the mating passive hooks, then the carriages pull the hook
            # tangent planes down: the interface closes, the seals compress and the Belleville stacks preload
            g = 0 if ph == "hook gang 1" else 1
            s1 = min(1.0, self.phase_t / a.hook_time)
            s2 = min(1.0, max(0.0, self.phase_t - a.hook_time) / a.hook_time)
            self.hook_cmd[g] = HOOK_CLOSE * s1
            self.pull_cmd[g] = HOOK_PULL * s2
            if self.phase_t > 2 * a.hook_time + 0.5:
                self._goto("hook gang 2" if g == 0 else "docked")
                if g == 1:
                    self._report_docked(self.measure())
        elif ph == "docked" and a.undock_after > 0 and self.phase_t > a.undock_after:
            self.undock()
        elif ph == "undock":
            s1 = min(1.0, self.phase_t / a.hook_time)
            s2 = min(1.0, max(0.0, self.phase_t - a.hook_time) / a.hook_time)
            self.pull_cmd = [HOOK_PULL * (1 - s1)] * 2
            self.hook_cmd = [HOOK_CLOSE * (1 - s2)] * 2
            if self.phase_t > 2 * a.hook_time and not self._released:
                self._released = True
                self._set_servo(self.nds_separators, 0.0, 0.0)   # separators: release the lock, the springs push off
                mdl.jnt_range[self.latch_j] = (0.15, 0.2)          # capture latches: motor drives the rockers in
                self.ring_cmd = np.zeros(6)
                self._set_leg_gains(a.leg_ke, a.leg_kd)
            if self._released and m["gap"] > 0.1:
                self._goto("separated")
        self._write_targets()

    def undock(self):
        if self.phase in ("docked", "hook gang 2"):
            self._released = False
            self._goto("undock")

    def key(self, keycode):
        if keycode == KEY_UNDOCK:
            self._pending = "undock"
        elif keycode == KEY_RESTART:
            self._pending = "restart"

    def _report_docked(self, m):
        pre = m["preload"]
        print(f"hard docked: HCS gap {m['gap'] * 1000:.2f} mm, misalignment {m['lateral'] * 1000:.2f} mm / "
              f"{m['tilt']:.3f} deg; passive hook loads {np.round(pre / 1000, 1)} kN (total {pre.sum() / 1000:.0f} kN)")

    # -------------------------------------------------------------- sim loop
    def step(self):
        pending, self._pending = getattr(self, "_pending", None), None
        if pending == "restart":
            self.restart()
        elif pending == "undock":
            self.undock()
        if self.phase == "diverged":
            return
        if not np.isfinite(self.d.qpos).all():
            print(f"[{self.sim_time:7.2f} s] simulation diverged in phase '{self.phase}' (try more --substeps)")
            self.phase = "diverged"
            return
        self.sequence()
        mujoco.mj_step(self.m, self.d, nstep=self.substeps)
        self.sim_time += self.frame_dt

    def status(self):
        m = self.meas
        if m is None:
            return f"Phase\n{self.phase}", ""
        rows = [("Phase", f"{self.phase} ({self.sim_time:.1f} s)"), ("HCS gap", f"{m['gap'] * 1000:.2f} mm"),
                ("Lateral / tilt", f"{m['lateral'] * 1000:.2f} mm / {m['tilt']:.2f} deg"),
                ("Latches", " ".join("LOCKED" if self.latched[k] else f"{math.degrees(m['latch'][k]):.1f}" for k in range(3))),
                ("Leg loads [N]", " ".join(f"{f:.0f}" for f in m["leg_f"])),
                ("Hook gangs", f"{math.degrees(self.hook_cmd[0]):.0f} / {math.degrees(self.hook_cmd[1]):.0f} deg, pull "
                               f"{self.pull_cmd[0] * 1000:.1f} / {self.pull_cmd[1] * 1000:.1f} mm"),
                ("Hook preload", f"{np.sum(m['preload']) / 1000:.1f} kN"), ("Contacts", str(m["contacts"]))]
        return "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)

    def report(self, wall):
        m = self.measure()
        print(f"idss docking: phase {self.phase}, HCS gap {m['gap'] * 1000:.2f} mm, misalignment {m['lateral'] * 1000:.2f} mm"
              f" / {m['tilt']:.3f} deg, hook preload {np.sum(m['preload']) / 1000:.0f} kN, peak leg load "
              f"{self.peak['ring_load']:.0f} N")
        print(f"idss docking: {self.sim_time:.1f} s sim in {wall:.1f} s wall (real-time factor {self.sim_time / wall:.2f}x, "
              f"dt {self.m.opt.timestep * 1000:.3f} ms), events {self.events}")


def main():
    parser = argparse.ArgumentParser(description="IDSS docking (plain MuJoCo).")
    parser.add_argument("--mjcf", default=MJCF, help="Folder holding idss_docking_scene.xml (and ida/, nds/).")
    parser.add_argument("--gap", type=float, default=0.62, help="Initial HCS-plane separation [m].")
    parser.add_argument("--closing-speed", type=float, default=0.08, help="Closing velocity [m/s] (IDSS 0.05-0.10).")
    parser.add_argument("--lateral-speed", type=float, default=0.0, help="Lateral velocity [m/s] (IDSS <= 0.04).")
    parser.add_argument("--offset", type=float, default=0.04, help="Lateral offset [m] (IDSS <= 0.10).")
    parser.add_argument("--misalign", type=float, default=2.0, help="Angular misalignment [deg] (IDSS <= 4).")
    parser.add_argument("--misalign-axis", default="x", choices=["x", "y"], help="Axis of the misalignment.")
    parser.add_argument("--rate", type=float, default=0.0, help="Angular rate about y [deg/s] (IDSS <= 0.2).")
    parser.add_argument("--leg-ke", type=float, default=3.0e3, help="Leg servo stiffness at capture [N/m].")
    parser.add_argument("--leg-kd", type=float, default=1.5e3, help="Leg servo damping at capture [N s/m].")
    parser.add_argument("--capture-push", type=float, default=0.015,
                        help="Axial offset of the compliant ring command that presses the capture rings together [m].")
    parser.add_argument("--align-ke", type=float, default=3.0e5,
                        help="Leg servo stiffness, align/retract [N/m] (ball-screw position servo).")
    parser.add_argument("--align-kd", type=float, default=4.5e5, help="Leg servo damping, align/retract [N s/m].")
    parser.add_argument("--pin-engage-gap", type=float, default=0.065,
                        help="HCS gap below which retraction slows for the guide pins [m].")
    parser.add_argument("--retract-speed-pins", type=float, default=0.01, help="Retraction speed as the pins seat [m/s].")
    parser.add_argument("--retract-force-limit", type=float, default=2500.0,
                        help="Actuator force limit during alignment and retraction [N].")
    parser.add_argument("--leg-force-limit", type=float, default=1500.0,
                        help="Actuator force limit after capture [N] (keeps the latch loads inside the IDSS rating).")
    parser.add_argument("--capture-timeout", type=float, default=15.0, help="Give up on soft capture after [s].")
    parser.add_argument("--attenuate-time", type=float, default=4.0, help="Attenuation phase [s].")
    parser.add_argument("--align-time", type=float, default=10.0, help="Alignment phase [s].")
    parser.add_argument("--retract-speed", type=float, default=0.03, help="Ring retraction speed [m/s].")
    parser.add_argument("--hook-time", type=float, default=2.0, help="Hook gang drive time [s].")
    parser.add_argument("--undock-after", type=float, default=0.0, help="Undock this long after hard capture [s] (0: stay).")
    parser.add_argument("--substeps", type=int, default=8, help="Physics substeps per 60 Hz frame (dt = 1/(60 n)).")
    parser.add_argument("--iterations", type=int, default=50, help="MuJoCo solver iterations.")
    parser.add_argument("--ls-iterations", type=int, default=25, help="MuJoCo line-search iterations.")
    parser.add_argument("--cone", default="elliptic", choices=["elliptic", "pyramidal"], help="Friction cone.")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer and print the results.")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="Simulated time for --headless [s]; stops early once docked (or separated with --undock-after).")
    args = parser.parse_args()
    ex = Example(args)
    if args.headless:
        final = "separated" if args.undock_after > 0 else "docked"
        t0 = time.perf_counter()
        while ex.sim_time < args.seconds - 1e-9 and ex.phase not in (final, "capture failed", "diverged"):
            ex.step()
        ex.report(time.perf_counter() - t0)
        return
    import mujoco.viewer

    with mujoco.viewer.launch_passive(ex.m, ex.d, key_callback=ex.key) as v:
        v.cam.lookat[:], v.cam.distance, v.cam.elevation, v.cam.azimuth = (0, 0, 0.9), 5.0, -12, 120
        t0 = time.perf_counter()
        while v.is_running():
            with v.lock():
                restarted = getattr(ex, "_pending", None) == "restart"
                ex.step()
                if restarted:
                    t0 = time.perf_counter()
                v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPRIGHT, *ex.status()))
            v.sync()
            lag = ex.sim_time - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)
        ex.report(time.perf_counter() - t0)


if __name__ == "__main__":
    main()
