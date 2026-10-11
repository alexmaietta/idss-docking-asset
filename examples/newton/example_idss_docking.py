# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example IDSS Docking
#
# A visiting vehicle with an active IDSS docking system (NASA Docking System
# Block 1) docks to the International Docking Adapter on the ISS. Both halves
# are loaded from USD with newton.ModelBuilder.add_usd(): the IDA is fixed to
# the world (the station is ~350 t), the NDS rides on a free-floating 13 t
# host vehicle in zero g.
#
# Every mechanism is simulated, with Newton's collision pipeline feeding
# SolverMuJoCo:
#   - soft capture: the capture ring sits on a Stewart platform (six actuated
#     legs, five of them closing kinematic loops); its three guide petals slide
#     over the IDA's truss petals and strikers (convex-hull and SDF contacts),
#     and the spring-loaded capture latches snap past the 2 mm striker steps;
#   - attenuation, alignment and retraction through closed-form hexapod
#     inverse kinematics with load-cell (actuator force) feedback;
#   - hard capture: two gangs of six active hooks close over the IDA's
#     Belleville-loaded passive hooks (SDF hook profiles from the IDSS IDD) and
#     preload the interface; the guide pins enter their receptacles and the
#     NDS seal beads compress against the IDA's flat sealing land;
#   - undocking: hooks open, latches release and the spring separators push
#     the vehicle away.
#
# The docking controller (sensing, phase sequencing, hexapod kinematics, latch
# ratchet, hook drives) runs in Warp kernels inside the frame's CUDA graph, and
# switches the solver's actuator gains and limits in place, so the host never
# waits on the GPU; telemetry comes back through asynchronous copies.
#
# Initial conditions default to the IDSS IDD capture envelope (closing
# 0.05-0.10 m/s, lateral offset up to 0.10 m, misalignment up to 4 deg).
#
# Keys: u undock, r restart. The GUI shows the docking phase and live loads.
#
# Command (from the idss repo root):
#   python examples/newton/example_idss_docking.py
#   python examples/newton/example_idss_docking.py --offset 0.06 --misalign 3 --closing-speed 0.1
#
###########################################################################

import math
import os
from collections import deque
from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples

wp.set_module_options({"enable_backward": False})   # controller kernels need no gradients (and compile far faster)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ASSETS = os.environ.get("IDSS_ASSETS", os.path.join(REPO, "assets"))

# IDSS interface (IDSS IDD Rev F, NDS IDD Rev H; see the idss repo scripts/params.py)
IDA_HCS_Z = 0.642  # HCS mating plane above the IDA's PMA interface [m]
NDS_HCS_Z = 0.2304  # HCS mating plane above the NDS host-vehicle interface [m]
SCS_PASSIVE_Z = -0.006  # passive capture-ring plane below the HCS plane [m]
RING_READY_Z = 0.33  # NDS capture ring extension at ready-to-capture [m]
SEAL_H = 0.0021  # seal bead height [m]
READY_TO_HOOK = 0.0042  # HCS gap at which the hooks are swung (IDD Fig 3.2.3.4-2) [m]
HOOK_CLOSE = math.radians(28.0)  # hook swing open -> tangent position [rad]
HOOK_PULL = 0.008  # pull-down carriage command [m]; the carriage force limit (40 kN) sets the preload
PASSIVE_HOOK_K = 8.0e7  # Belleville stack stiffness [N/m] (IDD: 69-94 kN/mm)
PASSIVE_HOOK_PRE = 1.0e-4  # Belleville preload compression [m]
STRIKER_X_R = 0.601  # point X: striker 45 deg surface meets the SCS plane on the Ø1202 circle [m]
STRIKER_STEP_S = 0.114  # catch step, along the 45 deg surface from point X [m] (IDD Fig 3.2.2.4-1)

# Host vehicle (Starliner-class crew vehicle): mass, inertia about its CoM, CoM behind the host plane
VEHICLE_MASS = 13000.0
VEHICLE_INERTIA = (26000.0, 26000.0, 33000.0)
VEHICLE_COM = -1.8

PHASES = ["approach", "contact", "capture failed", "captured", "attenuate", "align", "retract", "hook gang 1", "hook gang 2",
          "docked", "undock", "separated", "diverged"]
(APPROACH, CONTACT, CAPTURE_FAILED, CAPTURED, ATTENUATE, ALIGN, RETRACT, HOOK1, HOOK2, DOCKED, UNDOCK, SEPARATED,
 DIVERGED) = range(len(PHASES))
EV_LATCH, EV_LOCKED, EV_DOCKED = 20, 30, 31  # non-phase event codes (latch k past its step: EV_LATCH + k)
# solver DOF-property sets (gains, effort limits, limit stiffness), one per stage of the sequence
CFG_CAPTURE, CFG_BRAKE, CFG_ATTENUATE, CFG_ALIGN, CFG_RELEASE = range(5)
DOF_FIELDS = ("actuator_gainprm", "actuator_biasprm", "jnt_actfrcrange", "jnt_solref")

# controller state on the device: ints, then float64 (sequence state followed by the measurements)
S_PHASE, S_FRAME, S_LATCHED, S_SEEN, S_RELEASED, S_CFG, S_UNDOCK, S_NEV = 0, 1, 2, 3, 6, 7, 8, 9
NI = 10
F_T, F_CMD, F_FROM, F_CREEP, F_HOOK, F_PULL, F_LO, F_HI, F_PEAK = 0, 1, 7, 13, 14, 16, 18, 21, 24
M_GAP, M_LAT, M_TILT, M_RING, M_LEGF, M_PRE, M_LATCH, M_SLANT, M_SENS, M_CONTACTS = 25, 26, 27, 28, 29, 35, 47, 50, 53, 59
NF = 60
MAXEV, EW = 64, 17  # event log rows: frame, code, up to 15 values

D = wp.float64
vec6d = wp.types.vector(6, wp.float64)
mat66d = wp.types.matrix((6, 6), wp.float64)


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def quat_rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    s = math.sin(angle / 2)
    return np.array([*(axis * s), math.cos(angle / 2)])


def rot_matrix(q):
    return np.stack([quat_rotate(q, e) for e in np.eye(3)], axis=1)


def vehicle_mesh(r_top=0.82, r_base=2.25, h=2.4, n=48):
    """Visiting-vehicle forebody (visual only): conical crew module below the NDS host plane."""
    pts, tris = [], []
    rings = [(r_top, 0.0), (r_top + 0.05, -0.06), (r_base * 0.97, -h + 0.25), (r_base, -h + 0.12), (r_base * 0.98, -h),
             (r_base * 0.6, -h - 0.22), (0.01, -h - 0.3)]
    for r, z in rings:
        for i in range(n):
            a = 2 * math.pi * i / n
            pts.append((r * math.cos(a), r * math.sin(a), z))
    for k in range(len(rings) - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            c, d = a + n, b + n
            tris += [a, c, b, b, c, d]
    top = len(pts)
    pts.append((0.0, 0.0, 0.0))
    for i in range(n):   # annular top deck (the NDS sits in its centre)
        tris += [top, i, (i + 1) % n]
    return newton.Mesh(np.array(pts, dtype=np.float32), np.array(tris, dtype=np.int32), compute_inertia=False)


def pma_mesh(n=48):
    """PMA stub under the IDA (visual only): the white MLI cone of Pressurized Mating Adapter 2."""
    pts, tris = [], []
    rings = [(0.80, 0.0), (0.86, -0.08), (1.05, -0.9), (1.18, -1.5), (1.18, -1.6)]
    for r, z in rings:
        for i in range(n):
            a = 2 * math.pi * i / n
            pts.append((r * math.cos(a), r * math.sin(a), z))
    for k in range(len(rings) - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            tris += [a, b, a + n, b, b + n, a + n]
    return newton.Mesh(np.array(pts, dtype=np.float32), np.array(tris, dtype=np.int32), compute_inertia=False)


# ------------------------------------------------------------------ docking controller (Warp)
@wp.struct
class Rig:
    """Mechanism indices and hexapod geometry."""
    nds_base: int
    leg_B: wp.array(dtype=wp.vec3d)  # leg base points, NDS frame
    leg_T: wp.array(dtype=wp.vec3d)  # leg top points relative to the ring centre at ready-to-capture
    leg_L0: wp.array(dtype=wp.float64)
    leg_q: wp.array(dtype=int)
    leg_d: wp.array(dtype=int)
    latch_q: wp.array(dtype=int)
    latch_body: wp.array(dtype=int)
    latch_catch: wp.array(dtype=wp.vec3d)  # catch face relative to the rocker pivot, latch body frame
    latch_jnt: wp.array(dtype=int)  # MuJoCo joint index (for the ratchet limit)
    passive_q: wp.array(dtype=int)
    hook_q: wp.array(dtype=int)  # [gang * 6 + i]
    pull_q: wp.array(dtype=int)
    sensor_q: wp.array(dtype=int)


@wp.struct
class Params:
    frame_dt: wp.float64
    capture_push: wp.float64
    capture_timeout: wp.float64
    attenuate_time: wp.float64
    align_time: wp.float64
    retract_speed: wp.float64
    retract_speed_pins: wp.float64
    pin_engage_gap: wp.float64
    hook_time: wp.float64
    undock_after: wp.float64


@wp.func
def qrot(q: wp.transform, v: wp.vec3d):
    u = wp.vec3d(D(q[3]), D(q[4]), D(q[5]))
    return v + D(2.0) * wp.cross(u, wp.cross(u, v) + D(q[6]) * v)


@wp.func
def tpos(q: wp.transform):
    return wp.vec3d(D(q[0]), D(q[1]), D(q[2]))


@wp.func
def leg_lengths(rig: Rig, x: vec6d):
    """Closed-form Stewart-platform IK: leg strokes (length - rest length) for a ring pose
    (dx, dy, dz, rx, ry, rz) relative to ready-to-capture, in the NDS frame."""
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
def solve6(A: mat66d, b: vec6d):
    M = A
    y = b
    for c in range(6):
        p = c
        for r in range(c + 1, 6):
            if wp.abs(M[r, c]) > wp.abs(M[p, c]):
                p = r
        if p != c:
            for j in range(6):
                t = M[c, j]
                M[c, j] = M[p, j]
                M[p, j] = t
            t = y[c]
            y[c] = y[p]
            y[p] = t
        for r in range(c + 1, 6):
            f = M[r, c] / M[c, c]
            for j in range(c, 6):
                M[r, j] = M[r, j] - f * M[c, j]
            y[r] = y[r] - f * y[c]
    x = vec6d()
    for k in range(6):
        i = 5 - k
        s = y[i]
        for j in range(i + 1, 6):
            s = s - M[i, j] * x[j]
        x[i] = s / M[i, i]
    return x


@wp.func
def ring_fk(rig: Rig, joint_q: wp.array(dtype=float), x0: vec6d):
    """Hexapod forward kinematics (Gauss-Newton on the IK, warm-started from the last command)."""
    target = vec6d()
    for i in range(6):
        target[i] = D(joint_q[rig.leg_q[i]])
    x = x0
    for _it in range(3):
        L = leg_lengths(rig, x)
        J = mat66d()
        for k in range(6):
            xe = x
            xe[k] = xe[k] + D(1.0e-6)
            Le = leg_lengths(rig, xe)
            for i in range(6):
                J[i, k] = (Le[i] - L[i]) / D(1.0e-6)
        x = x - solve6(J, L - target)
    return x


@wp.func
def log_event(ist: wp.array(dtype=int), ev: wp.array2d(dtype=wp.float64), code: int):
    n = ist[S_NEV]
    if n < MAXEV:
        ev[n, 0] = D(ist[S_FRAME])
        ev[n, 1] = D(code)
        ist[S_NEV] = n + 1
        return n
    return -1


@wp.func
def goto(ist: wp.array(dtype=int), fst: wp.array(dtype=wp.float64), ev: wp.array2d(dtype=wp.float64), phase: int):
    ist[S_PHASE] = phase
    fst[F_T] = D(0.0)
    return log_event(ist, ev, phase)


@wp.kernel
def nan_check(body_q: wp.array(dtype=wp.transform), flag: wp.array(dtype=int)):
    x = body_q[wp.tid()]
    for j in range(7):
        if not wp.isfinite(x[j]):
            flag[0] = 1


@wp.kernel
def measure_kernel(rig: Rig, body_q: wp.array(dtype=wp.transform), joint_q: wp.array(dtype=float),
                   qfrc: wp.array(dtype=float), ncontact: wp.array(dtype=int), fst: wp.array(dtype=wp.float64)):
    bq = body_q[rig.nds_base]
    hcs = tpos(bq) + qrot(bq, wp.vec3d(D(0.0), D(0.0), D(NDS_HCS_Z)))
    axis = qrot(bq, wp.vec3d(D(0.0), D(0.0), D(1.0)))
    fst[M_GAP] = hcs[2] - D(IDA_HCS_Z)
    fst[M_LAT] = wp.sqrt(hcs[0] * hcs[0] + hcs[1] * hcs[1])
    fst[M_TILT] = wp.acos(wp.clamp(-axis[2], D(-1.0), D(1.0))) * D(180.0) / D(math.pi)
    ring = D(0.0)
    for i in range(6):
        f = D(qfrc[rig.leg_d[i]])
        fst[M_LEGF + i] = f
        ring = ring + f
    fst[M_RING] = ring
    for k in range(12):
        fst[M_PRE + k] = D(PASSIVE_HOOK_K) * (D(joint_q[rig.passive_q[k]]) + D(PASSIVE_HOOK_PRE))
    for k in range(6):
        fst[M_SENS + k] = D(joint_q[rig.sensor_q[k]])
    # capture sensor: each latch's catch face, as a position along the IDA striker's 45 deg surface measured
    # from point X (the IDA is fixed at the world origin, so its frame is the world frame)
    for k in range(3):
        fst[M_LATCH + k] = D(joint_q[rig.latch_q[k]])
        lb = body_q[rig.latch_body[k]]
        p = tpos(lb) + qrot(lb, rig.latch_catch[k])
        inset = D(STRIKER_X_R) - wp.sqrt(p[0] * p[0] + p[1] * p[1])
        depth = (D(IDA_HCS_Z) + D(SCS_PASSIVE_Z)) - p[2]
        fst[M_SLANT + k] = (inset + depth) / wp.sqrt(D(2.0))
    fst[M_CONTACTS] = D(ncontact[0])


@wp.kernel
def sequence_kernel(rig: Rig, prm: Params, joint_q: wp.array(dtype=float), nan: wp.array(dtype=int),
                    ist: wp.array(dtype=int), fst: wp.array(dtype=wp.float64), ev: wp.array2d(dtype=wp.float64),
                    target_q: wp.array(dtype=float)):
    """Docking sequence (IDSS / NDS ConOps), one call per frame."""
    ph = ist[S_PHASE]
    if ph == DIVERGED:
        return
    if nan[0] != 0:
        n = log_event(ist, ev, DIVERGED)
        if n >= 0:
            ev[n, 2] = D(ph)
        ist[S_PHASE] = DIVERGED
        return
    if ist[S_UNDOCK] != 0:
        ist[S_UNDOCK] = 0
        if ph == DOCKED or ph == HOOK2:
            ist[S_RELEASED] = 0
            goto(ist, fst, ev, UNDOCK)
            ph = UNDOCK
    dt = prm.frame_dt
    t = fst[F_T] + dt
    fst[F_T] = t
    gap, lateral, tilt = fst[M_GAP], fst[M_LAT], fst[M_TILT]
    for i in range(6):
        fst[F_PEAK] = wp.max(fst[F_PEAK], wp.abs(fst[M_LEGF + i]))
    latched = ist[S_LATCHED] != 0
    # capture latches: armed rockers ride in on the striker's upper planar surface, then drop past the 2 mm catch
    # step. Soft capture is declared when all three are past together (ring seated and centred), and only then are
    # the latches locked (NDS capture-latch brakes); a latch locked early would wedge the ring off-centre.
    if latched and ph != UNDOCK and ph != SEPARATED:   # ratchet the pawls as they drop onto the striker
        for k in range(3):
            if fst[M_LATCH + k] + D(0.001) < fst[F_HI + k] - D(0.002):
                fst[F_HI + k] = fst[M_LATCH + k] + D(0.001)
    if ph == CONTACT and not latched:
        npast = int(0)
        for k in range(3):
            if fst[M_SLANT + k] > D(STRIKER_STEP_S) + D(0.0003):
                npast += 1
                if ist[S_SEEN + k] == 0:
                    ist[S_SEEN + k] = 1
                    log_event(ist, ev, EV_LATCH + k)
        if npast == 3:
            latched = True
            ist[S_LATCHED] = 1
            ist[S_CFG] = CFG_BRAKE   # latch brake: stiff stop
            n = log_event(ist, ev, EV_LOCKED)
            for k in range(3):   # pawl: free to swing out, never back in
                fst[F_HI + k] = fst[M_LATCH + k] + D(0.001)
                if n >= 0:
                    ev[n, 2 + k] = fst[M_LATCH + k]
    cmd = vec6d(fst[F_CMD], fst[F_CMD + 1], fst[F_CMD + 2], fst[F_CMD + 3], fst[F_CMD + 4], fst[F_CMD + 5])
    if ph == CONTACT or ph == ATTENUATE:
        # compliance mode (NDS load-sensing ring control): the commanded ring pose follows the measured pose
        # (hexapod forward kinematics), so the ring floats laterally and in rotation and aligns itself to the
        # passive ring; during capture an axial offset presses the capture rings together so the latches seat
        cmd = ring_fk(rig, joint_q, cmd)
        if ph == CONTACT:
            cmd[2] = cmd[2] + prm.capture_push
    if ph == APPROACH:
        if fst[M_CONTACTS] > D(0.0):
            goto(ist, fst, ev, CONTACT)
    elif ph == CONTACT:
        if latched:
            goto(ist, fst, ev, CAPTURED)
        elif t > prm.capture_timeout:
            goto(ist, fst, ev, CAPTURE_FAILED)
    elif ph == CAPTURED:
        # load-limited from here on (the soft-capture interface is rated for ~3.5 kN / 2.8 kN m, IDSS IDD);
        # attenuation: soft and heavily damped
        ist[S_CFG] = CFG_ATTENUATE
        goto(ist, fst, ev, ATTENUATE)
    elif ph == ATTENUATE:
        if t > prm.attenuate_time:
            cmd = ring_fk(rig, joint_q, cmd)
            for i in range(6):
                fst[F_FROM + i] = cmd[i]
            ist[S_CFG] = CFG_ALIGN   # position-hold servo for alignment and retraction (latches hold > 4 kN)
            goto(ist, fst, ev, ALIGN)
    elif ph == ALIGN:
        # null the lateral and angular ring offsets (this swings the vehicle into line); hold the axial stroke
        s = wp.min(D(1.0), t / prm.align_time)
        s = s * s * (D(3.0) - D(2.0) * s)
        for i in range(6):
            cmd[i] = fst[F_FROM + i] * (D(1.0) - s)
        cmd[2] = fst[F_FROM + 2]
        # retract only once the vehicle is actually in line (tilt and lateral offset nulled), or after a timeout
        aligned = tilt < D(0.1) and lateral < D(0.0005)
        if t > prm.align_time + D(0.5) and (aligned or t > prm.align_time + D(15.0)):
            goto(ist, fst, ev, RETRACT)
    elif ph == RETRACT:
        # pull the vehicle in to the ready-to-hook gap (4.2 mm); slow down while the guide pins seat
        # (Ø24.7 pins in Ø25.2 bores) so they square the interface, then swing the hooks only once the
        # interface is square (a cocked interface lets the swinging hooks pry it apart)
        stop = D(READY_TO_HOOK) - D(RING_READY_Z) - D(SCS_PASSIVE_Z) - D(0.0005)
        creep = fst[F_CREEP]
        if cmd[2] <= stop - creep and gap > D(READY_TO_HOOK) + D(0.0005):
            # the nominal stroke leaves the latch / striker slack open (nothing loads the latches in zero g):
            # keep creeping in on the measured gap, up to 5 mm past the nominal stop
            creep = wp.min(D(0.005), creep + D(0.2) * prm.retract_speed_pins * dt)
            fst[F_CREEP] = creep
        stop = stop - creep
        v = prm.retract_speed_pins
        if gap > prm.pin_engage_gap:
            v = prm.retract_speed
        cmd[2] = wp.max(stop, cmd[2] - v * dt)
        square = tilt < D(0.15) and lateral < D(0.001)
        if cmd[2] <= stop and gap < D(READY_TO_HOOK) + D(0.0008) and square:
            goto(ist, fst, ev, HOOK1)
    elif ph == HOOK1 or ph == HOOK2:
        # the gang swings its six hooks over the mating passive hooks, then the carriages pull the hook
        # tangent planes down: the interface closes, the seals compress and the Belleville stacks preload
        g = ph - HOOK1
        s1 = wp.min(D(1.0), t / prm.hook_time)
        s2 = wp.min(D(1.0), wp.max(D(0.0), t - prm.hook_time) / prm.hook_time)
        fst[F_HOOK + g] = D(HOOK_CLOSE) * s1
        fst[F_PULL + g] = D(HOOK_PULL) * s2
        if t > D(2.0) * prm.hook_time + D(0.5):
            if g == 0:
                goto(ist, fst, ev, HOOK2)
            else:
                goto(ist, fst, ev, DOCKED)
                n = log_event(ist, ev, EV_DOCKED)
                if n >= 0:
                    ev[n, 2] = gap
                    ev[n, 3] = lateral
                    ev[n, 4] = tilt
                    for k in range(12):
                        ev[n, 5 + k] = fst[M_PRE + k]
    elif ph == DOCKED:
        if prm.undock_after > D(0.0) and t > prm.undock_after:
            ist[S_RELEASED] = 0
            goto(ist, fst, ev, UNDOCK)
    elif ph == UNDOCK:
        s1 = wp.min(D(1.0), t / prm.hook_time)
        s2 = wp.min(D(1.0), wp.max(D(0.0), t - prm.hook_time) / prm.hook_time)
        for g in range(2):
            fst[F_PULL + g] = D(HOOK_PULL) * (D(1.0) - s1)
            fst[F_HOOK + g] = D(HOOK_CLOSE) * (D(1.0) - s2)
        if t > D(2.0) * prm.hook_time and ist[S_RELEASED] == 0:
            # separators: release the lock, the springs push off; capture latches: motor drives the rockers in
            ist[S_RELEASED] = 1
            ist[S_CFG] = CFG_RELEASE
            for k in range(3):
                fst[F_LO + k] = D(0.15)
                fst[F_HI + k] = D(0.2)
            cmd = vec6d()
        if ist[S_RELEASED] != 0 and gap > D(0.1):
            goto(ist, fst, ev, SEPARATED)
    for i in range(6):
        fst[F_CMD + i] = cmd[i]
    L = leg_lengths(rig, cmd)
    for i in range(6):
        target_q[rig.leg_q[i]] = float(L[i])
    for g in range(2):
        for i in range(6):
            target_q[rig.hook_q[g * 6 + i]] = float(fst[F_HOOK + g])
            target_q[rig.pull_q[g * 6 + i]] = float(fst[F_PULL + g])
    ist[S_FRAME] = ist[S_FRAME] + 1


@wp.kernel
def select_row(table: wp.array2d(dtype=Any), ist: wp.array(dtype=int), out: wp.array2d(dtype=Any)):
    i = wp.tid()
    out[0, i] = table[ist[S_CFG], i]


@wp.kernel
def latch_range(rig: Rig, fst: wp.array(dtype=wp.float64), jnt_range: wp.array2d(dtype=wp.vec2)):
    k = wp.tid()
    jnt_range[0, rig.latch_jnt[k]] = wp.vec2(float(fst[F_LO + k]), float(fst[F_HI + k]))


@wp.kernel
def request_undock(ist: wp.array(dtype=int)):
    ist[S_UNDOCK] = 1


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

        def stage(name):
            st = Usd.Stage.Open(os.path.join(args.assets, name, f"{name}.usda"))
            root = st.GetDefaultPrim()
            root.GetVariantSet("Physics").SetVariantSelection("physics")
            root.GetVariantSet("Fidelity").SetVariantSelection(args.fidelity)
            root.GetVariantSet("Contact").SetVariantSelection(args.contact)
            return st

        b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.rigid_gap = 0.0015
        # --- IDA on the station (fixed), docking axis along world +Z
        b.add_usd(stage("ida"), floating=False, collapse_fixed_joints=False, enable_self_collisions=False,
                  hide_collision_shapes=True)
        b.add_shape_mesh(-1, mesh=pma_mesh(), cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False),
                         color=(0.72, 0.72, 0.70), label="pma_stub")
        # --- visiting vehicle: NDS mirrored about the line of androgyny (+Y), approaching from +Z
        self.mirror = quat_axis_angle((0, 1, 0), math.pi)
        tilt = quat_axis_angle((1, 0, 0) if args.misalign_axis == "x" else (0, 1, 0), math.radians(args.misalign))
        q0 = quat_mul(tilt, self.mirror)
        hcs = np.array([args.offset, 0.0, IDA_HCS_Z + args.gap])
        p0 = hcs - quat_rotate(q0, (0, 0, NDS_HCS_Z))
        nds_stage = stage("nds")
        b.add_usd(nds_stage, xform=wp.transform(wp.vec3(*p0), wp.quat(*q0)), collapse_fixed_joints=False,
                  enable_self_collisions=False, hide_collision_shapes=True)
        self._index(b, nds_stage)
        self._add_vehicle(b)
        b.request_state_attributes("mujoco:qfrc_actuator")
        self.model = b.finalize()

        # initial relative motion (IDSS IDD 3.3 capture conditions)
        qd = self.model.joint_qd.numpy()
        d0 = self.model.joint_qd_start.numpy()[self.free_joint]
        qd[d0:d0 + 3] = [args.lateral_speed, 0.0, -args.closing_speed]
        qd[d0 + 3:d0 + 6] = [0.0, math.radians(args.rate), 0.0]
        self.model.joint_qd.assign(qd)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_contacts=False, solver="newton", integrator="implicitfast", cone=args.cone,
            impratio=10, iterations=args.iterations, ls_iterations=args.ls_iterations, njmax=args.nconmax * 6, nconmax=args.nconmax)
        # stiff loop closures (Stewart platform ball joints): equality time constant 2 dt
        eq = self.solver.mjw_model.eq_solref.numpy()
        eq[..., 0] = 2.0 * self.sim_dt
        self.solver.mjw_model.eq_solref.assign(eq)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.collision_pipeline = newton.CollisionPipeline(self.model, broad_phase="sap")
        self.contacts = self.collision_pipeline.contacts()

        self._build_dof_configs()
        self._init_controller()
        self.viewer.set_model(self.model)
        self.viewer.show_contacts = args.show_contacts
        self.capture()

    # -------------------------------------------------------------- model lookup
    def _index(self, b, nds_stage):
        lab = list(b.joint_label)

        def j(name):
            return lab.index(name)

        def qi(name):
            return b.joint_q_start[j(name)]

        def di(name):
            return b.joint_qd_start[j(name)]

        self.free_joint = next(i for i in range(b.joint_count) if b.joint_type[i] == newton.JointType.FREE)
        self.nds_base = list(b.body_label).index("/nds/Geometry/nds_base")
        self.ida_base = list(b.body_label).index("/ida/Geometry/ida_base")
        self.legs = []
        for i in range(6):
            p = nds_stage.GetPrimAtPath(f"/nds/Physics/leg{i + 1}_rod")
            self.legs.append({"q": qi(f"/nds/Physics/leg{i + 1}_rod"), "d": di(f"/nds/Physics/leg{i + 1}_rod"),
                              "B": np.array(p.GetAttribute("idss:base").Get()),
                              "T": np.array(p.GetAttribute("idss:top").Get()),
                              "L0": p.GetAttribute("idss:restLength").Get()})
        self.ring_centre = np.array([0.0, 0.0, NDS_HCS_Z + RING_READY_Z])
        bodies = list(b.body_label)
        self.latch = []
        for k in (1, 2, 3):
            p = nds_stage.GetPrimAtPath(f"/nds/Physics/latch_{k}")
            self.latch.append({"q": qi(f"/nds/Physics/latch_{k}"), "d": di(f"/nds/Physics/latch_{k}"),
                               "body": next(i for i, lb in enumerate(bodies) if lb.endswith(f"/latch_{k}")),
                               "catch": np.array(p.GetAttribute("idss:catch").Get())
                               - np.array(p.GetAttribute("idss:pivot").Get())})
        gangs = ([0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11])
        self.hooks = [[qi(f"/nds/Physics/hook_a_{k:02d}") for k in g] for g in gangs]
        self.hook_pulls = [[qi(f"/nds/Physics/hook_c_{k:02d}") for k in g] for g in gangs]
        self.ida_passive = [qi(f"/ida/Physics/hook_p_{k:02d}") for k in range(12)]
        self.nds_separators = [(qi(f"/nds/Physics/separator_{k}"), di(f"/nds/Physics/separator_{k}")) for k in (1, 2, 3)]
        self.nds_sensors = [qi(f"/nds/Physics/sensor_{k}") for k in (1, 2, 3)]
        self.ida_sensors = [qi(f"/ida/Physics/sensor_{k}") for k in (1, 2, 3)]

    def _add_vehicle(self, b):
        """Put the host vehicle's mass on the NDS base body (rigid attachment) and draw its forebody."""
        i = self.nds_base
        m0, c0 = b.body_mass[i], np.array(b.body_com[i])
        cv = np.array([0.0, 0.0, VEHICLE_COM])
        m = m0 + VEHICLE_MASS
        com = (m0 * c0 + VEHICLE_MASS * cv) / m
        I0 = np.array(b.body_inertia[i]).reshape(3, 3)

        def shift(mass, d):
            return mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

        inertia = I0 + shift(m0, c0 - com) + np.diag(VEHICLE_INERTIA) + shift(VEHICLE_MASS, cv - com)
        b.body_mass[i], b.body_inv_mass[i] = m, 1.0 / m
        b.body_com[i] = wp.vec3(*com)
        b.body_inertia[i] = wp.mat33(*inertia.flatten())
        b.body_inv_inertia[i] = wp.mat33(*np.linalg.inv(inertia).flatten())
        b.add_shape_mesh(i, mesh=vehicle_mesh(), cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False),
                         color=(0.86, 0.86, 0.84), label="visiting_vehicle")

    # -------------------------------------------------------------- controller setup
    def _build_dof_configs(self):
        """Tabulate the solver's actuator gains, effort limits and limit stiffness for every stage of the sequence.
        Each set goes through notify_model_changed once here; at run time the sequence kernel picks a row and
        select_row copies it into the MuJoCo-Warp model inside the graph."""
        a, mj, m = self.args, self.solver.mjw_model, self.model
        base = {k: getattr(m, k).numpy() for k in ("joint_target_ke", "joint_target_kd", "joint_effort_limit",
                                                    "joint_limit_ke", "joint_limit_kd")}
        legs = [leg["d"] for leg in self.legs]
        latches = [lt["d"] for lt in self.latch]
        seps = [d for _q, d in self.nds_separators]
        stages = [(a.leg_ke, a.leg_kd, None, False, False),   # CFG_CAPTURE
                  (a.leg_ke, a.leg_kd, None, True, False),   # CFG_BRAKE: latch brakes on
                  (a.leg_ke * 0.25, a.leg_kd * 4.0, a.leg_force_limit, True, False),   # CFG_ATTENUATE
                  (a.align_ke, a.align_kd, a.retract_force_limit, True, False),   # CFG_ALIGN
                  (a.leg_ke, a.leg_kd, a.retract_force_limit, True, True)]   # CFG_RELEASE: separators free
        rows, values = {f: [] for f in DOF_FIELDS}, []
        for ke, kd, effort, brake, release in stages:
            v = {k: x.copy() for k, x in base.items()}
            values.append(v)
            v["joint_target_ke"][legs], v["joint_target_kd"][legs] = ke, kd
            if effort is not None:
                v["joint_effort_limit"][legs] = effort
            if brake:
                v["joint_limit_ke"][latches], v["joint_limit_kd"][latches] = 2.0e5, 2.0e3
            if release:
                v["joint_target_ke"][seps], v["joint_target_kd"][seps] = 0.0, 0.0
            for k, x in v.items():
                getattr(m, k).assign(x)
            self.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
            for f in DOF_FIELDS:
                rows[f].append(getattr(mj, f).numpy()[0])
        for k, x in values[CFG_CAPTURE].items():   # leave the Newton model in the capture configuration
            getattr(m, k).assign(x)
        self.dof_tables = {f: wp.array(np.stack(r), dtype=getattr(mj, f).dtype) for f, r in rows.items()}

    def _init_controller(self):
        a, dev = self.args, self.model.device
        rig = Rig()
        rig.nds_base = self.nds_base
        rig.leg_B = wp.array([leg["B"] for leg in self.legs], dtype=wp.vec3d)
        rig.leg_T = wp.array([leg["T"] - self.ring_centre for leg in self.legs], dtype=wp.vec3d)
        rig.leg_L0 = wp.array([leg["L0"] for leg in self.legs], dtype=wp.float64)
        rig.leg_q = wp.array([leg["q"] for leg in self.legs], dtype=int)
        rig.leg_d = wp.array([leg["d"] for leg in self.legs], dtype=int)
        rig.latch_q = wp.array([lt["q"] for lt in self.latch], dtype=int)
        rig.latch_body = wp.array([lt["body"] for lt in self.latch], dtype=int)
        rig.latch_catch = wp.array([lt["catch"] for lt in self.latch], dtype=wp.vec3d)
        jnt_dof = list(self.solver.mjc_jnt_to_newton_dof.numpy()[0])
        rig.latch_jnt = wp.array([jnt_dof.index(lt["d"]) for lt in self.latch], dtype=int)
        rig.passive_q = wp.array(self.ida_passive, dtype=int)
        rig.hook_q = wp.array(self.hooks[0] + self.hooks[1], dtype=int)
        rig.pull_q = wp.array(self.hook_pulls[0] + self.hook_pulls[1], dtype=int)
        rig.sensor_q = wp.array(self.nds_sensors + self.ida_sensors, dtype=int)
        self.rig = rig
        prm = Params()
        prm.frame_dt = self.frame_dt
        for k in ("capture_push", "capture_timeout", "attenuate_time", "align_time", "retract_speed",
                  "retract_speed_pins", "pin_engage_gap", "hook_time", "undock_after"):
            setattr(prm, k, getattr(a, k))
        self.prm = prm
        lo, hi = self.model.joint_limit_lower.numpy(), self.model.joint_limit_upper.numpy()
        fst = np.zeros(NF)
        fst[F_LO:F_LO + 3] = [lo[lt["d"]] for lt in self.latch]
        fst[F_HI:F_HI + 3] = [hi[lt["d"]] for lt in self.latch]
        self.ist = wp.zeros(NI, dtype=int)
        self.fst = wp.array(fst, dtype=wp.float64)
        self.ev = wp.zeros((MAXEV, EW), dtype=wp.float64)
        self.nan_flag = wp.zeros(1, dtype=int)
        qfrc = getattr(getattr(self.state_0, "mujoco", None), "qfrc_actuator", None)
        self.qfrc = qfrc if qfrc is not None else wp.zeros(self.model.joint_dof_count, dtype=float)
        self.rec_body_q = wp.empty_like(self.state_0.body_q) if a.record else None
        # host side: telemetry snapshots land in a ring of pinned buffers, each tagged with a CUDA event
        pinned = wp.get_device().is_cuda

        def host(arr):
            return wp.empty(arr.shape, dtype=arr.dtype, device="cpu", pinned=pinned)

        self._slots = [SimpleNamespace(ist=host(self.ist), fst=host(self.fst), ev=host(self.ev),
                                       body_q=host(self.rec_body_q) if a.record else None, evt=None, busy=False)
                       for _ in range(4)]
        self._pending = deque()
        self._launched = 0
        self._nev = 0
        self._events = []
        self._snap = {"phase": "approach", "meas": None, "latched": [False] * 3, "hook_cmd": [0.0, 0.0],
                      "pull_cmd": [0.0, 0.0], "peak": 0.0}
        self._apply_dof_config()

    def _apply_dof_config(self):
        mj = self.solver.mjw_model
        for f, table in self.dof_tables.items():
            wp.launch(select_row, dim=table.shape[1], inputs=[table, self.ist], outputs=[getattr(mj, f)])
        wp.launch(latch_range, dim=3, inputs=[self.rig, self.fst], outputs=[mj.jnt_range])

    # -------------------------------------------------------------- telemetry (host)
    def _readback(self):
        """Queue an asynchronous copy of the controller state for the frame just launched."""
        s = self._slots[self._launched % len(self._slots)]
        while s.busy:
            self._poll(wait=True, limit=1)
        wp.copy(s.ist, self.ist)
        wp.copy(s.fst, self.fst)
        wp.copy(s.ev, self.ev)
        if s.body_q is not None:
            wp.copy(s.body_q, self.rec_body_q)
        s.evt = wp.record_event() if wp.get_device().is_cuda else None
        s.busy = True
        self._pending.append(s)
        self._launched += 1

    def _poll(self, wait=False, limit=None):
        """Consume the snapshots that have arrived (all queued ones if wait)."""
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

    def _meas(self, f):
        return {"gap": f[M_GAP], "lateral": f[M_LAT], "tilt": f[M_TILT], "leg_f": f[M_LEGF:M_LEGF + 6].copy(),
                "ring_load": f[M_RING], "preload": f[M_PRE:M_PRE + 12].copy(), "latch": f[M_LATCH:M_LATCH + 3].copy(),
                "latch_slant": f[M_SLANT:M_SLANT + 3].copy(), "sensors": f[M_SENS:M_SENS + 6].copy(),
                "contacts": int(f[M_CONTACTS])}

    def _consume(self, s):
        i, f = s.ist.numpy(), s.fst.numpy()
        ev = s.ev.numpy()
        for row in ev[self._nev:min(int(i[S_NEV]), MAXEV)]:
            self._event(row)
        self._nev = max(self._nev, min(int(i[S_NEV]), MAXEV))
        phase = PHASES[int(i[S_PHASE])]
        self._snap = {"phase": phase, "meas": self._meas(f), "latched": [bool(i[S_LATCHED])] * 3,
                      "hook_cmd": [float(f[F_HOOK]), float(f[F_HOOK + 1])],
                      "pull_cmd": [float(f[F_PULL]), float(f[F_PULL + 1])], "peak": float(f[F_PEAK])}
        if self.args.record and phase != "diverged":
            m = self._snap["meas"]
            if not hasattr(self, "_rec"):
                self._rec = {"body_q": [], "t": [], "phase": [], "gap": [], "preload": [], "ring_load": []}
            self._rec["body_q"].append(s.body_q.numpy().copy())
            self._rec["t"].append((int(i[S_FRAME]) - 1) * self.frame_dt)
            self._rec["phase"].append(phase)
            self._rec["gap"].append(m["gap"])
            self._rec["preload"].append(float(np.sum(m["preload"])))
            self._rec["ring_load"].append(m["ring_load"])

    def _event(self, row):
        t, code = row[0] * self.frame_dt, int(row[1])
        if code == DIVERGED:
            print(f"[{t:7.2f} s] simulation diverged in phase '{PHASES[int(row[2])]}' (try more --substeps)")
        elif code < len(PHASES):
            self._events.append((round(t, 2), PHASES[code]))
            print(f"[{t:7.2f} s] {PHASES[code]}")
        elif EV_LATCH <= code < EV_LATCH + 3:
            print(f"[{t:7.2f} s] latch {code - EV_LATCH + 1} past the striker step")
        elif code == EV_LOCKED:
            print(f"[{t:7.2f} s] latches locked (rockers {np.round(np.degrees(row[2:5]), 1)} deg)")
        elif code == EV_DOCKED:
            pre = row[5:17]
            self._docked = {"gap": float(row[2]), "preload": pre.copy()}
            print(f"hard docked: HCS gap {row[2] * 1000:.2f} mm, misalignment {row[3] * 1000:.2f} mm / "
                  f"{row[4]:.3f} deg; passive hook loads {np.round(pre / 1000, 1)} kN (total {pre.sum() / 1000:.0f} kN)")

    # telemetry of the last stepped frame (waits for its snapshot; the viewer uses the latest arrived one instead)
    @property
    def phase(self):
        self._poll(wait=True)
        return self._snap["phase"]

    @property
    def meas(self):
        self._poll(wait=True)
        return self._snap["meas"]

    @property
    def latched(self):
        self._poll(wait=True)
        return self._snap["latched"]

    @property
    def hook_cmd(self):
        self._poll(wait=True)
        return self._snap["hook_cmd"]

    @property
    def pull_cmd(self):
        self._poll(wait=True)
        return self._snap["pull_cmd"]

    @property
    def events(self):
        self._poll(wait=True)
        return self._events

    @property
    def peak(self):
        self._poll(wait=True)
        return {"ring_load": self._snap["peak"]}

    def measure(self):
        """Measurements of the current state (synchronous)."""
        wp.launch(measure_kernel, dim=1, inputs=[self.rig, self.state_0.body_q, self.state_0.joint_q, self.qfrc,
                                                 self.contacts.rigid_contact_count, self.fst])
        return self._meas(self.fst.numpy())

    def undock(self):
        wp.launch(request_undock, dim=1, inputs=[self.ist])   # honoured in docked / hook gang 2

    # -------------------------------------------------------------- sim loop
    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.nan_flag.zero_()
        wp.launch(nan_check, dim=self.model.body_count, inputs=[self.state_0.body_q, self.nan_flag])
        wp.launch(measure_kernel, dim=1, inputs=[self.rig, self.state_0.body_q, self.state_0.joint_q, self.qfrc,
                                                 self.contacts.rigid_contact_count, self.fst])
        wp.launch(sequence_kernel, dim=1, inputs=[self.rig, self.prm, self.state_0.joint_q, self.nan_flag, self.ist,
                                                  self.fst, self.ev, self.control.joint_target_q])
        self._apply_dof_config()
        if self.rec_body_q is not None:
            wp.copy(self.rec_body_q, self.state_0.body_q)
        for k in range(self.sim_substeps):
            if k % self.args.collide_every == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def save_recording(self):
        if not self.args.record or not hasattr(self, "_rec"):
            return
        r = self._rec
        np.savez_compressed(self.args.record, body_q=np.array(r["body_q"], dtype=np.float32), t=np.array(r["t"]),
                            phase=np.array(r["phase"]), gap=np.array(r["gap"]), preload=np.array(r["preload"]),
                            ring_load=np.array(r["ring_load"]), body_label=np.array(list(self.model.body_label)))
        print(f"recorded {len(r['t'])} frames -> {self.args.record}")

    def step(self):
        self._poll()
        if self._snap["phase"] == "diverged":
            return
        if self.viewer.is_key_down("u"):
            self.undock()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self._readback()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        m = self._snap["meas"]
        if m is not None:
            self.viewer.log_scalar("HCS gap [mm]", m["gap"] * 1000.0)
            self.viewer.log_scalar("Lateral misalignment [mm]", m["lateral"] * 1000.0)
            self.viewer.log_scalar("Ring axial load [N]", m["ring_load"])
            self.viewer.log_scalar("Hook preload, total [kN]", float(np.sum(m["preload"])) / 1000.0)
        self.viewer.end_frame()

    def gui(self, ui):
        sn = self._snap
        m = sn["meas"]
        ui.text(f"Phase: {sn['phase']}  ({self.sim_time:.1f} s)")
        if m is None:
            return
        ui.text(f"HCS gap {m['gap'] * 1000:8.2f} mm   lateral {m['lateral'] * 1000:6.2f} mm   tilt {m['tilt']:5.2f} deg")
        ui.text("Latches " + "  ".join(("LOCKED" if sn["latched"][k] else f"{math.degrees(m['latch'][k]):4.1f} deg")
                                       for k in range(3)))
        ui.text("Leg loads [N] " + " ".join(f"{f:7.0f}" for f in m["leg_f"]))
        hook, pull = sn["hook_cmd"], sn["pull_cmd"]
        ui.text(f"Hook gangs {math.degrees(hook[0]):5.1f} / {math.degrees(hook[1]):5.1f} deg, pull "
                f"{pull[0] * 1000:4.1f} / {pull[1] * 1000:4.1f} mm   "
                f"preload {np.sum(m['preload']) / 1000:6.1f} kN")
        ui.text(f"Contacts {m['contacts']}")
        if sn["phase"] == "docked" and ui.button("Undock (u)"):
            self.undock()

    def test_final(self):
        self._poll(wait=True)
        self.save_recording()
        m = self.measure()
        assert all(self.latched), f"soft capture failed: latches {self.latched}"
        docked = getattr(self, "_docked", None)   # measured at the hard-docked event
        if docked is not None:
            assert docked["gap"] < 0.004, f"HCS gap {docked['gap']:.4f} m after hard capture"
            assert np.sum(docked["preload"]) > 50e3, "hooks are not preloaded"
        if self.phase == "separated":
            assert m["gap"] > 0.1, f"HCS gap {m['gap']:.4f} m after separation"
        print(f"idss docking: phase {self.phase}, gap {m['gap'] * 1000:.2f} mm, events {self.events}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--assets", default=ASSETS, help="Folder holding nds/nds.usda and ida/ida.usda.")
        parser.add_argument("--fidelity", default="full", choices=["full", "proxy"], help="Visual detail variant.")
        parser.add_argument("--contact", default="exact", choices=["exact", "sdf", "hydro"],
                            help="Contact model variant: exact (convex/mesh for the sub-mm fits + SDF hooks), "
                                 "sdf (all sparse SDF) or hydro (SDF hydroelastic pressure-field contact).")
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
        parser.add_argument("--pin-engage-gap", type=float, default=0.065,
                            help="HCS gap below which retraction slows for the guide pins [m].")
        parser.add_argument("--retract-speed-pins", type=float, default=0.01, help="Retraction speed as the pins seat [m/s].")
        parser.add_argument("--retract-force-limit", type=float, default=2500.0,
                            help="Actuator force limit during alignment and retraction [N].")
        parser.add_argument("--align-kd", type=float, default=4.5e5, help="Leg servo damping, align/retract [N s/m].")
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
        parser.add_argument("--collide-every", type=int, default=2,
                            help="Run collision detection every N substeps (1 = every substep, slowest).")
        parser.add_argument("--nconmax", type=int, default=6000, help="MuJoCo contact buffer (raise if it overflows).")
        parser.add_argument("--show-contacts", action="store_true", help="Draw contact points and normals.")
        parser.add_argument("--record", default="", help="Save per-frame body poses and docking telemetry (.npz).")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
