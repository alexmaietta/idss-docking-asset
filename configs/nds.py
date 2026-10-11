"""NASA Docking System Block 1 (active IDSS docking system), NDS IDD JSC 65795 Rev H.

Drive units are SI (N, N m, N/m, N m/rad); the packager converts angular values to the USD per-degree
convention. Masses split the 324 kg NDS Block 1 mass (NDS papers) between the moving parts (ESTIMATE
by part volume x density) and the base.
"""
from math import radians

from common import MATERIALS, PHYSICS_MATERIALS, TEXTURE_SETS  # noqa: F401

ASSET_ID = "nds"
FAMILY = "nds"
TITLE = "NASA Docking System Block 1 (active IDSS)"
DOC = ("Active IDSS-compliant docking system (NASA Docking System Block 1): six-DOF soft-capture ring on six "
       "electromechanical actuators (Stewart platform) with three guide petals and three capture latches, and the "
       "hard-capture ring with 12 active/passive hook pairs, guide pins, dual seals, separators and sensors.")
VERSION = open(__file__.replace("configs/nds.py", "VERSION")).read().strip()
ROBOT_TYPE = "Manipulator"
SEMANTIC_CLASS = ["docking_system", "idss_active"]

SENSOR_ANGLES = [radians(32.0), radians(152.0), radians(272.0)]

MASSES = {   # kg
    "nds_base": 224.2,          # 324 kg NDS Block 1 minus the moving parts
    "scs_ring": 40.0,          # ring channel + frame + 3 petals + 3 strikers
    "latch": 0.5,
    "leg_ujoint": 0.4,
    "leg_body": 5.0,           # housing, motor, gearbox
    "leg_rod": 1.4,
    "hook_active": 0.5,
    "hook_carriage": 0.4,
    "hook_passive": 0.4,
    "separator": 0.4,
    "sensor": 0.05,
}

# joint drives / springs ("drive" in the joint spec; SI units, target in joint units)
HOOK_DRIVE = {"stiffness": 2.0e4, "damping": 100.0, "max_force": 600.0, "target": 0.0, "armature": 0.5}   # gang swing drive
HOOK_PULL = {"stiffness": 5.0e7, "damping": 4.0e4, "max_force": 40.0e3, "target": 0.0, "armature": 1000.0}   # preload carriage
PASSIVE_HOOK_SPRING = {"stiffness": 8.0e7, "damping": 2.0e4, "max_force": 6.0e4, "target": -1.0e-4, "armature": 1500.0}
SEPARATOR_SPRING = {"stiffness": 2.0e5, "damping": 400.0, "max_force": 2000.0, "target": 0.0, "armature": 1.0,
                    "spring": {"stiffness": 700.0, "free_length": 1.017}}   # 712 N mated -> 689 N at full stroke
SENSOR_SPRING = {"stiffness": 2000.0, "damping": 5.0, "max_force": 50.0, "target": -0.002, "armature": 0.05}
LATCH_SPRING = {"stiffness": 8.0, "damping": 0.3, "max_force": 200.0, "target": -0.15, "armature": 0.05}   # armature: latch gear train
LEG_DRIVE = {"stiffness": 2.0e4, "damping": 4.0e3, "max_force": 4000.0, "target": 0.0, "armature": 2.0}
PASSIVE_JOINT = {"damping": 0.5, "armature": 0.01}

# collider contact parameters per physics material (Newton ke/kd per shape, metres for margins)
CONTACT = {"ke": 2.0e7, "kd": 2.0e4, "gap": 0.0015, "margin": 0.0}

RENDER_VIEWS = {
    "studio_front": ("studio", (2.4, -2.0, 1.25), (0, 0, 0.42), 38),
    "studio_low": ("studio", (1.6, -1.25, 0.25), (0, 0, 0.45), 30),
    "studio_top": ("studio", (1.3, -0.9, 1.9), (0, 0, 0.45), 32),
    "latch_closeup": ("studio", (1.1, 0.8, 0.95), (0.27, 0.46, 0.68), 60),
    "hook_closeup": ("studio", (1.05, 0.25, 0.42), (0.64, 0.17, 0.23), 70),
    "thumb": ("studio", (2.4, -2.0, 1.25), (0, 0, 0.42), 40),
}
