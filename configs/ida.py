"""International Docking Adapter (IDA-2 / IDA-3), the passive IDSS docking port on ISS.

Masses split the 526 kg IDA (NDS Block 2 paper) between the hooks, plungers and the fixed base.
Hardware clocking (handrails, targets, trunnions, reflectors, connector panel) is ESTIMATE from the
KSC photographs in refs/images/photos/ida; the IDSS interface itself follows params.py.
"""
from math import radians

from common import MATERIALS, PHYSICS_MATERIALS, TEXTURE_SETS  # noqa: F401

ASSET_ID = "ida"
FAMILY = "ida"
TITLE = "International Docking Adapter (passive IDSS)"
DOC = ("International Docking Adapter (IDA-2/IDA-3) on the ISS: passive IDSS docking port with the fixed capture "
       "ring, three truss guide petals and three strikers, the hard-capture ring with 12 active/passive hook pairs, "
       "guide pins, a flat sealing land, retractable separators and interface sensors, under white MLI with "
       "handrails, peripheral docking targets, retro-reflectors and trunnions.")
VERSION = open(__file__.replace("configs/ida.py", "VERSION")).read().strip()
ROBOT_TYPE = "Manipulator"
SEMANTIC_CLASS = ["docking_system", "idss_passive", "docking_adapter"]

SENSOR_ANGLES = [radians(152.0), radians(272.0), radians(32.0)]
HANDRAIL_ANGLES = [radians(a) for a in (22.0, 68.0, 112.0, 200.0, 248.0, 292.0, 338.0)]
TARGET_ANGLES = [radians(158.0)]
TRUNNION_ANGLES = [radians(a) for a in (0.0, 180.0)]
REFLECTOR_ANGLES = [radians(a) for a in (45.0, 165.0, 285.0)]
CONNECTOR_ANGLE = radians(90.0)

MASSES = {
    "ida_base": 507.2,
    "hook_active": 0.5,
    "hook_carriage": 0.4,
    "hook_passive": 0.4,
    "separator": 0.4,
    "sensor": 0.05,
}

HOOK_DRIVE = {"stiffness": 2.0e4, "damping": 100.0, "max_force": 600.0, "target": 0.0, "armature": 0.5}   # gang swing drive
HOOK_PULL = {"stiffness": 5.0e7, "damping": 4.0e4, "max_force": 40.0e3, "target": 0.0, "armature": 1000.0}   # preload carriage
PASSIVE_HOOK_SPRING = {"stiffness": 8.0e7, "damping": 2.0e4, "max_force": 6.0e4, "target": -1.0e-4, "armature": 1500.0}
SEPARATOR_SPRING = {"stiffness": 2.0e5, "damping": 400.0, "max_force": 2000.0, "target": 0.0, "armature": 1.0,
                    "spring": {"stiffness": 700.0, "free_length": 1.017}}
SENSOR_SPRING = {"stiffness": 2000.0, "damping": 5.0, "max_force": 50.0, "target": -0.002, "armature": 0.05}
PASSIVE_JOINT = {"damping": 0.5, "armature": 0.01}
CONTACT = {"ke": 2.0e7, "kd": 2.0e4, "gap": 0.0015, "margin": 0.0}

RENDER_VIEWS = {
    "studio_front": ("studio", (3.0, -2.6, 1.9), (0, 0, 0.45), 38),
    "studio_top": ("studio", (1.7, -1.3, 2.4), (0, 0, 0.6), 35),
    "studio_low": ("studio", (2.6, -2.1, 0.3), (0, 0, 0.35), 35),
    "petal_closeup": ("studio", (1.1, -0.4, 1.25), (0.45, 0.0, 0.75), 50),
    "hook_closeup": ("studio", (1.0, 0.3, 0.85), (0.64, 0.17, 0.645), 70),
    "thumb": ("studio", (3.0, -2.6, 1.9), (0, 0, 0.45), 40),
}
