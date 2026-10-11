"""Single source of truth for IDSS docking-system dimensions (SI units: metres, kg, radians).

References are to the IDSS Interface Definition Document Rev F (July 2022, "IDD") and the NDS IDD
JSC 65795 Rev H (Nov 2013, "NDS IDD"); see refs/IDSS_SPEC.md. Values marked ESTIMATE are not defined
by either document and were scaled from photos or chosen for plausibility (reasoning given inline).

Interface frame (shared by both halves): origin on the docking axis in the HCS mating plane, +Z out
of the interface (towards the mating vehicle), +X along the IDD "line of symmetry" (through the
mechanical latch / guide petal at 0 deg), +Y along the "line of androgyny" (guide pin 1 / receptacle 1).
Clock angles are measured from +X towards +Y. The mating interface is this one rotated 180 deg about
the line of androgyny: (x, y, z) -> (-x, y, -z). With that mirror every petal lands on a striker, every
active hook on a passive hook and every guide pin in a receptacle (IDD Fig 3.1.1.1-1, 3.2.3-1).
"""
from math import atan, degrees, radians, sqrt, tan

MM = 1e-3
IN = 0.0254


def deg(a):
    return radians(a)


# ------------------------------------------------------------------ HCS: hard-capture structural ring
TUNNEL_ID = 1255.0 * MM          # IDD Fig 3.1.1.1-2 (+0.65/-0.5)
PASSAGE_D = 800.0 * MM           # transfer passageway, IDD Fig 3.1.1.1-2
HCS_OD = 1420.0 * MM             # HCS flange OD (NDS IDD static envelope 56.00 in = 1422.4)
HCS_RING_H = 46.0 * MM           # ESTIMATE: axial depth of the machined HCS ring (hooks pivot 25.2 below plane)
TUNNEL_WALL = 6.0 * MM           # ESTIMATE: tunnel skin under the ring
SEAL_D = (1273.0 * MM, 1304.0 * MM)  # dual pressure seal beads, IDD Fig 3.2.3-1 (+-1)
SEAL_H = 2.1 * MM                # seal bead height above the HCS plane (max), IDD Fig 3.2.3-1
SEAL_W = 2.2 * MM                # ESTIMATE: bead width (scaled from IDD Fig 3.2.3-1 detail and the 2023 seal poster)
SEAL_CLOSURE_FORCE = 97150.0    # N, seal closure force (IDD 3.2.3.2)
# Seal compliance as a MuJoCo contact time constant: the station side is effectively immovable, so the seal
# contact's effective mass is the visiting vehicle's; k ~ 97 kN / 2.1 mm = 46 MN/m with a 13.3 t vehicle gives
# sqrt(m/k) ~ 17 ms; 8 ms (~200 MN/m) lets the 12 x ~36 kN hook preload close the beads fully onto the lands,
# as on the real interface (ESTIMATE; scales with vehicle mass).
SEAL_SOLREF = (0.008, 1.0)
SEAL_ON_SEAL_GAP = 4.2 * MM      # HCS planes apart at seal-on-seal ready-to-hook, IDD Fig 3.2.3.4-2
STRIKER_ZONE_D = (1330.0 * MM, 1410.0 * MM)   # HCS sensor/separator striker zones, IDD Fig 3.2.3-2
STRIKER_ZONE_RECESS = 1.0 * MM
STRIKER_ZONE_W = deg(9.0)        # 9 zones, 9 deg wide (IDD Fig 3.2.3-2 / D.1.1-1)
SENSOR_ZONES = [deg(a) for a in (2.0, 32.0, 62.0, 122.0, 152.0, 182.0, 242.0, 272.0, 302.0)]  # 30k+2, skipping the guide pins

# ------------------------------------------------------------------ hooks: 12 active/passive pairs
HOOK_R = 663.5 * MM              # hook locating radius, IDD Fig 3.2.3-1 (2X 663.5 +-0.3)
HOOK_ANGLES = [deg(15.0 + 30.0 * k) for k in range(12)]   # 12 equally spaced, 15 deg off the lines
HOOK_W = 20.0 * MM               # hook thickness (radial), IDD Fig 3.2.3.4-4 / -5
HOOK_OFFSET = 42.5 * MM          # local centreline to passive hook leading edge, IDD Fig 3.2.3.4-1
HOOK_TANGENT_Z = 2.8 * MM        # hook tangent (catch) plane above the HCS plane, IDD Fig 3.2.3.4-1
HOOK_ACTIVE_LE = 95.0 * MM - HOOK_OFFSET   # active leading edge 52.5 from the centreline (95 REF)
HOOK_PIVOT_DX = 23.0 * MM        # pivot to active leading edge, IDD Fig 3.2.3.4-4
HOOK_PIVOT_DZ = 28.0 * MM        # pivot below the hook tangent line
HOOK_PIVOT_T = HOOK_ACTIVE_LE - HOOK_PIVOT_DX   # pivot 29.5 from the local centreline (tangential)
HOOK_PIVOT_Z = HOOK_TANGENT_Z - HOOK_PIVOT_DZ  # -25.2 below the HCS plane
HOOK_OPEN = deg(-28.0)           # ESTIMATE: open (ready-to-dock) angle; puts the leading edge at ~37 (79.5 REF)
HOOK_CLOSED = deg(0.0)           # hook tangent line horizontal: ready to preload
HOOK_OVERTRAVEL = deg(3.0)       # ESTIMATE: swing range past the tangent position (stop)
HOOK_PULL_STROKE = 8.0 * MM      # ESTIMATE: pull-down carriage stroke (closes the 4.2 mm ready-to-hook gap,
                                 # compresses the seals and stretches the passive Belleville stacks)
PASSIVE_HOOK_TOP = 17.8 * MM     # passive hook top above the HCS plane, IDD Fig 3.2.3.4-1
PASSIVE_HOOK_H = 41.0 * MM       # IDD Fig 3.2.3.4-5
PASSIVE_HOOK_BASE_Z = PASSIVE_HOOK_TOP - PASSIVE_HOOK_H   # -23.2
HOOK_POCKET = (172.0 * MM, 20.6 * MM, 40.0 * MM)  # ESTIMATE: pocket length (tangential), width (radial), depth
# hook loads, IDD 3.2.3.4.1-.3
HOOK_PRELOAD = (31.3e3, 44.34e3)        # N per hook system
HOOK_LIMIT_LOAD = 50.0e3                # N
HOOK_ACTIVE_K = (49.05e6, 55.72e6)      # N/m, active hook stiffness
HOOK_PASSIVE_K1 = (69.26e6, 94.325e6)   # N/m up to the knee (Belleville stack)
HOOK_PASSIVE_KNEE = (23.5e3, 25.5e3)    # N
HOOK_PASSIVE_K2 = (10.5e6, 12.02e6)     # N/m after the knee
HOOK_GANGS = ([0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11])   # NDS: two gangs of six alternating hooks

# Hook side profiles, IDD Fig 3.2.3.4-4 (active) and -5/-6 (passive), in the hook plane:
# x = tangential (towards the leading edge), y = axial. Arcs are flattened to a few segments.
# Active hook: origin at the pivot; the catch (beak underside) is the y=28 edge from x=12 to 23.
ACTIVE_HOOK_PROFILE = [
    (23.0, 28.0), (23.0, 31.5), (21.6, 34.4), (14.0, 42.0), (10.0, 43.6), (5.0, 44.0), (0.0, 43.5),
    (-5.0, 41.6), (-9.5, 38.2), (-12.6, 34.0), (-14.0, 30.0), (-19.0, 10.5), (-17.5, 4.5), (-12.0, -1.0),
    (-5.0, -5.5), (2.0, -6.0), (7.5, -3.5), (11.0, 2.0), (12.0, 8.0), (12.0, 27.0), (13.0, 28.0)]
# Passive hook: origin at the leading edge on the base line (y=0 is the hook base, 23.2 below the
# HCS plane); the catch is the y=26 ledge from x=0 to x=-11, the throat opens below it.
PASSIVE_HOOK_PROFILE = [
    (0.0, 26.0), (0.0, 33.0), (-1.5, 36.0), (-6.0, 39.6), (-12.0, 41.0), (-19.0, 40.6), (-25.0, 38.2),
    (-30.5, 33.5), (-34.0, 27.5), (-35.0, 21.0), (-33.5, 14.5), (-30.0, 11.0), (-21.0, 5.6), (-16.0, 2.5),
    (-14.0, 0.0), (8.0, 0.0), (8.0, 1.2), (6.5, 3.0), (-9.5, 17.5), (-11.0, 20.0), (-11.0, 25.0), (-10.0, 26.0)]

# ------------------------------------------------------------------ guide pins and receptacles
GUIDE_R = 1375.0 * MM / 2        # locating circle, IDD Fig 3.2.3-1 (687.5 +-0.075)
GUIDE_OFFSET = 25.0 * MM         # pin 1 / receptacle 1 either side of the line of androgyny
GUIDE_PINS = [(deg(90.0), -GUIDE_OFFSET), (deg(-30.0), 0.0)]       # (clock angle, tangential offset)
GUIDE_RECEPTACLES = [(deg(90.0), +GUIDE_OFFSET), (deg(210.0), 0.0)]
GUIDE_PIN_D = 24.7 * MM          # 25 -0.2/-0.4, IDD Fig 3.2.3.3-2
GUIDE_PIN_H = 51.0 * MM          # 51 +-0.5 above the HCS plane
GUIDE_PIN_CYL_H = 40.0 * MM      # 40 REF parallel length, then a 90 deg cone with SR3 tip
GUIDE_PIN_TIP_R = 3.0 * MM
GUIDE_RECEPT_SLOT = (30.2 * MM, 25.2 * MM)  # obround hole (tangential x radial), IDD Fig 3.2.3.3-3
GUIDE_RECEPT_CBORE_D = 39.0 * MM
GUIDE_RECEPT_DEPTH = 12.0 * MM       # precision bore; the hole then clears through the ring
GUIDE_RECEPT_CLEAR = 62.0 * MM       # ESTIMATE: through-clearance under the bore for the 51 mm pin
GUIDE_RECEPT_RECESS = 1.0 * MM

# ------------------------------------------------------------------ separators (push-off springs)
SEPARATOR_R = 1375.0 * MM / 2    # IDD Fig D.1.1-5 (IDA separators on the 1375 circle)
IDA_SEPARATORS = [deg(a) for a in (61.5, 181.5, 301.5)]   # IDD Fig D.1.1-5
NDS_SEPARATORS = [deg(a) for a in (2.0, 122.0, 242.0)]    # ESTIMATE: the remaining striker-zone family
SEPARATOR_STROKE = 32.9 * MM     # Block 2 / NDS IDD
SEPARATOR_FORCE = (689.0, 712.0) # N each, at the mated / extended ends
SEPARATOR_D = 28.0 * MM          # ESTIMATE: plunger diameter (IDD Fig D.1.1-6 cutout ~ 34)
SEPARATOR_TOTAL_MAX = 2670.0     # N, IDD 3.2.3.5
SEPARATOR_ENERGY = (39.2, 47.5)  # J

# ------------------------------------------------------------------ SCS: soft-capture ring and petals
SCS_RING_OD = 1250.0 * MM        # IDD Fig 3.2.2.1-5 (+-0.2)
SCS_FLAT_OD = 1220.0 * MM        # SCS mating-plane flat, OD
SCS_FLAT_ID = 1202.0 * MM        # SCS mating-plane flat, ID (point X circle for the strikers)
SCS_RING_ID = 1160.0 * MM        # {1160} inner diameter, IDD Fig 3.2.2.4-3
SCS_RING_H = 62.0 * MM           # ESTIMATE: channel depth (Fig 3.2.2.1-5 proportions: ~1.1 x the 56 wall spacing)
SCS_SHEET = 3.0 * MM             # ESTIMATE: formed channel wall
SCS_CHAMFER = deg(45.0)
SCS_PASSIVE_Z = -6.0 * MM        # SCS mating plane below the HCS plane in passive mode, IDD Fig 3.1.1.1-2

PETAL_ANGLES = [deg(0.0), deg(120.0), deg(240.0)]
STRIKER_ANGLES = [deg(60.0), deg(180.0), deg(300.0)]
PETAL_BASE_R = 600.0 * MM        # petal base on the SCS mating plane, Ø1200
PETAL_TIP_R = 400.0 * MM         # petal tip on the Ø800 passage
PETAL_H = 200.0 * MM             # 200 REF
PETAL_CONE = deg(45.0)           # contact surface (A-B-C) on a 45 deg cone
PETAL_BASE_HALF = deg(30.0)      # base spans +-30 deg about the petal centre line (60 deg REF)
PETAL_TIP_HALF = deg(11.3)       # tip spans +-11.3 deg (measured; consistent with the 32 deg 30' edges)
PETAL_TIP_CORNER_R = 5.0 * MM    # 2X R5
PETAL_ROOT_R = 470.0 * MM        # ESTIMATE: inner (back) web meets the ring at r=470 (Fig 3.1.1.1-3)
PETAL_SKIN = 8.0 * MM            # ESTIMATE: NDS petal skin / IDA truss member depth
PETAL_TRUSS_W = 16.0 * MM        # ESTIMATE: IDA truss petal member width (photos)

# mechanical capture latches (active petals) and strikers, IDD 3.2.2.4
LATCH_SLANT = 115.0 * MM         # "bottom of latch" (catch face) along the petal slant from the base, Fig 3.2.2.4-4
LATCH_TOOTH = 12.0 * MM          # ESTIMATE: catch face to roller centre (roller lead-in on the tip side)
LATCH_ROLLER_D = 10.0 * MM
LATCH_PROTRUSION = 6.0 * MM      # ESTIMATE: free roller protrusion past the petal face (>2 mm step + margin)
LATCH_ARM = 70.0 * MM            # ESTIMATE: rocker length, catch face to pivot (up the slant)
LATCH_WIDTH = 40.0 * MM          # ESTIMATE: rocker width (tangential)
LATCH_WINDOW = (70.0 * MM, 48.0 * MM)  # ESTIMATE: petal window (slant x tangential)
LATCH_SPRING_K = 30.0            # ESTIMATE: N m/rad rocker spring
LATCH_SPRING_PRELOAD = deg(10.0)
STRIKER_X_SLANT = 114.0 * MM     # point X to the catch step along the 45 deg chamfer (114 +0/-0.2)
STRIKER_STEP = 2.0 * MM          # catch step (2 +0.5/-0), Fig 3.2.2.4.1.1-1
STRIKER_UPPER_W = 21.0 * MM      # upper planar surface max height above the catch line
STRIKER_LOWER_W = 35.0 * MM      # lower planar surface min
STRIKER_WIDTH = 232.0 * MM       # min, tangential
STRIKER_FRONT = 70.0 * MM        # straight inner edge (top view)
STRIKER_LOAD = 3000.0            # N, latch-striker design load

# ------------------------------------------------------------------ NDS Block 1 (active), NDS IDD Rev H
NDS_HCS_Z = 230.4 * MM           # HCS mating plane above the host-vehicle mating plane (9.07 in)
NDS_MOUNT_D = 1249.68 * MM       # host-vehicle interface diameter (49.20 in)
NDS_STOWED_TOP = 431.8 * MM      # stowed envelope top above the host-vehicle plane (17.00 in)
NDS_ENVELOPE_D = 1727.2 * MM     # 68.00 in
NDS_KINEMATIC_TOP = 673.1 * MM   # above the HCS plane (26.50 in)
NDS_MASS = 324.0                 # kg, NDS Block 1
# Soft-capture ring travel (ESTIMATES: the IDD leaves the active ring stroke to the implementation).
# Stowed: SCS plane 6 mm below the HCS plane (passive mode). Ready to capture: extended so the petal
# tips clear the passive petals with margin. The kinematic envelope (petal tips 673 above the HCS
# plane) bounds the extension at ~0.47 m.
RING_STOWED_Z = SCS_PASSIVE_Z
RING_READY_Z = 0.33              # ESTIMATE (NDS Block 1 test footage: ring extended ~13 in)
RING_MAX_Z = NDS_KINEMATIC_TOP - PETAL_H
# Stewart platform: 6 Moog electromechanical linear actuators in 3 pairs; struts 1 and 2 on petal 1
# (NDS Block 1 paper). Geometry is ESTIMATE, fitted inside the Ø1255 tunnel (outside the Ø800 passage).
LEG_BASE_R = 0.565               # base U-joint radius
LEG_BASE_Z = -0.155              # base U-joints below the HCS plane (inside the tunnel)
LEG_TOP_R = 0.545                # ring ball-joint radius
LEG_TOP_DZ = -0.040              # ring ball joints below the SCS mating plane
LEG_TOP_SPLIT = deg(9.0)         # ring joints of a pair either side of the petal centre line
LEG_BASE_SPLIT = deg(9.0)        # base joints of a pair either side of the striker line (60 deg off)
LEG_BODY_D = 0.070               # actuator housing
LEG_ROD_D = 0.032
LEG_BODY_L = 0.30
LEG_STROKE_MAX = 0.42
LEG_MAX_FORCE = 4000.0           # N, ESTIMATE (IDD: 3500 N static compression on the SCS)
# Soft-capture load limits, IDD 3.3.2 (SCS interface loads)
SCS_LOADS = {"tension": 3900.0, "compression_static": 3500.0, "compression_dynamic": 6500.0,
             "shear": 3200.0, "bending": 2800.0, "torsion": 1500.0}

# ------------------------------------------------------------------ IDA (passive), NDS IDD Fig 4.2.1-2
IDA_H = 642.0 * MM               # PMA-to-IDA plane to VV-to-IDA plane (25.276 in)
IDA_HCS_Z = IDA_H
IDA_ENVELOPE_H = 1077.0 * MM     # incl. petals and the MAPAS guides (42.4 in)
IDA_ENVELOPE_D = 2388.0 * MM     # incl. trunnions / targets (94 in)
IDA_MASS = 526.0                 # kg (Block 2 paper); passive hard-mate assembly ~165 kg
IDA_BODY_R = 0.800               # ESTIMATE: EIPS shell outer radius under the MLI (photos vs the 1.42 m flange)
IDA_MLI_T = 0.040                # ESTIMATE: blanket stand-off
IDA_TOP_CAP_Z = IDA_H - 0.070    # ESTIMATE: top of the EAR / shell, below the HCS flange (photo)
MAPAS_GUIDES = 8                 # dark guide blades round the PMA interface (KSC-20190620-PH_ILW01_0072)
MAPAS_GUIDE_H = 0.235            # ESTIMATE: 1077 - 642 - 200 (petals)
MAPAS_RING_R = (0.640, 0.800)

# ------------------------------------------------------------------ docking initial conditions, IDD 3.3
CONTACT_VEL_AXIAL = (0.05, 0.10)    # m/s closing
CONTACT_VEL_LATERAL = 0.04          # m/s
CONTACT_RATE = deg(0.2)             # rad/s
CONTACT_OFFSET = 0.10               # m radial misalignment
CONTACT_ANGLE = deg(4.0)            # angular misalignment
VEHICLE_MASS_RANGE = (5.0e3, 25.0e3)
STATION_MASS = 350.0e3

# friction (ESTIMATES: the IDD only requires low-friction finishes)
MU_PTFE_ANODIZE = 0.10           # NDS petals: Teflon-impregnated hard anodize
MU_ANODIZE = 0.15                # IDA petals / rings: Type III hard anodize
MU_STEEL = 0.20                  # A286 strikers, Inconel pins, titanium hooks with dry-film lube


def petal_slant(h):
    """Distance along the 45 deg petal slant from the base for a height h above the SCS plane."""
    return h * sqrt(2.0)


def cone_r(h):
    """Radius of the 45 deg petal/striker cone at height h above the SCS mating plane."""
    return PETAL_BASE_R - h * tan(PETAL_CONE)


assert abs(degrees(atan(1.0)) - 45.0) < 1e-9
