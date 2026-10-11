"""Materials shared by the IDSS assets: texture sets (scripts/texgen.py generators) and Blender materials.

Colours are linear RGB. Finishes follow the hardware papers (refs/IDSS_SPEC.md "Materials") and were
matched by eye to the KSC/ISS photographs in refs/images/photos (lighting there is cleanroom fluorescent
or full sun, so values are judged on mid-tones, not highlights):

- HCS rings: machined AMg6 (IDA) / 2219 (NDS) aluminium, clear anodize: light satin grey with turning
  marks (KSC-20160713-PH_SPX01, ida1_ksc_2015_sspf_top_angle).
- IDA capture ring and truss petals: 6061 Type III hard anodize: dark gunmetal with a warm cast.
- NDS petals: 7075 PTFE-impregnated hard anodize: mid grey satin (NDS docking test footage).
- NDS capture ring: black anodize. Strikers: passivated A286 (bright). Hooks: titanium with dry-film
  lubricant (dark matte grey). Guide pins: Inconel 718. Seals: red-orange silicone (APAS heritage).
- MLI: Teflon-coated beta cloth (white), ISS handrails: gold (yellow) anodize.
"""

TEXTURE_SETS = {
    "hcs_turned": ("turned", dict(colour=(0.74, 0.745, 0.75), tile_m=0.3, rough0=0.30)),
    "aluminium": ("brushed", dict(colour=(0.78, 0.78, 0.79), tile_m=0.4, rough0=0.32)),
    "aluminium_machined": ("anodised", dict(colour=(0.70, 0.705, 0.71), tile_m=0.25, rough0=0.38)),
    "hard_anodize_dark": ("hard_anodize", dict(colour=(0.030, 0.027, 0.024), tile_m=0.25, rough0=0.48)),
    "ptfe_anodize": ("hard_anodize", dict(colour=(0.16, 0.162, 0.165), tile_m=0.25, rough0=0.38, metallic=0.15)),
    "black_anodize": ("hard_anodize", dict(colour=(0.018, 0.018, 0.02), tile_m=0.25, rough0=0.36)),
    "a286": ("brushed", dict(colour=(0.62, 0.61, 0.60), tile_m=0.15, rough0=0.20)),
    "titanium_dfl": ("hard_anodize", dict(colour=(0.065, 0.066, 0.07), tile_m=0.08, rough0=0.62, metallic=0.25)),
    "inconel": ("brushed", dict(colour=(0.60, 0.58, 0.55), tile_m=0.08, rough0=0.22)),
    "steel": ("brushed", dict(colour=(0.56, 0.56, 0.57), tile_m=0.1, rough0=0.30)),
    "seal_silicone": ("rubber", dict(colour=(0.36, 0.035, 0.016), tile_m=0.1, rough0=0.55)),
    "beta_cloth": ("beta_cloth", dict(colour=(0.74, 0.74, 0.71), tile_m=0.6, rough0=0.78)),
    "gold_anodize": ("anodised", dict(colour=(0.62, 0.38, 0.045), tile_m=0.2, rough0=0.32)),
    "iridite": ("anodised", dict(colour=(0.70, 0.645, 0.46), tile_m=0.3, rough0=0.40)),
    "target_white": ("painted", dict(colour=(0.80, 0.80, 0.78), tile_m=0.3, rough0=0.55)),
}

MATERIALS = {
    "hcs": {"set": "hcs_turned", "normal": 0.5},
    "aluminium": {"set": "aluminium", "normal": 0.5},
    "aluminium_machined": {"set": "aluminium_machined", "normal": 0.5},
    "hard_anodize": {"set": "hard_anodize_dark", "normal": 0.6},
    "ptfe_anodize": {"set": "ptfe_anodize", "normal": 0.5},
    "black_anodize": {"set": "black_anodize", "normal": 0.5},
    "a286": {"set": "a286", "normal": 0.4},
    "titanium_dfl": {"set": "titanium_dfl", "normal": 0.5},
    "inconel": {"set": "inconel", "normal": 0.3},
    "steel": {"set": "steel", "normal": 0.4},
    "seal_silicone": {"set": "seal_silicone", "normal": 0.4},
    "seal_land": {"color": (0.82, 0.82, 0.83), "metallic": 1.0, "roughness": 0.12},
    "mli": {"set": "beta_cloth", "normal": 1.0},
    "handrail": {"set": "gold_anodize", "normal": 0.4},
    "iridite": {"set": "iridite", "normal": 0.5},
    "target_white": {"set": "target_white", "normal": 0.3},
    "target_black": {"color": (0.012, 0.012, 0.013), "roughness": 0.7},
    "retroreflector": {"color": (0.9, 0.9, 0.92), "metallic": 1.0, "roughness": 0.03},
    "latch_body": {"set": "steel", "normal": 0.3},
    "chrome": {"color": (0.85, 0.85, 0.86), "metallic": 1.0, "roughness": 0.08},
    "connector": {"color": (0.70, 0.70, 0.68), "metallic": 1.0, "roughness": 0.28},
    "cable": {"color": (0.06, 0.06, 0.065), "roughness": 0.55},
    "cable_white": {"color": (0.72, 0.72, 0.70), "roughness": 0.6},
    "placard": {"color": (0.78, 0.78, 0.74), "roughness": 0.5},
    "kapton": {"color": (0.55, 0.22, 0.02), "metallic": 0.6, "roughness": 0.25},
}

# physics materials: name -> (static friction, dynamic friction, restitution)
PHYSICS_MATERIALS = {
    "anodize": (0.15, 0.12, 0.0),      # Type III hard anodize on anodize / steel (dry, vacuum)
    "ptfe": (0.10, 0.08, 0.0),         # PTFE-impregnated hard anodize (NDS petals)
    "steel": (0.20, 0.16, 0.0),        # A286 strikers, Inconel guide pins, latch rollers
    "titanium": (0.18, 0.15, 0.0),     # hooks with dry-film lubricant
    "seal": (0.8, 0.7, 0.0),           # silicone seal beads
}
