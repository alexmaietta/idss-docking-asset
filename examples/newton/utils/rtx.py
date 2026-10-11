# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# OVRTX rendering helpers for the IDSS docking renders
#
# Shared by render_docking_rtx.py (the README docking GIFs),
# render_stills_rtx.py (gallery stills and USD thumbnails) and
# record_docking_video.py (the example in the stock RTX viewer):
#   - ProductionRTX: the Newton RTX viewer in OVRTX's production PathTracing
#     mode (OptiX denoiser, fixed sample count per frame), lit by the orbital
#     HDRI or a neutral studio rig, with per-shape opacity for x-ray views;
#   - the x-ray rule, camera aiming, the recorded-run helpers and the small
#     phase / sim-time caption chip.
#
###########################################################################

import math
import os

import numpy as np
import warp as wp
from PIL import Image, ImageDraw, ImageFont

import newton
import newton.viewer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HDRI = os.path.join(REPO, "refs", "env", "RenderCrate-HDRI_Orbital_34.exr")   # git-ignored, see the README
HDRI_SUN_AZIMUTH = 135.5   # azimuth of the HDRI's sun disc with no rotation [deg]
SUN_AZIMUTH = math.degrees(math.atan2(-0.35, 0.55))   # where the sun sits in every space render [deg]
HDRI_ROTATION = SUN_AZIMUTH - HDRI_SUN_AZIMUTH

# lighting rigs: dome (colour, intensity) and distant lights (direction the light travels, intensity, angle [deg],
# colour). In space the HDRI is the dome; the close-ups add a low key under the NDS base (the HDRI sun is 32 deg up),
# earthshine from below and a rim.
EARTHSHINE = ((0.25, -0.35, 0.90), 400.0, 40.0, (0.70, 0.82, 1.0))
RIGS = {
    "space": {"dome": None, "lights": [EARTHSHINE]},
    "space_detail": {"dome": None, "lights": [((-0.62, 0.70, -0.30), 1500.0, 2.0, (1.0, 0.97, 0.92)), EARTHSHINE,
                                              ((0.80, 0.45, -0.40), 900.0, 2.0, (1.0, 1.0, 1.0))]},
    "studio": {"dome": ((0.30, 0.30, 0.31), 300.0),
               "lights": [((-0.55, 0.45, -0.70), 2600.0, 8.0, (1.0, 0.97, 0.93)),
                          ((0.70, 0.35, -0.30), 700.0, 20.0, (0.92, 0.95, 1.0)),
                          ((0.30, -0.80, 0.25), 900.0, 4.0, (1.0, 1.0, 1.0))]},
}

# fixed structure that turns see-through in x-ray views (material of the merged base-body meshes)
GHOST_TOKENS = {
    "ida_base": ("mli", "hcs", "aluminium", "aluminium_machined", "seal_land", "target", "handrail", "connector"),
    "nds_base": ("hcs", "iridite", "aluminium_machined", "black_anodize", "seal_silicone", "steel"),
}
GHOST_DRESSING = ("visiting_vehicle", "pma_stub")


class ProductionRTX(newton.viewer.ViewerRTX):
    """ViewerRTX in OVRTX's PathTracing mode: each frame accumulates `spp` samples (denoised) before it is read.

    rig: a RIGS entry (or a dict of the same shape); with dome None the orbital HDRI is the dome.
    shape_opacity: fn(model) -> per-shape opacity, applied before the stage is built (it cannot change later).
    """

    def __init__(self, width, height, rig="space", spp=64, spp_per_step=16, hdri_intensity=500.0, light_scale=1.0,
                 shape_opacity=None, fov=None):
        self.spp, self.spp_per_step = spp, spp_per_step
        self.rig = RIGS[rig] if isinstance(rig, str) else rig
        self.hdri = HDRI if self.rig["dome"] is None and os.path.exists(HDRI) else None
        self.hdri_intensity, self.light_scale = hdri_intensity, light_scale
        self.shape_opacity = shape_opacity
        super().__init__(width=width, height=height, headless=True, async_rendering=False, environment="none")
        if fov is not None:
            self.camera.fov = fov

    def set_model(self, model):
        if model is not None and self.shape_opacity is not None:
            model.shape_opacity.assign(self.shape_opacity(model))
        super().set_model(model)

    def _add_camera_lights_and_render_product(self):
        super()._add_camera_lights_and_render_product()
        from pxr import Gf, Sdf, UsdGeom, UsdLux

        rp = self.stage.GetPrimAtPath(self._render_product_path)
        rp.GetAttribute("omni:rtx:rendermode").Set("PathTracing")
        rp.GetAttribute("omni:rtx:pt:samplesPerPixel").Set(self.spp)
        rp.GetAttribute("omni:rtx:pt:denoising:enabled").Set(True)
        rp.CreateAttribute("omni:rtx:pt:clampSpp", Sdf.ValueTypeNames.Int).Set(self.spp_per_step)
        rp.CreateAttribute("omni:rtx:pt:maxBounces", Sdf.ValueTypeNames.Int).Set(8)
        rp.GetAttribute("omni:rtx:dlss:frameGeneration").Set(False)
        xf = UsdGeom.Xform.Define(self.stage, "/root/_RTXDomeLight")
        xf.ClearXformOpOrder()
        dome = UsdLux.DomeLight.Define(self.stage, "/root/_RTXDomeLight/dome")
        if self.hdri:
            xf.AddRotateZOp().Set(HDRI_ROTATION)
            dome.CreateTextureFileAttr().Set(Sdf.AssetPath(self.hdri))
            dome.CreateTextureFormatAttr().Set("latlong")
            dome.CreateIntensityAttr().Set(self.hdri_intensity)
        else:
            color, intensity = self.rig["dome"] or ((0.02, 0.02, 0.02), 1.0)
            dome.CreateColorAttr().Set(Gf.Vec3f(*color))
            dome.CreateIntensityAttr().Set(intensity)
        for k, (d, intensity, angle, color) in enumerate(self.rig["lights"]):   # a distant light shines along -Z
            light = UsdLux.DistantLight.Define(self.stage, f"/root/_RTXLight{k}")
            light.CreateIntensityAttr().Set(intensity * self.light_scale)
            light.CreateAngleAttr().Set(angle)
            light.CreateColorAttr().Set(Gf.Vec3f(*color))
            rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, -1.0), Gf.Vec3d(*d).GetNormalized())
            UsdGeom.Xformable(light.GetPrim()).AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(rot.GetQuat())

    def converge(self, max_steps=256):
        """Keep stepping the unchanged scene until the path tracer reaches its sample count."""
        for _ in range(max_steps):
            if all(f.converged for p in self._render_products.values() for f in p.frames):
                return
            self._render_and_display()

    def pixels(self):
        return np.asarray(self._capture_screenshot_pixels())[..., :3]

    def render_state(self, state, time=0.0):
        self.begin_frame(time)
        self.log_state(state)
        self.end_frame()
        self.converge()
        return self.pixels()


def ghost_opacity(opacity):
    """Per-shape opacity: the fixed IDA/NDS structure (and the vehicle and PMA dressing) at `opacity`, the rest solid."""
    def fn(model):
        op = model.shape_opacity.numpy()
        if opacity >= 1.0:
            return op
        body, labels, blabels = model.shape_body.numpy(), list(model.shape_label), list(model.body_label)
        for i, lab in enumerate(labels):
            name = lab.rsplit("/", 1)[-1]
            b = blabels[body[i]].rsplit("/", 1)[-1] if body[i] >= 0 else ""
            mat = name.split("__", 1)[1] if "__" in name else ""
            if (b in GHOST_TOKENS and mat.startswith(GHOST_TOKENS[b])) or name in GHOST_DRESSING:
                op[i] = opacity
        return op
    return fn


def lens_fov(lens, aspect):
    """Vertical field of view [deg] of a lens [mm] on a 36 mm wide sensor."""
    return math.degrees(2.0 * math.atan(36.0 / aspect / 2.0 / lens))


def look(viewer, pos, target):
    d = np.subtract(target, pos)
    d /= np.linalg.norm(d)
    viewer.set_camera(wp.vec3(*pos), math.degrees(math.asin(d[2])), math.degrees(math.atan2(d[1], d[0])))


# overall docking framing (the RTX viewer video): a slow orbit around the HCS interface
ORBIT_TARGET = (0.0, 0.0, 0.75)


def orbit_camera(t):
    az = math.radians(-50.0 + 0.6 * t)
    return (ORBIT_TARGET[0] + 2.8 * math.cos(az), ORBIT_TARGET[1] + 2.8 * math.sin(az), 1.35), ORBIT_TARGET


# hero camera: per-phase framing (target, distance, height above the target, x-ray amount), eased between phases.
# The capture latch at azimuth -60 deg and the hook station at -45 deg (active/passive pairs at -47.5 / -42.5 deg on
# r = 0.664 m) face the orbit when their phases come up.
LATCH = (0.26, -0.45, 0.68)
HOOK_STATION = (0.664 * math.cos(math.radians(-45.0)), 0.664 * math.sin(math.radians(-45.0)), 0.645)
HERO_KEYS = {
    "approach": ((0.0, 0.0, 0.80), 2.9, 0.55, 0.0),
    "contact": (LATCH, 1.15, 0.35, 0.8),
    "captured": (LATCH, 1.15, 0.35, 0.8),
    "attenuate": ((0.0, 0.0, 0.72), 2.0, 0.45, 0.35),
    "align": ((0.0, 0.0, 0.75), 2.4, 0.50, 0.0),
    "retract": ((0.0, 0.0, 0.66), 1.5, 0.30, 0.8),
    "hook gang 1": (HOOK_STATION, 0.45, 0.03, 0.9),
    "hook gang 2": (HOOK_STATION, 0.45, 0.03, 0.9),
    "docked": ((0.0, 0.0, 0.75), 2.7, 0.55, 0.0),
}


def hero_path(sim, idx, fps=12, ease=0.3):
    """Per frame (camera position, target, x-ray amount 0..1) for the hero: a slow orbit whose framing follows the
    phase, smoothed with a Gaussian of `ease` seconds (output time) so the moves and the x-ray fade ease in and out."""
    n = len(idx)
    az = np.radians(-50.0 + 2.0 * np.arange(n) / fps)   # 2 deg per output second
    raw = []
    for k, i in enumerate(idx):
        tgt, dist, h, ghost = HERO_KEYS.get(str(sim["phase"][i]), HERO_KEYS["align"])
        raw.append([*tgt, dist, h, ghost])
    raw = np.array(raw)
    sigma = max(1.0, ease * fps)
    kern = np.exp(-0.5 * (np.arange(-3 * int(sigma), 3 * int(sigma) + 1) / sigma) ** 2)
    kern /= kern.sum()
    pad = len(kern) // 2
    sm = np.stack([np.convolve(np.pad(raw[:, j], pad, mode="edge"), kern, mode="valid") for j in range(raw.shape[1])],
                  axis=1)
    # the x-ray fade is causal (it starts when its phase starts, never before): exponential ease over the past frames
    decay = np.exp(-np.arange(len(kern)) / sigma)
    decay /= decay.sum()
    g = np.pad(raw[:, 5], (len(decay) - 1, 0), mode="edge")
    sm[:, 5] = np.convolve(g, decay, mode="valid")
    out = []
    for k in range(n):
        tx, ty, tz, dist, h, ghost = sm[k]
        pos = (tx + dist * math.cos(az[k]), ty + dist * math.sin(az[k]), tz + h)
        out.append((pos, (tx, ty, tz), float(np.clip(ghost, 0.0, 1.0))))
    return out


# ---------------------------------------------------------------- recorded runs
# playback speed per phase (sim seconds per output second): capture plays slowly, alignment and retraction quickly
SPEED = {"approach": 2.0, "contact": 0.6, "captured": 0.6, "attenuate": 3.0, "align": 5.0, "retract": 4.0,
         "hook gang 1": 1.5, "hook gang 2": 1.5, "docked": 1.0}


def load_recording(path):
    return dict(np.load(path, allow_pickle=True))


def sim_index(sim, phase, offset=0.0):
    """Recorded frame `offset` seconds after the start of `phase` (the last frame if the phase never happened)."""
    t, ph = sim["time"], sim["phase"]
    idx = np.nonzero(ph == phase)[0]
    return len(t) - 1 if len(idx) == 0 else int(np.argmin(np.abs(t - (t[idx[0]] + offset))))


def frame_times(sim, n, speed=SPEED, start=("contact", -1.5), end=("docked", 1.5)):
    """n recorded frame indices, evenly spaced in output time with the per-phase playback speed."""
    t, ph = sim["time"], sim["phase"]
    i0, i1 = sim_index(sim, *start), sim_index(sim, *end)
    dt = np.diff(t[i0:i1 + 1], prepend=t[i0])
    g = np.cumsum(np.array([1.0 / speed.get(str(p), 2.0) for p in ph[i0:i1 + 1]]) * dt)
    return [i0 + int(np.searchsorted(g, x).clip(0, len(g) - 1)) for x in np.linspace(0, g[-1], n)]


def playback(sim, fps, speed=SPEED, start=("approach", 0.0), end=("docked", 2.0)):
    """Recorded frame indices for an output at `fps`, with the per-phase playback speed."""
    t, ph = sim["time"], sim["phase"]
    i0, i1 = sim_index(sim, *start), sim_index(sim, *end)
    dt = np.diff(t[i0:i1 + 1], prepend=t[i0])
    duration = float(np.sum(np.array([1.0 / speed.get(str(p), 2.0) for p in ph[i0:i1 + 1]]) * dt))
    return frame_times(sim, max(2, round(duration * fps)), speed, start, end)


def frame_meta(sim, idx):
    """Telemetry of the recorded frames, as scripts/make_gif.py reads it from frames.json."""
    keys = [k for k in ("ring_load", "hook_swing", "hook_pull", "latched", "lateral", "tilt") if k in sim]
    return [{"frame": k, "time": float(sim["time"][i]), "phase": str(sim["phase"][i]), "gap": float(sim["gap"][i]),
             "preload": float(sim["preload"][i]), "latch": [float(x) for x in sim["latch"][i]],
             **{key: float(sim[key][i]) for key in keys}} for k, i in enumerate(idx)]


# ---------------------------------------------------------------- caption chip
def _font(size, weight="Regular"):
    p = f"/usr/share/fonts/opentype/inter/Inter-{weight}.otf"
    return ImageFont.truetype(p, size) if os.path.exists(p) else ImageFont.load_default()


def caption(img, phase, t, scale=1.0):
    """The small phase / sim-time chip, top left, on a translucent dark background."""
    img = img.convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    fb, fr = _font(round(22 * scale), "SemiBold"), _font(round(20 * scale))
    name, tm = phase[:1].upper() + phase[1:], f"{t:.1f} s"
    pad = int(10 * scale)
    w = pad + d.textlength(name, font=fb) + int(10 * scale) + d.textlength(tm, font=fr, features=["tnum"]) + pad
    h = fb.size + 2 * int(7 * scale)
    x = y = int(12 * scale)
    d.rounded_rectangle([x, y, x + w, y + h], radius=int(6 * scale), fill=(12, 12, 11, 190))
    d.text((x + pad, y + h / 2), name, font=fb, fill=(240, 239, 236), anchor="lm")
    d.text((x + w - pad, y + h / 2), tm, font=fr, fill=(195, 194, 183), anchor="rm", features=["tnum"])
    return Image.alpha_composite(img, ov).convert("RGB")
