"""Blender materials driven by the tileable texture sets from texgen.py (Cycles / EEVEE / USD export).

An asset config declares materials as:
    MATERIALS = {name: {"set": texture_set, "normal": strength, "alpha": bool, "emission": (r,g,b,strength)}}
or {"color": (r,g,b), "metallic": m, "roughness": r} for plain untextured materials (decals, glass ...).
"""
import json
import os

import bpy

_tex_dir = None
_tiles = None
_specs = {}
_cache = {}


def configure(tex_dir, specs):
    """Point at the asset's generated texture folder and its MATERIALS table."""
    global _tex_dir, _tiles, _specs
    _tex_dir = os.path.abspath(tex_dir)
    with open(os.path.join(_tex_dir, "tiles.json")) as f:
        _tiles = json.load(f)
    _specs = specs
    _cache.clear()


def _image(nt, set_name, kind, vec, colour=False):
    path = os.path.join(_tex_dir, set_name, f"{set_name}_{kind}.png")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path, check_existing=True)
    tex.image.colorspace_settings.name = "sRGB" if colour else "Non-Color"
    tex.interpolation = "Smart"
    nt.links.new(vec, tex.inputs["Vector"])
    return tex


def _principled(name, color=(0.5, 0.5, 0.5), metallic=0.0, rough=0.5):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except AttributeError:
        pass
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = rough
    m.diffuse_color = (*color, 1.0)
    return m, bsdf


def _textured(name, spec):
    set_name = spec["set"]
    meta = _tiles[set_name]
    m, bsdf = _principled(name)
    nt = m.node_tree
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    mp = nt.nodes.new("ShaderNodeMapping")
    s = 1.0 / meta["tile_m"] * spec.get("uv_scale", 1.0)
    mp.inputs["Scale"].default_value = (s, s, 1.0)
    nt.links.new(uv.outputs["UV"], mp.inputs["Vector"])
    vec = mp.outputs["Vector"]
    base = _image(nt, set_name, "basecolor", vec, True)
    nt.links.new(base.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(_image(nt, set_name, "roughness", vec).outputs["Color"], bsdf.inputs["Roughness"])
    if "metallic" in meta["maps"] and "metallic" not in spec:
        nt.links.new(_image(nt, set_name, "metallic", vec).outputs["Color"], bsdf.inputs["Metallic"])
    else:
        bsdf.inputs["Metallic"].default_value = spec.get("metallic", 0.0)
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nm.uv_map = "UVMap"
    nm.inputs["Strength"].default_value = spec.get("normal", 1.0)
    nt.links.new(_image(nt, set_name, "normal", vec).outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if spec.get("alpha") and "opacity" in meta["maps"]:
        nt.links.new(_image(nt, set_name, "opacity", vec).outputs["Color"], bsdf.inputs["Alpha"])
        _cutout(m)
    m.diffuse_color = (*spec.get("preview", (0.6, 0.6, 0.6)), 1.0)
    return m


def _cutout(m):
    try:
        m.surface_render_method = "DITHERED"
    except AttributeError:
        pass
    m.use_backface_culling = False


def _plain(name, spec):
    m, bsdf = _principled(name, spec.get("color", (0.5, 0.5, 0.5)), spec.get("metallic", 0.0),
                          spec.get("roughness", 0.5))
    if "emission" in spec:
        r, g, b, k = spec["emission"]
        bsdf.inputs["Emission Color"].default_value = (r, g, b, 1.0)
        bsdf.inputs["Emission Strength"].default_value = k
    if "image" in spec:   # decal: image base colour, clipped
        nt = m.node_tree
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(os.path.join(_tex_dir, spec["image"]), check_existing=True)
        tex.interpolation = "Cubic"
        tex.extension = "CLIP"
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        if spec.get("alpha"):
            nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
            _cutout(m)
    return m


def get(name):
    m = _cache.get(name) or bpy.data.materials.get(name)
    if m is None:
        spec = _specs[name]
        m = _textured(name, spec) if "set" in spec else _plain(name, spec)
    _cache[name] = m
    return m
