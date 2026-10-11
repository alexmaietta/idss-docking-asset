"""Locate and load an asset's config module (configs/<id>.py) and its build folders (build/<id>/)."""
import sys
import importlib.util
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(asset_id):
    if os.path.join(ROOT, "configs") not in sys.path:
        sys.path.append(os.path.join(ROOT, "configs"))
    path = os.path.join(ROOT, "configs", f"{asset_id}.py")
    spec = importlib.util.spec_from_file_location(f"cfg_{asset_id}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_dir(asset_id, *parts):
    d = os.path.join(ROOT, "build", asset_id, *parts)
    os.makedirs(d if not os.path.splitext(d)[1] else os.path.dirname(d), exist_ok=True)
    return d


def arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default
