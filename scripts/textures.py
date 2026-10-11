"""Generate an asset's tileable texture sets.  Run: $PY scripts/textures.py --asset nds [--res 2048] [--only a,b]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assetcfg  # noqa: E402
import texgen  # noqa: E402

argv = sys.argv[1:]
asset = assetcfg.arg(argv, "--asset")
cfg = assetcfg.load(asset)
only = assetcfg.arg(argv, "--only")
res = int(assetcfg.arg(argv, "--res", os.environ.get("IDSS_TEX_RES", 2048)))
texgen.generate(assetcfg.build_dir(asset, "textures"), cfg.TEXTURE_SETS, res=res, only=only.split(",") if only else None)
