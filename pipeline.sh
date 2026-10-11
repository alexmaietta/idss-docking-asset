#!/usr/bin/env bash
# Build the IDSS docking assets end to end: textures -> Blender build -> export (full + proxy) -> USD package
# -> OVRTX thumbnail -> URDF/MJCF; then the Newton test and the OVRTX renders (README stills, GIFs, video).
# Usage: ./pipeline.sh [asset ...] [step ...]
#   assets: nds ida (default: both)
#   steps:  textures build export package thumb robots sim record stills rtx video assets update release example all
#           (default: all)
#   e.g.    ./pipeline.sh                       everything for both assets
#           ./pipeline.sh nds build export package
#           IDSS_DRAFT=1 ./pipeline.sh textures build   (1024 px textures for fast iteration)
set -euo pipefail
cd "$(dirname "$0")"
export BLENDER=${BLENDER:-$(command -v blender || echo "$HOME/opt/blender-5.2.2-linux-x64/blender")}   # Blender 5.2
PY=${PY:-.venv/bin/python}
RES=$([ "${IDSS_DRAFT:-0}" = 1 ] && echo 1024 || echo 2048)

assets=(); steps=()
for a in "$@"; do case $a in nds|ida) assets+=("$a") ;; *) steps+=("$a") ;; esac; done
[ ${#assets[@]} -eq 0 ] && assets=(nds ida)
[ ${#steps[@]} -eq 0 ] && steps=(all)

textures() { $PY scripts/textures.py --asset "$1" --res "$RES"; }
build()    { $BLENDER -b --factory-startup -P scripts/build.py -- --asset "$1"; }
export_()  { for lod in full proxy; do $BLENDER -b "build/$1/$1.blend" -P scripts/export.py -- --asset "$1" --lod $lod; done; }
package()  { $PY scripts/package.py --asset "$1"; }
# 256x256 USD thumbnail, path traced by OVRTX from the packaged asset (build/<id>/usd/<id>/.thumbs/256x256/)
thumb()    { $PY examples/newton/utils/render_stills_rtx.py --thumb "$1" --assets "build/$1/usd"; }
robots()   { $PY scripts/export_robots.py --asset "$1"; }      # URDF + MJCF (build/urdf/<id>, build/mjcf/<id>)
scene()    { $PY scripts/export_robots.py --scene; }            # MJCF docking scene (needs both assets)
sim()      { $PY examples/newton/example_idss_docking.py --viewer null --num-frames 3600 --test; }
record()   { $PY examples/newton/utils/record_docking.py --out build/sim/docking.npz; }
# README renders, all OVRTX (Newton RTX viewer) from the recorded run: gallery stills; the hero GIF and the x-ray GIF;
# the example in the stock RTX viewer
stills()   { $PY examples/newton/utils/render_stills_rtx.py; }
rtx()      { $PY examples/newton/utils/render_docking_rtx.py --out build/anim_rtx \
               && $PY scripts/make_gif.py --frames build/anim_rtx --single hero --width 1280 --out docs/images/docking.gif \
               && $PY scripts/make_gif.py --frames build/anim_rtx --plain --out docs/images/docking_xray.gif; }
video()    { $PY examples/newton/utils/record_docking_video.py --out docs/images/newton_docking; }
# copy the packaged USD, URDF, MJCF and the .blend into the committed assets/ folder (Git LFS)
assets_()  { mkdir -p assets/urdf assets/mjcf; rsync -a --delete "build/$1/usd/$1/" "assets/$1/"
             rsync -a --delete "build/urdf/$1/" "assets/urdf/$1/"; rsync -a --delete "build/mjcf/$1/" "assets/mjcf/$1/"
             [ -f build/mjcf/idss_docking_scene.xml ] && cp build/mjcf/idss_docking_scene.xml assets/mjcf/
             $BLENDER -b "build/$1/$1.blend" -P scripts/export_blend.py -- --out "assets/blender/$1.blend"; echo "assets: $1"; }
update()   { textures "$1"; build "$1"; export_ "$1"; package "$1"; thumb "$1"; robots "$1"; }
example()  { $PY examples/newton/example_idss_docking.py; }
release()  { for a in "${assets[@]}"; do update "$a"; done; scene; for a in "${assets[@]}"; do assets_ "$a"; done
             $PY scripts/make_release.py; v=$(cat VERSION)
             echo "next: git add -A assets VERSION && git commit -m \"Release v$v\" && git tag v$v && git push origin main v$v"; }

for step in "${steps[@]}"; do
  case $step in
    stills|rtx|video|record|scene|sim|example|release) $step ;;
    *)
      for a in "${assets[@]}"; do
        case $step in
          all) textures "$a"; build "$a"; export_ "$a"; package "$a"; thumb "$a"; robots "$a" ;;
          export) export_ "$a" ;;
          update) update "$a"; assets_ "$a" ;;
          assets) assets_ "$a" ;;
          *) $step "$a" ;;
        esac
      done ;;
  esac
done
