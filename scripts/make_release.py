"""Build the GitHub release files for the current VERSION from assets/ (run after ./pipeline.sh update).

Run: .venv/bin/python scripts/make_release.py [--out build/release]

Writes build/release/v<VERSION>/:
  idss-usd.zip     ida/, nds/ (SimReady-style USD assets: <id>.usda, payloads/, textures)  + LICENSE, NOTICE
  idss-urdf.zip    urdf/ida, urdf/nds (URDF, OBJ meshes, convex collision STLs)            + LICENSE, NOTICE
  idss-mjcf.zip    mjcf/ (ida/, nds/, idss_docking_scene.xml)                               + LICENSE, NOTICE
  idss-blend.zip   blender/ida.blend, blender/nds.blend (textures packed)                   + LICENSE, NOTICE
  SHA256SUMS, RELEASE_NOTES.md
File names carry no version, so .../releases/latest/download/idss-usd.zip always points at the newest release.
"""
import argparse
import hashlib
import os
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FORMATS = {"usd": ["ida", "nds"], "urdf": ["urdf"], "mjcf": ["mjcf"], "blend": ["blender"]}   # zip -> assets/ folders


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "release"))
    a = ap.parse_args()
    version = open(os.path.join(ROOT, "VERSION")).read().strip()
    out = os.path.join(a.out, f"v{version}")
    os.makedirs(out, exist_ok=True)
    lic, notice = os.path.join(ROOT, "assets", "LICENSE"), os.path.join(ROOT, "assets", "NOTICE")
    sums = []
    for name, folders in FORMATS.items():
        zp = os.path.join(out, f"idss-{name}.zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for folder in folders:
                src = os.path.join(ROOT, "assets", folder)
                if not os.path.isdir(src):
                    raise SystemExit(f"missing assets/{folder}: run ./pipeline.sh update first")
                for dirpath, dirnames, files in os.walk(src):
                    dirnames.sort()
                    for f in sorted(x for x in files if not x.endswith(".blend1")):
                        p = os.path.join(dirpath, f)
                        z.write(p, os.path.join(folder, os.path.relpath(p, src)))
            z.write(lic, "LICENSE")
            z.write(notice, "NOTICE")
        sums.append(f"{sha256(zp)}  {os.path.basename(zp)}")
        print(f"wrote {zp} ({os.path.getsize(zp) / 1e6:.0f} MB)")
    with open(os.path.join(out, "SHA256SUMS"), "w") as f:
        f.write("\n".join(sums) + "\n")
    with open(os.path.join(out, "RELEASE_NOTES.md"), "w") as f:
        f.write(f"IDSS docking system assets v{version}: the passive International Docking Adapter (IDA) and the active "
                "NASA Docking System Block 1 (NDS).\n\n"
                "| File | Contents |\n|---|---|\n"
                "| `idss-usd.zip` | SimReady/Isaac-style USD assets `ida/ida.usda`, `nds/nds.usda` (Physics, Fidelity and "
                "Contact variants) |\n"
                "| `idss-urdf.zip` | URDF (`urdf/ida/ida.urdf`, `urdf/nds/nds.urdf`) with OBJ meshes and convex collision |\n"
                "| `idss-mjcf.zip` | MJCF models and the docking scene `mjcf/idss_docking_scene.xml` |\n"
                "| `idss-blend.zip` | Blender source models (`blender/ida.blend`, `blender/nds.blend`), textures packed |\n\n"
                "Assets are licensed CC BY 4.0 (LICENSE and NOTICE in each zip). Checksums: `SHA256SUMS`.\n")
    print(f"release files in {out}")


if __name__ == "__main__":
    main()
