"""Upload a preprocessed folder (npz files + manifest) to Kaggle as a PRIVATE dataset.

Needs a Kaggle API token: Kaggle -> Settings -> API -> "Create New Token", which downloads
kaggle.json. Put it at %USERPROFILE%\\.kaggle\\kaggle.json (Windows) or ~/.kaggle/kaggle.json.

  python scripts/kaggle_upload.py --dir C:/Users/me/Downloads/afriseg-africa-npz --user <kaggle-username> \
      --slug afriseg-brats-africa-npz --title "afriseg BraTS-Africa glioma (preprocessed)"

New version of an existing dataset:  add --update "short note".
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--user", required=True, help="your Kaggle username")
ap.add_argument("--slug", default="afriseg-brats-africa-npz")
ap.add_argument("--title", default="afriseg BraTS-Africa glioma (preprocessed)")
ap.add_argument("--update", metavar="NOTE", help="upload a new version instead of creating")
a = ap.parse_args()

d = Path(a.dir)
if not any(d.glob("*_manifest.csv")):
    sys.exit(f"No *_manifest.csv in {d}; is this a preprocessed folder?")
meta = {
    "title": a.title,
    "id": f"{a.user}/{a.slug}",
    "licenses": [{"name": "CC-BY-4.0"}],
    "description": (
        "Brain-cropped float16 .npz volumes (t1n, t1c, t2w, t2f) and harmonised labels for the 95 "
        "BraTS-Africa glioma cases, made with https://github.com/AJmoe/afriseg. Source: TCIA "
        "BraTS-Africa collection, DOI 10.7937/v8h6-8x67, CC BY 4.0. Cite the BraTS-Africa papers."
    ),
}
(d / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
kaggle = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
cmd = [str(kaggle), "datasets"]
if a.update:
    cmd += ["version", "-p", str(d), "-m", a.update, "--dir-mode", "zip"]
else:
    cmd += ["create", "-p", str(d), "--dir-mode", "zip"]  # private unless --public is given
print(" ".join(cmd))
sys.exit(subprocess.call(cmd))
