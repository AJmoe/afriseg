"""Case discovery, preprocessing to compact .npz files, and patient-level splits.

Usage:
  python -m afriseg.data preprocess --root /data/BraTS2021 --out /work/npz --dataset brats2021
  python -m afriseg.data split --manifest /work/npz/brats2021_manifest.csv --mode holdout --out splits/brats2021.json
  python -m afriseg.data split --manifest /work/npz/africa_manifest.csv  --mode kfold --k 5 --out splits/africa.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from .labels import detect_scheme, harmonize

MODALITIES = ("t1n", "t1c", "t2w", "t2f")  # channel order used everywhere
ALIASES = {
    "t1": "t1n", "t1n": "t1n",
    "t1ce": "t1c", "t1c": "t1c", "t1gd": "t1c",
    "t2": "t2w", "t2w": "t2w",
    "flair": "t2f", "t2f": "t2f",
    "seg": "seg",
}


def strip_nii(name: str) -> str:
    for suf in (".nii.gz", ".nii"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def modality_of(path: Path) -> str | None:
    if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        return None
    token = re.split(r"[-_]", strip_nii(path.name))[-1].lower()
    return ALIASES.get(token)


def discover(root: str | Path) -> list[dict]:
    """Find case folders that contain all four modalities (+ optional seg)."""
    groups: dict[Path, dict[str, Path]] = {}
    for p in Path(root).rglob("*.nii*"):
        m = modality_of(p)
        if m:
            groups.setdefault(p.parent, {})[m] = p
    cases, seen, dupes = [], {}, []
    for folder, files in sorted(groups.items()):
        if all(m in files for m in MODALITIES):
            if folder.name in seen:  # same case id in two places (e.g. data extracted twice)
                dupes.append((folder.name, str(folder), seen[folder.name]))
                continue
            seen[folder.name] = str(folder)
            cases.append({
                "id": folder.name,
                "images": [str(files[m]) for m in MODALITIES],
                "label": str(files["seg"]) if "seg" in files else None,
            })
    if dupes:
        print(f"WARNING skipped {len(dupes)} duplicate case folders, e.g. {dupes[0][1]} "
              f"(already have {dupes[0][2]})", flush=True)
    return cases


def brain_bbox(img: np.ndarray, margin: int = 2) -> tuple[slice, ...]:
    mask = (img > 0).any(axis=0)
    if not mask.any():
        raise ValueError("Empty volume")
    idx = np.nonzero(mask)
    return tuple(
        slice(max(int(i.min()) - margin, 0), min(int(i.max()) + margin + 1, n))
        for i, n in zip(idx, mask.shape)
    )


def load_case(case: dict) -> tuple[np.ndarray, np.ndarray | None, dict]:
    import nibabel as nib

    vols, spacing = [], None
    for path in case["images"]:
        im = nib.load(path)
        v = np.squeeze(np.asanyarray(im.dataobj)).astype(np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v[v < 0] = 0.0
        vols.append(v)
        spacing = tuple(float(s) for s in im.header.get_zooms()[:3])
    shapes = {v.shape for v in vols}
    if len(shapes) != 1:
        raise ValueError(f"{case['id']}: modality shapes differ {shapes}")
    img = np.stack(vols)
    lbl, scheme = None, None
    if case.get("label"):
        seg = np.squeeze(np.asanyarray(nib.load(case["label"]).dataobj))
        scheme = detect_scheme(seg)
        lbl = harmonize(seg, scheme)
        if lbl.shape != img.shape[1:]:
            raise ValueError(f"{case['id']}: label shape {lbl.shape} != image {img.shape[1:]}")
    return img, lbl, {"spacing": spacing, "scheme": scheme, "shape": img.shape[1:]}


def read_id_list(path: str) -> set[str]:
    """Case ids from a .json list or a text file with one id per line."""
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        return {str(x) for x in json.loads(text)}
    return {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")}


def _process_one(job) -> list | None:
    try:
        return _process_one_unsafe(job)
    except Exception as e:  # one damaged case must not kill a 1-hour preprocessing run
        print(f"WARNING skipped {job[0]['id']}: {type(e).__name__}: {e}", flush=True)
        return None


def _process_one_unsafe(job) -> list:
    case, out_dir, dataset = job
    out = Path(out_dir)
    img, lbl, meta = load_case(case)
    if meta["spacing"] and any(abs(s - 1.0) > 0.05 for s in meta["spacing"]):
        print(f"WARNING {case['id']}: spacing {meta['spacing']} is not 1 mm isotropic", flush=True)
    bb = brain_bbox(img)
    img_c = img[(slice(None),) + bb]
    if img_c.max() > 65000:  # float16 overflow guard
        img_c = img_c * (65000.0 / img_c.max())
    img_c = img_c.astype(np.float16)
    path = out / dataset / f"{case['id']}.npz"
    if lbl is not None:
        np.savez_compressed(path, image=img_c, label=lbl[bb])
    else:
        np.savez_compressed(path, image=img_c)
    return [case["id"], dataset, f"{dataset}/{path.name}", int(lbl is not None), meta["scheme"],
            "x".join(f"{s:.2f}" for s in meta["spacing"]), "x".join(map(str, meta["shape"]))]


def _collect(results, n: int) -> list:
    rows, skipped = [], 0
    for i, row in enumerate(results):
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        print(f"[{i + 1}/{n}] {row[0]}", flush=True)
    if skipped:
        print(f"WARNING {skipped} of {n} cases skipped (see messages above)", flush=True)
    return rows


def preprocess(root: str, out_dir: str, dataset: str, limit: int | None = None,
               ids: set[str] | None = None, workers: int = 1) -> Path:
    """Crop every case to its brain bounding box; save float16 .npz plus a manifest CSV.

    Intensities stay RAW (not normalised) so physics-based augmentation can act on
    magnitude images. Normalisation happens on the fly in the training transform.
    """
    out = Path(out_dir)
    (out / dataset).mkdir(parents=True, exist_ok=True)
    cases = discover(root)
    if ids is not None:
        missing = ids - {c["id"] for c in cases}
        if missing:
            print(f"WARNING {len(missing)} listed ids not found, e.g. {sorted(missing)[:3]}")
        cases = [c for c in cases if c["id"] in ids]
    if limit:
        cases = cases[:limit]
    if not cases:
        raise SystemExit(f"No complete cases found under {root}")
    manifest = out / f"{dataset}_manifest.csv"
    jobs = [(case, str(out), dataset) for case in cases]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            it = pool.imap(_process_one, jobs)
            rows = _collect(it, len(cases))
    else:
        rows = _collect(map(_process_one, jobs), len(cases))
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "dataset", "path", "has_label", "scheme", "spacing", "shape"])
        w.writerows(rows)
    return manifest


def read_manifest(path: str) -> list[dict]:
    """Rows of a manifest. Relative `path` entries are resolved against the manifest's folder,
    so a preprocessed folder can be moved (e.g. uploaded to Kaggle) without rewriting it."""
    base = Path(path).resolve().parent
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        p = Path(r["path"])
        if not p.is_absolute():
            r["path"] = str(base / p)
    return rows


def make_folds(ids: list[str], k: int, seed: int = 2026) -> dict[str, int]:
    order = np.random.default_rng(seed).permutation(sorted(ids))
    return {str(cid): int(i % k) for i, cid in enumerate(order)}


def make_holdout(ids: list[str], fractions=(0.8, 0.1, 0.1), seed: int = 2026) -> dict[str, str]:
    order = list(np.random.default_rng(seed).permutation(sorted(ids)))
    n_tr = int(round(fractions[0] * len(order)))
    n_va = int(round(fractions[1] * len(order)))
    return {str(c): "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
            for i, c in enumerate(order)}


def load_split(path: str) -> dict:
    return json.loads(Path(path).read_text())["split"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess BraTS-style NIfTI folders and make splits")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preprocess")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", required=True, help="e.g. brats2021 or africa")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=1, help="parallel processes (Kaggle CPU: 4)")
    p.add_argument("--ids", help="only these case ids (.json list or one id per line)")
    s = sub.add_parser("split")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--mode", choices=["holdout", "kfold"], required=True)
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    if a.cmd == "preprocess":
        ids = read_id_list(a.ids) if a.ids else None
        print("Manifest:", preprocess(a.root, a.out, a.dataset, a.limit, ids, a.workers))
    else:
        ids = [r["id"] for r in read_manifest(a.manifest) if r["has_label"] == "1"]
        split = make_holdout(ids, seed=a.seed) if a.mode == "holdout" else make_folds(ids, a.k, a.seed)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"mode": a.mode, "seed": a.seed, "split": split}, indent=1))
        print(f"Wrote {a.out} ({len(split)} cases)")


if __name__ == "__main__":
    main()
