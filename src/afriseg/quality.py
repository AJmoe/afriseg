"""Label-free image-quality features, used to (1) quantify how BraTS 2021 and BraTS-Africa
differ and (2) calibrate Aug15T so augmented source images match the target distribution.

Features per sequence (computed on raw, brain-cropped volumes):
  noise       robust high-pass residual sigma / median brain intensity   (~ 1 / SNR)
  hf_inplane  share of in-plane spectral energy above 0.25 cycles/voxel  (in-plane resolution)
  slice_ratio high-freq energy share along slices / same in-plane        (slice thickness)
  bias        std of log heavily-smoothed intensity inside eroded brain   (field inhomogeneity)
Plus, only when a label is given (calibration on TRAINING folds only):
  enhancement median T1c/T1n ratio in ET relative to normal brain          (contrast uptake)

  python -m afriseg.quality --manifest npz/africa_manifest.csv --out quality/africa.csv
  python -m afriseg.quality --manifest npz/brats2021_manifest.csv --limit 100 \
      --augment 15t --aug-config configs/aug15t.json --out quality/brats_aug.csv
  python -m afriseg.quality --compare quality/brats_aug.csv quality/africa.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import fft as sfft
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.stats import wasserstein_distance

from .data import MODALITIES, read_manifest

FEATURES = ("noise", "hf_inplane", "slice_ratio", "bias")


def hf_share_1d(vol: np.ndarray, axis: int, cutoff: float = 0.25) -> float:
    """Share of spectral energy above `cutoff` cycles/voxel along one axis (DC excluded)."""
    P = np.abs(sfft.rfft(vol, axis=axis, workers=-1)) ** 2
    k = sfft.rfftfreq(vol.shape[axis])
    shp = [1, 1, 1]; shp[axis] = -1
    k = np.broadcast_to(k.reshape(shp), P.shape)
    tot = P[k > 0].sum()
    return float(P[k > cutoff].sum() / tot) if tot > 0 else 0.0


def volume_features(vol: np.ndarray, mask: np.ndarray, inplane=(0, 1)) -> dict:
    core = binary_erosion(mask, iterations=3)
    if core.sum() < 100:
        core = mask
    med = float(np.median(vol[core])) or 1.0
    resid = vol - gaussian_filter(vol, 1.0)
    r = resid[core]
    noise = 1.4826 * float(np.median(np.abs(r - np.median(r)))) / med

    P = np.abs(sfft.fft2(vol, axes=inplane, workers=-1)) ** 2
    k0 = sfft.fftfreq(vol.shape[inplane[0]])
    k1 = sfft.fftfreq(vol.shape[inplane[1]])
    shp0 = [1, 1, 1]; shp0[inplane[0]] = -1
    shp1 = [1, 1, 1]; shp1[inplane[1]] = -1
    rad = np.sqrt(k0.reshape(shp0) ** 2 + k1.reshape(shp1) ** 2)
    rad = np.broadcast_to(rad, P.shape)
    tot = P.sum() - P[rad == 0].sum()
    hf = float(P[rad > 0.25].sum() / tot) if tot > 0 else 0.0

    slice_axis = 3 - sum(inplane)
    inpl = np.mean([hf_share_1d(vol, ax) for ax in inplane])
    slice_ratio = hf_share_1d(vol, slice_axis) / inpl if inpl > 0 else 0.0

    sm = gaussian_filter(vol, 8.0)
    bias = float(np.std(np.log(np.clip(sm[core], 1e-3, None))))
    return {"noise": noise, "hf_inplane": hf, "slice_ratio": slice_ratio, "bias": bias}


def case_features(img: np.ndarray, lbl: np.ndarray | None = None) -> dict:
    mask = (img > 0).any(axis=0)
    out = {}
    for c, m in enumerate(MODALITIES):
        for k, v in volume_features(img[c], mask).items():
            out[f"{m}_{k}"] = v
    if lbl is not None and (lbl == 3).sum() > 20:
        normal = mask & (lbl == 0)
        t1n, t1c = img[0], img[1]
        def ratio(sel):
            return float(np.median(t1c[sel]) / max(np.median(t1n[sel]), 1e-6))
        out["enhancement"] = ratio(lbl == 3) / max(ratio(normal), 1e-6)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--augment", help="apply an augmentation arm first (e.g. 15t) to measure its effect")
    ap.add_argument("--aug-config")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--use-labels", action="store_true", help="also compute ET enhancement (training folds only!)")
    ap.add_argument("--compare", nargs=2, metavar=("SOURCE_CSV", "TARGET_CSV"))
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if a.compare:
        s, t = pd.read_csv(a.compare[0]), pd.read_csv(a.compare[1])
        cols = [c for c in t.columns if c in s.columns and c not in ("id", "rep")]
        rows = []
        for c in cols:
            sd = t[c].std() or 1.0
            rows.append({"feature": c, "source_median": s[c].median(), "target_median": t[c].median(),
                         "wasserstein_std": wasserstein_distance(s[c] / sd, t[c] / sd)})
        df = pd.DataFrame(rows)
        print(df.round(4).to_string(index=False))
        print(f"mean standardised Wasserstein distance: {df['wasserstein_std'].mean():.4f}")
        return

    from .dataset import load_npz
    from .aug15t import Aug15TConfig
    from .transforms import parse_aug

    physics = None
    if a.augment:
        cfg = Aug15TConfig.from_json(a.aug_config) if a.aug_config else Aug15TConfig()
        physics, generic = parse_aug(a.augment, cfg)
        if physics is None:
            raise SystemExit("--augment must be an arm containing the physics augmentation (15t...)")
    rows = [r for r in read_manifest(a.manifest)][: a.limit]
    rng = np.random.default_rng(a.seed)
    recs = []
    for i, r in enumerate(rows):
        img, lbl = load_npz(r["path"])
        for rep in range(a.repeats if physics else 1):
            x = img
            if physics is not None:
                x, _ = physics(img, (img > 0).any(axis=0), rng)
            f = case_features(x, lbl if a.use_labels else None)
            recs.append({"id": r["id"], "rep": rep, **f})
        print(f"[{i + 1}/{len(rows)}] {r['id']}")
    pd.DataFrame(recs).to_csv(a.out, index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
