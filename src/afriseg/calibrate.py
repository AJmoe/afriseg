"""Label-free calibration of Aug15T parameter ranges.

Random search over Aug15T configurations. Each candidate is scored by how closely the
image-quality features (afriseg.quality, label-free only) of AUGMENTED source cases match the
features of UNLABELLED target cases: mean standardised 1-D Wasserstein distance over features.
No target labels are used, so the source-only claim stays clean.

  # 1) target feature table (unlabelled images; calibration half only, see PROTOCOL.md)
  python -m afriseg.quality --manifest npz/africa_manifest.csv --out quality/africa.csv
  # 2) search
  python -m afriseg.calibrate --source-manifest npz/brats2021_manifest.csv \
      --target-features quality/africa.csv --target-ids splits/africa_calib_ids.json \
      --n-candidates 40 --n-source 30 --out configs/aug15t_calibrated.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .aug15t import Aug15T, Aug15TConfig
from .data import read_manifest
from .dataset import load_npz
from .quality import case_features


def sample_config(rng: np.random.Generator) -> Aug15TConfig:
    def rng_pair(lo, hi, min_width):
        a = rng.uniform(lo, hi - min_width)
        return (round(a, 3), round(rng.uniform(a + min_width, hi), 3))

    return replace(
        Aug15TConfig(),
        p_bias=round(rng.uniform(0, 1), 2), bias_coeff=rng_pair(0.0, 0.6, 0.05),
        p_motion=round(rng.uniform(0, 0.6), 2), motion_frac=rng_pair(0.01, 0.2, 0.02),
        motion_shift=rng_pair(0.5, 6.0, 0.5),
        p_resolution=round(rng.uniform(0, 1), 2), kspace_keep=rng_pair(0.3, 1.0, 0.1),
        p_thickness=round(rng.uniform(0, 1), 2), slice_mm=rng_pair(1.5, 7.0, 0.5),
        p_noise=round(rng.uniform(0, 1), 2), snr=rng_pair(4.0, 50.0, 4.0),
    )


def feature_distance(src: pd.DataFrame, tgt: pd.DataFrame) -> float:
    cols = [c for c in tgt.columns if c in src.columns and c not in ("id", "rep", "enhancement")]
    d = []
    for c in cols:
        sd = tgt[c].std() or 1.0
        d.append(wasserstein_distance(src[c] / sd, tgt[c] / sd))
    return float(np.mean(d))


def score(cfg: Aug15TConfig | None, cases, tgt, seed: int, repeats: int) -> float:
    rng = np.random.default_rng(seed)
    aug = Aug15T(cfg) if cfg is not None else None
    recs = []
    for img in cases:
        mask = (img > 0).any(axis=0)
        for _ in range(repeats if aug else 1):
            x = aug(img, mask, rng)[0] if aug else img
            recs.append(case_features(x))
    return feature_distance(pd.DataFrame(recs), tgt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-manifest", required=True)
    ap.add_argument("--target-features", required=True, help="CSV from afriseg.quality on target images")
    ap.add_argument("--target-ids", help="JSON list: restrict target rows to this calibration subset")
    ap.add_argument("--n-candidates", type=int, default=40)
    ap.add_argument("--n-source", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    tgt = pd.read_csv(a.target_features)
    if a.target_ids:
        keep = set(json.loads(Path(a.target_ids).read_text()))
        tgt = tgt[tgt["id"].isin(keep)]
    rows = read_manifest(a.source_manifest)
    rows = [rows[i] for i in np.random.default_rng(a.seed).permutation(len(rows))[: a.n_source]]
    cases = [load_npz(r["path"])[0] for r in rows]

    rng = np.random.default_rng(a.seed)
    results = []
    base = score(None, cases, tgt, a.seed, 1)
    prior = score(Aug15TConfig(), cases, tgt, a.seed, a.repeats)
    print(f"no augmentation: {base:.4f}   default prior: {prior:.4f}")
    results.append({"name": "none", "distance": base})
    results.append({"name": "prior", "distance": prior, "config": Aug15TConfig().__dict__})
    best_cfg, best = Aug15TConfig(), prior
    for i in range(a.n_candidates):
        cfg = sample_config(rng)
        d = score(cfg, cases, tgt, a.seed, a.repeats)
        results.append({"name": f"cand{i}", "distance": d, "config": cfg.__dict__})
        flag = ""
        if d < best:
            best, best_cfg, flag = d, cfg, "  <- best"
        print(f"[{i + 1}/{a.n_candidates}] distance={d:.4f}{flag}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    best_cfg.to_json(a.out)
    Path(a.out).with_suffix(".search.json").write_text(json.dumps(results, indent=1, default=list))
    print(f"best distance {best:.4f} (no-aug {base:.4f}, prior {prior:.4f}) -> {a.out}")


if __name__ == "__main__":
    main()
