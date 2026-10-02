"""Paired comparison of two arms evaluated on the same cases.

  python -m afriseg.stats --a results/A1_generic_africa.csv --b results/A2_15t_africa.csv

Per region: mean difference (b - a) with a percentile-bootstrap 95% CI, two-sided Wilcoxon
signed-rank test, matched-pairs rank-biserial effect size, Holm correction across the tested
regions. Several CSVs per arm (e.g. one per fold) can be given; they are concatenated.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .labels import REGIONS


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def bootstrap_ci(d: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rank_biserial(d: np.ndarray) -> float:
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def compare(a: pd.DataFrame, b: pd.DataFrame, metric: str = "dice") -> pd.DataFrame:
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        print(f"WARNING: only {len(m)} cases in common (a={len(a)}, b={len(b)})")
    rows = []
    for r in REGIONS:
        col = f"{metric}_{r}"
        if f"{col}_a" not in m:
            continue
        d = (m[f"{col}_b"] - m[f"{col}_a"]).to_numpy(dtype=float)
        lo, hi = bootstrap_ci(d)
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        rows.append({"region": r, "n": len(d), "mean_a": m[f"{col}_a"].mean(), "mean_b": m[f"{col}_b"].mean(),
                     "diff": d.mean(), "ci_lo": lo, "ci_hi": hi, "p": p, "r_rb": rank_biserial(d)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].tolist())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="baseline arm CSV(s)")
    ap.add_argument("--b", nargs="+", required=True, help="comparison arm CSV(s)")
    ap.add_argument("--metric", default="dice", choices=["dice", "hd95"])
    a = ap.parse_args(argv)
    A = pd.concat([pd.read_csv(f) for f in a.a])
    B = pd.concat([pd.read_csv(f) for f in a.b])
    res = compare(A, B, a.metric)
    print(f"a = {A['arm'].iloc[0]}   b = {B['arm'].iloc[0]}   metric = {a.metric}   diff = b - a")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
