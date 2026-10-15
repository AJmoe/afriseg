"""Pre-registered comparison of two arms (PROTOCOL.md section 6), plus clearly labelled exploratory
tail metrics.

  python scripts/compare_arms.py --a A1_generic --b A2_15t --results results

Reads results/<arm>_africa.csv and results/<arm>_brats_test.csv for both arms.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from afriseg.stats import compare, rank_biserial

REG = ["dice_WT", "dice_TC", "dice_ET"]


def load(results, arm, split):
    d = pd.read_csv(f"{results}/{arm}_{split}.csv")
    d["dice_MEAN"] = d[REG].mean(axis=1)
    return d


def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    return d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)


ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="reference arm, e.g. A1_generic")
ap.add_argument("--b", required=True, help="comparison arm, e.g. A2_15t")
ap.add_argument("--results", default="results")
ap.add_argument("--margin", type=float, default=0.01, help="non-inferiority margin on the source test set")
a = ap.parse_args()

A, B = load(a.results, a.a, "africa"), load(a.results, a.b, "africa")
m = A.merge(B, on="id", suffixes=("_a", "_b"))
assert len(m) == len(A) == len(B), "arms were not evaluated on the same cases"
d = (m.dice_MEAN_b - m.dice_MEAN_a).to_numpy()
bs = boot(d)
print(f"=== {a.b} vs {a.a} on BraTS-Africa (n={len(m)}) ===")
print("PRIMARY (pre-registered): per-case mean Dice over WT/TC/ET, two-sided Wilcoxon signed-rank")
print(f"  mean {a.a} {m.dice_MEAN_a.mean():.4f} | {a.b} {m.dice_MEAN_b.mean():.4f} | diff {d.mean():+.4f} "
      f"95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
print(f"  median diff {np.median(d):+.4f} | {a.b} better in {(d > 0).sum()}, worse in {(d < 0).sum()}")
print(f"  Wilcoxon p = {wilcoxon(d).pvalue:.4f}, rank-biserial r = {rank_biserial(d):+.3f}")

print("\nSECONDARY (pre-registered): per region, Holm-corrected")
for metric in ("dice", "hd95"):
    print(f"  {metric}:")
    print(compare(A, B, metric).round(4).to_string(index=False))

SA, SB = load(a.results, a.a, "brats_test"), load(a.results, a.b, "brats_test")
s = SA.merge(SB, on="id", suffixes=("_a", "_b"))
ds = (s.dice_MEAN_b - s.dice_MEAN_a).to_numpy()
lo = np.percentile(boot(ds), 5)  # one-sided 95% lower bound
print(f"\nSAFETY (pre-registered): non-inferiority on BraTS 2021 test (n={len(s)}), margin {a.margin}")
print(f"  diff {ds.mean():+.4f}, one-sided 95% lower bound {lo:+.4f} -> "
      f"{'NON-INFERIOR' if lo > -a.margin else 'NOT non-inferior'}")

print("\nEXPLORATORY (not pre-registered): lower tail of per-case mean Dice on BraTS-Africa")
rows = []
for name, x in ((a.a, m.dice_MEAN_a), (a.b, m.dice_MEAN_b)):
    p = np.percentile(x, [5, 10, 25, 50, 75])
    rows.append([name, int((x < 0.5).sum()), int((x < 0.7).sum()), *np.round(p, 3)])
print(pd.DataFrame(rows, columns=["arm", "n<0.5", "n<0.7", "p5", "p10", "p25", "median", "p75"]).to_string(index=False))
low = m.dice_MEAN_a < np.percentile(m.dice_MEAN_a, 25)
for name, sel in (("hardest quarter for the reference arm", low), ("other three quarters", ~low)):
    dd = d[sel.to_numpy()]
    print(f"  {name}: n={sel.sum()} mean diff {dd.mean():+.4f}, Wilcoxon p={wilcoxon(dd).pvalue:.4f}")
print("  (subgroups defined by the reference arm's own score: regression to the mean inflates the hard-quarter"
      " gain; treat as descriptive only)")
