"""Exploratory (post hoc) analysis of per-case African results for one arm.

Joins per-case Dice with ground-truth label composition and label-free image-quality features,
then compares TC failures (Dice_TC < 0.5) against the rest and correlates every feature with the
per-case mean Dice. Labelled exploratory in the paper: none of these tests were pre-registered.

  python scripts/analyse_failures.py --results results/A1_generic_africa.csv \
      --manifest <npz>/africa_manifest.csv --quality quality/africa.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from afriseg.data import read_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--quality", required=True)
ap.add_argument("--out")
a = ap.parse_args()

af = pd.read_csv(a.results)
af["mean"] = af[["dice_WT", "dice_TC", "dice_ET"]].mean(axis=1)
paths = {r["id"]: r["path"] for r in read_manifest(a.manifest)}
comp = []
for cid in af.id:
    with np.load(paths[cid]) as z:
        lbl = z["label"]
    netc, snfh, et = (int((lbl == k).sum()) for k in (1, 2, 3))
    comp.append({"id": cid, "netc": netc, "snfh": snfh, "et": et, "tc_vol": netc + et,
                 "wt_vol": netc + snfh + et})
af = af.merge(pd.DataFrame(comp), on="id")
af["netc_frac_tc"] = af.netc / af.tc_vol.clip(lower=1)
af["tc_frac_wt"] = af.tc_vol / af.wt_vol.clip(lower=1)
q = pd.read_csv(a.quality).drop(columns=["rep"], errors="ignore")
af = af.merge(q, on="id")

fail = af.dice_TC < 0.5
cols = ["netc", "et", "snfh", "tc_vol", "wt_vol", "netc_frac_tc", "tc_frac_wt"] + \
       [c for c in q.columns if c != "id"]
print(f"TC failures: {fail.sum()} of {len(af)}")
print(af.loc[fail, ["id", "dice_WT", "dice_TC", "dice_ET", "netc", "et", "snfh", "tc_frac_wt"]]
      .round(3).to_string(index=False))
rows = [(c, af.loc[fail, c].median(), af.loc[~fail, c].median(),
         mannwhitneyu(af.loc[fail, c], af.loc[~fail, c]).pvalue) for c in cols]
print(pd.DataFrame(rows, columns=["feature", "fail_median", "rest_median", "p_MWU"]).round(4).to_string(index=False))
rows = [(c,) + tuple(spearmanr(af[c], af["mean"])) for c in cols]
print(pd.DataFrame(rows, columns=["feature", "rho", "p"]).sort_values("p").round(3).to_string(index=False))
if a.out:
    af.to_csv(a.out, index=False)
