"""Evaluate a checkpoint on a set of cases; writes one CSV row per case.

  python -m afriseg.evaluate --ckpt runs/A2_15t/best.pt --manifest npz/africa_manifest.csv \
      --out results/A2_15t_africa_all.csv                      # all 95 African cases (source-only)
  python -m afriseg.evaluate --ckpt runs/B_15t_k10_f0/best.pt --manifest npz/africa_manifest.csv \
      --split splits/africa.json --subset 0 --out results/B_15t_k10_f0.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import load_npz, select_rows
from .infer import predict_probs, to_binary_regions
from .labels import REGIONS, to_regions
from .metrics import region_scores
from .model import build_model
from .transforms import eval_transform


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split")
    ap.add_argument("--subset", default="test")
    ap.add_argument("--tta", action="store_true", help="8-fold flip test-time augmentation")
    ap.add_argument("--no-hd", action="store_true")
    ap.add_argument("--arm", help="label written into the CSV (defaults to run folder name)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    st = torch.load(a.ckpt, map_location=device, weights_only=False)
    targs = st["args"]
    model = build_model(targs["model"], small=targs.get("small", False)).to(device)
    model.load_state_dict(st["model"])
    rows = select_rows(a.manifest, a.split, a.subset) if a.split else select_rows(a.manifest, None, "")
    arm = a.arm or Path(a.ckpt).parent.name
    roi = tuple(targs["patch"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["arm", "id", "dataset"] + [f"dice_{r}" for r in REGIONS] + \
             ([] if a.no_hd else [f"hd95_{r}" for r in REGIONS]) + ["sec"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            t = time.time()
            img, lbl = load_npz(r["path"])
            probs = predict_probs(model, eval_transform(img), roi, device,
                                  amp=device.type == "cuda", flip_tta=a.tta)
            s = region_scores(to_binary_regions(probs), to_regions(lbl), with_hd=not a.no_hd)
            w.writerow({"arm": arm, "id": r["id"], "dataset": r["dataset"], **s,
                        "sec": round(time.time() - t, 1)})
            fh.flush()
            print(f"[{i + 1}/{len(rows)}] {r['id']} " + " ".join(f"{k}={v:.3f}" for k, v in s.items()))
    import pandas as pd
    df = pd.read_csv(a.out)
    print(df[[c for c in df.columns if c.startswith(("dice", "hd95"))]].describe().loc[["mean", "50%", "std"]].round(3))


if __name__ == "__main__":
    main()
