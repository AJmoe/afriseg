"""PyTorch datasets over the preprocessed .npz cases and split selection."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import load_split, read_manifest


def load_npz(path: str):
    with np.load(path) as z:
        img = z["image"].astype(np.float32)
        lbl = z["label"] if "label" in z.files else None
    return img, lbl


def select_rows(manifest: str, split_path: str | None, subset: str, exclude: bool = False,
                n_labelled: int | None = None, seed: int = 2026) -> list[dict]:
    """Rows of `manifest` whose split value equals `subset` (holdout: train/val/test;
    k-fold: fold index as string). With exclude=True, every row EXCEPT that subset.
    n_labelled: deterministic subsample (label-efficiency experiments)."""
    rows = [r for r in read_manifest(manifest) if r["has_label"] == "1"]
    if split_path:
        split = load_split(split_path)
        rows = [r for r in rows if r["id"] in split and ((str(split[r["id"]]) == subset) != exclude)]
    if n_labelled is not None:
        if n_labelled > len(rows):
            raise ValueError(f"Asked for {n_labelled} labelled cases, only {len(rows)} available")
        order = np.random.default_rng(seed).permutation(sorted(r["id"] for r in rows))
        keep = set(order[:n_labelled].tolist())
        rows = [r for r in rows if r["id"] in keep]
    return rows


class PatchDataset(Dataset):
    """Each item = one random augmented patch from a random case."""

    def __init__(self, rows: list[dict], transform, samples_per_epoch: int):
        if not rows:
            raise ValueError("No training cases selected")
        self.rows, self.transform, self.n = rows, transform, samples_per_epoch

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))
        row = self.rows[int(rng.integers(len(self.rows)))]
        img, lbl = load_npz(row["path"])
        x, y = self.transform(img, lbl, rng)
        return {"image": torch.from_numpy(x), "label": torch.from_numpy(y)}
