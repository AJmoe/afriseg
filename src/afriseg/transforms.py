"""Training / evaluation transforms shared by every experimental arm.

Arms (--aug):
  none              crop + flips only
  generic           + nnU-Net-style intensity augmentation (control)
  15t               + acquisition-physics augmentation (Aug15T) on raw volumes
  15t+generic       both
  15t-no-<comp>     Aug15T without one component (ablation), e.g. 15t-no-motion
  15t-only-<comp>   Aug15T with a single component
"""
from __future__ import annotations

import numpy as np

from .aug15t import COMPONENTS, Aug15T, Aug15TConfig
from .generic import GenericAug
from .labels import to_regions


def brain_mask(img: np.ndarray) -> np.ndarray:
    return (img > 0).any(axis=0)


def zscore(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[0]):
        v = img[c][mask]
        if v.size == 0:
            continue
        m, s = float(v.mean()), float(v.std())
        out[c] = (img[c] - m) / max(s, 1e-6)
        out[c][~mask] = 0.0
    return out


def pad_to(arr: np.ndarray, size, spatial_start: int) -> np.ndarray:
    pads = [(0, 0)] * spatial_start
    for n, s in zip(arr.shape[spatial_start:], size):
        total = max(s - n, 0)
        pads.append((total // 2, total - total // 2))
    return np.pad(arr, pads) if any(p != (0, 0) for p in pads) else arr


def random_crop(img, lbl, size, rng, fg_prob: float = 0.33):
    img = pad_to(img, size, 1)
    lbl = pad_to(lbl, size, 0)
    shape = lbl.shape
    if rng.random() < fg_prob and (lbl > 0).any():
        fg = np.argwhere(lbl > 0)
        centre = fg[rng.integers(len(fg))]
        start = [int(np.clip(c - s // 2, 0, n - s)) for c, s, n in zip(centre, size, shape)]
    else:
        start = [int(rng.integers(0, n - s + 1)) for s, n in zip(size, shape)]
    sl = tuple(slice(a, a + s) for a, s in zip(start, size))
    return img[(slice(None),) + sl], lbl[sl]


def parse_aug(name: str, cfg: Aug15TConfig | None = None):
    """-> (Aug15T | None, GenericAug | None)"""
    cfg = cfg or Aug15TConfig()
    if name == "none":
        return None, None
    if name == "generic":
        return None, GenericAug()
    if name == "15t":
        return Aug15T(cfg), None
    if name == "15t+generic":
        return Aug15T(cfg), GenericAug()
    for prefix, method in (("15t-no-", cfg.without), ("15t-only-", cfg.only)):
        if name.startswith(prefix):
            comp = name[len(prefix):]
            if comp not in COMPONENTS:
                raise ValueError(f"Unknown component in {name}; choose from {COMPONENTS}")
            return Aug15T(method(comp)), None
    raise ValueError(f"Unknown augmentation arm: {name}")


class TrainTransform:
    def __init__(self, aug: str = "none", patch=(128, 128, 128), cfg: Aug15TConfig | None = None,
                 fg_prob: float = 0.33):
        self.physics, self.generic = parse_aug(aug, cfg)
        self.patch = tuple(patch)
        self.fg_prob = fg_prob

    def __call__(self, img: np.ndarray, lbl: np.ndarray, rng: np.random.Generator):
        mask = brain_mask(img)
        if self.physics is not None:
            img, _ = self.physics(img, mask, rng)
        img = zscore(img, mask)
        img, lbl = random_crop(img, lbl, self.patch, rng, self.fg_prob)
        if self.generic is not None:
            img, _ = self.generic(img, rng)
        for ax in range(3):
            if rng.random() < 0.5:
                img = np.flip(img, axis=ax + 1)
                lbl = np.flip(lbl, axis=ax)
        return np.ascontiguousarray(img, dtype=np.float32), np.ascontiguousarray(to_regions(lbl))


def eval_transform(img: np.ndarray) -> np.ndarray:
    return zscore(img.astype(np.float32), brain_mask(img))
