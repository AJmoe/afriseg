"""Render each Aug15T component (forced on) on one case, as a PNG grid of mid-axial slices.

  python scripts/preview_aug.py --out preview.png                    # synthetic phantom
  python scripts/preview_aug.py --npz npz/brats2021/BraTS2021_00000.npz --out preview.png
"""
import argparse
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from afriseg.aug15t import COMPONENTS, Aug15T, Aug15TConfig
from afriseg.data import MODALITIES

ap = argparse.ArgumentParser()
ap.add_argument("--npz")
ap.add_argument("--out", default="aug_preview.png")
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()

if a.npz:
    with np.load(a.npz) as z:
        img = z["image"].astype(np.float32)
else:
    from afriseg.phantom import make_phantom
    img, _ = make_phantom()
mask = (img > 0).any(axis=0)
z = img.shape[3] // 2

base = Aug15TConfig()
forced = replace(base, p_contrast=1, p_bias=1, p_motion=1, p_resolution=1, p_thickness=1, p_noise=1,
                 p_slice_axis_is_z=1.0)
variants = [("original", None)] + [(c, forced.only(c)) for c in COMPONENTS] + [("all", forced)]

fig, axes = plt.subplots(len(MODALITIES), len(variants), figsize=(2.1 * len(variants), 2.2 * len(MODALITIES)))
for j, (name, cfg) in enumerate(variants):
    x = img if cfg is None else Aug15T(cfg)(img, mask, np.random.default_rng(a.seed))[0]
    for i, m in enumerate(MODALITIES):
        sl = x[i, :, :, z].T
        vmax = np.percentile(img[i][mask], 99.5)
        axes[i, j].imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=vmax)
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        if i == 0:
            axes[i, j].set_title(name, fontsize=9)
        if j == 0:
            axes[i, j].set_ylabel(m, fontsize=9)
plt.tight_layout()
plt.savefig(a.out, dpi=110)
print("wrote", a.out)
