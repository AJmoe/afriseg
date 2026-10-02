"""Synthetic 4-sequence brain-with-glioma phantom for tests and augmentation previews.

Not anatomy: just enough structure (tissue classes, an enhancing rim, oedema, texture) to
check that transforms behave and to visualise each degradation without real data.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

# rough relative intensities per tissue: (t1n, t1c, t2w, t2f)
TISSUE = {
    "csf": (0.25, 0.25, 1.00, 0.15),
    "gm": (0.60, 0.62, 0.70, 0.65),
    "wm": (0.85, 0.86, 0.50, 0.55),
    "necrosis": (0.35, 0.38, 0.95, 0.60),
    "et": (0.60, 1.40, 0.70, 0.80),
    "oedema": (0.55, 0.57, 0.95, 1.00),
}


def make_phantom(shape=(96, 112, 80), seed: int = 0):
    """-> img (4,X,Y,Z) float32 raw, lbl (X,Y,Z) uint8 in the unified scheme."""
    rng = np.random.default_rng(seed)
    X, Y, Z = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    r = np.sqrt((X / 0.85) ** 2 + (Y / 0.9) ** 2 + (Z / 0.8) ** 2)
    brain = r < 1.0
    cortex = brain & (r > 0.78)
    ventricles = (np.sqrt((X / 0.12) ** 2 + (Y / 0.3) ** 2 + (Z / 0.25) ** 2) < 1.0)
    tissue = np.full(shape, "", dtype=object)
    tissue[brain] = "wm"
    tissue[cortex] = "gm"
    tissue[ventricles & brain] = "csf"

    cx, cy, cz = rng.uniform(-0.35, 0.35, 3)
    d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    lbl = np.zeros(shape, dtype=np.uint8)
    lbl[(d < 0.35) & brain] = 2   # oedema
    lbl[(d < 0.20) & brain] = 3   # enhancing rim
    lbl[(d < 0.12) & brain] = 1   # necrotic core
    tissue[lbl == 2] = "oedema"
    tissue[lbl == 3] = "et"
    tissue[lbl == 1] = "necrosis"

    img = np.zeros((4,) + shape, dtype=np.float32)
    for name, vals in TISSUE.items():
        sel = tissue == name
        for c in range(4):
            img[c][sel] = vals[c]
    tex = gaussian_filter(rng.standard_normal(shape), 1.5).astype(np.float32)
    tex /= tex.std() + 1e-6
    for c in range(4):
        img[c] = gaussian_filter(img[c], 0.7) * (1 + 0.05 * tex)
        img[c] += np.abs(rng.normal(0, 0.01, shape)).astype(np.float32)
        img[c] *= brain
    return (img * 1000).astype(np.float32), lbl
