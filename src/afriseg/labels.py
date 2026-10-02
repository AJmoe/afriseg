"""Label harmonisation between BraTS label schemes.

BraTS 2021 (GLI):   0 bg, 1 NCR (necrotic/non-enhancing core), 2 ED (oedema), 4 ET
BraTS 2023+/Africa: 0 bg, 1 NETC, 2 SNFH, 3 ET

Both map to one unified scheme: 0 bg, 1 non-enhancing core, 2 oedema/SNFH, 3 ET.
Evaluation regions (same definition in both challenges):
  WT = {1,2,3}, TC = {1,3}, ET = {3}
"""
from __future__ import annotations

import numpy as np

REGIONS = ("WT", "TC", "ET")

SCHEMES = {
    "brats2021": {0: 0, 1: 1, 2: 2, 4: 3},
    "brats2023": {0: 0, 1: 1, 2: 2, 3: 3},
}


def detect_scheme(seg: np.ndarray) -> str:
    vals = set(np.unique(seg).astype(int).tolist())
    if 4 in vals and 3 in vals:
        raise ValueError(f"Label volume mixes both schemes: {sorted(vals)}")
    if 4 in vals:
        return "brats2021"
    # {0,1,2} or {0,1,2,3}: labels 0-2 map identically in both schemes
    return "brats2023"


def harmonize(seg: np.ndarray, scheme: str | None = None) -> np.ndarray:
    seg = np.asarray(seg).astype(np.int16)
    scheme = scheme or detect_scheme(seg)
    mapping = SCHEMES[scheme]
    unknown = set(np.unique(seg).tolist()) - set(mapping)
    if unknown:
        raise ValueError(f"Labels {sorted(unknown)} are not valid for scheme {scheme}")
    out = np.zeros(seg.shape, dtype=np.uint8)
    for src, dst in mapping.items():
        out[seg == src] = dst
    return out


def to_regions(lbl: np.ndarray) -> np.ndarray:
    """Unified label (X,Y,Z) -> binary regions (3,X,Y,Z) ordered WT, TC, ET."""
    wt = lbl > 0
    tc = (lbl == 1) | (lbl == 3)
    et = lbl == 3
    return np.stack([wt, tc, et]).astype(np.uint8)
