"""Generic intensity augmentation, modelled on nnU-Net's default pipeline.

This is the CONTROL arm. It already contains noise, blur and low-resolution simulation, so any
gain from Aug15T has to come from acquisition realism and target calibration, not from merely
"more augmentation". Operates on z-score-normalised patches (C, X, Y, Z).

nnU-Net defaults reproduced (probabilities per sample / per channel as in nnU-Net v2):
  gaussian noise  p=0.10  variance U(0, 0.1)
  gaussian blur   p=0.20  sigma U(0.5, 1.0), per channel p=0.5
  brightness mult p=0.15  U(0.75, 1.25)
  contrast        p=0.15  U(0.75, 1.25), range preserved
  low resolution  p=0.25  zoom U(0.5, 1), per channel p=0.5
  gamma           p=0.30  U(0.7, 1.5), retain stats; inverted gamma p=0.1
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def _fit(vol: np.ndarray, shape) -> np.ndarray:
    out = np.zeros(shape, dtype=vol.dtype)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(vol.shape, shape))
    out[sl] = vol[sl]
    return out


def simulate_lowres(vol: np.ndarray, factor: float) -> np.ndarray:
    small = zoom(vol, factor, order=0)
    up = zoom(small, [s / t for s, t in zip(vol.shape, small.shape)], order=3)
    return _fit(up, vol.shape)


def _gamma(ch: np.ndarray, g: float, invert: bool) -> np.ndarray:
    if invert:
        ch = -ch
    mn, mx = ch.min(), ch.max()
    rng_ = mx - mn
    if rng_ < 1e-8:
        return -ch if invert else ch
    m, s = ch.mean(), ch.std()
    out = np.power((ch - mn) / rng_, g) * rng_ + mn
    out = (out - out.mean()) / max(out.std(), 1e-8) * s + m
    return -out if invert else out


class GenericAug:
    def __call__(self, img: np.ndarray, rng: np.random.Generator):
        img = img.astype(np.float32, copy=True)
        applied = []
        if rng.random() < 0.10:
            img += rng.normal(0, np.sqrt(rng.uniform(0, 0.1)), img.shape).astype(np.float32)
            applied.append("noise")
        if rng.random() < 0.20:
            for c in range(img.shape[0]):
                if rng.random() < 0.5:
                    img[c] = gaussian_filter(img[c], rng.uniform(0.5, 1.0))
            applied.append("blur")
        if rng.random() < 0.15:
            img *= np.float32(rng.uniform(0.75, 1.25))
            applied.append("brightness")
        if rng.random() < 0.15:
            for c in range(img.shape[0]):
                m, lo, hi = img[c].mean(), img[c].min(), img[c].max()
                img[c] = np.clip((img[c] - m) * rng.uniform(0.75, 1.25) + m, lo, hi)
            applied.append("contrast")
        if rng.random() < 0.25:
            for c in range(img.shape[0]):
                if rng.random() < 0.5:
                    img[c] = simulate_lowres(img[c], rng.uniform(0.5, 1.0))
            applied.append("lowres")
        if rng.random() < 0.10:
            for c in range(img.shape[0]):
                img[c] = _gamma(img[c], rng.uniform(0.7, 1.5), invert=True)
            applied.append("gamma_inv")
        if rng.random() < 0.30:
            for c in range(img.shape[0]):
                img[c] = _gamma(img[c], rng.uniform(0.7, 1.5), invert=False)
            applied.append("gamma")
        return img.astype(np.float32), applied
