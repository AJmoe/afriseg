"""Acquisition-physics augmentation: degrade high-quality (mostly 3T) BraTS scans so they
look like low-field / resource-constrained acquisitions, as seen in BraTS-Africa.

Operates on RAW magnitude volumes (C, X, Y, Z) with channel order t1n, t1c, t2w, t2f,
BEFORE intensity normalisation. Components are applied in the order they arise physically:

  contrast    reduced gadolinium enhancement (dose/timing)          tissue
  bias        smooth multiplicative coil-sensitivity field          receive coil
  motion      rigid in-plane translation during a subset of
              phase-encode lines (ghosting / blurring)              acquisition
  resolution  in-plane k-space truncation (smaller matrix, Gibbs)   acquisition
  thickness   thick slices: slab averaging + resample to 1 mm       acquisition
  noise       Rician noise at a sampled SNR (SNR ~ B0)              thermal noise

Every parameter range in Aug15TConfig is a PRIOR. The protocol calibrates the ranges against
image-quality statistics measured on unlabelled BraTS-Africa training images (afriseg.quality),
so the defaults here are placeholders until that calibration has been run.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import uniform_filter1d

COMPONENTS = ("contrast", "bias", "motion", "resolution", "thickness", "noise")


@dataclass
class Aug15TConfig:
    # reduced contrast enhancement on T1c: t1c' = t1n_scaled + alpha * (t1c - t1n_scaled)
    p_contrast: float = 0.3
    enhance_alpha: tuple = (0.3, 0.9)
    # bias field: exp(polynomial), coefficients ~ U(-c, c), c ~ U(range)
    p_bias: float = 0.5
    bias_coeff: tuple = (0.05, 0.35)
    bias_order: int = 3
    # motion: fraction of phase-encode lines acquired after a shift of `motion_shift` voxels
    p_motion: float = 0.25
    motion_frac: tuple = (0.02, 0.12)
    motion_shift: tuple = (1.0, 4.0)
    # resolution: fraction of in-plane Nyquist radius kept
    p_resolution: float = 0.5
    kspace_keep: tuple = (0.45, 0.85)
    # slice thickness in voxels (1 mm grid)
    p_thickness: float = 0.4
    slice_mm: tuple = (3.0, 6.0)
    p_slice_axis_is_z: float = 0.7  # else sagittal/coronal acquisition
    # Rician noise: sigma = mean(brain) / SNR
    p_noise: float = 0.7
    snr: tuple = (8.0, 30.0)
    # acquisition geometry on the BraTS grid (axial plane = axes X,Y; slices along Z)
    inplane_axes: tuple = (0, 1)
    enabled: tuple = field(default_factory=lambda: COMPONENTS)

    def without(self, component: str) -> "Aug15TConfig":
        if component not in COMPONENTS:
            raise ValueError(f"Unknown component {component}")
        return replace(self, enabled=tuple(c for c in self.enabled if c != component))

    def only(self, component: str) -> "Aug15TConfig":
        if component not in COMPONENTS:
            raise ValueError(f"Unknown component {component}")
        return replace(self, enabled=(component,))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def from_json(cls, path: str | Path) -> "Aug15TConfig":
        d = json.loads(Path(path).read_text())
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def _u(rng: np.random.Generator, lo_hi) -> float:
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


# ---------------------------------------------------------------- components (single volume)

def reduce_enhancement(t1n: np.ndarray, t1c: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend T1c toward an intensity-matched T1n, shrinking contrast uptake by factor alpha."""
    p_n = np.percentile(t1n[mask], 99) if mask.any() else 1.0
    p_c = np.percentile(t1c[mask], 99) if mask.any() else 1.0
    base = t1n * (p_c / max(p_n, 1e-6))
    return np.clip(base + alpha * (t1c - base), 0, None).astype(np.float32)


def bias_field(shape, order: int, coeff: float, rng: np.random.Generator) -> np.ndarray:
    axes = [np.linspace(-1, 1, n, dtype=np.float32) for n in shape]
    x, y, z = axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :]
    log_f = np.zeros(shape, dtype=np.float32)
    for i in range(order + 1):
        for j in range(order + 1 - i):
            for k in range(order + 1 - i - j):
                if i == j == k == 0:
                    continue
                log_f += np.float32(rng.uniform(-coeff, coeff)) * (x ** i) * (y ** j) * (z ** k)
    return np.exp(log_f)


def motion_ghosting(vol: np.ndarray, frac: float, shift: float, inplane, rng) -> np.ndarray:
    a0, a1 = inplane
    pe = inplane[int(rng.integers(2))]  # phase-encode axis
    F = sfft.fft2(vol, axes=inplane, workers=-1)
    n_pe = vol.shape[pe]
    n_lines = max(1, int(round(frac * n_pe)))
    lines = rng.choice(n_pe, size=n_lines, replace=False)
    ang = rng.uniform(0, 2 * np.pi)
    d0, d1 = shift * np.cos(ang), shift * np.sin(ang)
    k0 = sfft.fftfreq(vol.shape[a0]).astype(np.float32)
    k1 = sfft.fftfreq(vol.shape[a1]).astype(np.float32)
    shp0 = [1, 1, 1]; shp0[a0] = -1
    shp1 = [1, 1, 1]; shp1[a1] = -1
    phase = np.exp(-2j * np.pi * (k0.reshape(shp0) * d0 + k1.reshape(shp1) * d1)).astype(np.complex64)
    sel = np.zeros(n_pe, dtype=bool)
    sel[lines] = True
    shp_sel = [1, 1, 1]; shp_sel[pe] = -1
    sel = sel.reshape(shp_sel)
    F = np.where(sel, F * phase, F)
    return np.abs(sfft.ifft2(F, axes=inplane, workers=-1)).astype(np.float32)


def kspace_truncate(vol: np.ndarray, keep: float, inplane, rng) -> np.ndarray:
    a0, a1 = inplane
    k0 = sfft.fftfreq(vol.shape[a0]).astype(np.float32)
    k1 = sfft.fftfreq(vol.shape[a1]).astype(np.float32)
    aniso = rng.uniform(0.9, 1.1)
    r0, r1 = 0.5 * keep * aniso, 0.5 * keep / aniso
    shp0 = [1, 1, 1]; shp0[a0] = -1
    shp1 = [1, 1, 1]; shp1[a1] = -1
    keep_mask = (k0.reshape(shp0) / r0) ** 2 + (k1.reshape(shp1) / r1) ** 2 <= 1.0
    F = sfft.fft2(vol, axes=inplane, workers=-1)
    return np.abs(sfft.ifft2(F * keep_mask, axes=inplane, workers=-1)).astype(np.float32)


def thick_slices(vol: np.ndarray, thickness: float, axis: int, rng) -> np.ndarray:
    """Average over a slab of `thickness` voxels, sample every `thickness` voxels, and
    linearly interpolate back to the 1 mm grid."""
    n = vol.shape[axis]
    width = max(1, int(round(thickness)))
    slab = uniform_filter1d(vol, size=width, axis=axis, mode="nearest")
    offset = rng.uniform(0, thickness)
    idx = np.unique(np.clip(np.round(np.arange(offset, n, thickness)).astype(int), 0, n - 1))
    if len(idx) < 2:
        return slab.astype(np.float32)
    low = np.take(slab, idx, axis=axis)
    t = np.arange(n, dtype=np.float32)
    j = np.clip(np.searchsorted(idx, t, side="right") - 1, 0, len(idx) - 2)
    w = np.clip((t - idx[j]) / (idx[j + 1] - idx[j]), 0.0, 1.0).astype(np.float32)
    shp = [1, 1, 1]; shp[axis] = -1
    w = w.reshape(shp)
    out = np.take(low, j, axis=axis) * (1 - w) + np.take(low, j + 1, axis=axis) * w
    return out.astype(np.float32)


def rician_noise(vol: np.ndarray, mask: np.ndarray, snr: float, rng) -> np.ndarray:
    mean = float(vol[mask].mean()) if mask.any() else float(vol.mean())
    sigma = np.float32(mean / max(snr, 1e-3))
    n1 = rng.standard_normal(vol.shape, dtype=np.float32) * sigma
    n2 = rng.standard_normal(vol.shape, dtype=np.float32) * sigma
    return np.sqrt((vol + n1) ** 2 + n2 ** 2).astype(np.float32)


# ---------------------------------------------------------------- the transform

class Aug15T:
    """Callable: (img (C,X,Y,Z) raw float, mask (X,Y,Z) bool, rng) -> (img, applied)."""

    def __init__(self, cfg: Aug15TConfig | None = None):
        self.cfg = cfg or Aug15TConfig()

    def _on(self, name: str, p: float, rng) -> bool:
        return name in self.cfg.enabled and rng.random() < p

    def __call__(self, img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
        c = self.cfg
        img = img.astype(np.float32, copy=True)
        applied: list[str] = []
        n_ch = img.shape[0]

        if n_ch >= 2 and self._on("contrast", c.p_contrast, rng):
            img[1] = reduce_enhancement(img[0], img[1], mask, _u(rng, c.enhance_alpha))
            applied.append("contrast")

        if self._on("bias", c.p_bias, rng):
            coeff = _u(rng, c.bias_coeff)
            shared = bias_field(img.shape[1:], c.bias_order, coeff, rng)
            for ch in range(n_ch):  # shared coil field + small per-sequence jitter
                jitter = bias_field(img.shape[1:], 1, 0.25 * coeff, rng)
                f = shared * jitter
                f /= f[mask].mean() if mask.any() else f.mean()
                img[ch] *= f
            applied.append("bias")

        # acquisition effects differ per sequence: each channel is a separate acquisition
        for ch in range(n_ch):
            if self._on("motion", c.p_motion, rng):
                img[ch] = motion_ghosting(img[ch], _u(rng, c.motion_frac), _u(rng, c.motion_shift),
                                          c.inplane_axes, rng)
                applied.append(f"motion:{ch}")
            if self._on("resolution", c.p_resolution, rng):
                img[ch] = kspace_truncate(img[ch], _u(rng, c.kspace_keep), c.inplane_axes, rng)
                applied.append(f"resolution:{ch}")
            if self._on("thickness", c.p_thickness, rng):
                slice_axis = 3 - sum(c.inplane_axes)
                axis = slice_axis if rng.random() < c.p_slice_axis_is_z else int(rng.choice(c.inplane_axes))
                img[ch] = thick_slices(img[ch], _u(rng, c.slice_mm), axis, rng)
                applied.append(f"thickness:{ch}")
            if self._on("noise", c.p_noise, rng):
                img[ch] = rician_noise(img[ch], mask, _u(rng, c.snr), rng)
                applied.append(f"noise:{ch}")

        img *= mask[None]  # target data are skull-stripped: background stays exactly 0
        np.clip(img, 0, None, out=img)
        return img, applied
