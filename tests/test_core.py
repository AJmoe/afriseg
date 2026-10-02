import json
from dataclasses import replace

import numpy as np
import pytest

from afriseg.aug15t import (COMPONENTS, Aug15T, Aug15TConfig, kspace_truncate, rician_noise,
                            reduce_enhancement, thick_slices)
from afriseg.generic import GenericAug
from afriseg.labels import harmonize, to_regions
from afriseg.metrics import dice, hd95, HD95_MAX
from afriseg.phantom import make_phantom
from afriseg.quality import case_features, volume_features
from afriseg.stats import holm
from afriseg.transforms import TrainTransform, parse_aug


@pytest.fixture(scope="module")
def case():
    img, lbl = make_phantom((64, 72, 56), seed=3)
    return img, lbl, (img > 0).any(axis=0)


def forced(**kw):
    return replace(Aug15TConfig(), p_contrast=1, p_bias=1, p_motion=1, p_resolution=1,
                   p_thickness=1, p_noise=1, **kw)


# ---------------------------------------------------------------- labels

def test_harmonize_2021_and_2023_agree():
    seg21 = np.array([0, 1, 2, 4])
    seg23 = np.array([0, 1, 2, 3])
    assert harmonize(seg21).tolist() == harmonize(seg23).tolist() == [0, 1, 2, 3]


def test_harmonize_rejects_mixed_schemes():
    with pytest.raises(ValueError):
        harmonize(np.array([0, 3, 4]))


def test_regions_nesting():
    reg = to_regions(np.array([0, 1, 2, 3]))
    assert reg[0].tolist() == [0, 1, 1, 1]  # WT
    assert reg[1].tolist() == [0, 1, 0, 1]  # TC
    assert reg[2].tolist() == [0, 0, 0, 1]  # ET


# ---------------------------------------------------------------- physics augmentation

def test_each_component_changes_image_and_keeps_invariants(case):
    img, _, mask = case
    for comp in COMPONENTS:
        out, applied = Aug15T(forced().only(comp))(img, mask, np.random.default_rng(0))
        assert out.shape == img.shape and out.dtype == np.float32
        assert applied and all(a.split(":")[0] == comp for a in applied), comp
        assert np.all(out >= 0), comp
        assert np.all(out[:, ~mask] == 0), f"{comp} leaked signal into background"
        assert not np.allclose(out, img), f"{comp} had no effect"


def test_deterministic_given_seed(case):
    img, _, mask = case
    a = Aug15T(forced())(img, mask, np.random.default_rng(7))[0]
    b = Aug15T(forced())(img, mask, np.random.default_rng(7))[0]
    np.testing.assert_array_equal(a, b)


def test_disabled_config_is_identity(case):
    img, _, mask = case
    out, applied = Aug15T(replace(Aug15TConfig(), enabled=()))(img, mask, np.random.default_rng(0))
    assert applied == []
    np.testing.assert_allclose(out, img * mask[None])


def test_reduced_enhancement_lowers_et_contrast(case):
    img, lbl, mask = case
    before = case_features(img, lbl)["enhancement"]
    out = img.copy()
    out[1] = reduce_enhancement(img[0], img[1], mask, alpha=0.3)
    after = case_features(out, lbl)["enhancement"]
    assert after < before


def test_noise_raises_measured_noise(case):
    img, _, mask = case
    v = img[2]
    noisy = rician_noise(v, mask, snr=8, rng=np.random.default_rng(0))
    assert volume_features(noisy, mask)["noise"] > 2 * volume_features(v, mask)["noise"]


def test_kspace_truncation_removes_high_frequencies(case):
    img, _, mask = case
    v = img[3]
    low = kspace_truncate(v, 0.4, (0, 1), np.random.default_rng(0))
    assert volume_features(low, mask)["hf_inplane"] < volume_features(v, mask)["hf_inplane"]


def test_thick_slices_raise_slice_anisotropy(case):
    img, _, mask = case
    v = img[0]
    thick = thick_slices(v, 5.0, axis=2, rng=np.random.default_rng(0))
    assert volume_features(thick, mask)["slice_ratio"] < volume_features(v, mask)["slice_ratio"]


def test_config_json_roundtrip(tmp_path):
    cfg = Aug15TConfig().without("motion")
    cfg.to_json(tmp_path / "c.json")
    back = Aug15TConfig.from_json(tmp_path / "c.json")
    assert back == cfg and "motion" not in back.enabled


# ---------------------------------------------------------------- pipeline

@pytest.mark.parametrize("arm", ["none", "generic", "15t", "15t+generic", "15t-no-noise", "15t-only-bias"])
def test_train_transform_shapes(case, arm):
    img, lbl, _ = case
    x, y = TrainTransform(arm, patch=(48, 48, 48))(img, lbl, np.random.default_rng(1))
    assert x.shape == (4, 48, 48, 48) and y.shape == (3, 48, 48, 48)
    assert np.isfinite(x).all()


def test_patch_larger_than_volume_is_padded(case):
    img, lbl, _ = case
    x, y = TrainTransform("none", patch=(80, 80, 80))(img, lbl, np.random.default_rng(0))
    assert x.shape == (4, 80, 80, 80)


def test_unknown_arm_rejected():
    with pytest.raises(ValueError):
        parse_aug("15t-no-banana")


def test_generic_aug_runs(case):
    img, _, _ = case
    out, _ = GenericAug()(img[:, :32, :32, :32] / 1000, np.random.default_rng(0))
    assert out.shape == (4, 32, 32, 32) and np.isfinite(out).all()


# ---------------------------------------------------------------- metrics / stats

def test_dice_and_hd95_conventions():
    z = np.zeros((10, 10, 10), bool)
    o = z.copy(); o[2:5, 2:5, 2:5] = True
    assert dice(z, z) == 1.0 and dice(o, z) == 0.0 and dice(o, o) == 1.0
    assert hd95(z, z) == 0.0 and hd95(o, z) == HD95_MAX and hd95(o, o) == 0.0


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
