"""Smoke test of the whole pipeline on synthetic NIfTI cases: preprocess -> split -> train
(tiny network, CPU) -> fine-tune -> evaluate -> stats. Checks plumbing, not accuracy."""
import json

import nibabel as nib
import numpy as np
import pandas as pd

from afriseg import data, evaluate, stats, train
from afriseg.phantom import make_phantom


def write_cases(root, n, scheme, seed0):
    names = {"brats2021": ["t1", "t1ce", "t2", "flair"], "brats2023": ["t1n", "t1c", "t2w", "t2f"]}[scheme]
    for i in range(n):
        img, lbl = make_phantom((40, 44, 36), seed=seed0 + i)
        if scheme == "brats2021":
            lbl = np.where(lbl == 3, 4, lbl)
        cid = f"{scheme}_{i:03d}"
        d = root / cid
        d.mkdir(parents=True)
        aff = np.eye(4)
        sep = "_" if scheme == "brats2021" else "-"
        for c, name in enumerate(names):
            nib.save(nib.Nifti1Image(img[c], aff), d / f"{cid}{sep}{name}.nii.gz")
        nib.save(nib.Nifti1Image(lbl.astype(np.uint8), aff), d / f"{cid}{sep}seg.nii.gz")


def test_pipeline(tmp_path):
    write_cases(tmp_path / "raw21", 6, "brats2021", 0)
    write_cases(tmp_path / "rawAF", 5, "brats2023", 100)
    npz = tmp_path / "npz"
    m21 = data.preprocess(str(tmp_path / "raw21"), str(npz), "brats2021")
    maf = data.preprocess(str(tmp_path / "rawAF"), str(npz), "africa")
    with np.load(next((npz / "brats2021").glob("*.npz"))) as z:
        assert set(np.unique(z["label"])) <= {0, 1, 2, 3}

    ids21 = [r["id"] for r in data.read_manifest(str(m21))]
    (tmp_path / "s21.json").write_text(json.dumps({"split": data.make_holdout(ids21)}))
    idsaf = [r["id"] for r in data.read_manifest(str(maf))]
    (tmp_path / "saf.json").write_text(json.dumps({"split": data.make_folds(idsaf, 5)}))

    common = ["--small", "--patch", "32", "32", "32", "--batch", "1", "--iters", "2",
              "--workers", "0", "--device", "cpu"]
    src = tmp_path / "runs" / "src"
    train.main(["--manifest", str(m21), "--split", str(tmp_path / "s21.json"), "--subset", "train",
                "--val-subset", "val", "--aug", "15t", "--epochs", "2", "--val-every", "1",
                "--out", str(src)] + common)
    assert (src / "best.pt").exists() and (src / "DONE").exists()

    ft = tmp_path / "runs" / "ft"
    train.main(["--manifest", str(maf), "--split", str(tmp_path / "saf.json"), "--subset", "0",
                "--exclude", "--n-labelled", "2", "--init", str(src / "best.pt"), "--aug", "generic",
                "--epochs", "1", "--out", str(ft)] + common)
    assert len(json.loads((ft / "train_ids.json").read_text())) == 2
    held_out = [k for k, v in json.loads((tmp_path / "saf.json").read_text())["split"].items() if v == 0]
    assert not set(held_out) & set(json.loads((ft / "train_ids.json").read_text()))

    out_a = tmp_path / "res" / "a.csv"
    out_b = tmp_path / "res" / "b.csv"
    evaluate.main(["--ckpt", str(src / "best.pt"), "--manifest", str(maf), "--out", str(out_a), "--device", "cpu"])
    evaluate.main(["--ckpt", str(ft / "best.pt"), "--manifest", str(maf), "--out", str(out_b),
                   "--device", "cpu", "--tta"])
    df = pd.read_csv(out_a)
    assert len(df) == 5 and df["dice_WT"].between(0, 1).all()
    res = stats.compare(pd.read_csv(out_a), pd.read_csv(out_b))
    assert list(res["region"]) == ["WT", "TC", "ET"]


def test_quality_and_calibration(tmp_path):
    from afriseg import calibrate, quality
    from afriseg.aug15t import Aug15TConfig

    write_cases(tmp_path / "raw21", 3, "brats2021", 0)
    write_cases(tmp_path / "rawAF", 3, "brats2023", 100)
    npz = tmp_path / "npz"
    m21 = data.preprocess(str(tmp_path / "raw21"), str(npz), "brats2021")
    maf = data.preprocess(str(tmp_path / "rawAF"), str(npz), "africa")
    qa = tmp_path / "qa.csv"
    quality.main(["--manifest", str(maf), "--out", str(qa)])
    assert {"t1n_noise", "t2f_slice_ratio"} <= set(pd.read_csv(qa).columns)
    out = tmp_path / "cfg.json"
    calibrate.main(["--source-manifest", str(m21), "--target-features", str(qa),
                    "--n-candidates", "2", "--n-source", "2", "--repeats", "1", "--out", str(out)])
    assert isinstance(Aug15TConfig.from_json(out), Aug15TConfig)


def test_preprocess_id_filter(tmp_path):
    write_cases(tmp_path / "raw", 3, "brats2023", 0)
    (tmp_path / "ids.txt").write_text("brats2023_000\nbrats2023_002\n")
    m = data.preprocess(str(tmp_path / "raw"), str(tmp_path / "npz"), "africa",
                        ids=data.read_id_list(str(tmp_path / "ids.txt")))
    assert [r["id"] for r in data.read_manifest(str(m))] == ["brats2023_000", "brats2023_002"]


def test_manifest_survives_moving_folder(tmp_path):
    import shutil
    write_cases(tmp_path / "raw", 2, "brats2023", 0)
    data.preprocess(str(tmp_path / "raw"), str(tmp_path / "npz"), "africa")
    shutil.move(str(tmp_path / "npz"), str(tmp_path / "moved"))
    rows = data.read_manifest(str(tmp_path / "moved" / "africa_manifest.csv"))
    assert all(__import__("pathlib").Path(r["path"]).exists() for r in rows)


def test_preprocess_parallel_matches_serial(tmp_path):
    write_cases(tmp_path / "raw", 3, "brats2021", 0)
    a = data.preprocess(str(tmp_path / "raw"), str(tmp_path / "s"), "x")
    b = data.preprocess(str(tmp_path / "raw"), str(tmp_path / "p"), "x", workers=2)
    assert open(a).read() == open(b).read()
