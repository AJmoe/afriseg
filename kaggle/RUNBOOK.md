# Kaggle runbook

Run each step in a Kaggle notebook with a GPU (T4 or P100) and internet access turned on.
Every `!` line runs in its own shell, so shared values go in Python variables and are inserted
with `{NAME}`.
Steps 1 to 3 run once, and their output becomes a private Kaggle dataset that later steps attach.

## 0. Get the code into the notebook

Push this repository to a private GitHub repo, then in the notebook run:

```bash
!git clone https://github.com/<you>/afriseg.git && pip install -q -e afriseg
```

The alternative is to upload `src/` as a Kaggle dataset and `pip install -e` it.

## 1. Preprocess, on CPU, once per dataset

```bash
!python -m afriseg.data preprocess --root /kaggle/input/<brats2021> --out /kaggle/working/npz --dataset brats2021
!python -m afriseg.data preprocess --root /kaggle/input/<brats-africa> --out /kaggle/working/npz --dataset africa
!python -m afriseg.data split --manifest /kaggle/working/npz/brats2021_manifest.csv --mode holdout --out /kaggle/working/splits/brats2021.json
!python -m afriseg.data split --manifest /kaggle/working/npz/africa_manifest.csv  --mode kfold --k 5 --out /kaggle/working/splits/africa.json
```

- **Check the output.** Every case should print a shape, and no `WARNING ... spacing` lines
  should appear.
- **If BraTS 2021 exceeds Kaggle's 20 GB output limit,** run it in two notebooks with `--limit`,
  or keep a fixed random subset of 600 cases. Record that choice in the protocol's Deviations
  section.
- **Paths inside the manifest are absolute.** If the files move, for example once you attach the
  output as a dataset under `/kaggle/input/...`, rewrite the paths in the manifests with sed.

## 2. Calibration, CPU only, before any training on African data

```python
import json
s = json.load(open('/kaggle/working/splits/africa.json'))['split']
json.dump([k for k, v in s.items() if v in (0, 1)], open('/kaggle/working/splits/africa_calib_ids.json', 'w'))
```

```bash
!python -m afriseg.quality --manifest npz/africa_manifest.csv --out quality/africa.csv
!python -m afriseg.calibrate --source-manifest npz/brats2021_manifest.csv --target-features quality/africa.csv \
    --target-ids splits/africa_calib_ids.json --n-candidates 40 --n-source 30 --out configs/aug15t_calibrated.json
```

Save `configs/aug15t_calibrated.json` and its `.search.json` file to the repo, then freeze and
post the protocol.

## 3. Source arms, one notebook session each

Choose `--epochs` so that one arm fits the budget. Time the first epoch first, then set it. Use
the same value for every arm.

```python
COMMON = "--manifest npz/brats2021_manifest.csv --split splits/brats2021.json --subset train --val-subset val --epochs 150 --iters 250 --workers 4"
!python -m afriseg.train {COMMON} --aug none        --out runs/A0_none
!python -m afriseg.train {COMMON} --aug generic     --out runs/A1_generic
!python -m afriseg.train {COMMON} --aug 15t         --aug-config configs/aug15t_calibrated.json --out runs/A2_15t
!python -m afriseg.train {COMMON} --aug 15t         --out runs/A2u_15t_prior
!python -m afriseg.train {COMMON} --aug 15t+generic --aug-config configs/aug15t_calibrated.json --out runs/A3_both
```

If a session hits the time limit, attach the `runs/...` output and repeat the same command with
`--resume`.

## 4. Evaluate

```bash
!python -m afriseg.evaluate --ckpt runs/A1_generic/best.pt --manifest npz/africa_manifest.csv --out results/A1_africa.csv
!python -m afriseg.evaluate --ckpt runs/A2_15t/best.pt     --manifest npz/africa_manifest.csv --out results/A2_africa.csv
!python -m afriseg.evaluate --ckpt runs/A2_15t/best.pt     --manifest npz/brats2021_manifest.csv --split splits/brats2021.json --subset test --out results/A2_brats_test.csv
!python -m afriseg.stats --a results/A1_africa.csv --b results/A2_africa.csv
```

## 5. Fine-tuning grid (RQ2)

Run every combination of fold f in 0 to 4, k in 5, 10, 20 or all, and parent A1 or A2.
For "all", omit `--n-labelled`. `F` and `K` are Python variables, for example in a loop.

```python
F, K = 0, 10
!python -m afriseg.train --manifest npz/africa_manifest.csv --split splits/africa.json --subset {F} --exclude \
    --n-labelled {K} --init runs/A2_15t/best.pt --aug 15t --aug-config configs/aug15t_calibrated.json \
    --lr 1e-4 --epochs 40 --iters 100 --out runs/B_A2_k{K}_f{F}
!python -m afriseg.evaluate --ckpt runs/B_A2_k{K}_f{F}/best.pt --manifest npz/africa_manifest.csv \
    --split splits/africa.json --subset {F} --arm B_A2_k{K} --out results/B_A2_k{K}_f{F}.csv
```

Pool the five fold CSVs for each arm:

```bash
!python -m afriseg.stats --a results/B_A1_k10_f*.csv --b results/B_A2_k10_f*.csv
```
