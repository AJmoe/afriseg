# Study protocol: acquisition-calibrated augmentation for glioma segmentation on BraTS-Africa

Status: DRAFT v0.1 (2 October 2026). Freeze this file and the code commit hash, and post both
on OSF, **before** any model is evaluated on BraTS-Africa. Changes after that point go in a dated
"Deviations" section at the end, never silently.

Working title: *Calibrated 1.5T Acquisition Simulation for Label-Efficient Glioma Segmentation
in Sub-Saharan Africa*

## 1. Background and gap

Models trained on BraTS 2021 lose a large share of their Dice on BraTS-Africa. The established
remedy is transfer learning: pre-train on BraTS 2021 / BraTS-GLI, then fine-tune on labelled
African cases. Challenge entries using this route report mean Dice of roughly 0.8 to 0.9
(Zhao et al., arXiv:2410.18698; Parameter-efficient fine-tuning, arXiv:2412.14100; SegFormer3D+,
arXiv:2511.02928). Generative style transfer has also been tried (arXiv:2501.04734).

All of these assume tens of expert-labelled local cases. A hospital adopting a model usually has
**zero** labelled local cases and no radiologist time to create them. What is missing is:

1. A source-only method that closes part of the gap **without any target labels**.
2. A measurement of how many labelled African cases a better-prepared source model saves.
3. Evidence about *which* acquisition differences cause the gap.

Generic augmentation (nnU-Net defaults, TorchIO artefacts, MixUp/Fourier augmentation) is not
tied to the actual target scanners. Our method simulates low-field acquisition physics and fits
its parameter ranges to label-free image-quality statistics of the target images.

Our MIRASOL 2026 paper (Submission 37) is a separate negative result on reliability estimation.
This study does not depend on it.

## 2. Research questions and hypotheses

- **RQ1 (primary): source-only gain.** Trained only on BraTS 2021, does calibrated physics
  augmentation (arm A2) give higher Dice on BraTS-Africa than nnU-Net-style generic augmentation
  (arm A1)?
  - H1: per-case mean Dice (average of WT, TC, ET) is higher for A2 than A1.
- **RQ2: label efficiency.** When fine-tuning with k labelled African cases, k in {0, 5, 10, 20, all},
  does starting from A2 reach a given Dice with fewer labels than starting from A1?
  - H2: at k = 5 and k = 10, the A2-initialised model beats the A1-initialised model.
- **RQ3: mechanism.** Which components matter, and does calibration matter?
  - H3a: removing the most important component (to be found in the ablation) costs Dice.
  - H3b: calibrated ranges (A2) beat the uncalibrated priors (A2u).
- **Safety check.** A2 is non-inferior to A1 on the BraTS 2021 test split, with a margin of
  0.01 mean Dice.

## 3. Data

| Dataset | Use | Cases | Notes |
|---|---|---|---|
| BraTS 2021 Task 1 | Source training | about 1,251 labelled | 80/10/10 holdout split, seed 2026. Labels 1, 2 and 4 are harmonised to 1, 2 and 3. |
| BraTS-Africa | Target evaluation and fine-tuning | 95 labelled glioma cases (the set used in our MIRASOL paper) | 5 folds, seed 2026. No case is ever used both to train and to test one model. |

- **BraTS-Africa source.** The Cancer Imaging Archive (TCIA) collection "BraTS-Africa", DOI
  10.7937/v8h6-8x67. Use only the *processed* NIfTI download, about 1.6 GB, licensed CC BY 4.0.
  The unprocessed images are under NIH controlled access and are not needed.
- **BraTS-Africa contents.** The collection has 146 patients: the 95 glioma cases plus 51 other
  brain tumours. Only the 95 gliomas are used. Select them with the collection's metadata
  spreadsheet, and record the selected ids in the repo.
- **BraTS 2021 source.** The official RSNA-ASNR-MICCAI BraTS 2021 Task 1 training data from
  Synapse, which requires registration and accepting the challenge's data terms. Cite the BraTS
  papers the terms require. Unofficial Kaggle re-uploads are not the cited source.
- **Kaggle copies.** Keep every Kaggle copy of either dataset **private**, and use it only by the
  authors of this study.
- **Preprocessing.** Both datasets are already co-registered to SRI24, at 1 mm isotropic, and
  skull-stripped. We only crop to the brain bounding box. Normalisation is a per-channel z-score
  inside the brain.
- **Region definitions.** WT = labels {1,2,3}, TC = {1,3}, ET = {3}. These are the same in both
  challenges.

## 4. Method: calibrated acquisition simulation (Aug15T)

Aug15T is applied to raw magnitude volumes before normalisation, in physical order:

1. Reduced gadolinium enhancement on T1c.
2. Polynomial coil bias field, shared across sequences with small per-sequence variation.
3. Motion: rigid in-plane shift during a random subset of phase-encode lines.
4. In-plane k-space truncation, giving a lower matrix size with Gibbs ringing.
5. Thick slices: slab averaging and re-interpolation to 1 mm.
6. Rician noise at a sampled SNR.

Steps 3 to 6 are applied independently per sequence. The background is re-masked to zero
because the target images are skull-stripped. Code: `src/afriseg/aug15t.py`.

**Calibration (label-free).**
- For each image we compute four quality features: relative noise, in-plane high-frequency
  energy, slice anisotropy and bias-field spread. Code: `src/afriseg/quality.py`.
- A random search over Aug15T ranges then minimises the mean standardised Wasserstein distance
  between the features of *augmented BraTS 2021 cases* and those of *African cases*. Code:
  `src/afriseg/calibrate.py`.
- To keep calibration separate from evaluation, calibrate on the images of folds 0 and 1 only.
  Report the main results on all 95 cases, plus a sensitivity analysis on folds 2 to 4, whose
  images calibration never saw.
- Calibration never uses target labels. The enhancement-reduction range stays at its prior
  because measuring enhancement needs ET labels.
- Report the feature distances for three settings: no augmentation, the prior, and calibrated.
  This shows whether calibration actually moved the source distribution toward the target.

## 5. Experimental arms

Backbone for all arms: MONAI SegResNet, 3-channel sigmoid output (WT, TC, ET), Dice + BCE loss,
AdamW, cosine schedule, 128³ patches, batch 2, foreground oversampling 0.33, random flips. All
arms share the same budget and differ only in `--aug`.

| Arm | `--aug` | Trained on | Purpose |
|---|---|---|---|
| A0 | none | BraTS 2021 | Reproduce the domain gap |
| A1 | generic | BraTS 2021 | Control: nnU-Net-style augmentation |
| A2 | 15t with calibrated config | BraTS 2021 | Proposed method |
| A2u | 15t with prior config | BraTS 2021 | Does calibration matter? (H3b) |
| A3 | 15t+generic with calibrated config | BraTS 2021 | Are the two complementary? |
| ABL-x | 15t-no-x, one per component | BraTS 2021 | Component ablation (H3a) |
| B-A1-k / B-A2-k | fine-tune from A1 or A2, same aug as the parent | African folds, k labelled | Label-efficiency curve (RQ2) |
| C | none or generic | African folds only, from scratch | In-domain reference |

**Model selection.**
- Source arms pick their best epoch on the BraTS 2021 *validation* split. The African data are
  never used for model selection.
- Fine-tuning arms on African folds use the **final** epoch, so there is no selection on the
  held-out fold.

**Fine-tuning subsets.**
- For each fold f and each k, `--n-labelled k` draws a fixed subsample, using the same seed, from
  the four training folds.
- B-A1-k and B-A2-k therefore see identical cases.

## 6. Outcomes and statistics

**Primary outcome.**
- Per-case mean Dice over WT, TC and ET, on all 95 African cases (RQ1).
- Test: two-sided Wilcoxon signed-rank, A2 vs A1, at alpha = 0.05.
- One primary test, so no multiplicity correction is applied to it.
- With 95 paired cases the test has about 80% power for a standardised paired effect of about 0.29.

**Secondary outcomes.**
- Dice and HD95 per region, with Holm correction across regions.
- Mean differences reported with 10,000-sample bootstrap 95% CIs and rank-biserial effect sizes.
- RQ2: for each k, Wilcoxon B-A2-k vs B-A1-k on the pooled out-of-fold predictions for all 95
  cases, Holm across k. Also report the area under the learning curve.
- Ablation: each ABL-x compared with A2, Holm across components. Labelled exploratory if
  compute forces shorter training.
- Non-inferiority on BraTS 2021 test: the one-sided 95% bootstrap CI of (A2 − A1) lies above −0.01.

**Tools.** `python -m afriseg.stats` produces every paired comparison.

**Seeds.** If compute allows, run A1 and A2 with two seeds and report the spread between seeds.
The primary test uses seed 2026.

## 7. Threats to validity

- **Label shift, not only image shift.** BraTS-Africa annotations may follow subtly different
  conventions. Augmentation cannot fix that. A per-region error analysis is needed, for example
  oedema boundaries and ET in low-enhancement cases.
- **Acquisition metadata.** TCIA's metadata spreadsheet lists scanner information for
  BraTS-Africa; check whether it gives field strength per case before relying on it. BraTS 2021
  has no per-case field strength, and some of its sites already scan at 1.5T.
- **Transductive calibration.** Calibration uses unlabelled target images. This is realistic for a
  deploying hospital, but it must be stated. The fold 2 to 4 sensitivity analysis addresses it.
- **Single backbone.** Our MIRASOL reviewers raised this. If compute allows, repeat A1 vs A2 with
  a second backbone (`--model unet` or `swinunetr`).
- **The proxy features are crude.** They only need to rank configurations, not to measure
  physical quantities exactly.

## 8. Compute plan (estimate; time the first epoch before committing)

| Item | Estimate |
|---|---|
| One source arm | 150 epochs × 250 iterations, about 6 to 8 GPU-hours on a Kaggle T4/P100 |
| Arms A0 to A3 and A2u | about 35 GPU-hours |
| Ablations, six components | about 45 GPU-hours, or about 20 at 60 epochs |
| Fine-tuning grid | 2 parents × 4 values of k × 5 folds, about 30 to 40 GPU-hours |
| Total | about 110 to 120 GPU-hours |

Kaggle gives about 30 GPU-hours per account per week and 12-hour sessions; training stops at
11.5 hours and resumes with `--resume`. With five co-authors' accounts the work can run in
parallel in about 1 to 2 weeks. CPU augmentation costs about 1.3 s per sample per worker, so
use 4 workers.

## 9. Timeline (indicative)

| Week | Work |
|---|---|
| 1 | Data access, preprocessing, splits, quality features, calibration. Freeze and post protocol. |
| 2 to 3 | Source arms A0 to A3 and A2u; BraTS 2021 test evaluation |
| 4 | Evaluate source arms on African data; primary analysis |
| 4 to 5 | Fine-tuning grid (RQ2) and ablations (RQ3) |
| 6 | Error analysis, figures, write-up |

Target venue: a MICCAI 2027 workshop or a journal such as MELBA. Check the deadlines.

## 10. Authorship and roles

To be agreed with the team before experiments start: who runs which arms on which Kaggle
account, who does the error analysis, and who writes which section.

## Deviations

(none yet)
