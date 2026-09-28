# TXAI Reference Pipeline

A reviewer-ready reference implementation for a multimodal 3D neuroimaging **Trustworthy Explainable AI (TXAI)** pipeline using OASIS-1 data.

## What this repository contains

- OASIS-1 discovery and manifest construction
- Configurable task setup:
  - `cn_vs_ad`
  - `cn_mci_ad`
- 3D MRI backbone
- 3D PET backbone
- Clinical, biomarker, and demographic MLP encoders
- Missing-modality masking
- Multi-head cross-modal attention fusion
- Multitask prediction heads
- Patient-level cross-validation
- Training-only normalization
- Early stopping and learning-rate scheduling
- Monte Carlo dropout uncertainty
- Calibration / ECE / temperature scaling / Brier score
- Robustness checks
- Integrated Gradients
- 3D Grad-CAM
- SHAP for tabular features
- Attention-consistency analysis
- Saved model checkpoints and evaluation artifacts

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── txai_pipeline.py
└── outputs/
    ├── experiment_config.json
    ├── fold_metrics.csv
    ├── pooled_metrics.json
    ├── oof_predictions.csv
    ├── pooled_roc.png
    ├── trustworthy_fold1.json
    ├── trustworthy_fold2.json
    ├── trustworthy_fold3.json
    ├── xai_fold1.json
    ├── xai_fold2.json
    ├── xai_fold3.json
    ├── txai_fold1.pt
    ├── txai_fold2.pt
    └── txai_fold3.pt
```

**Important:** `outputs/` contains the actual uploaded Colab result artifacts. The source code does not fabricate or regenerate these result files.

## Dataset

### OASIS-1

The pipeline is configured around the **OASIS-1 cross-sectional neuroimaging dataset**, using the `disc1` archive supplied to the Colab workflow.

The original Colab expects the archive at:

```text
/content/drive/MyDrive/oasis_cross-sectional_disc1.tar.gz
```

It extracts the archive under:

```text
/content/oasis_data
```

and searches for the OASIS participant directory (`disc1`).

### Participant discovery

The manifest builder searches participant directories beginning with:

```text
OAS1_
```

For each participant, the pipeline attempts to collect:

- participant ID
- session ID
- visit/month information when available
- CDR
- age
- sex
- education
- MMSE
- MRI path
- PET path
- clinical JSON path
- biomarker JSON path
- demographic JSON path

The supplied OASIS construction populates MRI, clinical, and demographic information. PET and biomarker paths are left empty by that construction, so those modalities are treated as missing rather than invented.

### Current task configuration

The uploaded Colab result was generated with:

```text
TASK_SETUP = cn_vs_ad
```

For this configuration:

- CDR 0.5 cases are excluded.
- CDR < 1.0 is mapped to class `0`.
- CDR >= 1.0 is mapped to class `1`.

The repository therefore reports the exact task configuration used for the saved results rather than implying that the results apply to every possible OASIS label definition.

## Data splitting and leakage control

The pipeline performs patient-level stratified cross-validation and asserts that patient IDs do not overlap between the training, validation, and test partitions.

Normalization statistics are computed from the training split and then applied to validation/test data.

This is intended to prevent patient-level leakage and test-set information from entering preprocessing.

## Actual uploaded Colab results

The `outputs/` directory contains the result files produced by the uploaded Colab run.

### Pooled results

From `outputs/pooled_metrics.json`:

| Metric | Value |
|---|---:|
| Patients | 16 |
| Diagnosis accuracy | 0.8125 (81.25%) |
| Diagnosis precision | 0.4062 |
| Diagnosis recall | 0.5000 |
| Diagnosis F1 | 0.4483 |
| Diagnosis AUC | 0.4615 |

Pooled confusion matrix:

```text
[
  [
    13,
    0
  ],
  [
    3,
    0
  ]
]
```

The uploaded result shows that all 16 pooled predictions were assigned to class 0:

- true class 0 predicted as 0: 13
- true class 0 predicted as 1: 0
- true class 1 predicted as 0: 3
- true class 1 predicted as 1: 0

Therefore, **accuracy should not be interpreted by itself** for this run. The saved pooled AUC is 0.4615 and pooled F1 is 0.4483, and the confusion matrix shows no positive-class predictions.

### Per-fold results

The saved `fold_metrics.csv` contains:

| Fold | Best epoch | Validation loss | Accuracy | Precision | Recall | F1 | AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 0.2295 | 83.33% | 0.4167 | 0.5000 | 0.4545 | 0.8000 |
| 2 | 12 | 0.3190 | 80.00% | 0.4000 | 0.5000 | 0.4444 | 0.2500 |
| 3 | 12 | 0.0810 | 80.00% | 0.4000 | 0.5000 | 0.4444 | 0.0000 |

These are **the actual uploaded Colab results**, not benchmark or expected performance.

## Output files

- `fold_metrics.csv` — fold-level diagnosis metrics.
- `pooled_metrics.json` — pooled out-of-fold metrics and confusion matrix.
- `oof_predictions.csv` — out-of-fold predictions.
- `pooled_roc.png` — pooled ROC visualization.
- `experiment_config.json` — experiment configuration.
- `trustworthy_fold*.json` — uncertainty/calibration/robustness outputs saved per fold.
- `xai_fold*.json` — explainability outputs saved per fold.
- `txai_fold*.pt` — saved PyTorch model checkpoints.

## Reproducibility

The experiment configuration uses seed `42` in the uploaded pipeline. The Colab workflow also derives fold-specific seeds and records configuration in `experiment_config.json`.

The default lightweight Colab CPU preset uses:

- input volume: `48 × 48 × 48`
- batch size: `2`
- maximum epochs: `12`
- early-stopping patience: `5`

A larger `colab_gpu` preset is also defined in the source.

## Running in Google Colab

1. Open the source pipeline in Google Colab.
2. Mount Google Drive.
3. Place the OASIS-1 archive at:

```text
MyDrive/oasis_cross-sectional_disc1.tar.gz
```

4. Run the cells in order.
5. The pipeline writes generated outputs under:

```text
/content/txai_trustworthy_outputs
```

6. Copy the generated artifacts into this repository's `outputs/` directory when preparing a new reviewer package.

## Reviewer checklist

A reviewer can inspect:

1. `src/txai_pipeline.py` for the complete pipeline.
2. `outputs/experiment_config.json` for the saved run configuration.
3. `outputs/fold_metrics.csv` for per-fold performance.
4. `outputs/pooled_metrics.json` for pooled out-of-fold performance.
5. `outputs/oof_predictions.csv` for individual out-of-fold predictions.
6. `outputs/trustworthy_fold*.json` for uncertainty/calibration/robustness analyses.
7. `outputs/xai_fold*.json` for explainability results.
8. `outputs/txai_fold*.pt` for the saved model checkpoints.

## Data and licensing

The repository does **not** redistribute the OASIS-1 imaging dataset. Reviewers should obtain and use the dataset according to the dataset provider's terms and permissions.

The files in `outputs/` are the experiment artifacts supplied with this reviewer package.

## Scope and interpretation

This repository is a research/reference pipeline and the included metrics are from the supplied Colab run. They should not be presented as a clinical validation study or as evidence of clinical utility.

In particular, the included run has a small evaluation cohort and a highly imbalanced prediction outcome in the pooled confusion matrix. Reviewers should consider the complete set of metrics and predictions rather than accuracy alone.
