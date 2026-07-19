# Example Kaggle commands

These commands reproduce the workflows used for the 300-example, three-annotator study. The annotation roles are named consistently as `internal1`, `internal2`, and `external`.

## 1. Main TheraTime experiment

```python
!python scripts/theratime_v06_neurocomputing.py
```

The default output directory is:

```text
/kaggle/working/theratime_v06_outputs
```

## 2. Pairwise inter-annotator agreement

Run `theratime_kappa.py` once for each annotator pair. A unique `--out-prefix` prevents later runs from overwriting earlier outputs.

### Internal 1 vs Internal 2

```python
!python scripts/theratime_kappa.py -f \
  annotations/theratime_annotation_sample_300_internal1.csv \
  annotations/theratime_annotation_sample_300_internal2.csv \
  --out-prefix /kaggle/working/internal1_internal2
```

### Internal 1 vs External

```python
!python scripts/theratime_kappa.py -f \
  annotations/theratime_annotation_sample_300_internal1.csv \
  annotations/theratime_annotation_sample_300_external.csv \
  --out-prefix /kaggle/working/internal1_external
```

### Internal 2 vs External

```python
!python scripts/theratime_kappa.py -f \
  annotations/theratime_annotation_sample_300_internal2.csv \
  annotations/theratime_annotation_sample_300_external.csv \
  --out-prefix /kaggle/working/internal2_external
```

These runs produce:

```text
/kaggle/working/internal1_internal2_iaa_report.json
/kaggle/working/internal1_internal2_disagreements.csv
/kaggle/working/internal1_external_iaa_report.json
/kaggle/working/internal1_external_disagreements.csv
/kaggle/working/internal2_external_iaa_report.json
/kaggle/working/internal2_external_disagreements.csv
```

## 3. Primary calibration and robustness analysis

`theratime_post_calibration.py` is a library and has no command-line interface. The executable entry point is `theratime_calibration_robustness.py`.

```python
!python scripts/theratime_calibration_robustness.py \
  --auto /kaggle/working/theratime_v06_outputs/all_judgments_mpnet.csv \
  --ann annotations/theratime_annotation_sample_300_internal1.csv \
        annotations/theratime_annotation_sample_300_internal2.csv \
        annotations/theratime_annotation_sample_300_external.csv \
  --out-dir /kaggle/working/theratime_robustness_outputs_v3 \
  --methods baseline conservative_human_recompute safe_keep_correct_review \
  --k-folds 5
```

## 4. Selective reliability analysis

The notebook evaluates both the isotonic reliability score and the conservative confidence-margin score using the pooled held-out predictions from the primary robustness analysis.

### Isotonic reliability

```python
!python scripts/theratime_selective_reliability.py \
  --input /kaggle/working/theratime_robustness_outputs_v3/theratime_kfold_pooled_safe_keep_correct_review.csv \
  --out-dir /kaggle/working/theratime_selective_reliability_outputs \
  --preferred-coverage 0.80 \
  --score isotonic_overall_reliability
```

### Conservative margin reliability

```python
!python scripts/theratime_selective_reliability.py \
  --input /kaggle/working/theratime_robustness_outputs_v3/theratime_kfold_pooled_safe_keep_correct_review.csv \
  --out-dir /kaggle/working/theratime_selective_reliability_outputs_conservative \
  --preferred-coverage 0.80 \
  --score margin_reliability_score
```

## 5. Exploratory pairwise correction analysis

The notebook applies `theratime_error_recall.py` as a secondary diagnostic to the Internal 1–External pair. These outputs are not used to construct the primary three-annotator majority-consensus reference or the pooled calibration result.

```python
!python scripts/theratime_error_recall.py \
  --disagreements /kaggle/working/internal1_external_disagreements.csv \
  --annotations1 annotations/theratime_annotation_sample_300_internal1.csv \
  --annotations2 annotations/theratime_annotation_sample_300_external.csv \
  --out /kaggle/working/internal1_external_screening_recall_report.csv \
  --json /kaggle/working/internal1_external_recall_summary.json
```
