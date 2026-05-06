# Example Kaggle commands

## 1. Main TheraTime experiment
```python
!python scripts/theratime_v06_neurocomputing.py
```

## 2. Inter-annotator agreement
```python
!python scripts/theratime_kappa.py -f \
  /kaggle/input/datasets/asmaeassmaebriouya/annotations/theratime_150_Hasnae_human_corrected_annotations.csv \
  /kaggle/input/datasets/asmaeassmaebriouya/annotations/theratime_human_annotations_Asmae_150_updated_reviewed.csv
```

## 3. Post-hoc calibration
```python
!python scripts/theratime_post_calibration.py \
  --auto /kaggle/working/theratime_v06_outputs/all_judgments_mpnet.csv \
  --ann /kaggle/input/datasets/asmaeassmaebriouya/annotations/theratime_150_Hasnae_human_corrected_annotations.csv \
        /kaggle/input/datasets/asmaeassmaebriouya/annotations/theratime_human_annotations_Asmae_150_updated_reviewed.csv \
  --out-dir /kaggle/working/theratime_post_calibration_outputs \
  --methods all
```

## 4. Selective reliability
```python
!python scripts/theratime_selective_reliability.py \
  --input /kaggle/working/theratime_post_calibration_outputs/theratime_hybrid_isotonic_conservative.csv \
  --out-dir /kaggle/working/theratime_selective_reliability_outputs
```

## 5. Error recall / correction-direction analysis
```python
!python scripts/theratime_error_recall.py \
  --disagreements /kaggle/working/theratime_disagreements.csv \
  --annotations1 /kaggle/input/datasets/asmaeassmaebriouya/annotations/theratime_150_Hasnae_human_corrected_annotations.csv \
  --annotations2 /kaggle/input/datasets/asmaeassmaebriouya/annotations/theratime_human_annotations_Asmae_150_updated_reviewed.csv \
  --out theratime_screening_recall_report.csv
```
