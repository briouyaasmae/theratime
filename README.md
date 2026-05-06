# TheraTime

TheraTime is an offline research software framework for therapeutic timing evaluation in mental-health response retrieval.

It evaluates whether retrieved responses are appropriately timed for a user's support stage, rather than only whether they are semantically similar. The framework separates:

1. support-stage classification for the user query,
2. support-move classification for the retrieved response,
3. rule-based timing evaluation using a transparent stage--move compatibility taxonomy.

## Responsible use

TheraTime is an offline research evaluation framework only.

It is not a therapy chatbot, not a clinical decision-support system, and not validated for deployment with real users or crisis situations. Human review is required for safety-sensitive or final timing judgments.

## Repository contents

```text
scripts/
  theratime_v06_neurocomputing.py        # main retrieval + timing evaluation pipeline
  theratime_kappa.py                     # inter-annotator agreement analysis
  theratime_post_calibration.py          # post-hoc calibration methods
  theratime_selective_reliability.py     # selective reliability / risk-coverage analysis
  theratime_error_recall.py              # correction-direction and screening reliability analysis
paper/
  theratime_neurocomputing_submission_ready.tex
examples/
  example_kaggle_commands.md
requirements.txt
LICENSE
CITATION.cff
```

## Installation

```bash
git clone https://github.com/USERNAME/theratime.git
cd theratime
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Kaggle, install dependencies in a notebook cell:

```python
!pip install -q -r requirements.txt
```

## Run the main experiment

```bash
python scripts/theratime_v06_neurocomputing.py
```

Default output directory in the script:

```text
/kaggle/working/theratime_v06_outputs
```

If running locally, edit `OUTPUT_DIR` in the script or adapt it to your local path.

## Inter-annotator agreement

```bash
python scripts/theratime_kappa.py -f \
  annotations/theratime_150_Hasnae_human_corrected_annotations.csv \
  annotations/theratime_human_annotations_Asmae_150_updated_reviewed.csv
```

Outputs:

```text
theratime_iaa_report.json
theratime_disagreements.csv
```

## Post-hoc calibration

```bash
python scripts/theratime_post_calibration.py \
  --auto /kaggle/working/theratime_v06_outputs/all_judgments_mpnet.csv \
  --ann annotations/theratime_150_Hasnae_human_corrected_annotations.csv \
        annotations/theratime_human_annotations_Asmae_150_updated_reviewed.csv \
  --out-dir /kaggle/working/theratime_post_calibration_outputs \
  --methods all
```

## Selective reliability analysis

```bash
python scripts/theratime_selective_reliability.py \
  --input /kaggle/working/theratime_post_calibration_outputs/theratime_hybrid_isotonic_conservative.csv \
  --out-dir /kaggle/working/theratime_selective_reliability_outputs
```

## Correction-direction / screening reliability analysis

```bash
python scripts/theratime_error_recall.py \
  --disagreements theratime_disagreements.csv \
  --annotations1 annotations/theratime_150_Hasnae_human_corrected_annotations.csv \
  --annotations2 annotations/theratime_human_annotations_Asmae_150_updated_reviewed.csv \
  --out theratime_screening_recall_report.csv
```

## Paper

The LaTeX manuscript is provided in `paper/`. Before submission, replace the repository placeholder and Zenodo placeholder with the final GitHub URL and Zenodo DOI.

## Citation

After archiving the repository on Zenodo, update `CITATION.cff` with the DOI.
