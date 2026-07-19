[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20058732.svg)](https://doi.org/10.5281/zenodo.20058732)

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
  theratime_kappa.py                     # pairwise Cohen's-kappa utility (run once per annotator pair)
  theratime_post_calibration.py          # v2 calibration library: consensus construction, held-out
                                          #   splitting, do-no-harm rule learning, keep/correct/review
                                          #   policy. No CLI -- imported by theratime_calibration_robustness.py.
  theratime_calibration_robustness.py    # primary driver script: bootstrap CIs, multi-seed stability,
                                          #   threshold sweep, rank-based coverage, k-fold pooled
                                          #   evaluation, paired significance test
  theratime_selective_reliability.py     # legacy risk-coverage utility (secondary diagnostic only,
                                          #   not used for the primary human-validated accuracy claim)
  theratime_error_recall.py              # exploratory two-annotator correction analysis for the
                                          #   earlier 150-example pilot only -- not used for the final
                                          #   three-annotator majority-consensus results
paper/
  theratime_neurocomputing_submission_ready.tex
examples/
  example_kaggle_commands.md
requirements.txt
LICENSE
CITATION.cff
```

**Note on `theratime_post_calibration.py`**: this module has no command-line interface. It is a library of consensus-construction, held-out-splitting, and calibration functions imported by `theratime_calibration_robustness.py`, which is the actual entry point for running the calibration pipeline.

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

`theratime_kappa.py` computes pairwise Cohen's kappa between exactly two annotation files. For the primary three-annotator study, run it once per pair to reproduce the three pairwise values reported in the paper:

```bash
python scripts/theratime_kappa.py -f \
  annotations/theratime_300_annotator1.csv \
  annotations/theratime_300_annotator2.csv

python scripts/theratime_kappa.py -f \
  annotations/theratime_300_annotator1.csv \
  annotations/theratime_300_external.csv

python scripts/theratime_kappa.py -f \
  annotations/theratime_300_annotator2.csv \
  annotations/theratime_300_external.csv
```

Each run outputs:

```text
theratime_iaa_report.json
theratime_disagreements.csv
```

## Primary calibration and robustness analysis (300-example study)

`theratime_calibration_robustness.py` is the entry point for the primary calibration results reported in the paper: pooled five-fold out-of-fold evaluation, 95% bootstrap confidence intervals, ten-seed repeated-split robustness, threshold sensitivity, and rank-based coverage. It internally uses `theratime_post_calibration.py` for consensus construction and held-out calibration.

```bash
python scripts/theratime_calibration_robustness.py \
  --auto /kaggle/working/theratime_v06_outputs/all_judgments_mpnet.csv \
  --ann annotations/theratime_300_annotator1.csv \
        annotations/theratime_300_annotator2.csv \
        annotations/theratime_300_external.csv \
  --out-dir /kaggle/working/theratime_robustness_outputs \
  --methods baseline conservative_human_recompute safe_keep_correct_review \
  --k-folds 5
```

This produces, in `--out-dir`:

```text
theratime_bootstrap_ci.csv           # step 1/6: headline bootstrap CIs
theratime_multiseed_per_seed.csv     # step 2/6: per-seed stability results
theratime_multiseed_aggregate.csv    # step 2/6: mean/SD/range across seeds
theratime_threshold_sweep.csv        # step 3/6: keep/correction threshold sensitivity
theratime_reliability_diagnostics.csv
theratime_coverage_target_report.csv # step 4/6: rank-based coverage at target %
theratime_kfold_pooled_summary.csv   # step 5/6: pooled five-fold out-of-fold accuracy
theratime_kfold_pooled_<method>.csv  # per-method pooled predictions
theratime_paired_significance.csv    # step 6/6: paired bootstrap test vs. baseline
theratime_robustness_report.json     # full run metadata and recommended paper wording
```

The `--consensus-mode auto` default resolves to unanimous consensus for two annotators and majority consensus for three or more, so the same script also supports the earlier 150-example, two-annotator design if needed.

## Correction-direction analysis (300-example study)

There is currently no standalone script for this table. It can be reproduced directly from `theratime_post_calibration.py`'s consensus output with the following snippet:

```python
import pandas as pd
from pathlib import Path
import theratime_post_calibration as tpc

annotation_paths = [
    Path("annotations/theratime_300_annotator1.csv"),
    Path("annotations/theratime_300_annotator2.csv"),
    Path("annotations/theratime_300_external.csv"),
]

consensus_df, _ = tpc.build_consensus(
    annotation_paths,
    require_same_correction=False,
    consensus_mode="auto",
)

# Majority-rejected cases with a valid, non-tied consensus correction
rejected_with_correction = consensus_df[
    (consensus_df["human_timing_correct"] == "no")
    & (consensus_df["human_timing"] != consensus_df["auto_timing"])
]

corrected_to_well_timed = (rejected_with_correction["human_timing"] == "well_timed").sum()
total_valid_corrections = len(rejected_with_correction)

print(f"Corrected to well_timed: {corrected_to_well_timed} / {total_valid_corrections}")
print(f"Still mistimed, different subtype: {total_valid_corrections - corrected_to_well_timed} / {total_valid_corrections}")
```

This reproduces Table 6 in the paper (112 of 120 valid corrections directed to `well_timed`, 93.3%).

## Exploratory two-annotator analysis (150-example pilot only)

`theratime_error_recall.py` computes subset recall, correction direction, and auto-precision for the earlier two-annotator, 150-example pilot. It is retained to document the development of the annotation protocol and is **not** used to compute the final three-annotator majority-consensus results reported as the paper's primary findings.

```bash
python scripts/theratime_error_recall.py \
  --disagreements theratime_disagreements.csv \
  --annotations1 annotations/theratime_150_Hasnae_human_corrected_annotations.csv \
  --annotations2 annotations/theratime_human_annotations_Asmae_150_updated_reviewed.csv \
  --out theratime_screening_recall_report.csv
```

## Selective reliability analysis (secondary diagnostic)

`theratime_selective_reliability.py` is a legacy descriptive risk-coverage utility. It is retained as a secondary diagnostic and is not used for the primary human-validated accuracy claim, which is computed by `theratime_calibration_robustness.py`.

```bash
python scripts/theratime_selective_reliability.py \
  --input /kaggle/working/theratime_robustness_outputs/theratime_kfold_pooled_safe_keep_correct_review.csv \
  --out-dir /kaggle/working/theratime_selective_reliability_outputs
```

## Data availability

The illustrative retrieval experiment uses public datasets (ESConv, CounselChat, MentalChat16K) obtainable from their original sources under their respective licenses.


## License

MIT License. See `LICENSE`. Dataset-specific licenses and terms must be respected separately.
