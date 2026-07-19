[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20058732.svg)](https://doi.org/10.5281/zenodo.20058732)

# TheraTime

TheraTime is an offline research software framework for therapeutic timing evaluation in mental-health response retrieval.

It evaluates whether retrieved responses are appropriately timed for a user's support stage, rather than only whether they are semantically similar. The framework separates:

1. support-stage classification for the user query;
2. support-move classification for the retrieved response;
3. rule-based timing evaluation using a transparent stage--move compatibility taxonomy.

## Responsible use

TheraTime is an offline research evaluation framework only.

It is not a therapy chatbot, not a clinical decision-support system, and not validated for deployment with real users or crisis situations. Human review is required for safety-sensitive or final timing judgments.

## Repository contents

```text
annotations/
  theratime_annotation_sample_300_internal1.csv           # completed 300-example annotations: Internal 1
  theratime_annotation_sample_300_internal2.csv           # completed 300-example annotations: Internal 2
  theratime_annotation_sample_300_external.csv            # completed 300-example annotations: External
  theratime_annotation_template.csv     # blank reusable annotation template

scripts/
  theratime_v06_neurocomputing.py       # main retrieval + automatic timing-evaluation pipeline
  theratime_kappa.py                    # pairwise Cohen's-kappa and disagreement-file utility
  theratime_post_calibration.py         # reusable consensus and calibration library; no CLI
  theratime_calibration_robustness.py   # primary calibration/robustness command-line driver
  theratime_selective_reliability.py    # secondary human-validated risk--coverage diagnostic
  theratime_error_recall.py             # exploratory selected-pair correction diagnostic

notebooks/
  theratime.ipynb                       # notebook used for the released analysis workflow

paper/
  theratime_neurocomputing_submission_ready.tex

examples/
  example_kaggle_commands.md

requirements.txt
LICENSE
CITATION.cff
```

## Annotation files and naming convention

The primary study uses three completed annotation CSV files:

- `theratime_annotation_sample_300_internal1.csv`
- `theratime_annotation_sample_300_internal2.csv`
- `theratime_annotation_sample_300_external.csv`

The labels `Internal 1`, `Internal 2`, and `External` identify annotation roles in the paper and repository without embedding personal names in analysis filenames.

The folder also includes:

```text
annotations/theratime_annotation_template.csv
```

The blank template follows the schema expected by the analysis scripts and can be reused for new annotation studies. Its fields include example identifiers, query and response text, automatic stage/move/timing labels, correctness judgments, optional corrections, notes, and the annotator label.

## Installation

```bash
git clone https://github.com/briouyaasmae/theratime.git
cd theratime
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Kaggle:

```python
!pip install -q -r requirements.txt
```

## Run the main experiment

```bash
python scripts/theratime_v06_neurocomputing.py
```

The default output directory in the released script is:

```text
/kaggle/working/theratime_v06_outputs
```

For local execution, edit `OUTPUT_DIR` or adapt it to a local path.

## Pairwise inter-annotator agreement

`theratime_kappa.py` compares exactly two annotation files. Run it once for each annotator pair.

```bash
python scripts/theratime_kappa.py -f \
  annotations/theratime_annotation_sample_300_internal1.csv \
  annotations/theratime_annotation_sample_300_internal2.csv \
  --out-prefix internal1_internal2

python scripts/theratime_kappa.py -f \
  annotations/theratime_annotation_sample_300_internal1.csv \
  annotations/theratime_annotation_sample_300_external.csv \
  --out-prefix internal1_external

python scripts/theratime_kappa.py -f \
  annotations/theratime_annotation_sample_300_internal2.csv \
  annotations/theratime_annotation_sample_300_external.csv \
  --out-prefix internal2_external
```

Each run produces a pair-specific agreement report and disagreement CSV:

```text
<prefix>_iaa_report.json
<prefix>_disagreements.csv
```

Using unique prefixes prevents the three runs from overwriting one another.

## Primary calibration and robustness analysis

`theratime_post_calibration.py` is a reusable library and has no command-line interface. The executable entry point is `theratime_calibration_robustness.py`.

The primary analysis includes pooled five-fold out-of-fold evaluation, bootstrap confidence intervals, ten-seed repeated-split robustness, threshold sensitivity, reliability diagnostics, rank-based coverage, and paired significance testing.

```bash
python scripts/theratime_calibration_robustness.py \
  --auto /kaggle/working/theratime_v06_outputs/all_judgments_mpnet.csv \
  --ann annotations/theratime_annotation_sample_300_internal1.csv \
        annotations/theratime_annotation_sample_300_internal2.csv \
        annotations/theratime_annotation_sample_300_external.csv \
  --out-dir /kaggle/working/theratime_robustness_outputs_v3 \
  --methods baseline conservative_human_recompute safe_keep_correct_review \
  --k-folds 5
```

Main outputs include:

```text
theratime_bootstrap_ci.csv
theratime_multiseed_per_seed.csv
theratime_multiseed_aggregate.csv
theratime_threshold_sweep.csv
theratime_reliability_diagnostics.csv
theratime_coverage_target_report.csv
theratime_kfold_pooled_summary.csv
theratime_kfold_pooled_<method>.csv
theratime_paired_significance.csv
theratime_robustness_report.json
```

## Correction-direction analysis

The correction-direction table is derived from the same three-annotator majority-consensus output produced by `build_consensus()` in `theratime_post_calibration.py`.

```python
from pathlib import Path
import sys

sys.path.insert(0, "scripts")
import theratime_post_calibration as tpc

annotation_paths = [
    Path("annotations/theratime_annotation_sample_300_internal1.csv"),
    Path("annotations/theratime_annotation_sample_300_internal2.csv"),
    Path("annotations/theratime_annotation_sample_300_external.csv"),
]

consensus_df, _ = tpc.build_consensus(
    annotation_paths,
    require_same_correction=False,
    consensus_mode="auto",
)

rejected_with_correction = consensus_df[
    (consensus_df["human_timing_correct"] == "no")
    & (consensus_df["human_timing"] != consensus_df["auto_timing"])
]

corrected_to_well_timed = (
    rejected_with_correction["human_timing"] == "well_timed"
).sum()
total_valid_corrections = len(rejected_with_correction)

print(
    "Corrected to well_timed:",
    corrected_to_well_timed,
    "/",
    total_valid_corrections,
)
print(
    "Still mistimed, different subtype:",
    total_valid_corrections - corrected_to_well_timed,
    "/",
    total_valid_corrections,
)
```

This reproduces the primary correction-direction result reported in the paper: 112 of 120 valid majority-consensus corrections were directed to `well_timed` (93.3%).

## Selective reliability analysis

`theratime_selective_reliability.py` evaluates human-validated outcomes from the pooled held-out prediction file generated by the primary robustness analysis.

### Isotonic reliability

```bash
python scripts/theratime_selective_reliability.py \
  --input /kaggle/working/theratime_robustness_outputs_v3/theratime_kfold_pooled_safe_keep_correct_review.csv \
  --out-dir /kaggle/working/theratime_selective_reliability_outputs \
  --preferred-coverage 0.80 \
  --score isotonic_overall_reliability
```

### Conservative margin reliability

```bash
python scripts/theratime_selective_reliability.py \
  --input /kaggle/working/theratime_robustness_outputs_v3/theratime_kfold_pooled_safe_keep_correct_review.csv \
  --out-dir /kaggle/working/theratime_selective_reliability_outputs_conservative \
  --preferred-coverage 0.80 \
  --score margin_reliability_score
```

These analyses are secondary diagnostics. They do not replace the pooled five-fold human-validated accuracy result.

## Exploratory pairwise correction analysis

`theratime_error_recall.py` is an exploratory selected-pair diagnostic. In the released notebook it is applied to the Internal 1–External pair from the 300-example study.

```bash
python scripts/theratime_error_recall.py \
  --disagreements internal1_external_disagreements.csv \
  --annotations1 annotations/theratime_annotation_sample_300_internal1.csv \
  --annotations2 annotations/theratime_annotation_sample_300_external.csv \
  --out internal1_external_screening_recall_report.csv \
  --json internal1_external_recall_summary.json
```

Its pairwise disagreement-subset outputs are not used to construct the primary three-annotator majority-consensus reference or the pooled calibration result.

## Example commands

The commands used in the released Kaggle workflow are collected in:

```text
examples/example_kaggle_commands.md
```

## Data availability

The repository includes the three completed 300-example annotation CSV files and the reusable blank annotation template in `annotations/`.

The illustrative retrieval experiment uses the public ESConv, CounselChat, and MentalChat16K datasets, which must be obtained from their original sources and used under their respective licenses and ethical requirements.

## License

MIT License. See `LICENSE`. Dataset-specific licenses and terms must be respected separately.
