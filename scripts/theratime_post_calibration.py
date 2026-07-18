#!/usr/bin/env python3
"""
theratime_post_calibration.py
=============================

Defensible do-no-harm and keep/correct/review calibration for TheraTime.

This version fixes the main validity problems of the earlier script:

1. Strict human consensus
   - Every annotator must provide yes/no.
   - Conservative corrections require every annotator to provide the same
     valid correction.

2. Held-out human evaluation
   - Human-validated examples are split into calibration-development and
     untouched held-out evaluation subsets.
   - Correction rules and isotonic reliability models are learned only from
     the development subset.
   - Claims of improved accuracy are based only on the held-out subset.

3. Conditional correction rules
   - Stage correction rules are conditioned on the automatic stage.
   - Move correction rules are conditioned on automatic stage + automatic move.
   - Direct timing correction remains an explicit high-overfit-risk ablation.

4. Clear metric naming
   - Full-corpus outputs report automatic well-timed rates, not accuracy.
   - Held-out outputs report human-validated stage, move, and timing accuracy.

5. Explicit ID validation
   - Annotation IDs must match automatic rank-1 judgment IDs.
   - Missing or duplicated IDs raise an error instead of silently reducing
     the calibration subset.

6. Reliability calibration without label inflation
   - Isotonic regression estimates correctness probability.
   - It does not change labels.
   - Reliability results are reported with coverage and held-out accuracy.

Recommended workflow
--------------------
The same script supports both:
- 2 annotators with approximately 150 examples;
- 3 annotators with approximately 300 examples.

In --consensus-mode auto, two annotators use unanimous consensus and
three or more annotators use majority consensus.

Example with three annotators and 300 examples:

python theratime_post_calibration.py \
  --auto /kaggle/working/theratime_v06_outputs/all_judgments_mpnet.csv \
  --ann /kaggle/input/.../theratime_300_Hasnae.csv \
        /kaggle/input/.../theratime_300_Asmae.csv \
  --out-dir /kaggle/working/theratime_post_calibration_v2

Important interpretation
------------------------
- "automatic well-timed rate" is the proportion of outputs assigned
  well_timed by the automatic evaluator.
- "human-validated accuracy" is agreement with untouched held-out human
  consensus.
- Only the latter can support a claim that calibration improved accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except Exception:
    IsotonicRegression = None
    brier_score_loss = None
    train_test_split = None
    SKLEARN_AVAILABLE = False


# =============================================================================
# Constants
# =============================================================================

YES_NO = {"yes", "no"}

STAGES = {
    "distress_disclosure",
    "high_emotional_intensity",
    "unclear_need",
    "advice_seeking",
    "psychoeducation_seeking",
    "crisis_safety",
    "followup_problem_solving",
}

MOVES = {
    "validation",
    "empathy",
    "reflective_listening",
    "clarification",
    "grounding",
    "practical_advice",
    "psychoeducation",
    "encouragement",
    "safety_referral",
}

TIMINGS = {
    "well_timed",
    "premature_advice",
    "delayed_safety",
    "over_validation",
    "missing_clarification",
    "stage_mismatch",
}

ALLOWED_MOVES: Dict[str, set] = {
    "distress_disclosure": {
        "validation",
        "empathy",
        "reflective_listening",
        "clarification",
        "encouragement",
    },
    "high_emotional_intensity": {
        "validation",
        "empathy",
        "reflective_listening",
        "grounding",
        "clarification",
    },
    "unclear_need": {
        "validation",
        "empathy",
        "reflective_listening",
        "clarification",
    },
    "advice_seeking": {
        "practical_advice",
        "psychoeducation",
        "encouragement",
        "validation",
    },
    "psychoeducation_seeking": {
        "psychoeducation",
        "validation",
        "clarification",
    },
    "crisis_safety": {
        "safety_referral",
        "grounding",
        "validation",
    },
    "followup_problem_solving": {
        "practical_advice",
        "psychoeducation",
        "encouragement",
    },
}

METHODS = [
    "baseline",
    "confidence_only",
    "isotonic_reliability",
    "human_rules_direct",
    "human_rules_recompute",
    "conservative_human_recompute",
    "conservative_recompute_with_isotonic_reliability",
    "safe_keep_correct_review",
]

RULE_METHODS = {
    "human_rules_direct",
    "human_rules_recompute",
    "conservative_human_recompute",
    "conservative_recompute_with_isotonic_reliability",
}

RECOMPUTE_METHODS = {
    "human_rules_recompute",
    "conservative_human_recompute",
    "conservative_recompute_with_isotonic_reliability",
}


# =============================================================================
# General utilities
# =============================================================================

def norm(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def normalise_answer(value: Any) -> str:
    value = norm(value)

    if value in {"y", "true", "1", "correct"}:
        return "yes"

    if value in {"n", "false", "0", "incorrect", "wrong"}:
        return "no"

    if value in {"unsure", "uncertain", "maybe", "not_sure"}:
        return "unsure"

    return value


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)

        if math.isnan(output) or math.isinf(output):
            return default

        return output

    except Exception:
        return default


def as_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def serializable_rule_dict(
    rules: Dict[Any, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}

    for key, value in rules.items():
        if isinstance(key, tuple):
            key_string = "|||".join(str(x) for x in key)
        else:
            key_string = str(key)

        output[key_string] = value

    return output


# =============================================================================
# Annotation loading
# =============================================================================

def load_annotation_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path).fillna("")
        rows = df.to_dict("records")

        annotator = ""
        if "annotator" in df.columns:
            values = [
                str(value).strip()
                for value in df["annotator"].tolist()
                if str(value).strip()
            ]
            if values:
                annotator = values[0]

        if not annotator:
            annotator = path.stem

    elif suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8-sig"))

        if isinstance(raw, list):
            rows = raw
            annotator = path.stem
        else:
            rows = (
                raw.get("annotations")
                or raw.get("rows")
                or raw.get("data")
                or []
            )
            annotator = str(
                raw.get("annotator")
                or path.stem
            ).strip()

    else:
        raise ValueError(
            f"Unsupported annotation format for {path}. Use CSV or JSON."
        )

    if not rows:
        raise ValueError(f"No annotation rows found in: {path}")

    cleaned: Dict[str, Dict[str, Any]] = {}

    for index, original_row in enumerate(rows, start=1):
        row = dict(original_row)

        sample_id = str(
            row.get("id")
            or row.get("query_id")
            or row.get("example_id")
            or f"row_{index}"
        ).strip()

        if not sample_id:
            raise ValueError(
                f"Empty sample ID in {path} at row {index}."
            )

        if sample_id in cleaned:
            raise ValueError(
                f"Duplicate annotation ID '{sample_id}' in {path}."
            )

        row["id"] = sample_id

        for field in [
            "stage_correct",
            "move_correct",
            "timing_correct",
        ]:
            row[field] = normalise_answer(row.get(field, ""))

        for field in [
            "auto_stage",
            "predicted_stage",
            "stage_correction",
        ]:
            if field in row:
                row[field] = norm(row.get(field))

        for field in [
            "auto_move",
            "predicted_move",
            "move_correction",
        ]:
            if field in row:
                row[field] = norm(row.get(field))

        for field in [
            "auto_timing",
            "timing_label",
            "predicted_timing",
            "timing_correction",
        ]:
            if field in row:
                row[field] = norm(row.get(field))

        cleaned[sample_id] = row

    return {
        "annotator": annotator,
        "path": str(path),
        "annotations": cleaned,
    }


def get_auto_stage(row: Dict[str, Any]) -> str:
    return norm(
        row.get("auto_stage")
        or row.get("predicted_stage")
        or row.get("stage")
    )


def get_auto_move(row: Dict[str, Any]) -> str:
    return norm(
        row.get("auto_move")
        or row.get("predicted_move")
        or row.get("move")
    )


def get_auto_timing(row: Dict[str, Any]) -> str:
    return norm(
        row.get("auto_timing")
        or row.get("timing_label")
        or row.get("predicted_timing")
    )


# =============================================================================
# Strict human consensus
# =============================================================================

def resolve_consensus_mode(
    requested_mode: str,
    n_annotators: int,
) -> str:
    """Resolve auto/unanimous/majority consensus for 2+ annotators."""
    if n_annotators < 2:
        raise ValueError("At least two annotators are required.")
    if requested_mode == "auto":
        return "unanimous" if n_annotators == 2 else "majority"
    if requested_mode not in {"unanimous", "majority"}:
        raise ValueError("consensus mode must be auto, unanimous, or majority.")
    return requested_mode


def strict_consensus_label(
    rows: Sequence[Dict[str, Any]],
    correct_field: str,
    correction_field: str,
    auto_value: str,
    allowed_values: set,
    require_same_correction: bool,
    consensus_mode: str,
) -> Dict[str, Any]:
    """
    Build consensus for two or more annotators.

    auto mode:
      - 2 annotators: unanimous 2/2
      - 3+ annotators: majority floor(n/2)+1

    Standard corrections require the same full-panel threshold.
    Conservative corrections require every annotator to reject and provide
    the same valid correction. Missing/unsure answers never count as votes.

    A rejection whose supplied correction normalizes to the same value as
    the original automatic label is not a real correction and is excluded
    before majority/conservative correction logic runs (mirrors the filter
    already used in learn_stage_rules / learn_conditional_move_rules /
    learn_direct_timing_rules).
    """
    n_annotators = len(rows)
    resolved_mode = resolve_consensus_mode(consensus_mode, n_annotators)
    answers = [normalise_answer(row.get(correct_field, "")) for row in rows]
    yes_count = sum(answer == "yes" for answer in answers)
    no_count = sum(answer == "no" for answer in answers)
    valid_count = sum(answer in YES_NO for answer in answers)
    majority_threshold = n_annotators // 2 + 1

    if resolved_mode == "unanimous":
        accepted = valid_count == n_annotators and yes_count == n_annotators
        rejected = valid_count == n_annotators and no_count == n_annotators
    else:
        accepted = yes_count >= majority_threshold
        rejected = no_count >= majority_threshold

    common = {
        "yes_count": int(yes_count),
        "no_count": int(no_count),
        "valid_count": int(valid_count),
        "n_annotators": int(n_annotators),
        "consensus_mode": resolved_mode,
    }

    if accepted:
        return {**common, "consensus_correct": "yes", "consensus_value": auto_value, "source": f"{resolved_mode}_accepted"}

    if not rejected:
        source = "incomplete_or_unsure_annotations" if valid_count < majority_threshold else "correctness_disagreement"
        return {
            **common,
            "consensus_correct": "missing" if valid_count < majority_threshold else "disagree",
            "consensus_value": auto_value,
            "source": source,
        }

    corrections = [
        norm(row.get(correction_field, ""))
        for row, answer in zip(rows, answers)
        if answer == "no"
        and norm(row.get(correction_field, "")) in allowed_values
        and norm(row.get(correction_field, "")) != auto_value  # <-- FIX: exclude self-referential "corrections"
    ]

    if require_same_correction:
        if no_count != n_annotators:
            return {**common, "consensus_correct": "no", "consensus_value": auto_value, "source": "conservative_not_all_rejected"}
        if len(corrections) != n_annotators:
            return {**common, "consensus_correct": "no", "consensus_value": auto_value, "source": "incomplete_corrections"}
        if len(set(corrections)) != 1:
            return {**common, "consensus_correct": "no", "consensus_value": auto_value, "source": "correction_disagreement"}
        return {**common, "consensus_correct": "no", "consensus_value": corrections[0], "source": "all_annotators_agree_on_correction"}

    if not corrections:
        return {**common, "consensus_correct": "no", "consensus_value": auto_value, "source": "no_valid_correction"}

    counts = Counter(corrections)
    most_common = counts.most_common()
    top_value, top_count = most_common[0]
    tied = len(most_common) > 1 and most_common[1][1] == top_count
    if tied:
        return {**common, "consensus_correct": "no", "consensus_value": auto_value, "source": "correction_tie"}

    correction_threshold = n_annotators if resolved_mode == "unanimous" else majority_threshold
    if top_count < correction_threshold:
        return {**common, "consensus_correct": "no", "consensus_value": auto_value, "source": "no_majority_correction"}

    return {**common, "consensus_correct": "no", "consensus_value": top_value, "source": f"{resolved_mode}_correction"}

def build_consensus(
    annotation_paths: Sequence[Path],
    require_same_correction: bool,
    consensus_mode: str = "auto",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if len(annotation_paths) < 2:
        raise ValueError(
            "At least two annotation files are required."
        )

    loaded = [
        load_annotation_file(path)
        for path in annotation_paths
    ]

    annotation_maps = [
        item["annotations"]
        for item in loaded
    ]

    resolved_consensus_mode = resolve_consensus_mode(
        consensus_mode,
        len(annotation_paths),
    )

    common_ids = sorted(
        set.intersection(
            *[
                set(annotation_map.keys())
                for annotation_map in annotation_maps
            ]
        )
    )

    if not common_ids:
        raise ValueError(
            "The annotation files have no common sample IDs."
        )

    rows_out: List[Dict[str, Any]] = []

    for sample_id in common_ids:
        annotator_rows = [
            annotation_map[sample_id]
            for annotation_map in annotation_maps
        ]

        base = annotator_rows[0]

        auto_stage = get_auto_stage(base)
        auto_move = get_auto_move(base)
        auto_timing = get_auto_timing(base)

        if auto_stage not in STAGES:
            raise ValueError(
                f"Invalid or missing automatic stage '{auto_stage}' "
                f"for sample {sample_id}."
            )

        if auto_move not in MOVES:
            raise ValueError(
                f"Invalid or missing automatic move '{auto_move}' "
                f"for sample {sample_id}."
            )

        if auto_timing not in TIMINGS:
            raise ValueError(
                f"Invalid or missing automatic timing '{auto_timing}' "
                f"for sample {sample_id}."
            )

        for other in annotator_rows[1:]:
            if get_auto_stage(other) != auto_stage:
                raise ValueError(
                    f"Automatic stage mismatch between annotation files "
                    f"for sample {sample_id}."
                )

            if get_auto_move(other) != auto_move:
                raise ValueError(
                    f"Automatic move mismatch between annotation files "
                    f"for sample {sample_id}."
                )

            if get_auto_timing(other) != auto_timing:
                raise ValueError(
                    f"Automatic timing mismatch between annotation files "
                    f"for sample {sample_id}."
                )

        stage_consensus = strict_consensus_label(
            rows=annotator_rows,
            correct_field="stage_correct",
            correction_field="stage_correction",
            auto_value=auto_stage,
            allowed_values=STAGES,
            require_same_correction=require_same_correction,
            consensus_mode=resolved_consensus_mode,
        )

        move_consensus = strict_consensus_label(
            rows=annotator_rows,
            correct_field="move_correct",
            correction_field="move_correction",
            auto_value=auto_move,
            allowed_values=MOVES,
            require_same_correction=require_same_correction,
            consensus_mode=resolved_consensus_mode,
        )

        timing_consensus = strict_consensus_label(
            rows=annotator_rows,
            correct_field="timing_correct",
            correction_field="timing_correction",
            auto_value=auto_timing,
            allowed_values=TIMINGS,
            require_same_correction=require_same_correction,
            consensus_mode=resolved_consensus_mode,
        )

        rows_out.append(
            {
                "id": sample_id,
                "query_id": str(
                    base.get("query_id")
                    or sample_id
                ).strip(),
                "query": base.get("query", ""),
                "response": base.get("response", ""),
                "auto_stage": auto_stage,
                "auto_move": auto_move,
                "auto_timing": auto_timing,
                "human_stage_correct": stage_consensus[
                    "consensus_correct"
                ],
                "human_stage": stage_consensus[
                    "consensus_value"
                ],
                "human_stage_source": stage_consensus["source"],
                "stage_yes_count": stage_consensus["yes_count"],
                "stage_no_count": stage_consensus["no_count"],
                "stage_valid_count": stage_consensus["valid_count"],
                "human_move_correct": move_consensus[
                    "consensus_correct"
                ],
                "human_move": move_consensus[
                    "consensus_value"
                ],
                "human_move_source": move_consensus["source"],
                "move_yes_count": move_consensus["yes_count"],
                "move_no_count": move_consensus["no_count"],
                "move_valid_count": move_consensus["valid_count"],
                "human_timing_correct": timing_consensus[
                    "consensus_correct"
                ],
                "human_timing": timing_consensus[
                    "consensus_value"
                ],
                "human_timing_source": timing_consensus["source"],
                "timing_yes_count": timing_consensus["yes_count"],
                "timing_no_count": timing_consensus["no_count"],
                "timing_valid_count": timing_consensus["valid_count"],
            }
        )

    consensus_df = pd.DataFrame(rows_out)

    meta = {
        "annotators": [
            item["annotator"]
            for item in loaded
        ],
        "annotation_files": [
            item["path"]
            for item in loaded
        ],
        "n_common": int(len(common_ids)),
        "n_annotators": int(len(annotation_paths)),
        "requested_consensus_mode": consensus_mode,
        "resolved_consensus_mode": resolved_consensus_mode,
        "require_same_correction": bool(require_same_correction),
        "n_stage_valid_consensus": int(
            consensus_df["human_stage_correct"].isin(
                YES_NO
            ).sum()
        ),
        "n_move_valid_consensus": int(
            consensus_df["human_move_correct"].isin(
                YES_NO
            ).sum()
        ),
        "n_timing_valid_consensus": int(
            consensus_df["human_timing_correct"].isin(
                YES_NO
            ).sum()
        ),
    }

    return consensus_df, meta


# =============================================================================
# Automatic judgment loading and ID matching
# =============================================================================

def prepare_auto_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Automatic judgment file not found: {path}"
        )

    df = pd.read_csv(path).fillna("")

    if df.empty:
        raise ValueError(
            f"Automatic judgment file is empty: {path}"
        )

    if "query_id" in df.columns:
        df["id_query_only"] = df["query_id"].astype(str).str.strip()
    else:
        df["id_query_only"] = ""

    if "query_id" in df.columns and "response_id" in df.columns:
        df["id_query_response"] = (
            df["query_id"].astype(str).str.strip()
            + "__"
            + df["response_id"].astype(str).str.strip()
        )
    else:
        df["id_query_response"] = ""

    if "id" in df.columns:
        df["id_explicit"] = df["id"].astype(str).str.strip()
    else:
        df["id_explicit"] = ""

    if df["id_explicit"].astype(bool).any():
        df["id_for_calibration"] = df["id_explicit"]
    elif df["id_query_response"].astype(bool).any():
        df["id_for_calibration"] = df["id_query_response"]
    elif df["id_query_only"].astype(bool).any():
        df["id_for_calibration"] = df["id_query_only"]
    else:
        raise ValueError(
            "Automatic CSV must contain id, query_id, or query_id with response_id."
        )

    if "predicted_stage" in df.columns:
        df["auto_stage_for_calibration"] = (
            df["predicted_stage"].map(norm)
        )
    elif "auto_stage" in df.columns:
        df["auto_stage_for_calibration"] = (
            df["auto_stage"].map(norm)
        )
    else:
        raise ValueError(
            "Automatic CSV must contain predicted_stage or auto_stage."
        )

    if "predicted_move" in df.columns:
        df["auto_move_for_calibration"] = (
            df["predicted_move"].map(norm)
        )
    elif "auto_move" in df.columns:
        df["auto_move_for_calibration"] = (
            df["auto_move"].map(norm)
        )
    else:
        raise ValueError(
            "Automatic CSV must contain predicted_move or auto_move."
        )

    if "timing_label" in df.columns:
        df["auto_timing_for_calibration"] = (
            df["timing_label"].map(norm)
        )
    elif "auto_timing" in df.columns:
        df["auto_timing_for_calibration"] = (
            df["auto_timing"].map(norm)
        )
    else:
        raise ValueError(
            "Automatic CSV must contain timing_label or auto_timing."
        )

    if "is_well_timed" in df.columns:
        df["auto_is_well_timed_for_calibration"] = (
            as_bool_series(df["is_well_timed"])
        )
    else:
        df["auto_is_well_timed_for_calibration"] = (
            df["auto_timing_for_calibration"].eq("well_timed")
        )

    if "rank" in df.columns:
        df["rank_for_calibration"] = (
            pd.to_numeric(
                df["rank"],
                errors="coerce",
            )
            .fillna(-1)
            .astype(int)
        )
    else:
        df["rank_for_calibration"] = 1

    for column in [
        "stage_confidence",
        "move_confidence",
        "retrieval_score",
    ]:
        if column not in df.columns:
            df[column] = 0.0

        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .fillna(0.0)
        )

    df["combined_confidence"] = df[
        ["stage_confidence", "move_confidence"]
    ].min(axis=1)

    df["mean_confidence"] = df[
        ["stage_confidence", "move_confidence"]
    ].mean(axis=1)

    return df


def configure_auto_id_column(
    auto_df: pd.DataFrame,
    annotation_ids: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select legacy query_id, query_id__response_id, or explicit id."""
    output = auto_df.copy()
    expected = {str(value).strip() for value in annotation_ids}
    candidates = [
        ("explicit_id", "id_explicit"),
        ("query_id__response_id", "id_query_response"),
        ("query_id", "id_query_only"),
    ]
    scores = []
    for name, column in candidates:
        if column not in output.columns:
            continue
        values = {
            str(value).strip()
            for value in output.loc[output["rank_for_calibration"] == 1, column].tolist()
            if str(value).strip()
        }
        scores.append({"name": name, "column": column, "matched": len(expected & values), "expected": len(expected)})
    if not scores:
        raise ValueError("No usable automatic ID representation was found.")
    best = max(scores, key=lambda item: item["matched"])
    if best["matched"] != len(expected):
        raise ValueError(
            f"No automatic ID format matched every annotation ID. Best format {best['name']} matched {best['matched']}/{len(expected)}. Candidate results: {scores}"
        )
    output["id_for_calibration"] = output[best["column"]].astype(str).str.strip()
    return output, {
        "selected_id_format": best["name"],
        "selected_id_column": best["column"],
        "n_annotation_ids": len(expected),
        "candidate_matches": scores,
    }


def validate_annotation_ids_against_auto(
    auto_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
) -> Dict[str, Any]:
    rank1 = auto_df[
        auto_df["rank_for_calibration"] == 1
    ].copy()

    if rank1.empty:
        raise ValueError(
            "No rank-1 automatic judgments were found."
        )

    duplicate_ids = rank1[
        "id_for_calibration"
    ][rank1["id_for_calibration"].duplicated()].unique()

    if len(duplicate_ids):
        raise ValueError(
            "Rank-1 automatic judgment IDs must be unique. "
            f"Duplicate examples: {duplicate_ids[:10].tolist()}"
        )

    auto_ids = set(
        rank1["id_for_calibration"]
        .astype(str)
        .str.strip()
    )

    annotation_ids = set(
        consensus_df["id"]
        .astype(str)
        .str.strip()
    )

    missing_ids = sorted(annotation_ids - auto_ids)

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} annotation IDs do not match rank-1 "
            f"automatic judgment IDs. Examples: {missing_ids[:10]}"
        )

    return {
        "n_rank1_auto_ids": int(len(auto_ids)),
        "n_annotation_ids": int(len(annotation_ids)),
        "n_matched_annotation_ids": int(
            len(annotation_ids & auto_ids)
        ),
        "n_missing_annotation_ids": int(len(missing_ids)),
    }


def attach_human_consensus_to_auto(
    auto_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
) -> pd.DataFrame:
    out = auto_df.copy()
    consensus = consensus_df.copy()

    out["id_for_calibration"] = (
        out["id_for_calibration"]
        .astype(str)
        .str.strip()
    )
    consensus["id_for_calibration"] = (
        consensus["id"]
        .astype(str)
        .str.strip()
    )

    keep_columns = [
        "id_for_calibration",
        "human_stage_correct",
        "human_stage",
        "human_move_correct",
        "human_move",
        "human_timing_correct",
        "human_timing",
    ]

    out = out.merge(
        consensus[keep_columns],
        on="id_for_calibration",
        how="left",
        validate="many_to_one",
    )

    out["has_human_consensus"] = (
        out["human_stage_correct"].notna()
        | out["human_move_correct"].notna()
        | out["human_timing_correct"].notna()
    )

    for column in keep_columns[1:]:
        out[column] = out[column].fillna("")

    return out


# =============================================================================
# Timing rule engine
# =============================================================================

def timing_from_stage_move(stage: str, move: str) -> str:
    stage = norm(stage)
    move = norm(move)

    allowed = ALLOWED_MOVES.get(stage, set())

    if (
        stage == "crisis_safety"
        and move not in {
            "safety_referral",
            "grounding",
            "validation",
        }
    ):
        return "delayed_safety"

    if (
        stage in {
            "distress_disclosure",
            "high_emotional_intensity",
            "unclear_need",
        }
        and move == "practical_advice"
    ):
        return "premature_advice"

    if (
        stage in {
            "advice_seeking",
            "followup_problem_solving",
        }
        and move in {
            "validation",
            "empathy",
            "reflective_listening",
        }
    ):
        return "over_validation"

    if (
        stage == "unclear_need"
        and move not in allowed
    ):
        return "missing_clarification"

    if move in allowed:
        return "well_timed"

    return "stage_mismatch"


# =============================================================================
# Development / held-out split
# =============================================================================

def choose_stratification_series(
    dataframe: pd.DataFrame,
) -> Optional[pd.Series]:
    candidates = [
        "auto_timing",
        "human_timing_correct",
    ]

    for column in candidates:
        if column not in dataframe.columns:
            continue

        counts = dataframe[column].value_counts()

        if (
            len(counts) >= 2
            and int(counts.min()) >= 2
        ):
            return dataframe[column]

    return None


def split_consensus_data(
    consensus_df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError(
            "scikit-learn is required for held-out splitting."
        )

    if not 0.10 <= test_size <= 0.50:
        raise ValueError(
            "--heldout-size must be between 0.10 and 0.50."
        )

    valid = consensus_df[
        consensus_df["human_timing_correct"].isin(
            YES_NO
        )
    ].copy()

    if len(valid) < 30:
        raise ValueError(
            "At least 30 examples with valid timing consensus are "
            "required for a held-out calibration evaluation."
        )

    stratification = choose_stratification_series(valid)

    development_df, heldout_df = train_test_split(
        valid,
        test_size=test_size,
        random_state=random_state,
        stratify=stratification,
    )

    development_df = (
        development_df
        .sort_values("id")
        .reset_index(drop=True)
    )

    heldout_df = (
        heldout_df
        .sort_values("id")
        .reset_index(drop=True)
    )

    split_meta = {
        "random_state": int(random_state),
        "requested_heldout_fraction": float(test_size),
        "n_total_valid": int(len(valid)),
        "n_development": int(len(development_df)),
        "n_heldout": int(len(heldout_df)),
        "actual_heldout_fraction": float(
            len(heldout_df) / len(valid)
        ),
        "stratification_column": (
            stratification.name
            if stratification is not None
            else None
        ),
        "development_auto_timing_distribution": (
            development_df["auto_timing"]
            .value_counts()
            .to_dict()
        ),
        "heldout_auto_timing_distribution": (
            heldout_df["auto_timing"]
            .value_counts()
            .to_dict()
        ),
    }

    return development_df, heldout_df, split_meta


# =============================================================================
# Correction rule learning
# =============================================================================

def learn_stage_rules(
    development_df: pd.DataFrame,
    min_support: int,
    min_error_rate: float,
) -> Dict[str, Dict[str, Any]]:
    rules: Dict[str, Dict[str, Any]] = {}

    for auto_stage, group in development_df.groupby(
        "auto_stage"
    ):
        valid = group[
            group["human_stage_correct"].isin(
                YES_NO
            )
        ]

        if valid.empty:
            continue

        wrong = valid[
            valid["human_stage_correct"] == "no"
        ]

        error_rate = len(wrong) / len(valid)

        corrections = [
            norm(value)
            for value in wrong["human_stage"].tolist()
            if (
                norm(value) in STAGES
                and norm(value) != norm(auto_stage)
            )
        ]

        if not corrections:
            continue

        correction, support = Counter(
            corrections
        ).most_common(1)[0]

        if (
            support >= min_support
            and error_rate >= min_error_rate
        ):
            rules[norm(auto_stage)] = {
                "to": correction,
                "support": int(support),
                "n_valid": int(len(valid)),
                "n_wrong": int(len(wrong)),
                "error_rate": round(
                    float(error_rate),
                    4,
                ),
            }

    return rules


def learn_conditional_move_rules(
    development_df: pd.DataFrame,
    min_support: int,
    min_error_rate: float,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    rules: Dict[Tuple[str, str], Dict[str, Any]] = {}

    grouped = development_df.groupby(
        ["auto_stage", "auto_move"]
    )

    for (auto_stage, auto_move), group in grouped:
        valid = group[
            group["human_move_correct"].isin(
                YES_NO
            )
        ]

        if valid.empty:
            continue

        wrong = valid[
            valid["human_move_correct"] == "no"
        ]

        error_rate = len(wrong) / len(valid)

        corrections = [
            norm(value)
            for value in wrong["human_move"].tolist()
            if (
                norm(value) in MOVES
                and norm(value) != norm(auto_move)
            )
        ]

        if not corrections:
            continue

        correction, support = Counter(
            corrections
        ).most_common(1)[0]

        if (
            support >= min_support
            and error_rate >= min_error_rate
        ):
            key = (
                norm(auto_stage),
                norm(auto_move),
            )

            rules[key] = {
                "to": correction,
                "support": int(support),
                "n_valid": int(len(valid)),
                "n_wrong": int(len(wrong)),
                "error_rate": round(
                    float(error_rate),
                    4,
                ),
            }

    return rules


def learn_direct_timing_rules(
    development_df: pd.DataFrame,
    min_support: int,
    min_error_rate: float,
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """
    High-overfit-risk ablation.

    The rule is conditioned on automatic stage + move + timing instead of only
    timing, but it still directly changes final labels and should not be used
    as the principal calibrated method.
    """
    rules: Dict[
        Tuple[str, str, str],
        Dict[str, Any],
    ] = {}

    grouped = development_df.groupby(
        ["auto_stage", "auto_move", "auto_timing"]
    )

    for key_values, group in grouped:
        valid = group[
            group["human_timing_correct"].isin(
                YES_NO
            )
        ]

        if valid.empty:
            continue

        wrong = valid[
            valid["human_timing_correct"] == "no"
        ]

        error_rate = len(wrong) / len(valid)

        corrections = [
            norm(value)
            for value in wrong["human_timing"].tolist()
            if (
                norm(value) in TIMINGS
                and norm(value) != norm(key_values[2])
            )
        ]

        if not corrections:
            continue

        correction, support = Counter(
            corrections
        ).most_common(1)[0]

        if (
            support >= min_support
            and error_rate >= min_error_rate
        ):
            key = tuple(
                norm(value)
                for value in key_values
            )

            rules[key] = {
                "to": correction,
                "support": int(support),
                "n_valid": int(len(valid)),
                "n_wrong": int(len(wrong)),
                "error_rate": round(
                    float(error_rate),
                    4,
                ),
            }

    return rules


def apply_stage_rule(
    stage: str,
    rules: Dict[str, Dict[str, Any]],
) -> Tuple[str, str]:
    stage = norm(stage)

    if stage in rules:
        return (
            rules[stage]["to"],
            "stage_rule_corrected",
        )

    return stage, "kept"


def apply_conditional_move_rule(
    original_stage: str,
    original_move: str,
    rules: Dict[
        Tuple[str, str],
        Dict[str, Any],
    ],
) -> Tuple[str, str]:
    key = (
        norm(original_stage),
        norm(original_move),
    )

    if key in rules:
        return (
            rules[key]["to"],
            "conditional_move_rule_corrected",
        )

    return norm(original_move), "kept"


def apply_direct_timing_rule(
    original_stage: str,
    original_move: str,
    original_timing: str,
    rules: Dict[
        Tuple[str, str, str],
        Dict[str, Any],
    ],
) -> Tuple[str, str]:
    key = (
        norm(original_stage),
        norm(original_move),
        norm(original_timing),
    )

    if key in rules:
        return (
            rules[key]["to"],
            "direct_timing_rule_corrected",
        )

    return norm(original_timing), "kept"



# =============================================================================
# Do-no-harm rule selection
# =============================================================================

def timing_accuracy_and_balanced_accuracy(
    dataframe: pd.DataFrame,
    prediction_column: str,
) -> Tuple[float, float]:
    valid = dataframe[
        dataframe["human_timing"].isin(TIMINGS)
    ].copy()

    if valid.empty:
        return float("nan"), float("nan")

    accuracy = float(
        (
            valid[prediction_column]
            == valid["human_timing"]
        ).mean()
    )

    recalls: List[float] = []
    for label in sorted(TIMINGS):
        subset = valid[
            valid["human_timing"] == label
        ]
        if subset.empty:
            continue
        recalls.append(
            float(
                (
                    subset[prediction_column]
                    == subset["human_timing"]
                ).mean()
            )
        )

    balanced = (
        float(np.mean(recalls))
        if recalls
        else float("nan")
    )

    return accuracy, balanced


def predict_recomputed_timing_on_consensus(
    dataframe: pd.DataFrame,
    stage_rules: Dict[str, Dict[str, Any]],
    move_rules: Dict[Tuple[str, str], Dict[str, Any]],
) -> pd.Series:
    predictions: List[str] = []

    for _, row in dataframe.iterrows():
        original_stage = norm(row["auto_stage"])
        original_move = norm(row["auto_move"])

        stage, _ = apply_stage_rule(
            original_stage,
            stage_rules,
        )
        move, _ = apply_conditional_move_rule(
            original_stage,
            original_move,
            move_rules,
        )

        predictions.append(
            timing_from_stage_move(stage, move)
        )

    return pd.Series(
        predictions,
        index=dataframe.index,
        dtype="object",
    )


def predict_direct_timing_on_consensus(
    dataframe: pd.DataFrame,
    timing_rules: Dict[
        Tuple[str, str, str],
        Dict[str, Any],
    ],
) -> pd.Series:
    predictions: List[str] = []

    for _, row in dataframe.iterrows():
        timing, _ = apply_direct_timing_rule(
            original_stage=norm(row["auto_stage"]),
            original_move=norm(row["auto_move"]),
            original_timing=norm(row["auto_timing"]),
            rules=timing_rules,
        )
        predictions.append(timing)

    return pd.Series(
        predictions,
        index=dataframe.index,
        dtype="object",
    )


def split_inner_development(
    development_df: pd.DataFrame,
    validation_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid = development_df[
        development_df["human_timing"].isin(TIMINGS)
    ].copy()

    if len(valid) < 20:
        return valid.copy(), valid.copy()

    stratification = choose_stratification_series(valid)

    inner_train, inner_validation = train_test_split(
        valid,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratification,
    )

    return (
        inner_train.reset_index(drop=True),
        inner_validation.reset_index(drop=True),
    )


def accept_candidate_without_harm(
    validation_df: pd.DataFrame,
    current_predictions: pd.Series,
    candidate_predictions: pd.Series,
    affected_mask: pd.Series,
    min_affected: int,
    min_accuracy_gain: float,
) -> Tuple[bool, Dict[str, Any]]:
    affected_count = int(affected_mask.sum())

    current_eval = validation_df.copy()
    candidate_eval = validation_df.copy()
    current_eval["_prediction"] = current_predictions
    candidate_eval["_prediction"] = candidate_predictions

    current_accuracy, current_balanced = (
        timing_accuracy_and_balanced_accuracy(
            current_eval,
            "_prediction",
        )
    )
    candidate_accuracy, candidate_balanced = (
        timing_accuracy_and_balanced_accuracy(
            candidate_eval,
            "_prediction",
        )
    )

    current_correct = int(
        (
            current_predictions
            == validation_df["human_timing"]
        ).sum()
    )
    candidate_correct = int(
        (
            candidate_predictions
            == validation_df["human_timing"]
        ).sum()
    )

    accuracy_gain = candidate_accuracy - current_accuracy
    balanced_gain = candidate_balanced - current_balanced

    accepted = bool(
        affected_count >= min_affected
        and candidate_correct > current_correct
        and accuracy_gain >= min_accuracy_gain
        and balanced_gain >= -1e-12
    )

    diagnostics = {
        "accepted": accepted,
        "affected_validation_examples": affected_count,
        "current_correct": current_correct,
        "candidate_correct": candidate_correct,
        "current_accuracy": round(current_accuracy, 6),
        "candidate_accuracy": round(candidate_accuracy, 6),
        "accuracy_gain": round(accuracy_gain, 6),
        "current_balanced_accuracy": round(current_balanced, 6),
        "candidate_balanced_accuracy": round(candidate_balanced, 6),
        "balanced_accuracy_gain": round(balanced_gain, 6),
    }

    return accepted, diagnostics


def learn_do_no_harm_recompute_rules(
    development_df: pd.DataFrame,
    min_support: int,
    min_error_rate: float,
    validation_size: float,
    random_state: int,
    min_affected: int,
    min_accuracy_gain: float,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[str, Any],
]:
    inner_train, inner_validation = split_inner_development(
        development_df=development_df,
        validation_size=validation_size,
        random_state=random_state,
    )

    candidate_stage_rules = learn_stage_rules(
        development_df=inner_train,
        min_support=min_support,
        min_error_rate=min_error_rate,
    )
    candidate_move_rules = learn_conditional_move_rules(
        development_df=inner_train,
        min_support=min_support,
        min_error_rate=min_error_rate,
    )

    accepted_stage_rules: Dict[str, Dict[str, Any]] = {}
    accepted_move_rules: Dict[
        Tuple[str, str],
        Dict[str, Any],
    ] = {}
    decisions: List[Dict[str, Any]] = []

    current_predictions = inner_validation[
        "auto_timing"
    ].map(norm)

    # Greedily retain only rules that improve validation correctness while
    # preserving balanced accuracy.
    candidates: List[Tuple[str, Any, Dict[str, Any]]] = []
    candidates.extend(
        ("stage", key, value)
        for key, value in candidate_stage_rules.items()
    )
    candidates.extend(
        ("move", key, value)
        for key, value in candidate_move_rules.items()
    )
    candidates.sort(
        key=lambda item: (
            -int(item[2].get("support", 0)),
            -float(item[2].get("error_rate", 0.0)),
            str(item[1]),
        )
    )

    for rule_type, key, rule in candidates:
        trial_stage = dict(accepted_stage_rules)
        trial_move = dict(accepted_move_rules)

        if rule_type == "stage":
            trial_stage[key] = rule
            affected_mask = (
                inner_validation["auto_stage"].map(norm)
                == norm(key)
            )
        else:
            trial_move[key] = rule
            affected_mask = (
                inner_validation["auto_stage"].map(norm)
                == norm(key[0])
            ) & (
                inner_validation["auto_move"].map(norm)
                == norm(key[1])
            )

        candidate_predictions = (
            predict_recomputed_timing_on_consensus(
                dataframe=inner_validation,
                stage_rules=trial_stage,
                move_rules=trial_move,
            )
        )

        accepted, diagnostics = accept_candidate_without_harm(
            validation_df=inner_validation,
            current_predictions=current_predictions,
            candidate_predictions=candidate_predictions,
            affected_mask=affected_mask,
            min_affected=min_affected,
            min_accuracy_gain=min_accuracy_gain,
        )

        decisions.append(
            {
                "rule_type": rule_type,
                "rule_key": (
                    "|||".join(key)
                    if isinstance(key, tuple)
                    else str(key)
                ),
                "rule_to": rule.get("to"),
                **diagnostics,
            }
        )

        if accepted:
            accepted_stage_rules = trial_stage
            accepted_move_rules = trial_move
            current_predictions = candidate_predictions

    final_eval = inner_validation.copy()
    final_eval["_prediction"] = current_predictions
    final_accuracy, final_balanced = (
        timing_accuracy_and_balanced_accuracy(
            final_eval,
            "_prediction",
        )
    )

    baseline_eval = inner_validation.copy()
    baseline_eval["_prediction"] = inner_validation[
        "auto_timing"
    ].map(norm)
    baseline_accuracy, baseline_balanced = (
        timing_accuracy_and_balanced_accuracy(
            baseline_eval,
            "_prediction",
        )
    )

    metadata = {
        "enabled": True,
        "inner_train_size": int(len(inner_train)),
        "inner_validation_size": int(len(inner_validation)),
        "candidate_stage_rules": int(len(candidate_stage_rules)),
        "candidate_move_rules": int(len(candidate_move_rules)),
        "accepted_stage_rules": int(len(accepted_stage_rules)),
        "accepted_move_rules": int(len(accepted_move_rules)),
        "baseline_inner_accuracy": round(baseline_accuracy, 6),
        "safe_inner_accuracy": round(final_accuracy, 6),
        "baseline_inner_balanced_accuracy": round(
            baseline_balanced,
            6,
        ),
        "safe_inner_balanced_accuracy": round(
            final_balanced,
            6,
        ),
        "decisions": decisions,
    }

    return accepted_stage_rules, accepted_move_rules, metadata


def learn_do_no_harm_direct_rules(
    development_df: pd.DataFrame,
    min_support: int,
    min_error_rate: float,
    validation_size: float,
    random_state: int,
    min_affected: int,
    min_accuracy_gain: float,
) -> Tuple[
    Dict[Tuple[str, str, str], Dict[str, Any]],
    Dict[str, Any],
]:
    inner_train, inner_validation = split_inner_development(
        development_df=development_df,
        validation_size=validation_size,
        random_state=random_state,
    )

    candidate_rules = learn_direct_timing_rules(
        development_df=inner_train,
        min_support=min_support,
        min_error_rate=min_error_rate,
    )

    accepted_rules: Dict[
        Tuple[str, str, str],
        Dict[str, Any],
    ] = {}
    decisions: List[Dict[str, Any]] = []
    current_predictions = inner_validation[
        "auto_timing"
    ].map(norm)

    ordered = sorted(
        candidate_rules.items(),
        key=lambda item: (
            -int(item[1].get("support", 0)),
            -float(item[1].get("error_rate", 0.0)),
            str(item[0]),
        ),
    )

    for key, rule in ordered:
        trial_rules = dict(accepted_rules)
        trial_rules[key] = rule

        candidate_predictions = predict_direct_timing_on_consensus(
            dataframe=inner_validation,
            timing_rules=trial_rules,
        )

        affected_mask = (
            inner_validation["auto_stage"].map(norm)
            == norm(key[0])
        ) & (
            inner_validation["auto_move"].map(norm)
            == norm(key[1])
        ) & (
            inner_validation["auto_timing"].map(norm)
            == norm(key[2])
        )

        accepted, diagnostics = accept_candidate_without_harm(
            validation_df=inner_validation,
            current_predictions=current_predictions,
            candidate_predictions=candidate_predictions,
            affected_mask=affected_mask,
            min_affected=min_affected,
            min_accuracy_gain=min_accuracy_gain,
        )

        decisions.append(
            {
                "rule_type": "direct_timing",
                "rule_key": "|||".join(key),
                "rule_to": rule.get("to"),
                **diagnostics,
            }
        )

        if accepted:
            enriched_rule = dict(rule)
            enriched_rule["validation_affected"] = diagnostics[
                "affected_validation_examples"
            ]
            enriched_rule["validation_accuracy_gain"] = diagnostics[
                "accuracy_gain"
            ]
            enriched_rule["validation_balanced_accuracy_gain"] = diagnostics[
                "balanced_accuracy_gain"
            ]
            enriched_rule["validation_current_correct"] = diagnostics[
                "current_correct"
            ]
            enriched_rule["validation_candidate_correct"] = diagnostics[
                "candidate_correct"
            ]
            accepted_rules = dict(accepted_rules)
            accepted_rules[key] = enriched_rule
            current_predictions = candidate_predictions

    metadata = {
        "enabled": True,
        "inner_train_size": int(len(inner_train)),
        "inner_validation_size": int(len(inner_validation)),
        "candidate_rules": int(len(candidate_rules)),
        "accepted_rules": int(len(accepted_rules)),
        "decisions": decisions,
    }

    return accepted_rules, metadata

# =============================================================================
# Confidence and isotonic reliability
# =============================================================================

def compute_confidence_flags(
    dataframe: pd.DataFrame,
    stage_percentile: float,
    move_percentile: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    output = dataframe.copy()

    stage_values = output.loc[
        output["stage_confidence"] >= 0,
        "stage_confidence",
    ]

    move_values = output.loc[
        output["move_confidence"] >= 0,
        "move_confidence",
    ]

    stage_threshold = (
        float(
            stage_values.quantile(
                stage_percentile / 100.0
            )
        )
        if len(stage_values)
        else None
    )

    move_threshold = (
        float(
            move_values.quantile(
                move_percentile / 100.0
            )
        )
        if len(move_values)
        else None
    )

    output["stage_low_confidence"] = (
        output["stage_confidence"] < stage_threshold
        if stage_threshold is not None
        else False
    )

    output["move_low_confidence"] = (
        output["move_confidence"] < move_threshold
        if move_threshold is not None
        else False
    )

    output["low_confidence_calibrated"] = (
        output["stage_low_confidence"]
        | output["move_low_confidence"]
    )

    meta = {
        "stage_percentile": float(stage_percentile),
        "move_percentile": float(move_percentile),
        "stage_threshold": stage_threshold,
        "move_threshold": move_threshold,
        "full_corpus_low_confidence_rate": float(
            output[
                "low_confidence_calibrated"
            ].mean()
        ),
    }

    return output, meta


def minmax_scale(
    values: np.ndarray,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    denominator = maximum - minimum

    if abs(denominator) < 1e-12:
        return np.full_like(
            values,
            0.5,
            dtype=float,
        )

    return (
        np.asarray(values, dtype=float)
        - minimum
    ) / denominator


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    targets = np.asarray(
        targets,
        dtype=float,
    )
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(targets) == 0:
        return float("nan")

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    total = 0.0

    for index in range(n_bins):
        lower = edges[index]
        upper = edges[index + 1]

        if index == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        if not np.any(mask):
            continue

        bin_accuracy = float(
            np.mean(targets[mask])
        )
        bin_confidence = float(
            np.mean(probabilities[mask])
        )

        total += float(
            np.mean(mask)
        ) * abs(
            bin_accuracy - bin_confidence
        )

    return float(total)


def fit_isotonic_model(
    development_df: pd.DataFrame,
    target_column: str,
    feature_column: str,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    if not SKLEARN_AVAILABLE:
        return None, {
            "available": False,
            "reason": "scikit-learn is unavailable",
        }

    train = development_df[
        development_df[target_column].isin(
            YES_NO
        )
    ].copy()

    if (
        len(train) < 20
        or train[target_column].nunique() < 2
    ):
        return None, {
            "available": False,
            "reason": (
                "Insufficient development labels or only one target class"
            ),
            "n_train": int(len(train)),
        }

    x_raw = (
        train[feature_column]
        .astype(float)
        .to_numpy()
    )

    y = (
        train[target_column]
        .eq("yes")
        .astype(int)
        .to_numpy()
    )

    minimum = float(np.min(x_raw))
    maximum = float(np.max(x_raw))
    x = minmax_scale(
        x_raw,
        minimum,
        maximum,
    )

    model = IsotonicRegression(
        out_of_bounds="clip",
        y_min=0.0,
        y_max=1.0,
    )
    model.fit(x, y)

    probabilities = model.predict(x)

    brier = (
        float(
            brier_score_loss(
                y,
                probabilities,
            )
        )
        if brier_score_loss is not None
        else None
    )

    wrapped = {
        "model": model,
        "minimum": minimum,
        "maximum": maximum,
        "feature_column": feature_column,
        "target_column": target_column,
    }

    meta = {
        "available": True,
        "n_train": int(len(train)),
        "positive_rate": float(np.mean(y)),
        "development_brier": brier,
        "development_ece": float(
            expected_calibration_error(
                y,
                probabilities,
            )
        ),
        "feature_minimum": minimum,
        "feature_maximum": maximum,
        "note": (
            "The isotonic model was fitted only on the development subset."
        ),
    }

    return wrapped, meta


def predict_isotonic(
    wrapped_model: Optional[Dict[str, Any]],
    dataframe: pd.DataFrame,
    output_column: str,
) -> pd.DataFrame:
    output = dataframe.copy()

    if wrapped_model is None:
        output[output_column] = 0.5
        return output

    feature_column = wrapped_model[
        "feature_column"
    ]

    raw = (
        output[feature_column]
        .astype(float)
        .to_numpy()
    )

    scaled = minmax_scale(
        raw,
        wrapped_model["minimum"],
        wrapped_model["maximum"],
    )

    output[output_column] = wrapped_model[
        "model"
    ].predict(scaled)

    return output


def add_isotonic_reliability(
    base_df: pd.DataFrame,
    development_df: pd.DataFrame,
    reliability_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    stage_model, stage_meta = fit_isotonic_model(
        development_df=development_df,
        target_column="human_stage_correct",
        feature_column="stage_confidence",
    )

    move_model, move_meta = fit_isotonic_model(
        development_df=development_df,
        target_column="human_move_correct",
        feature_column="move_confidence",
    )

    timing_model, timing_meta = fit_isotonic_model(
        development_df=development_df,
        target_column="human_timing_correct",
        feature_column="combined_confidence",
    )

    output = predict_isotonic(
        stage_model,
        base_df,
        "isotonic_stage_reliability",
    )

    output = predict_isotonic(
        move_model,
        output,
        "isotonic_move_reliability",
    )

    output = predict_isotonic(
        timing_model,
        output,
        "isotonic_timing_reliability",
    )

    output["isotonic_overall_reliability"] = output[
        [
            "isotonic_stage_reliability",
            "isotonic_move_reliability",
            "isotonic_timing_reliability",
        ]
    ].min(axis=1)

    output["isotonic_high_reliability"] = (
        output[
            "isotonic_overall_reliability"
        ]
        >= reliability_threshold
    )

    meta = {
        "reliability_threshold": float(
            reliability_threshold
        ),
        "stage": stage_meta,
        "move": move_meta,
        "timing": timing_meta,
        "full_corpus_high_reliability_rate": float(
            output[
                "isotonic_high_reliability"
            ].mean()
        ),
    }

    model_bundle = {
        "stage": stage_model,
        "move": move_model,
        "timing": timing_model,
    }

    return output, meta, model_bundle


# =============================================================================
# Apply calibration methods
# =============================================================================

def apply_safe_keep_correct_review_policy(
    base_df: pd.DataFrame,
    timing_rules: Dict[
        Tuple[str, str, str],
        Dict[str, Any],
    ],
    keep_threshold: float,
    correction_max_original_reliability: float,
    min_rule_support: int,
    min_rule_error_rate: float,
    min_rule_validation_gain: float,
) -> pd.DataFrame:
    """
    Three-way safety policy:
      1. KEEP high-reliability original labels.
      2. CORRECT only with an internally validated direct rule and when the
         original label is not already highly reliable.
      3. REVIEW all remaining uncertain cases while preserving the original
         label in the exported prediction.

    Human review is an abstention decision, not an automatic relabelling.
    """
    output = base_df.copy()

    actions: List[str] = []
    final_timings: List[str] = []
    sources: List[str] = []
    review_flags: List[bool] = []
    rule_supports: List[int] = []
    rule_gains: List[float] = []

    for _, row in output.iterrows():
        stage = norm(row["auto_stage_for_calibration"])
        move = norm(row["auto_move_for_calibration"])
        original = norm(row["auto_timing_for_calibration"])
        reliability = safe_float(
            row.get("isotonic_overall_reliability", 0.5),
            0.5,
        )

        key = (stage, move, original)
        rule = timing_rules.get(key)

        if reliability >= keep_threshold:
            action = "keep"
            final = original
            source = "high_reliability_keep"
            needs_review = False
            support = 0
            gain = 0.0

        elif rule is not None:
            support = int(rule.get("support", 0))
            error_rate = float(rule.get("error_rate", 0.0))
            gain = float(rule.get("validation_accuracy_gain", 0.0))
            corrected = norm(rule.get("to", original))

            strong_rule = (
                support >= min_rule_support
                and error_rate >= min_rule_error_rate
                and gain >= min_rule_validation_gain
                and reliability <= correction_max_original_reliability
                and corrected in TIMINGS
                and corrected != original
            )

            if strong_rule:
                action = "correct"
                final = corrected
                source = "validated_direct_rule"
                needs_review = False
            else:
                action = "review"
                final = original
                source = "rule_not_strong_enough_review"
                needs_review = True

        else:
            action = "review"
            final = original
            source = "uncertain_no_validated_rule"
            needs_review = True
            support = 0
            gain = 0.0

        actions.append(action)
        final_timings.append(final)
        sources.append(source)
        review_flags.append(needs_review)
        rule_supports.append(support)
        rule_gains.append(gain)

    output["calibration_method"] = "safe_keep_correct_review"
    output["calibrated_stage"] = output[
        "auto_stage_for_calibration"
    ].map(norm)
    output["stage_calibration_source"] = "kept"
    output["calibrated_move"] = output[
        "auto_move_for_calibration"
    ].map(norm)
    output["move_calibration_source"] = "kept"
    output["calibrated_timing"] = final_timings
    output["timing_calibration_source"] = sources
    output["policy_action"] = actions
    output["needs_human_review"] = review_flags
    output["policy_accepted"] = ~output["needs_human_review"]
    output["policy_rule_support"] = rule_supports
    output["policy_rule_validation_gain"] = rule_gains
    output["calibrated_is_well_timed"] = output[
        "calibrated_timing"
    ].eq("well_timed")
    output["stage_changed"] = False
    output["move_changed"] = False
    output["timing_changed"] = (
        output["calibrated_timing"]
        != output["auto_timing_for_calibration"]
    )

    return output


def apply_calibration_method(
    base_df: pd.DataFrame,
    method: str,
    stage_rules: Optional[
        Dict[str, Dict[str, Any]]
    ] = None,
    move_rules: Optional[
        Dict[Tuple[str, str], Dict[str, Any]]
    ] = None,
    timing_rules: Optional[
        Dict[
            Tuple[str, str, str],
            Dict[str, Any],
        ]
    ] = None,
) -> pd.DataFrame:
    stage_rules = stage_rules or {}
    move_rules = move_rules or {}
    timing_rules = timing_rules or {}

    output = base_df.copy()

    calibrated_stages: List[str] = []
    stage_sources: List[str] = []
    calibrated_moves: List[str] = []
    move_sources: List[str] = []
    calibrated_timings: List[str] = []
    timing_sources: List[str] = []

    for _, row in output.iterrows():
        original_stage = norm(
            row["auto_stage_for_calibration"]
        )
        original_move = norm(
            row["auto_move_for_calibration"]
        )
        original_timing = norm(
            row["auto_timing_for_calibration"]
        )

        if method in RULE_METHODS:
            stage, stage_source = apply_stage_rule(
                original_stage,
                stage_rules,
            )

            move, move_source = (
                apply_conditional_move_rule(
                    original_stage,
                    original_move,
                    move_rules,
                )
            )
        else:
            stage = original_stage
            stage_source = "kept"
            move = original_move
            move_source = "kept"

        if method == "human_rules_direct":
            timing, timing_source = (
                apply_direct_timing_rule(
                    original_stage,
                    original_move,
                    original_timing,
                    timing_rules,
                )
            )

        elif method in RECOMPUTE_METHODS:
            timing = timing_from_stage_move(
                stage,
                move,
            )
            timing_source = (
                "recomputed_from_calibrated_stage_move"
            )

        else:
            timing = original_timing
            timing_source = "kept"

        calibrated_stages.append(stage)
        stage_sources.append(stage_source)
        calibrated_moves.append(move)
        move_sources.append(move_source)
        calibrated_timings.append(timing)
        timing_sources.append(timing_source)

    output["calibration_method"] = method
    output["calibrated_stage"] = calibrated_stages
    output["stage_calibration_source"] = stage_sources
    output["calibrated_move"] = calibrated_moves
    output["move_calibration_source"] = move_sources
    output["calibrated_timing"] = calibrated_timings
    output["timing_calibration_source"] = timing_sources
    output["calibrated_is_well_timed"] = output[
        "calibrated_timing"
    ].eq("well_timed")

    output["stage_changed"] = (
        output["calibrated_stage"]
        != output["auto_stage_for_calibration"]
    )

    output["move_changed"] = (
        output["calibrated_move"]
        != output["auto_move_for_calibration"]
    )

    output["timing_changed"] = (
        output["calibrated_timing"]
        != output["auto_timing_for_calibration"]
    )

    return output


# =============================================================================
# Full-corpus descriptive summaries
# =============================================================================

def summarize_full_corpus(
    dataframe: pd.DataFrame,
    method: str,
) -> Dict[str, Any]:
    rank1 = dataframe[
        dataframe["rank_for_calibration"] == 1
    ].copy()

    if rank1.empty:
        raise ValueError(
            "No rank-1 rows were available for full-corpus summary."
        )

    original_rate = float(
        rank1[
            "auto_is_well_timed_for_calibration"
        ].mean()
    )

    calibrated_rate = float(
        rank1[
            "calibrated_is_well_timed"
        ].mean()
    )

    high_confidence = rank1[
        ~rank1["low_confidence_calibrated"]
    ]

    high_confidence_rate = (
        float(
            high_confidence[
                "calibrated_is_well_timed"
            ].mean()
        )
        if len(high_confidence)
        else None
    )

    isotonic_subset = (
        rank1[
            rank1["isotonic_high_reliability"]
        ]
        if "isotonic_high_reliability" in rank1.columns
        else pd.DataFrame()
    )

    isotonic_rate = (
        float(
            isotonic_subset[
                "calibrated_is_well_timed"
            ].mean()
        )
        if len(isotonic_subset)
        else None
    )

    isotonic_coverage = (
        float(
            rank1[
                "isotonic_high_reliability"
            ].mean()
        )
        if "isotonic_high_reliability" in rank1.columns
        else None
    )

    return {
        "method": method,
        "original_automatic_well_timed_rate_at_1": round(
            original_rate,
            4,
        ),
        "calibrated_automatic_well_timed_rate_at_1": round(
            calibrated_rate,
            4,
        ),
        "delta_automatic_well_timed_rate_at_1": round(
            calibrated_rate - original_rate,
            4,
        ),
        "high_confidence_automatic_well_timed_rate_at_1": (
            round(high_confidence_rate, 4)
            if high_confidence_rate is not None
            else None
        ),
        "high_confidence_coverage": round(
            float(
                (
                    ~rank1[
                        "low_confidence_calibrated"
                    ]
                ).mean()
            ),
            4,
        ),
        "isotonic_high_reliability_automatic_well_timed_rate_at_1": (
            round(isotonic_rate, 4)
            if isotonic_rate is not None
            else None
        ),
        "isotonic_high_reliability_coverage": (
            round(isotonic_coverage, 4)
            if isotonic_coverage is not None
            else None
        ),
        "stage_changes_rank1": int(
            rank1["stage_changed"].sum()
        ),
        "move_changes_rank1": int(
            rank1["move_changed"].sum()
        ),
        "timing_changes_rank1": int(
            rank1["timing_changed"].sum()
        ),
        "stage_changes_all_rows": int(
            dataframe["stage_changed"].sum()
        ),
        "move_changes_all_rows": int(
            dataframe["move_changed"].sum()
        ),
        "timing_changes_all_rows": int(
            dataframe["timing_changed"].sum()
        ),
        "policy_keep_count_rank1": int(
            rank1["policy_action"].eq("keep").sum()
        ) if "policy_action" in rank1.columns else None,
        "policy_correct_count_rank1": int(
            rank1["policy_action"].eq("correct").sum()
        ) if "policy_action" in rank1.columns else None,
        "policy_review_count_rank1": int(
            rank1["policy_action"].eq("review").sum()
        ) if "policy_action" in rank1.columns else None,
        "policy_accepted_coverage_rank1": round(
            float(rank1["policy_accepted"].mean()), 4
        ) if "policy_accepted" in rank1.columns else None,
        "n_rank1": int(len(rank1)),
        "n_rows": int(len(dataframe)),
        "interpretation": (
            "These are automatic label-distribution statistics, not "
            "human-validated accuracy."
        ),
    }


# =============================================================================
# Held-out human evaluation
# =============================================================================

def build_human_reference(
    row: pd.Series,
    label_type: str,
) -> Optional[str]:
    auto_value = norm(row[f"auto_{label_type}"])
    correctness = norm(row[f"human_{label_type}_correct"])
    human_value = norm(row[f"human_{label_type}"])

    if correctness == "yes":
        return auto_value

    if (
        correctness == "no"
        and human_value
        and human_value != auto_value  # <-- FIX: a tie/no-correction fallback isn't a real reference
    ):
        return human_value

    return None


def make_heldout_auto_rows(
    auto_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
) -> pd.DataFrame:
    rank1 = auto_df[
        auto_df["rank_for_calibration"] == 1
    ].copy()

    wanted_ids = set(
        heldout_df["id"].astype(str)
    )

    selected = rank1[
        rank1["id_for_calibration"].isin(
            wanted_ids
        )
    ].copy()

    if len(selected) != len(wanted_ids):
        selected_ids = set(
            selected["id_for_calibration"]
        )

        missing = sorted(
            wanted_ids - selected_ids
        )

        raise ValueError(
            "Held-out examples could not all be matched to rank-1 "
            f"automatic judgments. Missing: {missing[:10]}"
        )

    return selected


def evaluate_method_on_heldout(
    method_output: pd.DataFrame,
    heldout_df: pd.DataFrame,
    method: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rank1 = method_output[
        method_output["rank_for_calibration"] == 1
    ].copy()

    human_columns = [
        "id",
        "auto_stage",
        "auto_move",
        "auto_timing",
        "human_stage_correct",
        "human_stage",
        "human_move_correct",
        "human_move",
        "human_timing_correct",
        "human_timing",
    ]

    human = heldout_df[
        human_columns
    ].copy()

    human["id_for_calibration"] = (
        human["id"]
        .astype(str)
        .str.strip()
    )

    evaluation = rank1.merge(
        human.drop(columns=["id"]),
        on="id_for_calibration",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_human"),
    )

    if len(evaluation) != len(heldout_df):
        raise ValueError(
            f"Held-out evaluation matched {len(evaluation)} of "
            f"{len(heldout_df)} examples for method {method}."
        )

    for label_type in [
        "stage",
        "move",
        "timing",
    ]:
        evaluation[
            f"human_reference_{label_type}"
        ] = evaluation.apply(
            lambda row: build_human_reference(
                row,
                label_type,
            ),
            axis=1,
        )

        reference_column = (
            f"human_reference_{label_type}"
        )

        evaluation[
            f"original_{label_type}_correct"
        ] = (
            evaluation[
                f"auto_{label_type}_for_calibration"
            ]
            == evaluation[reference_column]
        )

        evaluation[
            f"calibrated_{label_type}_correct"
        ] = (
            evaluation[
                f"calibrated_{label_type}"
            ]
            == evaluation[reference_column]
        )

    def accuracy(column: str) -> float:
        return float(
            evaluation[column].mean()
        )

    original_stage_accuracy = accuracy(
        "original_stage_correct"
    )
    calibrated_stage_accuracy = accuracy(
        "calibrated_stage_correct"
    )

    original_move_accuracy = accuracy(
        "original_move_correct"
    )
    calibrated_move_accuracy = accuracy(
        "calibrated_move_correct"
    )

    original_timing_accuracy = accuracy(
        "original_timing_correct"
    )
    calibrated_timing_accuracy = accuracy(
        "calibrated_timing_correct"
    )

    high_confidence = evaluation[
        ~evaluation["low_confidence_calibrated"]
    ]

    high_confidence_accuracy = (
        float(
            high_confidence[
                "calibrated_timing_correct"
            ].mean()
        )
        if len(high_confidence)
        else None
    )

    isotonic_subset = (
        evaluation[
            evaluation["isotonic_high_reliability"]
        ]
        if "isotonic_high_reliability" in evaluation.columns
        else pd.DataFrame()
    )

    isotonic_accuracy = (
        float(
            isotonic_subset[
                "calibrated_timing_correct"
            ].mean()
        )
        if len(isotonic_subset)
        else None
    )

    isotonic_coverage = (
        float(
            evaluation[
                "isotonic_high_reliability"
            ].mean()
        )
        if "isotonic_high_reliability" in evaluation.columns
        else None
    )

    policy_metrics: Dict[str, Any] = {}
    if "policy_action" in evaluation.columns:
        accepted = evaluation[evaluation["policy_accepted"]].copy()
        reviewed = evaluation[evaluation["needs_human_review"]].copy()
        corrected = evaluation[evaluation["policy_action"] == "correct"].copy()

        policy_metrics = {
            "policy_accepted_coverage": round(
                float(len(accepted) / len(evaluation)), 4
            ),
            "policy_review_rate": round(
                float(len(reviewed) / len(evaluation)), 4
            ),
            "policy_accepted_timing_accuracy": (
                round(float(accepted["calibrated_timing_correct"].mean()), 4)
                if len(accepted) else None
            ),
            "policy_n_keep": int(
                evaluation["policy_action"].eq("keep").sum()
            ),
            "policy_n_correct": int(len(corrected)),
            "policy_n_review": int(len(reviewed)),
            "policy_corrected_case_accuracy": (
                round(float(corrected["calibrated_timing_correct"].mean()), 4)
                if len(corrected) else None
            ),
        }

    summary = {
        "method": method,
        "n_heldout": int(len(evaluation)),
        "original_stage_accuracy": round(
            original_stage_accuracy,
            4,
        ),
        "calibrated_stage_accuracy": round(
            calibrated_stage_accuracy,
            4,
        ),
        "stage_accuracy_gain": round(
            calibrated_stage_accuracy
            - original_stage_accuracy,
            4,
        ),
        "original_move_accuracy": round(
            original_move_accuracy,
            4,
        ),
        "calibrated_move_accuracy": round(
            calibrated_move_accuracy,
            4,
        ),
        "move_accuracy_gain": round(
            calibrated_move_accuracy
            - original_move_accuracy,
            4,
        ),
        "original_timing_accuracy": round(
            original_timing_accuracy,
            4,
        ),
        "calibrated_timing_accuracy": round(
            calibrated_timing_accuracy,
            4,
        ),
        "timing_accuracy_gain": round(
            calibrated_timing_accuracy
            - original_timing_accuracy,
            4,
        ),
        "high_confidence_timing_accuracy": (
            round(high_confidence_accuracy, 4)
            if high_confidence_accuracy is not None
            else None
        ),
        "high_confidence_coverage": round(
            float(
                (
                    ~evaluation[
                        "low_confidence_calibrated"
                    ]
                ).mean()
            ),
            4,
        ),
        "isotonic_high_reliability_timing_accuracy": (
            round(isotonic_accuracy, 4)
            if isotonic_accuracy is not None
            else None
        ),
        "isotonic_high_reliability_coverage": (
            round(isotonic_coverage, 4)
            if isotonic_coverage is not None
            else None
        ),
        "interpretation": (
            "These accuracies are measured on an untouched held-out "
            "human-consensus subset."
        ),
    }

    summary.update(policy_metrics)
    return summary, evaluation


# =============================================================================
# Selective held-out reliability analysis
# =============================================================================

def heldout_risk_coverage_table(
    evaluation_df: pd.DataFrame,
    score_column: str,
    coverage_points: Optional[
        Sequence[float]
    ] = None,
) -> pd.DataFrame:
    if coverage_points is None:
        coverage_points = [
            1.00,
            0.90,
            0.80,
            0.70,
            0.60,
            0.50,
            0.40,
            0.30,
            0.20,
            0.10,
        ]

    if score_column not in evaluation_df.columns:
        return pd.DataFrame()

    ranked = (
        evaluation_df
        .sort_values(
            score_column,
            ascending=False,
        )
        .reset_index(drop=True)
    )

    total = len(ranked)
    rows: List[Dict[str, Any]] = []

    for target_coverage in coverage_points:
        accepted_count = max(
            1,
            int(round(
                target_coverage * total
            )),
        )

        subset = ranked.iloc[
            :accepted_count
        ]

        accuracy = float(
            subset[
                "calibrated_timing_correct"
            ].mean()
        )

        rows.append(
            {
                "score_column": score_column,
                "target_coverage": float(
                    target_coverage
                ),
                "actual_coverage": float(
                    accepted_count / total
                ),
                "n_accepted": int(
                    accepted_count
                ),
                "n_total": int(total),
                "threshold": float(
                    subset[score_column].min()
                ),
                "selective_human_validated_accuracy": round(
                    accuracy,
                    4,
                ),
                "selective_human_validated_risk": round(
                    1.0 - accuracy,
                    4,
                ),
            }
        )

    return pd.DataFrame(rows)
