"""
theratime_post_calibration.py
─────────────────────────────
Defensible multi-method post-calibration for TheraTime automatic labels.

Purpose
-------
This script does NOT train or fine-tune the sentence-transformer model.
It keeps the original TheraTime pre-trained/prototype system unchanged.

It compares several calibration strategies:

  0. baseline
     No calibration. Uses the original automatic labels.

  1. confidence_only
     Does not change labels. It only adds percentile-based low-confidence flags.

  2. isotonic_confidence
     Does not change labels. It learns calibrated reliability scores from human
     consensus using isotonic regression. This is confidence calibration, not
     label correction.

  3. isotonic_filter
     Does not change labels. It reports TTA@1 on high-reliability predictions
     after isotonic calibration. This improves trustworthiness at lower coverage.

  4. human_rules_direct
     Learns correction rules from human annotations and applies them directly
     to stage, move, and timing labels. This is included for ablation only and
     should usually not be the main paper result because it can overfit.

  5. human_rules_recompute
     Learns human correction rules for stage and move, then recomputes timing
     from the calibrated stage/move pair.

  6. conservative_human_recompute
     Same as human_rules_recompute, but more conservative:
       - requires annotators to agree on the correction
       - uses stronger support/error thresholds

  7. hybrid_isotonic_conservative
     Conservative human stage/move correction + recomputed timing + isotonic
     reliability scores and confidence filtering. This is the recommended
     defensible method for reporting.

Key defensibility features
--------------------------
  - Direct timing correction is marked as overfit-risk.
  - Recommended method is conservative/hybrid, not the method with maximum TTA.
  - Isotonic calibration is used as reliability calibration, not as label fixing.
  - Human annotations are used only as a lightweight calibration layer.
  - Underlying sentence-transformer encoders are not fine-tuned.
  - The report saves calibration rules, reliability thresholds, and comparison tables.

Typical Kaggle usage
--------------------
!python theratime_post_calibration.py \
  --auto /kaggle/working/theratime_v06_outputs/all_judgments_mpnet.csv \
  --ann /kaggle/input/datasets/asmaeassmaebriouya/annotations/theratime_150_Hasnae_human_corrected_annotations.csv \
        /kaggle/input/datasets/asmaeassmaebriouya/annotations/theratime_human_annotations_Asmae_150_updated_reviewed.csv \
  --out-dir /kaggle/working/theratime_post_calibration_outputs \
  --methods all

Recommended paper wording
-------------------------
"Human annotations were used for lightweight post-hoc calibration and reliability
analysis. The underlying sentence-transformer encoders were not fine-tuned.
Calibration was applied as a transparent correction and uncertainty layer over
automatic stage, move, and timing outputs."
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss
    SKLEARN_AVAILABLE = True
except Exception:
    IsotonicRegression = None
    brier_score_loss = None
    SKLEARN_AVAILABLE = False


YES_NO = {"yes", "no"}
ANSWER_VALUES = {"yes", "no", "unsure"}

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

ALLOWED_MOVES = {
    "distress_disclosure": {
        "validation", "empathy", "reflective_listening", "clarification", "encouragement"
    },
    "high_emotional_intensity": {
        "validation", "empathy", "reflective_listening", "grounding", "clarification"
    },
    "unclear_need": {
        "validation", "empathy", "reflective_listening", "clarification"
    },
    "advice_seeking": {
        "practical_advice", "psychoeducation", "encouragement", "validation"
    },
    "psychoeducation_seeking": {
        "psychoeducation", "validation", "clarification"
    },
    "crisis_safety": {
        "safety_referral", "grounding", "validation"
    },
    "followup_problem_solving": {
        "practical_advice", "psychoeducation", "encouragement"
    },
}


METHODS = [
    "baseline",
    "confidence_only",
    "isotonic_confidence",
    "isotonic_filter",
    "human_rules_direct",
    "human_rules_recompute",
    "conservative_human_recompute",
    "hybrid_isotonic_conservative",
]


def norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def normalise_answer(value: Any) -> str:
    value = norm(value)
    if value in {"y", "true", "1", "correct"}:
        return "yes"
    if value in {"n", "false", "0", "incorrect", "wrong"}:
        return "no"
    if value in {"unsure", "uncertain", "maybe", "not_sure"}:
        return "unsure"
    return value


def as_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_annotation_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        rows = df.fillna("").to_dict("records")
        annotator = ""
        if "annotator" in df.columns:
            vals = [
                str(x).strip()
                for x in df["annotator"].dropna().unique()
                if str(x).strip()
            ]
            annotator = vals[0] if vals else ""
        if not annotator:
            annotator = path.stem

    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            rows = data
            annotator = path.stem
        else:
            rows = data.get("annotations") or data.get("rows") or data.get("data") or []
            annotator = str(data.get("annotator") or path.stem).strip()

    else:
        raise ValueError(f"Unsupported annotation file type: {path}")

    cleaned = {}
    for idx, row in enumerate(rows, start=1):
        row = dict(row)
        sid = str(
            row.get("id")
            or row.get("query_id")
            or row.get("example_id")
            or f"row_{idx}"
        ).strip()
        row["id"] = sid

        for field in ["stage_correct", "move_correct", "timing_correct"]:
            row[field] = normalise_answer(row.get(field, ""))

        for field in ["auto_stage", "predicted_stage", "stage_correction"]:
            if field in row:
                row[field] = norm(row.get(field))

        for field in ["auto_move", "predicted_move", "move_correction"]:
            if field in row:
                row[field] = norm(row.get(field))

        for field in ["auto_timing", "timing_label", "timing_correction"]:
            if field in row:
                row[field] = norm(row.get(field))

        cleaned[sid] = row

    return {"annotator": annotator, "annotations": cleaned}


def get_auto_stage(row: Dict[str, Any]) -> str:
    return norm(row.get("auto_stage") or row.get("predicted_stage") or row.get("stage"))


def get_auto_move(row: Dict[str, Any]) -> str:
    return norm(row.get("auto_move") or row.get("predicted_move") or row.get("move"))


def get_auto_timing(row: Dict[str, Any]) -> str:
    return norm(row.get("auto_timing") or row.get("timing_label") or row.get("predicted_timing"))


def consensus_label(
    rows: List[Dict[str, Any]],
    correct_field: str,
    correction_field: str,
    auto_value: str,
    allowed_values: set,
    require_same_correction: bool = False,
) -> Dict[str, Any]:
    answers = [
        r.get(correct_field, "")
        for r in rows
        if r.get(correct_field, "") in YES_NO
    ]

    if not answers:
        return {
            "consensus_correct": "missing",
            "consensus_value": auto_value,
            "source": "missing",
        }

    if all(a == "yes" for a in answers):
        return {
            "consensus_correct": "yes",
            "consensus_value": auto_value,
            "source": "human_agreed_auto_correct",
        }

    if all(a == "no" for a in answers):
        corrections = [
            norm(r.get(correction_field, ""))
            for r in rows
            if r.get(correct_field, "") == "no"
            and norm(r.get(correction_field, "")) in allowed_values
        ]

        if corrections:
            counts = Counter(corrections)
            top_value, top_count = counts.most_common(1)[0]

            if require_same_correction and top_count < len(corrections):
                return {
                    "consensus_correct": "no",
                    "consensus_value": auto_value,
                    "source": "human_agreed_wrong_but_correction_disagreed",
                }

            return {
                "consensus_correct": "no",
                "consensus_value": top_value,
                "source": "human_agreed_auto_wrong",
            }

        return {
            "consensus_correct": "no",
            "consensus_value": auto_value,
            "source": "human_agreed_wrong_no_correction",
        }

    return {
        "consensus_correct": "disagree",
        "consensus_value": auto_value,
        "source": "human_disagreement",
    }


def timing_from_stage_move(stage: str, move: str) -> str:
    stage = norm(stage)
    move = norm(move)
    allowed = ALLOWED_MOVES.get(stage, set())

    if stage == "crisis_safety" and move not in {"safety_referral", "grounding", "validation"}:
        return "delayed_safety"

    if stage in {"distress_disclosure", "high_emotional_intensity", "unclear_need"} and move == "practical_advice":
        return "premature_advice"

    if stage in {"advice_seeking", "followup_problem_solving"} and move in {"validation", "empathy", "reflective_listening"}:
        return "over_validation"

    if stage == "unclear_need" and move not in allowed:
        return "missing_clarification"

    if move in allowed:
        return "well_timed"

    return "stage_mismatch"


def build_consensus(
    ann_files: List[Path],
    require_same_correction: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    loaded = [load_annotation_file(path) for path in ann_files]
    annotators = [x["annotator"] for x in loaded]
    ann_maps = [x["annotations"] for x in loaded]

    common_ids = sorted(set.intersection(*[set(m.keys()) for m in ann_maps]))
    rows_out = []

    for sid in common_ids:
        rows = [m[sid] for m in ann_maps]
        base = rows[0]

        auto_stage = get_auto_stage(base)
        auto_move = get_auto_move(base)
        auto_timing = get_auto_timing(base)

        stage_cons = consensus_label(
            rows,
            "stage_correct",
            "stage_correction",
            auto_stage,
            STAGES,
            require_same_correction=require_same_correction,
        )
        move_cons = consensus_label(
            rows,
            "move_correct",
            "move_correction",
            auto_move,
            MOVES,
            require_same_correction=require_same_correction,
        )
        timing_cons = consensus_label(
            rows,
            "timing_correct",
            "timing_correction",
            auto_timing,
            TIMINGS,
            require_same_correction=require_same_correction,
        )

        rows_out.append({
            "id": sid,
            "query_id": base.get("query_id", sid),
            "query": base.get("query", ""),
            "response": base.get("response", ""),
            "auto_stage": auto_stage,
            "auto_move": auto_move,
            "auto_timing": auto_timing,
            "human_stage_correct": stage_cons["consensus_correct"],
            "human_stage": stage_cons["consensus_value"],
            "human_stage_source": stage_cons["source"],
            "human_move_correct": move_cons["consensus_correct"],
            "human_move": move_cons["consensus_value"],
            "human_move_source": move_cons["source"],
            "human_timing_correct": timing_cons["consensus_correct"],
            "human_timing": timing_cons["consensus_value"],
            "human_timing_source": timing_cons["source"],
        })

    consensus_df = pd.DataFrame(rows_out)
    meta = {
        "annotators": annotators,
        "n_common": len(common_ids),
        "require_same_correction": require_same_correction,
    }
    return consensus_df, meta


def learn_correction_map(
    consensus_df: pd.DataFrame,
    auto_col: str,
    human_correct_col: str,
    human_col: str,
    min_support: int,
    min_error_rate: float,
) -> Dict[str, Dict[str, Any]]:
    rules = {}

    if consensus_df.empty:
        return rules

    for auto_value, group in consensus_df.groupby(auto_col):
        auto_value = norm(auto_value)
        if not auto_value:
            continue

        valid = group[group[human_correct_col].isin(["yes", "no"])]
        if len(valid) == 0:
            continue

        wrong = valid[valid[human_correct_col] == "no"]
        error_rate = len(wrong) / len(valid)

        corrections = [
            x
            for x in wrong[human_col].astype(str).map(norm).tolist()
            if x and x != auto_value
        ]

        if not corrections:
            continue

        top_correction, top_count = Counter(corrections).most_common(1)[0]
        support = top_count

        if support >= min_support and error_rate >= min_error_rate:
            rules[auto_value] = {
                "to": top_correction,
                "support": int(support),
                "n_valid": int(len(valid)),
                "n_wrong": int(len(wrong)),
                "error_rate": round(float(error_rate), 4),
            }

    return rules


def apply_map(value: str, rules: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    value = norm(value)
    if value in rules:
        return rules[value]["to"], "rule_corrected"
    return value, "kept"


def prepare_auto_df(auto_path: Path) -> pd.DataFrame:
    df = pd.read_csv(auto_path).fillna("")

    if "id" in df.columns:
        df["id_for_calibration"] = df["id"].astype(str)
    elif "query_id" in df.columns:
        df["id_for_calibration"] = df["query_id"].astype(str)
    else:
        df["id_for_calibration"] = [f"row_{i + 1}" for i in range(len(df))]

    if "predicted_stage" in df.columns:
        df["auto_stage_for_calibration"] = df["predicted_stage"].map(norm)
    elif "auto_stage" in df.columns:
        df["auto_stage_for_calibration"] = df["auto_stage"].map(norm)
    else:
        raise ValueError("Automatic CSV must contain predicted_stage or auto_stage.")

    if "predicted_move" in df.columns:
        df["auto_move_for_calibration"] = df["predicted_move"].map(norm)
    elif "auto_move" in df.columns:
        df["auto_move_for_calibration"] = df["auto_move"].map(norm)
    else:
        raise ValueError("Automatic CSV must contain predicted_move or auto_move.")

    if "timing_label" in df.columns:
        df["auto_timing_for_calibration"] = df["timing_label"].map(norm)
    elif "auto_timing" in df.columns:
        df["auto_timing_for_calibration"] = df["auto_timing"].map(norm)
    else:
        raise ValueError("Automatic CSV must contain timing_label or auto_timing.")

    if "is_well_timed" in df.columns:
        df["auto_is_well_timed_for_calibration"] = as_bool_series(df["is_well_timed"])
    else:
        df["auto_is_well_timed_for_calibration"] = df["auto_timing_for_calibration"].eq("well_timed")

    if "rank" in df.columns:
        df["rank_for_calibration"] = pd.to_numeric(df["rank"], errors="coerce").fillna(-1).astype(int)
    else:
        df["rank_for_calibration"] = 1

    for col in ["stage_confidence", "move_confidence", "retrieval_score"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["combined_confidence"] = df[["stage_confidence", "move_confidence"]].min(axis=1)
    df["mean_confidence"] = df[["stage_confidence", "move_confidence"]].mean(axis=1)

    return df


def compute_confidence_flags(
    df: pd.DataFrame,
    stage_pct: float = 10.0,
    move_pct: float = 10.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()

    stage_vals = pd.to_numeric(out["stage_confidence"], errors="coerce")
    stage_vals = stage_vals[stage_vals >= 0]
    if len(stage_vals):
        stage_threshold = float(stage_vals.quantile(stage_pct / 100.0))
    else:
        stage_threshold = None

    move_vals = pd.to_numeric(out["move_confidence"], errors="coerce")
    move_vals = move_vals[move_vals >= 0]
    if len(move_vals):
        move_threshold = float(move_vals.quantile(move_pct / 100.0))
    else:
        move_threshold = None

    if stage_threshold is None:
        out["stage_low_confidence_calibrated"] = False
    else:
        out["stage_low_confidence_calibrated"] = out["stage_confidence"] < stage_threshold

    if move_threshold is None:
        out["move_low_confidence_calibrated"] = False
    else:
        out["move_low_confidence_calibrated"] = out["move_confidence"] < move_threshold

    out["low_confidence_calibrated"] = (
        out["stage_low_confidence_calibrated"]
        | out["move_low_confidence_calibrated"]
    )

    meta = {
        "stage_confidence_percentile": stage_pct,
        "move_confidence_percentile": move_pct,
        "stage_confidence_threshold": stage_threshold,
        "move_confidence_threshold": move_threshold,
        "low_confidence_rate": float(out["low_confidence_calibrated"].mean()),
    }
    return out, meta


def attach_human_consensus_to_auto(auto_df: pd.DataFrame, consensus_df: pd.DataFrame) -> pd.DataFrame:
    out = auto_df.copy()

    if consensus_df.empty:
        out["has_human_consensus"] = False
        return out

    cons = consensus_df.copy()
    cons["id_for_calibration"] = cons["id"].astype(str)

    keep_cols = [
        "id_for_calibration",
        "human_stage_correct",
        "human_stage",
        "human_move_correct",
        "human_move",
        "human_timing_correct",
        "human_timing",
    ]
    keep_cols = [c for c in keep_cols if c in cons.columns]

    out = out.merge(cons[keep_cols], on="id_for_calibration", how="left")
    out["has_human_consensus"] = out["human_stage_correct"].notna()

    for col in [
        "human_stage_correct",
        "human_stage",
        "human_move_correct",
        "human_move",
        "human_timing_correct",
        "human_timing",
    ]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("")

    return out


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        if not np.any(mask):
            continue

        bin_acc = float(np.mean(y_true[mask]))
        bin_conf = float(np.mean(y_prob[mask]))
        ece += float(np.mean(mask)) * abs(bin_acc - bin_conf)

    return float(ece)


def scale_to_01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if abs(vmax - vmin) < 1e-12:
        return np.full_like(values, 0.5, dtype=float)
    return (values - vmin) / (vmax - vmin)


def fit_isotonic_reliability(
    labeled_df: pd.DataFrame,
    all_df: pd.DataFrame,
    target_col: str,
    feature_col: str,
    output_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit isotonic regression on human consensus and predict reliability on all rows.

    target_col should contain yes/no/missing labels where yes means the auto
    label was accepted by human consensus.
    """
    out = all_df.copy()

    if not SKLEARN_AVAILABLE:
        out[output_col] = 0.5
        return out, {
            "available": False,
            "reason": "sklearn is not available",
            "target_col": target_col,
            "feature_col": feature_col,
        }

    train = labeled_df[labeled_df[target_col].isin(["yes", "no"])].copy()

    if len(train) < 20 or train[target_col].nunique() < 2:
        out[output_col] = 0.5
        return out, {
            "available": False,
            "reason": "not enough labeled yes/no data or only one class",
            "target_col": target_col,
            "feature_col": feature_col,
            "n_train": int(len(train)),
        }

    x_train_raw = train[feature_col].astype(float).values
    y_train = train[target_col].eq("yes").astype(int).values

    x_all_raw = out[feature_col].astype(float).values

    all_min = float(np.nanmin(np.concatenate([x_train_raw, x_all_raw])))
    all_max = float(np.nanmax(np.concatenate([x_train_raw, x_all_raw])))
    denom = all_max - all_min if abs(all_max - all_min) > 1e-12 else 1.0

    x_train = (x_train_raw - all_min) / denom
    x_all = (x_all_raw - all_min) / denom

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(x_train, y_train)
    pred_train = iso.predict(x_train)
    pred_all = iso.predict(x_all)

    out[output_col] = pred_all

    brier = None
    if brier_score_loss is not None:
        try:
            brier = float(brier_score_loss(y_train, pred_train))
        except Exception:
            brier = None

    meta = {
        "available": True,
        "target_col": target_col,
        "feature_col": feature_col,
        "output_col": output_col,
        "n_train": int(len(train)),
        "positive_rate": round(float(np.mean(y_train)), 4),
        "train_brier": round(brier, 4) if brier is not None else None,
        "train_ece": round(expected_calibration_error(y_train, pred_train), 4),
        "feature_min": all_min,
        "feature_max": all_max,
        "note": (
            "This is in-sample isotonic reliability calibration on the human consensus subset. "
            "Use it as an uncertainty estimate, not as proof of improved label accuracy."
        ),
    }
    return out, meta


def add_isotonic_reliability(
    base_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
    reliability_threshold: float = 0.50,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Adds:
      isotonic_stage_reliability
      isotonic_move_reliability
      isotonic_timing_reliability
      isotonic_overall_reliability
      isotonic_high_reliability
    """
    merged = attach_human_consensus_to_auto(base_df, consensus_df)

    # Fit only on rows with human consensus. Predict on all rows.
    labeled = merged[merged["has_human_consensus"]].copy()

    out, stage_meta = fit_isotonic_reliability(
        labeled_df=labeled,
        all_df=merged,
        target_col="human_stage_correct",
        feature_col="stage_confidence",
        output_col="isotonic_stage_reliability",
    )

    out, move_meta = fit_isotonic_reliability(
        labeled_df=out[out["has_human_consensus"]].copy(),
        all_df=out,
        target_col="human_move_correct",
        feature_col="move_confidence",
        output_col="isotonic_move_reliability",
    )

    # Timing reliability depends on both stage and move confidence. The minimum is conservative.
    out, timing_meta = fit_isotonic_reliability(
        labeled_df=out[out["has_human_consensus"]].copy(),
        all_df=out,
        target_col="human_timing_correct",
        feature_col="combined_confidence",
        output_col="isotonic_timing_reliability",
    )

    out["isotonic_overall_reliability"] = out[
        [
            "isotonic_stage_reliability",
            "isotonic_move_reliability",
            "isotonic_timing_reliability",
        ]
    ].min(axis=1)

    out["isotonic_high_reliability"] = out["isotonic_overall_reliability"] >= reliability_threshold

    meta = {
        "sklearn_available": SKLEARN_AVAILABLE,
        "reliability_threshold": reliability_threshold,
        "stage": stage_meta,
        "move": move_meta,
        "timing": timing_meta,
        "overall_high_reliability_rate": float(out["isotonic_high_reliability"].mean()),
    }
    return out, meta


def apply_calibration_method(
    base_df: pd.DataFrame,
    method: str,
    stage_rules: Optional[Dict[str, Dict[str, Any]]] = None,
    move_rules: Optional[Dict[str, Dict[str, Any]]] = None,
    timing_rules: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    stage_rules = stage_rules or {}
    move_rules = move_rules or {}
    timing_rules = timing_rules or {}

    df = base_df.copy()

    calibrated_stages = []
    stage_sources = []
    calibrated_moves = []
    move_sources = []
    calibrated_timings = []
    timing_sources = []

    for _, row in df.iterrows():
        original_stage = row["auto_stage_for_calibration"]
        original_move = row["auto_move_for_calibration"]
        original_timing = row["auto_timing_for_calibration"]

        if method in {
            "human_rules_direct",
            "human_rules_recompute",
            "conservative_human_recompute",
            "hybrid_isotonic_conservative",
        }:
            stage, stage_src = apply_map(original_stage, stage_rules)
            move, move_src = apply_map(original_move, move_rules)
        else:
            stage, stage_src = original_stage, "kept"
            move, move_src = original_move, "kept"

        if method == "human_rules_direct":
            timing, timing_src = apply_map(original_timing, timing_rules)
        elif method in {
            "human_rules_recompute",
            "conservative_human_recompute",
            "hybrid_isotonic_conservative",
        }:
            timing = timing_from_stage_move(stage, move)
            timing_src = "recomputed_from_calibrated_stage_move"
        else:
            timing = original_timing
            timing_src = "kept"

        calibrated_stages.append(stage)
        stage_sources.append(stage_src)
        calibrated_moves.append(move)
        move_sources.append(move_src)
        calibrated_timings.append(timing)
        timing_sources.append(timing_src)

    df["calibration_method"] = method
    df["calibrated_stage"] = calibrated_stages
    df["stage_calibration_source"] = stage_sources
    df["calibrated_move"] = calibrated_moves
    df["move_calibration_source"] = move_sources
    df["calibrated_timing"] = calibrated_timings
    df["timing_calibration_source"] = timing_sources
    df["calibrated_is_well_timed"] = df["calibrated_timing"].eq("well_timed")

    df["stage_changed"] = df["calibrated_stage"] != df["auto_stage_for_calibration"]
    df["move_changed"] = df["calibrated_move"] != df["auto_move_for_calibration"]
    df["timing_changed"] = df["calibrated_timing"] != df["auto_timing_for_calibration"]

    return df


def summarize_method(df: pd.DataFrame, method: str) -> Dict[str, Any]:
    rank1 = df[df["rank_for_calibration"] == 1].copy()
    if len(rank1) == 0:
        rank1 = df.copy()

    original_tta1 = float(rank1["auto_is_well_timed_for_calibration"].mean())
    calibrated_tta1 = float(rank1["calibrated_is_well_timed"].mean())

    high_conf_rank1 = rank1[~rank1["low_confidence_calibrated"]].copy()
    if len(high_conf_rank1):
        high_conf_tta1 = float(high_conf_rank1["calibrated_is_well_timed"].mean())
    else:
        high_conf_tta1 = None

    if "isotonic_high_reliability" in rank1.columns:
        iso_rank1 = rank1[rank1["isotonic_high_reliability"]].copy()
        iso_coverage = float(rank1["isotonic_high_reliability"].mean())
        iso_tta1 = float(iso_rank1["calibrated_is_well_timed"].mean()) if len(iso_rank1) else None
        mean_iso_rel = float(rank1["isotonic_overall_reliability"].mean())
    else:
        iso_coverage = None
        iso_tta1 = None
        mean_iso_rel = None

    overfit_risk = "high" if method == "human_rules_direct" else "low_to_moderate"

    return {
        "method": method,
        "original_TTA_at_1": round(original_tta1, 4),
        "calibrated_TTA_at_1": round(calibrated_tta1, 4),
        "delta_TTA_at_1": round(calibrated_tta1 - original_tta1, 4),
        "percentile_high_conf_TTA_at_1": round(high_conf_tta1, 4) if high_conf_tta1 is not None else None,
        "percentile_conf_coverage": round(float((~rank1["low_confidence_calibrated"]).mean()), 4),
        "isotonic_high_rel_TTA_at_1": round(iso_tta1, 4) if iso_tta1 is not None else None,
        "isotonic_high_rel_coverage": round(iso_coverage, 4) if iso_coverage is not None else None,
        "mean_isotonic_reliability_rank1": round(mean_iso_rel, 4) if mean_iso_rel is not None else None,
        "stage_changes_rank1": int(rank1["stage_changed"].sum()),
        "move_changes_rank1": int(rank1["move_changed"].sum()),
        "timing_changes_rank1": int(rank1["timing_changed"].sum()),
        "stage_changes_all_rows": int(df["stage_changed"].sum()),
        "move_changes_all_rows": int(df["move_changed"].sum()),
        "timing_changes_all_rows": int(df["timing_changed"].sum()),
        "n_rank1": int(len(rank1)),
        "n_rows": int(len(df)),
        "overfit_risk": overfit_risk,
    }


def df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = list(df.columns)
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        return "\n".join(lines)


def choose_paper_safe_method(requested_methods: List[str]) -> str:
    if "hybrid_isotonic_conservative" in requested_methods:
        return "hybrid_isotonic_conservative"
    if "conservative_human_recompute" in requested_methods:
        return "conservative_human_recompute"
    if "human_rules_recompute" in requested_methods:
        return "human_rules_recompute"
    if "isotonic_filter" in requested_methods:
        return "isotonic_filter"
    return "baseline"


def main():
    parser = argparse.ArgumentParser(
        description="Run defensible post-calibration methods for TheraTime automatic labels."
    )
    parser.add_argument(
        "--auto",
        required=True,
        help="Automatic TheraTime judgments CSV, for example all_judgments_mpnet.csv.",
    )
    parser.add_argument(
        "--ann",
        nargs="+",
        required=True,
        help="Two or more human annotation CSV/JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        default="theratime_post_calibration_outputs",
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help=f"Methods to run. Use 'all' or any of: {', '.join(METHODS)}",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=3,
        help="Minimum number of human corrections needed to create a standard rule.",
    )
    parser.add_argument(
        "--min-error-rate",
        type=float,
        default=0.50,
        help="Minimum error rate for an automatic label before standard correction.",
    )
    parser.add_argument(
        "--conservative-min-support",
        type=int,
        default=5,
        help="Minimum support for conservative human calibration.",
    )
    parser.add_argument(
        "--conservative-min-error-rate",
        type=float,
        default=0.60,
        help="Minimum error rate for conservative human calibration.",
    )
    parser.add_argument(
        "--stage-confidence-percentile",
        type=float,
        default=10.0,
        help="Percentile threshold for stage low-confidence flag.",
    )
    parser.add_argument(
        "--move-confidence-percentile",
        type=float,
        default=10.0,
        help="Percentile threshold for move low-confidence flag.",
    )
    parser.add_argument(
        "--isotonic-reliability-threshold",
        type=float,
        default=0.50,
        help="Minimum isotonic overall reliability for high-reliability subset.",
    )
    args = parser.parse_args()

    auto_path = Path(args.auto)
    ann_paths = [Path(x) for x in args.ann]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_methods = METHODS if "all" in args.methods else args.methods
    unknown = [m for m in requested_methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed methods: {METHODS}")

    base_df = prepare_auto_df(auto_path)
    base_df, confidence_meta = compute_confidence_flags(
        base_df,
        stage_pct=args.stage_confidence_percentile,
        move_pct=args.move_confidence_percentile,
    )

    standard_consensus_df, consensus_meta = build_consensus(
        ann_paths,
        require_same_correction=False,
    )
    standard_consensus_path = out_dir / "theratime_human_consensus_standard.csv"
    standard_consensus_df.to_csv(standard_consensus_path, index=False)

    conservative_consensus_df, conservative_meta = build_consensus(
        ann_paths,
        require_same_correction=True,
    )
    conservative_consensus_path = out_dir / "theratime_human_consensus_conservative.csv"
    conservative_consensus_df.to_csv(conservative_consensus_path, index=False)

    stage_rules = learn_correction_map(
        standard_consensus_df,
        auto_col="auto_stage",
        human_correct_col="human_stage_correct",
        human_col="human_stage",
        min_support=args.min_support,
        min_error_rate=args.min_error_rate,
    )
    move_rules = learn_correction_map(
        standard_consensus_df,
        auto_col="auto_move",
        human_correct_col="human_move_correct",
        human_col="human_move",
        min_support=args.min_support,
        min_error_rate=args.min_error_rate,
    )
    timing_rules = learn_correction_map(
        standard_consensus_df,
        auto_col="auto_timing",
        human_correct_col="human_timing_correct",
        human_col="human_timing",
        min_support=args.min_support,
        min_error_rate=args.min_error_rate,
    )

    conservative_stage_rules = learn_correction_map(
        conservative_consensus_df,
        auto_col="auto_stage",
        human_correct_col="human_stage_correct",
        human_col="human_stage",
        min_support=args.conservative_min_support,
        min_error_rate=args.conservative_min_error_rate,
    )
    conservative_move_rules = learn_correction_map(
        conservative_consensus_df,
        auto_col="auto_move",
        human_correct_col="human_move_correct",
        human_col="human_move",
        min_support=args.conservative_min_support,
        min_error_rate=args.conservative_min_error_rate,
    )
    conservative_timing_rules = learn_correction_map(
        conservative_consensus_df,
        auto_col="auto_timing",
        human_correct_col="human_timing_correct",
        human_col="human_timing",
        min_support=args.conservative_min_support,
        min_error_rate=args.conservative_min_error_rate,
    )

    # Isotonic reliability is learned from the standard human consensus.
    isotonic_df, isotonic_meta = add_isotonic_reliability(
        base_df,
        standard_consensus_df,
        reliability_threshold=args.isotonic_reliability_threshold,
    )

    summaries = []
    method_files = {}

    for method in requested_methods:
        if method in {"isotonic_confidence", "isotonic_filter"}:
            method_base = isotonic_df.copy()
        elif method == "hybrid_isotonic_conservative":
            method_base = isotonic_df.copy()
        else:
            method_base = base_df.copy()

        if method == "baseline":
            df_method = apply_calibration_method(method_base, method="baseline")

        elif method == "confidence_only":
            df_method = apply_calibration_method(method_base, method="confidence_only")

        elif method == "isotonic_confidence":
            df_method = apply_calibration_method(method_base, method="isotonic_confidence")

        elif method == "isotonic_filter":
            df_method = apply_calibration_method(method_base, method="isotonic_filter")

        elif method == "human_rules_direct":
            df_method = apply_calibration_method(
                method_base,
                method=method,
                stage_rules=stage_rules,
                move_rules=move_rules,
                timing_rules=timing_rules,
            )

        elif method == "human_rules_recompute":
            df_method = apply_calibration_method(
                method_base,
                method=method,
                stage_rules=stage_rules,
                move_rules=move_rules,
                timing_rules=timing_rules,
            )

        elif method == "conservative_human_recompute":
            df_method = apply_calibration_method(
                method_base,
                method=method,
                stage_rules=conservative_stage_rules,
                move_rules=conservative_move_rules,
                timing_rules=conservative_timing_rules,
            )

        elif method == "hybrid_isotonic_conservative":
            df_method = apply_calibration_method(
                method_base,
                method=method,
                stage_rules=conservative_stage_rules,
                move_rules=conservative_move_rules,
                timing_rules=conservative_timing_rules,
            )

        else:
            raise ValueError(f"Unhandled method: {method}")

        out_file = out_dir / f"theratime_{method}.csv"
        df_method.to_csv(out_file, index=False)
        method_files[method] = str(out_file)
        summaries.append(summarize_method(df_method, method))

    comparison_df = pd.DataFrame(summaries)

    preferred_order = [
        "method",
        "original_TTA_at_1",
        "calibrated_TTA_at_1",
        "delta_TTA_at_1",
        "percentile_high_conf_TTA_at_1",
        "percentile_conf_coverage",
        "isotonic_high_rel_TTA_at_1",
        "isotonic_high_rel_coverage",
        "mean_isotonic_reliability_rank1",
        "stage_changes_rank1",
        "move_changes_rank1",
        "timing_changes_rank1",
        "stage_changes_all_rows",
        "move_changes_all_rows",
        "timing_changes_all_rows",
        "n_rank1",
        "n_rows",
        "overfit_risk",
    ]
    comparison_df = comparison_df[[c for c in preferred_order if c in comparison_df.columns]]

    comparison_path = out_dir / "theratime_calibration_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    markdown_table = df_to_markdown(comparison_df)
    markdown_path = out_dir / "theratime_calibration_comparison.md"
    markdown_path.write_text(markdown_table, encoding="utf-8")

    best_by_tta = comparison_df.sort_values(
        ["calibrated_TTA_at_1", "isotonic_high_rel_coverage"],
        ascending=[False, False],
    ).iloc[0].to_dict()

    paper_safe_method = choose_paper_safe_method(requested_methods)

    report = {
        "purpose": "Defensible multi-method post-hoc calibration of TheraTime automatic labels.",
        "important_note": "The sentence-transformer encoders are not trained or fine-tuned by this script.",
        "auto_file": str(auto_path),
        "annotation_files": [str(x) for x in ann_paths],
        "annotators": consensus_meta["annotators"],
        "n_common_human_annotated_items": consensus_meta["n_common"],
        "methods_run": requested_methods,
        "confidence_calibration": confidence_meta,
        "isotonic_reliability_calibration": isotonic_meta,
        "standard_rule_settings": {
            "min_support": args.min_support,
            "min_error_rate": args.min_error_rate,
            "require_same_correction": False,
        },
        "conservative_rule_settings": {
            "min_support": args.conservative_min_support,
            "min_error_rate": args.conservative_min_error_rate,
            "require_same_correction": True,
        },
        "standard_rules": {
            "stage_rules": stage_rules,
            "move_rules": move_rules,
            "timing_rules": timing_rules,
            "n_stage_rules": len(stage_rules),
            "n_move_rules": len(move_rules),
            "n_timing_rules": len(timing_rules),
        },
        "conservative_rules": {
            "stage_rules": conservative_stage_rules,
            "move_rules": conservative_move_rules,
            "timing_rules": conservative_timing_rules,
            "n_stage_rules": len(conservative_stage_rules),
            "n_move_rules": len(conservative_move_rules),
            "n_timing_rules": len(conservative_timing_rules),
        },
        "output_files": {
            "standard_consensus": str(standard_consensus_path),
            "conservative_consensus": str(conservative_consensus_path),
            "comparison_csv": str(comparison_path),
            "comparison_markdown": str(markdown_path),
            "method_outputs": method_files,
        },
        "best_method_by_calibrated_TTA_at_1": best_by_tta,
        "paper_safe_recommended_method": paper_safe_method,
        "warning_about_direct_rules": (
            "human_rules_direct can produce inflated results because it directly maps "
            "timing labels from a small annotated sample. Treat it as an ablation, not "
            "as the main calibrated estimate."
        ),
        "recommended_paper_wording": (
            "Post-hoc calibration was evaluated using several lightweight strategies, "
            "including percentile confidence filtering, isotonic reliability calibration, "
            "human-derived correction rules, timing recomputation from calibrated "
            "stage-move pairs, and a conservative hybrid variant. The underlying "
            "sentence-transformer encoders were not fine-tuned."
        ),
    }

    report_path = out_dir / "theratime_post_calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 90)
    print("TheraTime defensible multi-method post-calibration complete")
    print("=" * 90)
    print(f"Output directory             : {out_dir}")
    print(f"Standard human consensus     : {standard_consensus_path}")
    print(f"Conservative human consensus : {conservative_consensus_path}")
    print(f"Comparison CSV               : {comparison_path}")
    print(f"Comparison Markdown          : {markdown_path}")
    print(f"Report JSON                  : {report_path}")
    print()
    print("Rules learned:")
    print(f"  Standard stage rules       : {len(stage_rules)}")
    print(f"  Standard move rules        : {len(move_rules)}")
    print(f"  Standard timing rules      : {len(timing_rules)}")
    print(f"  Conservative stage rules   : {len(conservative_stage_rules)}")
    print(f"  Conservative move rules    : {len(conservative_move_rules)}")
    print(f"  Conservative timing rules  : {len(conservative_timing_rules)}")
    print()
    print("Isotonic reliability:")
    print(f"  sklearn available          : {SKLEARN_AVAILABLE}")
    print(f"  reliability threshold      : {args.isotonic_reliability_threshold}")
    print(f"  high-reliability rate      : {isotonic_meta.get('overall_high_reliability_rate')}")
    print()
    print("Comparison table:")
    print(markdown_table)
    print()
    print(f"Best by calibrated TTA@1     : {best_by_tta['method']}")
    print(f"Paper-safe recommendation    : {paper_safe_method}")
    print()
    print("NOTE: If human_rules_direct gives a perfect score, do not use it as the main claim.")
    print("Use the conservative or hybrid isotonic-conservative method for defensible reporting.")
    print("=" * 90)


if __name__ == "__main__":
    main()
