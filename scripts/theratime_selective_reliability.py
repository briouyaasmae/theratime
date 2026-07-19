%%writefile theratime_selective_reliability.py
"""
theratime_selective_reliability.py
──────────────────────────────────
Selective reliability analysis for TheraTime.

Purpose
-------
This script does not retrain TheraTime.
It analyzes calibrated TheraTime outputs and asks:

  1. Which predictions should we trust?
  2. What TTA@1 do we get if we keep only high-confidence cases?
  3. How much coverage do we keep?
  4. Which cases should be sent to human review?

Inputs
------
Use one calibrated output file from theratime_post_calibration.py.

Recommended input:
  theratime_hybrid_isotonic_conservative.csv

or:
  theratime_conservative_human_recompute.csv

Outputs
-------
  - theratime_selective_risk_coverage.csv
  - theratime_selective_summary.json
  - theratime_selective_review_flags.csv
  - theratime_risk_coverage_curve.png
  - theratime_coverage_tta_curve.png

Kaggle usage
------------
!python theratime_selective_reliability.py \
  --input /kaggle/working/theratime_post_calibration_outputs/theratime_hybrid_isotonic_conservative.csv \
  --out-dir /kaggle/working/theratime_selective_reliability_outputs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def as_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def prepare_df(path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path).fillna("")

    if "rank" in df.columns:
        df["rank_for_analysis"] = safe_numeric(df["rank"], -1).astype(int)
    elif "rank_for_calibration" in df.columns:
        df["rank_for_analysis"] = safe_numeric(
            df["rank_for_calibration"], -1
        ).astype(int)
    else:
        df["rank_for_analysis"] = 1

    if "calibrated_is_well_timed" in df.columns:
        df["final_is_well_timed"] = as_bool_series(
            df["calibrated_is_well_timed"]
        )
    elif "is_well_timed" in df.columns:
        df["final_is_well_timed"] = as_bool_series(df["is_well_timed"])
    elif "calibrated_timing" in df.columns:
        df["final_is_well_timed"] = (
            df["calibrated_timing"].astype(str).str.strip().str.lower()
            == "well_timed"
        )
    elif "timing_label" in df.columns:
        df["final_is_well_timed"] = (
            df["timing_label"].astype(str).str.strip().str.lower()
            == "well_timed"
        )
    else:
        raise ValueError(
            "Could not find calibrated_is_well_timed, is_well_timed, "
            "calibrated_timing, or timing_label."
        )

    # Prefer the human-validated correctness column from pooled held-out
    # evaluation files. Fall back to a descriptive well-timed indicator only
    # for legacy output files that do not contain human correctness.
    if "calibrated_timing_correct" in df.columns:
        df["analysis_outcome"] = as_bool_series(
            df["calibrated_timing_correct"]
        )
        outcome_type = "human_validated_timing_accuracy"
    elif "original_timing_correct" in df.columns:
        df["analysis_outcome"] = as_bool_series(
            df["original_timing_correct"]
        )
        outcome_type = "human_validated_original_timing_accuracy"
    else:
        df["analysis_outcome"] = df["final_is_well_timed"].astype(bool)
        outcome_type = "descriptive_well_timed_rate"

    if "stage_confidence" not in df.columns:
        df["stage_confidence"] = 0.0
    if "move_confidence" not in df.columns:
        df["move_confidence"] = 0.0
    if "retrieval_score" not in df.columns:
        df["retrieval_score"] = 0.0

    df["stage_confidence"] = safe_numeric(df["stage_confidence"], 0.0)
    df["move_confidence"] = safe_numeric(df["move_confidence"], 0.0)
    df["retrieval_score"] = safe_numeric(df["retrieval_score"], 0.0)

    df["margin_reliability_score"] = df[
        ["stage_confidence", "move_confidence"]
    ].min(axis=1)

    if "isotonic_overall_reliability" in df.columns:
        df["isotonic_overall_reliability"] = safe_numeric(
            df["isotonic_overall_reliability"], 0.0
        )
    else:
        df["isotonic_overall_reliability"] = np.nan

    valid_iso = (
        df["isotonic_overall_reliability"].notna()
        & (df["isotonic_overall_reliability"] > 0)
    )
    df["hybrid_reliability_score"] = df["margin_reliability_score"]
    df.loc[valid_iso, "hybrid_reliability_score"] = df.loc[
        valid_iso, "isotonic_overall_reliability"
    ]

    margin = df["margin_reliability_score"].to_numpy(dtype=float)
    if len(margin) and np.max(margin) > np.min(margin):
        df["margin_reliability_score_01"] = (
            margin - np.min(margin)
        ) / (np.max(margin) - np.min(margin))
    else:
        df["margin_reliability_score_01"] = 0.5

    return df, outcome_type

def risk_coverage_table(
    df_rank1: pd.DataFrame,
    score_col: str,
    coverage_points=None,
) -> pd.DataFrame:
    if coverage_points is None:
        coverage_points = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]

    rows = []
    n = len(df_rank1)

    if n == 0:
        return pd.DataFrame()

    ranked = df_rank1.sort_values(score_col, ascending=False).reset_index(drop=True)

    for cov in coverage_points:
        k = max(1, int(round(cov * n)))
        subset = ranked.iloc[:k]

        outcome = float(subset["analysis_outcome"].mean())
        risk = 1.0 - outcome
        threshold = float(subset[score_col].min())

        rows.append({
            "score_column": score_col,
            "target_coverage": round(cov, 4),
            "actual_coverage": round(k / n, 4),
            "n_accepted": int(k),
            "n_total": int(n),
            "threshold": round(threshold, 6),
            "selective_outcome": round(outcome, 4),
            "selective_risk": round(risk, 4),
        })

    return pd.DataFrame(rows)


def area_under_risk_coverage(rc_df: pd.DataFrame) -> float:
    if rc_df.empty:
        return float("nan")

    d = rc_df.sort_values("actual_coverage")
    x = d["actual_coverage"].values
    y = d["selective_risk"].values
    return float(np.trapezoid(y, x))


def add_review_flags(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    crisis_review: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    out["review_reliability_score"] = safe_numeric(out[score_col], 0.0)
    out["review_threshold"] = threshold

    out["review_recommendation"] = np.where(
        out["review_reliability_score"] >= threshold,
        "accept",
        "human_review",
    )

    # Safety-sensitive override.
    if crisis_review:
        stage_col = None
        for candidate in ["calibrated_stage", "predicted_stage", "auto_stage", "auto_stage_for_calibration"]:
            if candidate in out.columns:
                stage_col = candidate
                break

        timing_col = None
        for candidate in ["calibrated_timing", "timing_label", "auto_timing", "auto_timing_for_calibration"]:
            if candidate in out.columns:
                timing_col = candidate
                break

        if stage_col is not None:
            crisis_mask = out[stage_col].astype(str).str.lower().eq("crisis_safety")
            out.loc[crisis_mask, "review_recommendation"] = "high_risk_review"

        if timing_col is not None:
            safety_mask = out[timing_col].astype(str).str.lower().eq("delayed_safety")
            out.loc[safety_mask, "review_recommendation"] = "high_risk_review"

    return out


def plot_risk_coverage(rc_df: pd.DataFrame, out_dir: Path):
    if rc_df.empty:
        return

    plt.figure(figsize=(8, 5))
    for score_col, group in rc_df.groupby("score_column"):
        group = group.sort_values("actual_coverage")
        plt.plot(group["actual_coverage"], group["selective_risk"], marker="o", label=score_col)

    plt.xlabel("Coverage")
    plt.ylabel("Selective risk, lower is better")
    plt.title("TheraTime selective risk-coverage curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = out_dir / "theratime_risk_coverage_curve.png"
    plt.savefig(path, dpi=200)
    plt.close()


def plot_coverage_tta(rc_df: pd.DataFrame, out_dir: Path):
    if rc_df.empty:
        return

    plt.figure(figsize=(8, 5))
    for score_col, group in rc_df.groupby("score_column"):
        group = group.sort_values("actual_coverage")
        plt.plot(group["actual_coverage"], group["selective_outcome"], marker="o", label=score_col)

    plt.xlabel("Coverage")
    plt.ylabel("Selective outcome, higher is better")
    plt.title("TheraTime selective outcome by coverage")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = out_dir / "theratime_coverage_tta_curve.png"
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Selective reliability analysis for calibrated TheraTime outputs."
    )
    parser.add_argument("--input", required=True, help="Calibrated TheraTime CSV file.")
    parser.add_argument("--out-dir", default="theratime_selective_reliability_outputs")
    parser.add_argument(
        "--preferred-coverage",
        type=float,
        default=0.80,
        help="Coverage level used to choose the human-review threshold.",
    )
    parser.add_argument(
        "--score",
        default="margin_reliability_score",
        choices=[
            "margin_reliability_score",
            "margin_reliability_score_01",
            "hybrid_reliability_score",
            "isotonic_overall_reliability",
        ],
        help="Reliability score used for the final review flag.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, outcome_type = prepare_df(input_path)
    rank1 = df[df["rank_for_analysis"] == 1].copy()

    if len(rank1) == 0:
        raise ValueError("No rank-1 rows found for selective reliability analysis.")

    score_columns = [
        "margin_reliability_score",
        "margin_reliability_score_01",
        "hybrid_reliability_score",
    ]

    # Only include isotonic if it actually has useful non-zero values.
    if "isotonic_overall_reliability" in rank1.columns:
        if rank1["isotonic_overall_reliability"].notna().sum() > 0:
            score_columns.append("isotonic_overall_reliability")

    tables = []
    for score_col in score_columns:
        if score_col not in rank1.columns:
            continue
        rc = risk_coverage_table(rank1, score_col)
        if not rc.empty:
            rc["AURC"] = round(area_under_risk_coverage(rc), 6)
            tables.append(rc)

    if tables:
        rc_all = pd.concat(tables, ignore_index=True)
    else:
        rc_all = pd.DataFrame()

    rc_path = out_dir / "theratime_selective_risk_coverage.csv"
    rc_all.to_csv(rc_path, index=False)

    # Select threshold for review flags at preferred coverage.
    selected_score = args.score
    if selected_score not in rank1.columns:
        selected_score = "margin_reliability_score"

    selected_rc = risk_coverage_table(rank1, selected_score, coverage_points=[args.preferred_coverage])
    if selected_rc.empty:
        threshold = float(rank1[selected_score].median())
        selected_tta = float(rank1["final_is_well_timed"].mean())
        selected_coverage = 1.0
    else:
        threshold = float(selected_rc.iloc[0]["threshold"])
        selected_tta = float(selected_rc.iloc[0]["selective_outcome"])
        selected_coverage = float(selected_rc.iloc[0]["actual_coverage"])

    flagged = add_review_flags(df, selected_score, threshold)
    flags_path = out_dir / "theratime_selective_review_flags.csv"
    flagged.to_csv(flags_path, index=False)

    rank1_flagged = flagged[flagged["rank_for_analysis"] == 1].copy()
    review_counts = rank1_flagged["review_recommendation"].value_counts().to_dict()

    n_rank1 = int(len(rank1_flagged))
    n_accept = int(review_counts.get("accept", 0))
    n_human_review = int(review_counts.get("human_review", 0))
    n_high_risk_review = int(review_counts.get("high_risk_review", 0))

    final_acceptance_coverage = (
        float(n_accept / n_rank1) if n_rank1 else 0.0
    )
    final_review_rate = (
        float((n_human_review + n_high_risk_review) / n_rank1)
        if n_rank1 else 0.0
    )

    baseline_tta = float(rank1["analysis_outcome"].mean())

    summary = {
        "input_file": str(input_path),
        "n_rows": int(len(df)),
        "n_rank1": int(len(rank1)),
        "outcome_type": outcome_type,
        "all_rank1_outcome": round(baseline_tta, 4),
        "preferred_coverage": args.preferred_coverage,
        "selected_score": selected_score,
        "selected_threshold": round(threshold, 6),
        "selective_outcome_at_preferred_coverage": round(selected_tta, 4),
        "nominal_rank_selected_coverage": round(selected_coverage, 4),
        "final_acceptance_coverage_after_safety_override": round(
            final_acceptance_coverage, 4
        ),
        "final_review_rate_after_safety_override": round(
            final_review_rate, 4
        ),
        "selective_gain_vs_all_rank1": round(selected_tta - baseline_tta, 4),
        "review_counts_rank1": review_counts,
        "risk_coverage_csv": str(rc_path),
        "review_flags_csv": str(flags_path),
        "paper_safe_interpretation": (
            "Selective reliability analysis evaluates when TheraTime judgments are more trustworthy. "
            "The nominal rank-selected coverage is reported separately from final acceptance coverage "
            "after safety-sensitive overrides. Low-reliability or safety-sensitive cases are sent to "
            "human review."
        ),
    }

    summary_path = out_dir / "theratime_selective_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_risk_coverage(rc_all, out_dir)
    plot_coverage_tta(rc_all, out_dir)

    print("=" * 80)
    print("TheraTime selective reliability analysis complete")
    print("=" * 80)
    print(f"Input file                   : {input_path}")
    print(f"Output directory             : {out_dir}")
    print(f"Risk-coverage CSV            : {rc_path}")
    print(f"Review flags CSV             : {flags_path}")
    print(f"Summary JSON                 : {summary_path}")
    print()
    print(f"All rank-1 TTA@1             : {baseline_tta:.4f}")
    print(f"Selected reliability score   : {selected_score}")
    print(f"Preferred coverage           : {args.preferred_coverage:.2f}")
    print(f"Selected threshold           : {threshold:.6f}")
    print(f"Selective TTA@1              : {selected_tta:.4f}")
    print(f"Selective gain               : {selected_tta - baseline_tta:.4f}")
    print(f"Nominal rank-selected coverage: {selected_coverage:.4f}")
    print(f"Final acceptance coverage     : {final_acceptance_coverage:.4f}")
    print(f"Final review rate             : {final_review_rate:.4f}")
    print()
    print("Rank-1 review counts:")
    for key, value in review_counts.items():
        print(f"  {key:18s}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    main()
