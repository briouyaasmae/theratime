%%writefile theratime_error_recall.py
"""
theratime_error_recall.py  v2.2
────────────────────────────────
Compute TheraTime automatic timing reliability from human annotations.

Answers four questions for the paper:

  Q1. SUBSET RECALL (original metric, now correctly scoped)
      Among cases where both annotators rejected the auto timing label
      AND both provided corrections, what fraction was still genuinely
      mistimed (possibly with a different label)?
      → Answers: does the auto system catch real errors even when wrong?

  Q2. CORRECTION DIRECTION (full-150 picture)
      Across all human corrections, what is the distribution of
      correction targets (well_timed vs which error types)?
      → Answers: which direction does the auto system err?

  Q3. AUTO PRECISION (most important for the paper)
      Of cases auto labeled as mistimed, what fraction did both
      annotators confirm as genuinely mistimed?
      → Answers: how trustworthy is an auto timing flag?

  Q4. PER-ERROR-TYPE BREAKDOWN
      For each auto error category, how often did humans agree it
      was genuinely that error type vs. corrected to well_timed?
      → Answers: which error types are most reliable?

Usage:
    python theratime_error_recall.py \\
        --disagreements theratime_disagreements.csv \\
        --annotations1  theratime_annotations_annotator1.csv \\
        --annotations2  theratime_annotations_annotator2.csv \\
        --out           theratime_recall_report.csv

    The --annotations1 and --annotations2 flags are optional.
    If provided, they enable Q2 and Q3 (full-150 analysis).
    If only --disagreements is provided, only Q1 is computed.

Output:
    - Console report with paper-ready sentences
    - CSV with per-case detail
    - theratime_recall_summary.json with all key numbers
"""

import argparse
import json
import sys
import csv
from pathlib import Path
from collections import Counter

import pandas as pd


# ── Label sets ────────────────────────────────────────────────────────────────

MISTIMED_LABELS = {
    "premature_advice",
    "delayed_safety",
    "over_validation",
    "missing_clarification",
    "stage_mismatch",
}

WELL_TIMED_LABELS = {"well_timed"}

ALL_TIMING_LABELS = MISTIMED_LABELS | WELL_TIMED_LABELS


# ── Utilities ─────────────────────────────────────────────────────────────────

def norm(x: str) -> str:
    return str(x or "").strip().lower().replace("-", "_").replace(" ", "_")


def find_col(df: pd.DataFrame, must_contain: list, must_exclude: list = None) -> str:
    """
    Return the first column whose lowercase name contains all strings in
    must_contain and none of the strings in must_exclude.
    """
    must_exclude = must_exclude or []
    for col in df.columns:
        low = col.lower()
        if (all(k.lower() in low for k in must_contain)
                and not any(k.lower() in low for k in must_exclude)):
            return col
    return None



def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate column names while preserving the first occurrence.

    Some CSV exports contain repeated auto_timing or correction columns.
    In pandas, selecting a duplicated name returns a DataFrame rather than
    a Series, which breaks .map() operations.
    """
    return df.loc[:, ~df.columns.duplicated()].copy()


def resolve_column(
    df: pd.DataFrame,
    exact_names: list[str],
    must_contain: list[str] | None = None,
    must_exclude: list[str] | None = None,
) -> str | None:
    """
    Resolve a column using exact-name priority, then a conservative fallback.
    Matching is case-insensitive.
    """
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for name in exact_names:
        match = lookup.get(name.strip().lower())
        if match is not None:
            return match

    if must_contain:
        return find_col(
            df,
            must_contain=must_contain,
            must_exclude=must_exclude or [],
        )

    return None


def resolve_annotation_columns(df: pd.DataFrame) -> dict:
    """
    Resolve the canonical columns required for Q2/Q3 from a full annotation
    file. Exact canonical names are preferred.
    """
    return {
        "id": resolve_column(
            df,
            ["id", "query_id", "example_id"],
            must_contain=["id"],
        ),
        "timing_correct": resolve_column(
            df,
            [
                "timing_correct",
                "timing correctness",
                "timing_correctness",
                "is_timing_correct",
            ],
            must_contain=["timing", "correct"],
            must_exclude=["correction", "notes", "auto", "predicted"],
        ),
        "timing_correction": resolve_column(
            df,
            [
                "timing_correction",
                "corrected_timing",
                "timing correction",
                "human_timing_correction",
            ],
            must_contain=["timing", "correction"],
            must_exclude=["notes"],
        ),
        "auto_timing": resolve_column(
            df,
            [
                "auto_timing",
                "timing_label",
                "predicted_timing",
                "automatic_timing",
            ],
            must_contain=["timing"],
            must_exclude=[
                "correct",
                "correction",
                "notes",
                "human",
                "annotator",
            ],
        ),
    }


def unique_columns(columns: list[str | None]) -> list[str]:
    """Return non-empty column names once, preserving order."""
    result: list[str] = []
    for column in columns:
        if column and column not in result:
            result.append(column)
    return result


def is_mistimed(label: str) -> bool | None:
    """Return True if label is a timing error, False if well_timed, None if unknown."""
    l = norm(label)
    if l in MISTIMED_LABELS:
        return True
    if l in WELL_TIMED_LABELS:
        return False
    return None


def detect_annotator_prefixes(df: pd.DataFrame) -> tuple[str, str]:
    """
    Detect the two annotator prefixes in a disagreements CSV generated by
    theratime_kappa.py.

    Expected columns include:
      <annotator>_timing
      <annotator>_timing_correction

    Generic columns such as auto_timing and timing_disagree are ignored.
    """
    suffix = "_timing"
    excluded = {
        "auto",
        "predicted",
        "human",
        "stage",
        "move",
        "timing",
    }

    prefixes = []
    for col in df.columns:
        low = col.lower()
        if not low.endswith(suffix):
            continue
        if low in {"auto_timing", "predicted_timing"}:
            continue

        prefix = col[:-len(suffix)]
        if not prefix or prefix.lower() in excluded:
            continue

        correction_col = f"{prefix}_timing_correction"
        if correction_col in df.columns:
            prefixes.append(prefix)

    # Preserve column order while removing duplicates.
    unique = []
    for prefix in prefixes:
        if prefix not in unique:
            unique.append(prefix)

    if len(unique) != 2:
        raise ValueError(
            "Could not uniquely detect two annotator prefixes from the "
            f"disagreements CSV. Detected: {unique}. "
            "Expected columns like '<annotator>_timing' and "
            "'<annotator>_timing_correction'."
        )

    return unique[0], unique[1]


# ── Q1: Subset recall (disagreements CSV only) ────────────────────────────────

def compute_subset_recall(df: pd.DataFrame,
                           h_timing_col: str,
                           a_timing_col: str,
                           h_corr_col: str,
                           a_corr_col: str) -> dict:
    """
    Compute recall on the subset where both annotators said timing was wrong.

    SCOPE: This metric applies only to cases in theratime_disagreements.csv
    (cases with at least one stage or move disagreement). It is NOT the
    system's overall precision or recall. Report with explicit scope.
    """
    df = df.copy()
    df["_h_timing"] = df[h_timing_col].map(norm)
    df["_a_timing"] = df[a_timing_col].map(norm)
    df["_h_corr"]   = df[h_corr_col].map(norm)
    df["_a_corr"]   = df[a_corr_col].map(norm)

    both_wrong = df[(df["_h_timing"] == "no") & (df["_a_timing"] == "no")].copy()

    rows = []
    for _, row in both_wrong.iterrows():
        h = row["_h_corr"]
        a = row["_a_corr"]
        corrections = [x for x in [h, a] if x and x in ALL_TIMING_LABELS]

        h_mt = is_mistimed(h)
        a_mt = is_mistimed(a)

        if not corrections:
            status = "unknown_no_correction"
            consensus_mt = None
        elif h == a and h in ALL_TIMING_LABELS:
            status = ("mistimed_same_label" if h_mt
                      else "well_timed_same_label")
            consensus_mt = h_mt
        elif h_mt is True and a_mt is True:
            status = "mistimed_different_label"
            consensus_mt = True
        elif h_mt is False and a_mt is False:
            status = "well_timed_different_label"
            consensus_mt = False
        elif h_mt is not None and a_mt is not None and h_mt != a_mt:
            status = "human_disagree_mistimed_vs_well_timed"
            consensus_mt = None
        else:
            status = "unknown_or_invalid_correction"
            consensus_mt = None

        rows.append({
            "id":               row.get("id", ""),
            "auto_timing":      norm(row.get("auto_timing", row.get(
                                    "timing_label", row.get("predicted_timing","")))),
            "annotator_1_correction": h,
            "annotator_2_correction": a,
            "human_status":      status,
            "consensus_mistimed": consensus_mt,
        })

    result = pd.DataFrame(rows)
    valid  = result[result["consensus_mistimed"].isin([True, False])]
    n_both_wrong = len(both_wrong)
    n_valid      = len(valid)
    n_mistimed   = int((valid["consensus_mistimed"] == True).sum())
    n_well_timed = int((valid["consensus_mistimed"] == False).sum())
    recall = n_mistimed / n_valid if n_valid else None

    return {
        "n_both_said_wrong":   n_both_wrong,
        "n_valid_corrections": n_valid,
        "n_actually_mistimed": n_mistimed,
        "n_actually_well_timed": n_well_timed,
        "subset_recall":       round(recall, 4) if recall is not None else None,
        "status_breakdown":    result["human_status"].value_counts().to_dict(),
        "detail_df":           result,
    }


# ── Q2: Correction direction (full annotation files) ─────────────────────────

def compute_correction_direction(
    ann1: pd.DataFrame,
    ann2: pd.DataFrame,
) -> dict:
    """
    Across all human timing corrections where an annotator rejected the
    automatic timing label, report the correction-target distribution.

    This is a pairwise annotation diagnostic. It counts annotator-level
    corrections, not three-annotator majority-consensus corrections.
    """
    ann1 = drop_duplicate_columns(ann1)
    ann2 = drop_duplicate_columns(ann2)

    corrections = []
    diagnostics = []

    for df, annotator in [
        (ann1, "annotator1"),
        (ann2, "annotator2"),
    ]:
        columns = resolve_annotation_columns(df)
        tc_col = columns["timing_correct"]
        corr_col = columns["timing_correction"]

        diagnostics.append(
            {
                "annotator": annotator,
                "timing_correct_column": tc_col,
                "timing_correction_column": corr_col,
            }
        )

        if tc_col is None or corr_col is None:
            continue

        wrong = df[df[tc_col].map(norm) == "no"]
        for _, row in wrong.iterrows():
            correction = norm(row[corr_col])
            if correction in ALL_TIMING_LABELS:
                corrections.append(
                    {
                        "annotator": annotator,
                        "correction": correction,
                    }
                )

    if not corrections:
        return {
            "error": (
                "Could not find usable timing-correction rows in the full "
                "annotation files."
            ),
            "column_diagnostics": diagnostics,
        }

    direction = Counter(row["correction"] for row in corrections)
    total = sum(direction.values())

    return {
        "total_corrections": int(total),
        "well_timed_count": int(direction.get("well_timed", 0)),
        "well_timed_pct": (
            round(direction.get("well_timed", 0) / total, 4)
            if total
            else 0.0
        ),
        "mistimed_corrections": {
            key: int(value)
            for key, value in direction.items()
            if key in MISTIMED_LABELS
        },
        "full_breakdown": {
            key: int(value)
            for key, value in direction.most_common()
        },
        "column_diagnostics": diagnostics,
    }


# ── Q3: Auto precision (needs full annotation + auto labels) ──────────────────

def compute_auto_precision(
    ann1: pd.DataFrame,
    ann2: pd.DataFrame,
) -> dict:
    """
    Among cases automatically labelled as mistimed, estimate strict pairwise
    precision: both annotators must confirm that the automatic timing label
    is correct.

    Ambiguous or incomplete pairwise judgments remain in the denominator,
    making this a conservative pairwise estimate.
    """
    ann1 = drop_duplicate_columns(ann1)
    ann2 = drop_duplicate_columns(ann2)

    cols1 = resolve_annotation_columns(ann1)
    cols2 = resolve_annotation_columns(ann2)

    id1 = cols1["id"]
    id2 = cols2["id"]
    tc1 = cols1["timing_correct"]
    tc2 = cols2["timing_correct"]
    corr1 = cols1["timing_correction"]
    corr2 = cols2["timing_correction"]
    auto1 = cols1["auto_timing"]
    auto2 = cols2["auto_timing"]

    required = {
        "annotator1_id": id1,
        "annotator2_id": id2,
        "annotator1_timing_correct": tc1,
        "annotator2_timing_correct": tc2,
        "annotator1_timing_correction": corr1,
        "annotator2_timing_correction": corr2,
    }

    missing = [name for name, value in required.items() if value is None]
    if missing:
        return {
            "error": f"Missing required columns: {missing}",
            "annotator1_columns": cols1,
            "annotator2_columns": cols2,
        }

    if auto1 is None and auto2 is None:
        return {
            "error": "Could not locate an automatic timing-label column.",
            "annotator1_columns": cols1,
            "annotator2_columns": cols2,
        }

    left_columns = unique_columns([id1, tc1, corr1, auto1])
    right_columns = unique_columns([id2, tc2, corr2, auto2])

    left = ann1[left_columns].copy()
    right = ann2[right_columns].copy()

    left_rename = {
        id1: "id",
        tc1: "h1_tc",
        corr1: "h1_corr",
    }
    if auto1 is not None:
        left_rename[auto1] = "auto_timing"

    right_rename = {
        id2: "id",
        tc2: "h2_tc",
        corr2: "h2_corr",
    }
    if auto2 is not None:
        right_rename[auto2] = "auto_timing_from_annotator2"

    left = left.rename(columns=left_rename)
    right = right.rename(columns=right_rename)

    merged = pd.merge(
        left,
        right,
        on="id",
        how="inner",
        validate="one_to_one",
    )

    if "auto_timing" not in merged.columns:
        merged["auto_timing"] = merged["auto_timing_from_annotator2"]
    elif "auto_timing_from_annotator2" in merged.columns:
        left_auto = merged["auto_timing"].map(norm)
        right_auto = merged["auto_timing_from_annotator2"].map(norm)

        mismatch = (
            left_auto.ne("")
            & right_auto.ne("")
            & left_auto.ne(right_auto)
        )
        if mismatch.any():
            examples = merged.loc[
                mismatch,
                ["id", "auto_timing", "auto_timing_from_annotator2"],
            ].head(10)
            return {
                "error": (
                    "Automatic timing labels disagree between the two full "
                    "annotation files."
                ),
                "n_auto_timing_mismatches": int(mismatch.sum()),
                "examples": examples.to_dict("records"),
            }

        merged["auto_timing"] = merged["auto_timing"].where(
            left_auto.ne(""),
            merged["auto_timing_from_annotator2"],
        )

    # At this point auto_timing is guaranteed to be one Series.
    merged["auto_mt"] = merged["auto_timing"].map(norm).map(is_mistimed)
    merged["h1_tc_n"] = merged["h1_tc"].map(norm)
    merged["h2_tc_n"] = merged["h2_tc"].map(norm)
    merged["h1_corr_n"] = merged["h1_corr"].map(norm)
    merged["h2_corr_n"] = merged["h2_corr"].map(norm)

    auto_mistimed = merged[merged["auto_mt"] == True].copy()
    n_auto_mistimed = int(len(auto_mistimed))

    if n_auto_mistimed == 0:
        return {
            "error": "No automatically mistimed cases were found.",
            "n_merged": int(len(merged)),
        }

    both_confirmed_mask = (
        (auto_mistimed["h1_tc_n"] == "yes")
        & (auto_mistimed["h2_tc_n"] == "yes")
    )

    both_rejected_to_wt_mask = (
        (auto_mistimed["h1_tc_n"] == "no")
        & (auto_mistimed["h2_tc_n"] == "no")
        & (auto_mistimed["h1_corr_n"] == "well_timed")
        & (auto_mistimed["h2_corr_n"] == "well_timed")
    )

    n_both_confirmed = int(both_confirmed_mask.sum())
    n_both_rejected_to_wt = int(both_rejected_to_wt_mask.sum())
    n_ambiguous = (
        n_auto_mistimed
        - n_both_confirmed
        - n_both_rejected_to_wt
    )

    precision = n_both_confirmed / n_auto_mistimed

    return {
        "n_merged": int(len(merged)),
        "n_auto_mistimed": n_auto_mistimed,
        "n_both_confirmed_tp": n_both_confirmed,
        "n_both_rejected_to_wt_fp": n_both_rejected_to_wt,
        "n_ambiguous": int(n_ambiguous),
        "precision": round(float(precision), 4),
        "precision_pct": round(float(precision * 100), 1),
        "annotator1_columns": cols1,
        "annotator2_columns": cols2,
        "scope_note": (
            "Strict pairwise precision: both annotators must confirm the "
            "automatic mistimed label. Ambiguous cases remain in the "
            "denominator."
        ),
    }


# ── Q4: Per-error-type breakdown ──────────────────────────────────────────────

def compute_per_error_breakdown(disagreements: pd.DataFrame,
                                 h_timing_col: str,
                                 a_timing_col: str,
                                 h_corr_col: str,
                                 a_corr_col: str,
                                 auto_col: str) -> pd.DataFrame:
    """
    For each auto error type, compute:
    - How many cases had that auto label
    - How many both annotators confirmed correct
    - How many were corrected to well_timed
    - How many were corrected to a different error type

    Clinical note: delayed_safety false positives are more concerning
    than stage_mismatch false positives. This breakdown shows which
    error types are most and least reliable.
    """
    df = disagreements.copy()
    df["_auto"]   = df[auto_col].map(norm) if auto_col in df.columns else ""
    df["_h_tc"]   = df[h_timing_col].map(norm)
    df["_a_tc"]   = df[a_timing_col].map(norm)
    df["_h_corr"] = df[h_corr_col].map(norm)
    df["_a_corr"] = df[a_corr_col].map(norm)

    rows = []
    for error_type in sorted(MISTIMED_LABELS):
        subset = df[df["_auto"] == error_type]
        if len(subset) == 0:
            continue
        n_total = len(subset)
        n_both_correct = ((subset["_h_tc"] == "yes") & (subset["_a_tc"] == "yes")).sum()
        n_both_to_wt   = (
            (subset["_h_tc"] == "no") & (subset["_a_tc"] == "no") &
            (subset["_h_corr"] == "well_timed") & (subset["_a_corr"] == "well_timed")
        ).sum()
        rows.append({
            "auto_error_type":          error_type,
            "n_in_disagreements":       n_total,
            "n_both_confirmed_correct": int(n_both_correct),
            "n_both_corrected_to_wt":   int(n_both_to_wt),
            "n_other":                  n_total - int(n_both_correct) - int(n_both_to_wt),
            "confirmation_rate":        round(int(n_both_correct)/n_total, 3),
            "fp_rate_to_wt":            round(int(n_both_to_wt)/n_total, 3),
            "clinical_note":            (
                "HIGHEST RISK — false positives hide safety failures"
                if error_type == "delayed_safety"
                else "")
        })
    return pd.DataFrame(rows)


# ── Paper reporting template ──────────────────────────────────────────────────

def paper_template(q1: dict, q2: dict, q3: dict) -> str:
    r1  = q1.get("subset_recall")
    nm  = q1.get("n_actually_mistimed", 0)
    nv  = q1.get("n_valid_corrections", 0)
    nwt = q1.get("n_actually_well_timed", 0)

    q2_available = bool(q2) and "error" not in q2 and q2.get("total_corrections", 0) > 0
    wt_count = q2.get("well_timed_count", 0)
    wt_pct   = q2.get("well_timed_pct", 0)
    total_c  = q2.get("total_corrections", 0)

    prec     = q3.get("precision_pct")
    n_tp     = q3.get("n_both_confirmed_tp")
    n_auto_m = q3.get("n_auto_mistimed")

    lines = []
    lines.append("── PAPER REPORTING TEMPLATE ─────────────────────────────────────────")
    lines.append("")
    lines.append("Section 3.3 paragraph (paste directly):")
    lines.append("")
    lines.append(
        "Analysis of correction patterns provides further diagnostic evidence."
    )
    if q2_available:
        lines.append(
            f" Across all {total_c} annotator-level timing corrections made when"
            f" annotators judged the automatic label incorrect, {wt_count}"
            f" ({wt_pct*100:.1f}\\%) were directed toward"
            f" \\texttt{{well\\_timed}}."
        )
    if prec is not None:
        lines.append(
            f" Among the {n_auto_m} rank-1 cases where the automatic system predicted"
            f" a timing error, both annotators confirmed the prediction correct in"
            f" {n_tp} cases ({prec:.1f}\\%), giving an estimated automatic timing"
            f" precision of {prec:.1f}\\%."
        )
    if r1 is not None:
        lines.append(
            f" Among the {nv} disagreement cases where both annotators provided"
            f" explicit corrections, {nm} correction{'s' if nm != 1 else ''}"
            f" ({nm/nv*100:.1f}\\%) confirmed a genuine timing error"
            f" (assigned a different label by the automatic system),"
            f" while {nwt} ({nwt/nv*100:.1f}\\%) were corrected to"
            f" \\texttt{{well\\_timed}}."
        )
    lines.append(
        " These results indicate that the automatic timing component should be"
        " treated as a conservative first-pass screener: it identifies candidate"
        " timing problems for human review rather than making reliable final judgments."
    )
    lines.append("")
    lines.append("── Limitations bullet (paste directly):")
    lines.append("")
    if prec is not None and q2_available:
        lines.append(
            f"\\item \\textbf{{Conservative automatic timing bias.}}"
            f" Strict pairwise validation estimated automatic timing precision"
            f" at {prec:.1f}\\%: both annotators confirmed {n_tp} of"
            f" {n_auto_m} automatically flagged timing errors."
            f" The dominant correction direction was"
            f" \\texttt{{well\\_timed}} ({wt_count} of {total_c}"
            f" annotator-level corrections, {wt_pct*100:.1f}\\%)."
            f" Final timing judgments require human review."
        )
    elif prec is not None:
        lines.append(
            f"\\item \\textbf{{Pairwise automatic timing precision.}}"
            f" Both annotators confirmed {n_tp} of {n_auto_m}"
            f" automatically flagged timing errors ({prec:.1f}\\%)."
            f" This pairwise diagnostic does not replace the primary"
            f" three-annotator majority-consensus evaluation."
        )
    elif q2_available:
        lines.append(
            f"\\item \\textbf{{Conservative automatic timing bias.}}"
            f" The dominant correction direction was"
            f" \\texttt{{well\\_timed}} ({wt_count} of {total_c}"
            f" annotator-level corrections, {wt_pct*100:.1f}\\%)."
            f" Automatic timing labels require human review."
        )
    else:
        lines.append(
            "\\item The pairwise disagreement-subset analysis is exploratory "
            "and does not replace the primary three-annotator "
            "majority-consensus evaluation."
        )
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute TheraTime automatic timing reliability from human annotations."
    )
    parser.add_argument("--disagreements", required=True,
                        help="theratime_disagreements.csv from theratime_kappa.py")
    parser.add_argument("--annotations1",  default=None,
                        help="Full annotation CSV for annotator 1 (optional, enables Q2/Q3)")
    parser.add_argument("--annotations2",  default=None,
                        help="Full annotation CSV for annotator 2 (optional, enables Q2/Q3)")
    parser.add_argument("--out",           default="theratime_recall_report.csv")
    parser.add_argument("--json",          default="theratime_recall_summary.json")
    args = parser.parse_args()

    SEP = "=" * 80

    # ── Load disagreements ────────────────────────────────────────────────────
    disagree_df = pd.read_csv(args.disagreements).fillna("")
    print(SEP)
    print("TheraTime Automatic Timing Reliability Report  v2.2")
    print(SEP)
    print(f"Disagreements file : {args.disagreements}")
    print(f"Disagreement rows  : {len(disagree_df)}")
    print()

    # Detect the two annotators dynamically from the column names generated
    # by theratime_kappa.py. This supports pairs such as:
    #   Asmae / external
    #   Hasnae / Asmae
    #   Internal_1 / External
    try:
        annotator_1, annotator_2 = detect_annotator_prefixes(disagree_df)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        print(f"Columns found: {list(disagree_df.columns)}")
        sys.exit(1)

    h_tc = f"{annotator_1}_timing"
    a_tc = f"{annotator_2}_timing"
    h_corr = f"{annotator_1}_timing_correction"
    a_corr = f"{annotator_2}_timing_correction"

    auto_c = next(
        (
            c
            for c in ["auto_timing", "timing_label", "predicted_timing"]
            if c in disagree_df.columns
        ),
        None,
    )

    print(f"Detected annotator pair: {annotator_1} vs {annotator_2}")
    print()

    # ── Q1: Subset recall ─────────────────────────────────────────────────────
    print("── Q1: Subset Recall (disagreements CSV only) ──────────────────────────")
    print("   SCOPE: Cases with stage/move disagreement where both annotators")
    print("   rejected the timing label AND provided corrections.")
    print("   NOT the system's overall precision — scoped to this subset only.")
    print()

    q1 = compute_subset_recall(disagree_df, h_tc, a_tc, h_corr, a_corr)

    print(f"   Both annotators rejected auto timing label : {q1['n_both_said_wrong']}")
    print(f"   Valid correction cases (both gave labels)  : {q1['n_valid_corrections']}")
    print(f"   Still genuinely mistimed (any error type)  : {q1['n_actually_mistimed']}")
    print(f"   Corrected to well_timed                    : {q1['n_actually_well_timed']}")
    if q1['subset_recall'] is not None:
        print(f"   Subset recall (within this scope)          : "
              f"{q1['subset_recall']:.4f} ({q1['subset_recall']*100:.1f}%)")
    print()
    print("   Status breakdown:")
    for k, v in sorted(q1['status_breakdown'].items(), key=lambda x: -x[1]):
        print(f"     {k:45s}: {v}")
    print()

    # ── Q2: Correction direction ──────────────────────────────────────────────
    q2 = {}
    if args.annotations1 and args.annotations2:
        print("── Q2: Correction Direction (full annotation files) ────────────────────")
        ann1 = drop_duplicate_columns(
            pd.read_csv(args.annotations1).fillna("")
        )
        ann2 = drop_duplicate_columns(
            pd.read_csv(args.annotations2).fillna("")
        )
        print(f"   Annotator 1 file : {args.annotations1} ({len(ann1)} rows)")
        print(f"   Annotator 2 file : {args.annotations2} ({len(ann2)} rows)")
        q2 = compute_correction_direction(ann1, ann2)
        if "error" in q2:
            print(f"   Warning: {q2['error']}")
            for diagnostic in q2.get("column_diagnostics", []):
                print(
                    "   Column detection "
                    f"({diagnostic['annotator']}): "
                    f"timing_correct={diagnostic['timing_correct_column']!r}, "
                    f"timing_correction={diagnostic['timing_correction_column']!r}"
                )
        else:
            print(f"   Total timing corrections  : {q2['total_corrections']}")
            print(f"   Toward well_timed         : {q2['well_timed_count']} "
                  f"({q2['well_timed_pct']*100:.1f}%)")
            print(f"   Toward other error types  :")
            for lbl, cnt in sorted(q2['mistimed_corrections'].items(),
                                    key=lambda x: -x[1]):
                print(f"     {lbl:35s}: {cnt}")
            print()

        # ── Q3: Auto precision ────────────────────────────────────────────────
        print("── Q3: Auto Timing Precision (most important for paper) ────────────────")
        q3 = compute_auto_precision(ann1, ann2)
        if "error" in q3:
            print(f"   Warning: {q3['error']}")
            q3 = {}
        else:
            print(f"   Auto-mistimed cases           : {q3['n_auto_mistimed']}")
            print(f"   Both confirmed correct (TP)   : {q3['n_both_confirmed_tp']}")
            print(f"   Both corrected to well_timed  : {q3['n_both_rejected_to_wt_fp']}")
            print(f"   Ambiguous                     : {q3['n_ambiguous']}")
            print(f"   AUTO TIMING PRECISION         : {q3['precision_pct']:.1f}%")
            print()
    else:
        print("── Q2 and Q3 skipped (no --annotations1 / --annotations2 provided) ────")
        print("   Provide full annotation CSVs to compute correction direction")
        print("   and auto precision — the most important numbers for the paper.")
        print()
        q3 = {}

    # ── Q4: Per-error breakdown ───────────────────────────────────────────────
    if auto_c:
        print("── Q4: Per-Error-Type Breakdown ────────────────────────────────────────")
        q4_df = compute_per_error_breakdown(
            disagree_df, h_tc, a_tc, h_corr, a_corr, auto_c)
        if not q4_df.empty:
            print(q4_df[["auto_error_type","n_in_disagreements",
                          "n_both_confirmed_correct","n_both_corrected_to_wt",
                          "confirmation_rate","fp_rate_to_wt",
                          "clinical_note"]].to_string(index=False))
            print()
            print("   Clinical priority note:")
            print("   delayed_safety false positives are MOST CONCERNING because they")
            print("   indicate the system flags safety responses as errors when they are not.")
            print("   Check these cases manually regardless of aggregate statistics.")
    else:
        q4_df = pd.DataFrame()
        print("── Q4 skipped (no auto_timing column in disagreements CSV) ─────────────")
    print()

    # ── Paper template ────────────────────────────────────────────────────────
    print(SEP)
    print(paper_template(q1, q2, q3))
    print(SEP)

    # ── Save outputs ──────────────────────────────────────────────────────────
    q1["detail_df"].to_csv(args.out, index=False)
    print(f"\nSaved detail CSV : {args.out}")

    summary = {
        "q1_subset_recall": {k: v for k, v in q1.items() if k != "detail_df"},
        "q2_correction_direction": q2,
        "q3_auto_precision": q3,
        "q4_per_error_rows": q4_df.to_dict(orient="records") if not q4_df.empty else [],
        "scope_note": (
            "subset_recall in Q1 applies only to cases in theratime_disagreements.csv "
            "where both annotators rejected the timing label. "
            "It is NOT the system's overall precision. "
            "Use Q3 auto_precision for the overall number."
        ),
    }
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary JSON: {args.json}")


if __name__ == "__main__":
    main()
