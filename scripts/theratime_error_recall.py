"""
theratime_error_recall.py  v2.0
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


def is_mistimed(label: str) -> bool | None:
    """Return True if label is a timing error, False if well_timed, None if unknown."""
    l = norm(label)
    if l in MISTIMED_LABELS:
        return True
    if l in WELL_TIMED_LABELS:
        return False
    return None


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
            "hasnae_correction": h,
            "asmae_correction":  a,
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

def compute_correction_direction(ann1: pd.DataFrame,
                                  ann2: pd.DataFrame) -> dict:
    """
    Across all human timing corrections (when annotator said auto was wrong),
    what is the distribution of correction targets?

    This gives the full-150 picture of which direction the auto system errs.
    """
    corrections = []

    for df, annotator in [(ann1, "annotator1"), (ann2, "annotator2")]:
        # Find timing_correct and timing_correction columns
        tc_col   = find_col(df, ["timing"], ["correction", "notes"])
        corr_col = find_col(df, ["timing", "correction"])
        if tc_col is None or corr_col is None:
            continue

        wrong = df[df[tc_col].map(norm) == "no"]
        for _, row in wrong.iterrows():
            c = norm(row[corr_col])
            if c in ALL_TIMING_LABELS:
                corrections.append({"annotator": annotator, "correction": c})

    if not corrections:
        return {"error": "Could not find timing correction columns in annotation files"}

    direction = Counter(r["correction"] for r in corrections)
    total = sum(direction.values())

    return {
        "total_corrections":    total,
        "well_timed_count":     direction.get("well_timed", 0),
        "well_timed_pct":       round(direction.get("well_timed", 0) / total, 4) if total else 0,
        "mistimed_corrections": {k: v for k, v in direction.items()
                                  if k in MISTIMED_LABELS},
        "full_breakdown":       dict(direction.most_common()),
    }


# ── Q3: Auto precision (needs full annotation + auto labels) ──────────────────

def compute_auto_precision(ann1: pd.DataFrame,
                            ann2: pd.DataFrame,
                            auto_col: str = "auto_timing") -> dict:
    """
    Of all cases where the auto system said MISTIMED, what fraction did
    BOTH annotators confirm as genuinely mistimed (any error type)?

    Precision = TP / (TP + FP)
    where TP = both said timing correct (auto correctly flagged an error)
          FP = both said timing wrong, correction was well_timed
               OR one said wrong, one said correct

    This is the most useful number for the paper.
    """
    # Merge on ID
    id1 = find_col(ann1, ["id"]) or ann1.columns[0]
    id2 = find_col(ann2, ["id"]) or ann2.columns[0]

    tc1  = find_col(ann1, ["timing"], ["correction","notes"])
    tc2  = find_col(ann2, ["timing"], ["correction","notes"])
    corr1 = find_col(ann1, ["timing","correction"])
    corr2 = find_col(ann2, ["timing","correction"])
    auto1 = auto_col if auto_col in ann1.columns else find_col(ann1, ["auto","timing"])

    if any(c is None for c in [tc1, tc2, corr1, corr2]):
        return {"error": "Missing required columns for precision computation"}

    merged = pd.merge(
        ann1[[id1, tc1, corr1] + ([auto1] if auto1 else [])].rename(
            columns={id1: "id", tc1: "h1_tc", corr1: "h1_corr",
                     **({auto1: "auto_timing"} if auto1 else {})}),
        ann2[[id2, tc2, corr2]].rename(
            columns={id2: "id", tc2: "h2_tc", corr2: "h2_corr"}),
        on="id", how="inner"
    )

    if "auto_timing" not in merged.columns:
        return {"error": "Could not locate auto_timing column"}

    merged["auto_mt"]  = merged["auto_timing"].map(norm).map(is_mistimed)
    merged["h1_tc_n"]  = merged["h1_tc"].map(norm)
    merged["h2_tc_n"]  = merged["h2_tc"].map(norm)
    merged["h1_corr_n"]= merged["h1_corr"].map(norm)
    merged["h2_corr_n"]= merged["h2_corr"].map(norm)

    # Only cases where auto said mistimed
    auto_mistimed = merged[merged["auto_mt"] == True].copy()
    n_auto_mistimed = len(auto_mistimed)

    if n_auto_mistimed == 0:
        return {"error": "No auto-mistimed cases found"}

    # TP: both said timing correct
    both_confirmed = (
        (auto_mistimed["h1_tc_n"] == "yes") &
        (auto_mistimed["h2_tc_n"] == "yes")
    ).sum()

    # FP_clear: both said timing wrong, both corrected to well_timed
    both_rejected_to_wt = (
        (auto_mistimed["h1_tc_n"] == "no") &
        (auto_mistimed["h2_tc_n"] == "no") &
        (auto_mistimed["h1_corr_n"] == "well_timed") &
        (auto_mistimed["h2_corr_n"] == "well_timed")
    ).sum()

    # Ambiguous: one agreed, one didn't; or corrections disagree
    ambiguous = n_auto_mistimed - both_confirmed - both_rejected_to_wt

    precision = int(both_confirmed) / n_auto_mistimed

    return {
        "n_auto_mistimed":          n_auto_mistimed,
        "n_both_confirmed_tp":      int(both_confirmed),
        "n_both_rejected_to_wt_fp": int(both_rejected_to_wt),
        "n_ambiguous":              int(ambiguous),
        "precision":                round(precision, 4),
        "precision_pct":            round(precision * 100, 1),
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
        f"Analysis of correction patterns provides further diagnostic evidence."
        f" Across all {total_c} timing corrections made when annotators judged the"
        f" automatic label incorrect, {wt_count} ({wt_pct*100:.1f}\\%) were directed"
        f" toward \\texttt{{well\\_timed}}, confirming a systematic conservative bias:"
        f" the rule engine flags responses as mistimed that humans judge as"
        f" contextually appropriate."
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
    if prec is not None:
        lines.append(
            f"\\item \\textbf{{Conservative automatic timing bias.}}"
            f" Human validation estimated automatic timing precision at {prec:.1f}\\%:"
            f" of cases flagged as timing errors, both annotators confirmed"
            f" {n_tp} of {n_auto_m} ({prec:.1f}\\%)."
            f" The dominant false-positive direction was \\texttt{{well\\_timed}}"
            f" ({wt_count} of {total_c} corrections, {wt_pct*100:.1f}\\%)."
            f" The framework therefore requires human review for final timing judgments."
        )
    else:
        lines.append(
            f"\\item \\textbf{{Conservative automatic timing bias.}}"
            f" The dominant correction direction was \\texttt{{well\\_timed}}"
            f" ({wt_count} of {total_c} corrections, {wt_pct*100:.1f}\\%)."
            f" Automatic timing labels require human review."
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
    print("TheraTime Automatic Timing Reliability Report  v2.0")
    print(SEP)
    print(f"Disagreements file : {args.disagreements}")
    print(f"Disagreement rows  : {len(disagree_df)}")
    print()

    # Locate columns in disagreements CSV
    h_tc   = find_col(disagree_df, ["hasnae","timing"],  ["correction","notes"])
    a_tc   = find_col(disagree_df, ["asmae","timing"],   ["correction","notes"])
    h_corr = find_col(disagree_df, ["hasnae","timing","correction"])
    a_corr = find_col(disagree_df, ["asmae","timing","correction"])
    auto_c = next((c for c in ["auto_timing","timing_label","predicted_timing"]
                   if c in disagree_df.columns), None)

    if None in [h_tc, a_tc, h_corr, a_corr]:
        print("ERROR: Cannot locate required columns in disagreements CSV.")
        print(f"Columns found: {list(disagree_df.columns)}")
        sys.exit(1)

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
        ann1 = pd.read_csv(args.annotations1).fillna("")
        ann2 = pd.read_csv(args.annotations2).fillna("")
        print(f"   Annotator 1 file : {args.annotations1} ({len(ann1)} rows)")
        print(f"   Annotator 2 file : {args.annotations2} ({len(ann2)} rows)")
        q2 = compute_correction_direction(ann1, ann2)
        if "error" in q2:
            print(f"   Warning: {q2['error']}")
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
