%%writefile theratime_kappa.py
"""
theratime_kappa.py
──────────────────
Compute inter-annotator agreement (Cohen's kappa) from two TheraTime
annotation files.

Accepted inputs:
  - CSV files exported by the HTML annotation tool
  - JSON files exported by the annotation tool

Usage examples:
  python theratime_kappa.py -f ann1.csv ann2.csv
  python theratime_kappa.py -f ann1.json ann2.json
  python theratime_kappa.py ann1.csv ann2.csv

Kaggle example:
  !python theratime_kappa.py -f \
    /kaggle/input/datasets/asmaeassmaebriouya/annotations/theratime_150_Hasnae_human_corrected_annotations.csv \
    /kaggle/input/datasets/asmaeassmaebriouya/annotations/theratime_human_annotations_Asmae_150_updated_reviewed.csv

Outputs:
  - kappa for stage_correct  (Q1)
  - kappa for move_correct   (Q2)
  - kappa for timing_correct (Q3)
  - per-label agreement breakdown
  - disagreement examples
  - saves: theratime_iaa_report.json + theratime_disagreements.csv
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

YES_NO = {"yes", "no"}
ANSWER_VALUES = {"yes", "no", "unsure"}


def normalise_answer(value):
    """Return normalized yes/no/unsure/empty annotation values."""
    value = str(value or "").strip().lower()
    if value in {"y", "true", "1", "correct"}:
        return "yes"
    if value in {"n", "false", "0", "incorrect", "wrong"}:
        return "no"
    if value in {"unsure", "uncertain", "maybe", "not sure"}:
        return "unsure"
    return value


def read_csv_annotations(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in CSV: {path}")

    annotator = ""
    for row in rows:
        annotator = str(row.get("annotator", "")).strip()
        if annotator:
            break
    if not annotator:
        annotator = Path(path).stem

    annotations = []
    for idx, row in enumerate(rows, start=1):
        clean = {str(k).strip(): v for k, v in row.items() if k is not None}
        clean["id"] = str(
            clean.get("id")
            or clean.get("query_id")
            or clean.get("example_id")
            or f"row_{idx}"
        ).strip()

        for field in ["stage_correct", "move_correct", "timing_correct"]:
            clean[field] = normalise_answer(clean.get(field, ""))

        annotations.append(clean)

    return {
        "annotator": annotator,
        "n_annotated": sum(
            1
            for a in annotations
            if a.get("stage_correct") in ANSWER_VALUES
            or a.get("move_correct") in ANSWER_VALUES
            or a.get("timing_correct") in ANSWER_VALUES
        ),
        "annotations": annotations,
    }


def read_json_annotations(path):
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, list):
        annotations = data
        annotator = Path(path).stem
    else:
        annotations = data.get("annotations") or data.get("rows") or data.get("data") or []
        annotator = str(data.get("annotator") or Path(path).stem).strip()

    if not annotations:
        raise ValueError(f"No annotations found in JSON: {path}")

    cleaned = []
    for idx, row in enumerate(annotations, start=1):
        clean = dict(row)
        clean["id"] = str(
            clean.get("id")
            or clean.get("query_id")
            or clean.get("example_id")
            or f"row_{idx}"
        ).strip()
        for field in ["stage_correct", "move_correct", "timing_correct"]:
            clean[field] = normalise_answer(clean.get(field, ""))
        cleaned.append(clean)

    n_annotated = data.get("n_annotated") if isinstance(data, dict) else None
    if n_annotated is None:
        n_annotated = sum(
            1
            for a in cleaned
            if a.get("stage_correct") in ANSWER_VALUES
            or a.get("move_correct") in ANSWER_VALUES
            or a.get("timing_correct") in ANSWER_VALUES
        )

    return {
        "annotator": annotator,
        "n_annotated": n_annotated,
        "annotations": cleaned,
    }


def load_annotations(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_annotations(path)
    if suffix == ".json":
        return read_json_annotations(path)

    # Fallback: try JSON first, then CSV.
    try:
        return read_json_annotations(path)
    except Exception:
        return read_csv_annotations(path)


# ── Cohen's kappa (unweighted) ───────────────────────────────

def cohen_kappa(labels_a, labels_b):
    """
    Compute Cohen's kappa between two annotators.
    Filters out pairs where either annotator answered unsure or left the field empty.
    Only yes/no pairs are used.
    """
    assert len(labels_a) == len(labels_b), "Label lists must be same length"

    valid = [
        (a, b)
        for a, b in zip(labels_a, labels_b)
        if a in YES_NO and b in YES_NO
    ]

    if not valid:
        return {"kappa": None, "po": None, "pe": None, "n_valid": 0, "n_skipped": len(labels_a)}

    n = len(valid)
    va, vb = zip(*valid)

    po = sum(a == b for a, b in valid) / n
    pe = sum((va.count(label) / n) * (vb.count(label) / n) for label in YES_NO)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    return {
        "kappa": round(kappa, 4),
        "po": round(po, 4),
        "pe": round(pe, 4),
        "n_valid": n,
        "n_skipped": len(labels_a) - n,
    }


def kappa_interpretation(k):
    if k is None:
        return "N/A"
    if k < 0:
        return "Poor (worse than chance)"
    if k < 0.20:
        return "Slight"
    if k < 0.40:
        return "Fair"
    if k < 0.60:
        return "Moderate"
    if k < 0.80:
        return "Substantial"
    return "Almost perfect"


def per_label_agreement(labels_a, labels_b):
    rows = []
    valid = [(a, b) for a, b in zip(labels_a, labels_b) if a in YES_NO and b in YES_NO]
    for val in sorted(YES_NO):
        pairs_with_val = [(a, b) for a, b in valid if a == val or b == val]
        if not pairs_with_val:
            continue
        agreed = sum(a == b for a, b in pairs_with_val)
        rows.append({
            "label": val,
            "n_pairs": len(pairs_with_val),
            "n_agreed": agreed,
            "agreement_rate": round(agreed / len(pairs_with_val), 4),
        })
    return rows


def print_kappa_block(title, result):
    print(title)
    print(f"  Cohen's kappa : {result['kappa']} ({kappa_interpretation(result['kappa'])})")
    print(f"  Observed po   : {result['po']}")
    print(f"  Expected pe   : {result['pe']}")
    print(f"  Valid pairs   : {result['n_valid']} (skipped unsure/missing: {result['n_skipped']})")
    print()


def collect_field(common_ids, ann1, ann2, field):
    return [ann1[i].get(field, "") for i in common_ids], [ann2[i].get(field, "") for i in common_ids]


def both_yes_count(common_ids, ann1, ann2, field):
    return sum(1 for i in common_ids if ann1[i].get(field) == "yes" and ann2[i].get(field) == "yes")


def correction_counter(common_ids, ann1, ann2, correct_field, correction_field):
    corrections = []
    for source in [ann1, ann2]:
        corrections.extend(
            source[i].get(correction_field, "")
            for i in common_ids
            if source[i].get(correct_field) == "no" and source[i].get(correction_field, "")
        )
    return Counter(corrections)


def main():
    parser = argparse.ArgumentParser(
        description="Compute Cohen's kappa for two TheraTime annotation CSV/JSON files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Two annotation files. Kept for backward compatibility: python theratime_kappa.py ann1.csv ann2.csv",
    )
    parser.add_argument(
        "-f",
        "--files-input",
        nargs=2,
        metavar=("ANNOTATOR_1", "ANNOTATOR_2"),
        help="Two annotation files, CSV or JSON.",
    )
    parser.add_argument(
        "--out-prefix",
        default="theratime",
        help="Prefix for output files. Default: theratime",
    )
    args = parser.parse_args()

    if args.files_input:
        input_files = args.files_input
    elif len(args.files) == 2:
        input_files = args.files
    else:
        parser.error("Provide exactly two files, either with -f ann1.csv ann2.csv or as positional arguments.")

    path1, path2 = Path(input_files[0]), Path(input_files[1])
    data1 = load_annotations(path1)
    data2 = load_annotations(path2)

    ann1 = {str(a["id"]): a for a in data1["annotations"]}
    ann2 = {str(a["id"]): a for a in data2["annotations"]}

    common_ids = sorted(set(ann1.keys()) & set(ann2.keys()))
    n_common = len(common_ids)

    print(f"\n{'=' * 60}")
    print("TheraTime Inter-Annotator Agreement Report")
    print(f"{'=' * 60}")
    print(f"Annotator 1 : {data1['annotator']} ({data1['n_annotated']} labelled)")
    print(f"Annotator 2 : {data2['annotator']} ({data2['n_annotated']} labelled)")
    print(f"File 1      : {path1}")
    print(f"File 2      : {path2}")
    print(f"Common IDs  : {n_common}")
    print()

    if n_common == 0:
        print("ERROR: No common sample IDs found. Check that both annotators used the same sample list.")
        sys.exit(1)

    stage_a, stage_b = collect_field(common_ids, ann1, ann2, "stage_correct")
    move_a, move_b = collect_field(common_ids, ann1, ann2, "move_correct")
    timing_a, timing_b = collect_field(common_ids, ann1, ann2, "timing_correct")

    stage_kappa = cohen_kappa(stage_a, stage_b)
    move_kappa = cohen_kappa(move_a, move_b)
    timing_kappa = cohen_kappa(timing_a, timing_b)

    print_kappa_block("── Q1: Stage Classification Agreement ──────────────────", stage_kappa)
    print_kappa_block("── Q2: Move Classification Agreement ───────────────────", move_kappa)
    print_kappa_block("── Q3: Timing Label Agreement ──────────────────────────", timing_kappa)

    for name, labels_a, labels_b in [
        ("stage", stage_a, stage_b),
        ("move", move_a, move_b),
        ("timing", timing_a, timing_b),
    ]:
        print(f"── Per-label breakdown ({name}) ─────────────────────────")
        for row in per_label_agreement(labels_a, labels_b):
            print(f"  {row['label']:8s}: {row['n_agreed']}/{row['n_pairs']} agreed ({row['agreement_rate'] * 100:.1f}%)")
        print()

    disagreements = []
    fields = ["stage_correct", "move_correct", "timing_correct"]
    for sid in common_ids:
        a1 = ann1[sid]
        a2 = ann2[sid]
        disagree_flags = {}
        has_disagreement = False

        for field in fields:
            agree = a1.get(field) == a2.get(field) and a1.get(field) in YES_NO
            disagree_flags[field.replace("_correct", "_disagree")] = not agree
            if not agree:
                has_disagreement = True

        if has_disagreement:
            disagreements.append({
                "id": sid,
                "query": a1.get("query", ""),
                "response": a1.get("response", ""),
                "auto_stage": a1.get("auto_stage", ""),
                "auto_move": a1.get("auto_move", ""),
                "auto_timing": a1.get("auto_timing", ""),
                f"{data1['annotator']}_stage": a1.get("stage_correct", ""),
                f"{data2['annotator']}_stage": a2.get("stage_correct", ""),
                f"{data1['annotator']}_stage_correction": a1.get("stage_correction", ""),
                f"{data2['annotator']}_stage_correction": a2.get("stage_correction", ""),
                f"{data1['annotator']}_move": a1.get("move_correct", ""),
                f"{data2['annotator']}_move": a2.get("move_correct", ""),
                f"{data1['annotator']}_move_correction": a1.get("move_correction", ""),
                f"{data2['annotator']}_move_correction": a2.get("move_correction", ""),
                f"{data1['annotator']}_timing": a1.get("timing_correct", ""),
                f"{data2['annotator']}_timing": a2.get("timing_correct", ""),
                f"{data1['annotator']}_timing_correction": a1.get("timing_correction", ""),
                f"{data2['annotator']}_timing_correction": a2.get("timing_correction", ""),
                f"{data1['annotator']}_notes": a1.get("notes", ""),
                f"{data2['annotator']}_notes": a2.get("notes", ""),
                **disagree_flags,
            })

    print("── Disagreements ────────────────────────────────────────")
    print(f"  Stage disagreements : {sum(1 for d in disagreements if d['stage_disagree'])}")
    print(f"  Move disagreements  : {sum(1 for d in disagreements if d['move_disagree'])}")
    print(f"  Timing disagreements: {sum(1 for d in disagreements if d['timing_disagree'])}")
    print(f"  Total items with at least one disagreement: {len(disagreements)}")
    print()

    stage_corrections = correction_counter(common_ids, ann1, ann2, "stage_correct", "stage_correction")
    move_corrections = correction_counter(common_ids, ann1, ann2, "move_correct", "move_correction")
    timing_corrections = correction_counter(common_ids, ann1, ann2, "timing_correct", "timing_correction")

    if stage_corrections:
        print("── Most common stage corrections ───────────────────────")
        for label, count in stage_corrections.most_common():
            print(f"  {label:35s}: {count}x")
        print()

    if move_corrections:
        print("── Most common move corrections ────────────────────────")
        for label, count in move_corrections.most_common():
            print(f"  {label:35s}: {count}x")
        print()

    if timing_corrections:
        print("── Most common timing corrections ──────────────────────")
        for label, count in timing_corrections.most_common():
            print(f"  {label:35s}: {count}x")
        print()

    both_stage_correct = both_yes_count(common_ids, ann1, ann2, "stage_correct")
    both_move_correct = both_yes_count(common_ids, ann1, ann2, "move_correct")
    both_timing_correct = both_yes_count(common_ids, ann1, ann2, "timing_correct")

    print("── Automatic label accuracy, human-validated ───────────")
    if stage_kappa["n_valid"]:
        print(f"  Stage accuracy : {both_stage_correct}/{stage_kappa['n_valid']} = {both_stage_correct / stage_kappa['n_valid'] * 100:.1f}%")
    if move_kappa["n_valid"]:
        print(f"  Move accuracy  : {both_move_correct}/{move_kappa['n_valid']} = {both_move_correct / move_kappa['n_valid'] * 100:.1f}%")
    if timing_kappa["n_valid"]:
        print(f"  Timing accuracy: {both_timing_correct}/{timing_kappa['n_valid']} = {both_timing_correct / timing_kappa['n_valid'] * 100:.1f}%")
    print("  Note: cases where annotators disagreed or used unsure/missing are excluded from these accuracy denominators.")
    print()

    report = {
        "annotator_1": data1["annotator"],
        "annotator_2": data2["annotator"],
        "file_1": str(path1),
        "file_2": str(path2),
        "n_common_samples": n_common,
        "stage_kappa": stage_kappa,
        "stage_kappa_interpretation": kappa_interpretation(stage_kappa["kappa"]),
        "move_kappa": move_kappa,
        "move_kappa_interpretation": kappa_interpretation(move_kappa["kappa"]),
        "timing_kappa": timing_kappa,
        "timing_kappa_interpretation": kappa_interpretation(timing_kappa["kappa"]),
        "stage_auto_accuracy_human_validated": round(both_stage_correct / stage_kappa["n_valid"], 4) if stage_kappa["n_valid"] else None,
        "move_auto_accuracy_human_validated": round(both_move_correct / move_kappa["n_valid"], 4) if move_kappa["n_valid"] else None,
        "timing_auto_accuracy_human_validated": round(both_timing_correct / timing_kappa["n_valid"], 4) if timing_kappa["n_valid"] else None,
        "n_disagreements": len(disagreements),
        "stage_corrections_most_common": stage_corrections.most_common(10),
        "move_corrections_most_common": move_corrections.most_common(10),
        "timing_corrections_most_common": timing_corrections.most_common(10),
    }

    report_path = Path(f"{args.out_prefix}_iaa_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved: {report_path}")

    if disagreements:
        disagree_path = Path(f"{args.out_prefix}_disagreements.csv")
        with open(disagree_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(disagreements[0].keys()))
            writer.writeheader()
            writer.writerows(disagreements)
        print(f"Saved: {disagree_path}")

    print(f"\n{'=' * 60}")
    print("PAPER REPORTING TEMPLATE:")
    print(f"{'=' * 60}")
    if stage_kappa["n_valid"] and move_kappa["n_valid"]:
        print(
            f"Inter-annotator agreement was assessed by two independent annotators "
            f"({data1['annotator']} and {data2['annotator']}) on {n_common} shared timing judgments. "
            f"Cohen's kappa for support-stage correctness was kappa = {stage_kappa['kappa']} "
            f"({kappa_interpretation(stage_kappa['kappa']).lower()} agreement), "
            f"for support-move correctness was kappa = {move_kappa['kappa']} "
            f"({kappa_interpretation(move_kappa['kappa']).lower()} agreement), "
            f"and for timing-label correctness was kappa = {timing_kappa['kappa']} "
            f"({kappa_interpretation(timing_kappa['kappa']).lower()} agreement)."
        )
    else:
        print("Complete both annotation files to generate the reporting template.")
    print()


if __name__ == "__main__":
    main()
