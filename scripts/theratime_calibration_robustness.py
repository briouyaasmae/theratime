%%writefile theratime_calibration_robustness.py
#!/usr/bin/env python3
"""
theratime_calibration_robustness.py
====================================

Robustness layer on top of theratime_post_calibration.py (v2).

This module does NOT reimplement the calibration pipeline. It imports the
v2 functions (consensus building, dev/held-out splitting, rule learning,
isotonic reliability, the safe_keep_correct_review policy) and adds three
things that a reviewer would reasonably ask for once you claim a calibration
improvement on a ~150-300 item human-annotated sample:

1. BOOTSTRAP CONFIDENCE INTERVALS
   Percentile bootstrap over the held-out evaluation rows, for each method's
   held-out timing/stage/move accuracy. Answers: "how much would this number
   move if we'd happened to sample a slightly different held-out set?"

2. MULTI-SEED STABILITY
   Re-runs the ENTIRE pipeline (dev/held-out split -> inner train/validation
   split -> rule learning with the do-no-harm gate -> isotonic fit ->
   held-out evaluation) under several different random seeds. Reports the
   mean / std / min / max of held-out timing accuracy per method across
   seeds. Answers: "is the reported gain a property of the method, or an
   artifact of one lucky 70/30 split?"

3. THRESHOLD SENSITIVITY SWEEP
   For the safe_keep_correct_review policy specifically, sweeps
   policy_keep_threshold and policy_correction_max_original_reliability over
   a grid and reports held-out accepted-coverage / accepted-accuracy /
   review-rate for each combination. Answers: "why these threshold values,
   and how much do results change if we'd picked slightly different ones?"

Nothing here changes labels beyond what v2 already does. This module only
adds uncertainty quantification and sensitivity analysis around v2's
existing, defensible pipeline.

Typical usage
-------------
python theratime_calibration_robustness.py \
  --auto all_judgments_mpnet.csv \
  --ann theratime_300_internal2.csv theratime_300_internal1.csv theratime_300_external.csv \
  --out-dir theratime_robustness_outputs \
  --methods baseline conservative_human_recompute safe_keep_correct_review \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --n-bootstrap 2000

Recommended paper wording
--------------------------
"Held-out timing accuracy is reported with 95% percentile bootstrap
confidence intervals (2000 resamples) over the held-out evaluation subset.
To assess sensitivity to the specific development/held-out split, the full
calibration pipeline -- including do-no-harm rule learning -- was repeated
under 10 random seeds; we report the mean and range of held-out timing
accuracy across seeds. The safe_keep_correct_review policy's reliability
and correction thresholds were selected via a sensitivity sweep reported in
the supplementary material, rather than a single untested default."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import theratime_post_calibration as tpc


# =============================================================================
# 1. Bootstrap confidence intervals
# =============================================================================

def bootstrap_ci_from_bool_series(
    correct: pd.Series,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap CI for a proportion (e.g. accuracy) over rows."""
    values = correct.astype(float).to_numpy()
    n = len(values)

    if n == 0:
        return {
            "point": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
            "n_bootstrap": int(n_bootstrap),
        }

    rng = np.random.default_rng(seed)
    point = float(values.mean())

    # Resample row indices with replacement, n_bootstrap times.
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = values[boot_indices].mean(axis=1)

    ci_low, ci_high = np.percentile(
        boot_means,
        [100 * alpha / 2, 100 * (1 - alpha / 2)],
    )

    return {
        "point": round(point, 4),
        "ci_low": round(float(ci_low), 4),
        "ci_high": round(float(ci_high), 4),
        "n": int(n),
        "n_bootstrap": int(n_bootstrap),
        "alpha": float(alpha),
    }


def bootstrap_ci_for_heldout_evaluation(
    evaluation_df: pd.DataFrame,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap CIs for stage/move/timing calibrated accuracy on one
    held-out evaluation dataframe (as returned by evaluate_method_on_heldout).
    """
    results: Dict[str, Dict[str, float]] = {}

    for label_type, seed_offset in [("stage", 0), ("move", 1), ("timing", 2)]:
        column = f"calibrated_{label_type}_correct"
        if column not in evaluation_df.columns:
            continue
        results[label_type] = bootstrap_ci_from_bool_series(
            evaluation_df[column],
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=seed + seed_offset,
        )

    # Also bootstrap the "accepted subset" accuracy for the safe policy,
    # since that's the number most likely to be quoted in the paper.
    if "policy_accepted" in evaluation_df.columns:
        accepted = evaluation_df[evaluation_df["policy_accepted"]]
        results["policy_accepted_timing"] = bootstrap_ci_from_bool_series(
            accepted["calibrated_timing_correct"],
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=seed + 3,
        )

    return results


# =============================================================================
# 2. Single full-pipeline run (one seed) -- mirrors v2's main(), as a function
# =============================================================================

def run_pipeline_once(
    auto_path: Path,
    annotation_paths: Sequence[Path],
    methods: Sequence[str],
    seed: int,
    consensus_mode: str = "auto",
    heldout_size: float = 0.30,
    standard_min_support: int = 3,
    standard_min_error_rate: float = 0.50,
    conservative_min_support: int = 5,
    conservative_min_error_rate: float = 0.60,
    do_no_harm: bool = True,
    inner_validation_size: float = 0.25,
    min_rule_affected: int = 3,
    min_rule_accuracy_gain: float = 0.0,
    stage_confidence_percentile: float = 10.0,
    move_confidence_percentile: float = 10.0,
    isotonic_reliability_threshold: float = 0.50,
    policy_keep_threshold: float = 0.75,
    policy_correction_max_original_reliability: float = 0.60,
    policy_min_rule_support: int = 5,
    policy_min_rule_error_rate: float = 0.60,
    policy_min_rule_validation_gain: float = 0.0,
) -> Dict[str, Any]:
    """Run the full v2 pipeline once under a given random seed.

    Returns a dict with:
      - 'heldout_summaries': {method: summary_dict}
      - 'heldout_evaluations': {method: evaluation_dataframe}
      - 'split_meta', 'standard_meta'
    """
    base_df = tpc.prepare_auto_df(auto_path)
    base_df, _confidence_meta = tpc.compute_confidence_flags(
        base_df,
        stage_percentile=stage_confidence_percentile,
        move_percentile=move_confidence_percentile,
    )

    standard_consensus_df, standard_meta = tpc.build_consensus(
        annotation_paths=annotation_paths,
        require_same_correction=False,
        consensus_mode=consensus_mode,
    )
    conservative_consensus_df, _conservative_meta = tpc.build_consensus(
        annotation_paths=annotation_paths,
        require_same_correction=True,
        consensus_mode=consensus_mode,
    )

    base_df, _id_config = tpc.configure_auto_id_column(
        auto_df=base_df,
        annotation_ids=standard_consensus_df["id"].tolist(),
    )
    tpc.validate_annotation_ids_against_auto(base_df, standard_consensus_df)

    standard_development, standard_heldout, split_meta = tpc.split_consensus_data(
        consensus_df=standard_consensus_df,
        test_size=heldout_size,
        random_state=seed,
    )

    development_ids = set(standard_development["id"])
    heldout_ids = set(standard_heldout["id"])

    conservative_development = (
        conservative_consensus_df[conservative_consensus_df["id"].isin(development_ids)]
        .copy()
        .reset_index(drop=True)
    )

    if do_no_harm:
        standard_stage_rules, standard_move_rules, _std_safe_meta = (
            tpc.learn_do_no_harm_recompute_rules(
                development_df=standard_development,
                min_support=standard_min_support,
                min_error_rate=standard_min_error_rate,
                validation_size=inner_validation_size,
                random_state=seed + 100,
                min_affected=min_rule_affected,
                min_accuracy_gain=min_rule_accuracy_gain,
            )
        )
        conservative_stage_rules, conservative_move_rules, _cons_safe_meta = (
            tpc.learn_do_no_harm_recompute_rules(
                development_df=conservative_development,
                min_support=conservative_min_support,
                min_error_rate=conservative_min_error_rate,
                validation_size=inner_validation_size,
                random_state=seed + 200,
                min_affected=min_rule_affected,
                min_accuracy_gain=min_rule_accuracy_gain,
            )
        )
        standard_timing_rules, _direct_safe_meta = tpc.learn_do_no_harm_direct_rules(
            development_df=standard_development,
            min_support=standard_min_support,
            min_error_rate=standard_min_error_rate,
            validation_size=inner_validation_size,
            random_state=seed + 300,
            min_affected=min_rule_affected,
            min_accuracy_gain=min_rule_accuracy_gain,
        )
        conservative_timing_rules: Dict[Any, Any] = {}
    else:
        standard_stage_rules = tpc.learn_stage_rules(
            standard_development, standard_min_support, standard_min_error_rate
        )
        standard_move_rules = tpc.learn_conditional_move_rules(
            standard_development, standard_min_support, standard_min_error_rate
        )
        standard_timing_rules = tpc.learn_direct_timing_rules(
            standard_development, standard_min_support, standard_min_error_rate
        )
        conservative_stage_rules = tpc.learn_stage_rules(
            conservative_development, conservative_min_support, conservative_min_error_rate
        )
        conservative_move_rules = tpc.learn_conditional_move_rules(
            conservative_development, conservative_min_support, conservative_min_error_rate
        )
        conservative_timing_rules = tpc.learn_direct_timing_rules(
            conservative_development, conservative_min_support, conservative_min_error_rate
        )

    development_auto_rows = tpc.make_heldout_auto_rows(base_df, standard_development)
    development_with_humans = development_auto_rows.merge(
        standard_development[
            ["id", "human_stage_correct", "human_move_correct", "human_timing_correct"]
        ].rename(columns={"id": "id_for_calibration"}),
        on="id_for_calibration",
        how="inner",
        validate="one_to_one",
    )

    isotonic_base_df, _iso_meta, _iso_models = tpc.add_isotonic_reliability(
        base_df=base_df,
        development_df=development_with_humans,
        reliability_threshold=isotonic_reliability_threshold,
    )

    heldout_summaries: Dict[str, Dict[str, Any]] = {}
    heldout_evaluations: Dict[str, pd.DataFrame] = {}

    for method in methods:
        # isotonic_base_df is base_df plus isotonic_*_reliability columns --
        # a pure addition, so using it as the source for EVERY method (not
        # just the ones whose label logic reads those columns) is safe and
        # ensures isotonic_overall_reliability is always available for
        # coverage_target_report, regardless of which methods were requested.
        method_base = isotonic_base_df.copy()

        if method == "human_rules_direct":
            stage_rules, move_rules, timing_rules = (
                standard_stage_rules, standard_move_rules, standard_timing_rules
            )
        elif method == "human_rules_recompute":
            stage_rules, move_rules, timing_rules = (
                standard_stage_rules, standard_move_rules, {}
            )
        elif method in {
            "conservative_human_recompute",
            "conservative_recompute_with_isotonic_reliability",
        }:
            stage_rules, move_rules, timing_rules = (
                conservative_stage_rules, conservative_move_rules, conservative_timing_rules
            )
        else:
            stage_rules, move_rules, timing_rules = {}, {}, {}

        if method == "safe_keep_correct_review":
            method_output = tpc.apply_safe_keep_correct_review_policy(
                base_df=method_base,
                timing_rules=standard_timing_rules,
                keep_threshold=policy_keep_threshold,
                correction_max_original_reliability=policy_correction_max_original_reliability,
                min_rule_support=policy_min_rule_support,
                min_rule_error_rate=policy_min_rule_error_rate,
                min_rule_validation_gain=policy_min_rule_validation_gain,
            )
        else:
            method_output = tpc.apply_calibration_method(
                base_df=method_base,
                method=method,
                stage_rules=stage_rules,
                move_rules=move_rules,
                timing_rules=timing_rules,
            )

        summary, evaluation = tpc.evaluate_method_on_heldout(
            method_output=method_output,
            heldout_df=standard_heldout,
            method=method,
        )
        heldout_summaries[method] = summary
        heldout_evaluations[method] = evaluation

    return {
        "seed": seed,
        "heldout_summaries": heldout_summaries,
        "heldout_evaluations": heldout_evaluations,
        "split_meta": split_meta,
        "standard_meta": standard_meta,
        "standard_timing_rules": standard_timing_rules,
        "isotonic_base_df": isotonic_base_df,
        "standard_development": standard_development,
        "standard_heldout": standard_heldout,
    }


# =============================================================================
# 2.5 K-fold pooled held-out evaluation (tightens CI WITHOUT new annotation)
# =============================================================================
#
# A single 70/30 development/held-out split only ever EVALUATES the 30% held
# out; the other 70% is only ever used to LEARN rules. That means a 300-item
# annotated sample gives an accuracy estimate with the precision of n=90,
# not n=300 -- most of the collected labels never contribute to the reported
# confidence interval.
#
# K-fold cross-validation fixes this without collecting a single new label:
# rotate which 1/K of the consensus data is held out, learn rules on the
# other (K-1)/K each time (with the same do-no-harm gate as a single split),
# and evaluate each fold's held-out portion. Every item is held out exactly
# once, by a model that never saw it during rule learning, so pooling all K
# folds' predictions gives a legitimate out-of-fold accuracy estimate over
# the FULL annotated sample (~300, not ~90) -- roughly halving the
# confidence interval width for the same annotation effort already spent.

def run_kfold_pooled_evaluation(
    auto_path: Path,
    annotation_paths: Sequence[Path],
    methods: Sequence[str],
    k_folds: int = 5,
    seed: int = 42,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    consensus_mode: str = "auto",
    standard_min_support: int = 3,
    standard_min_error_rate: float = 0.50,
    conservative_min_support: int = 5,
    conservative_min_error_rate: float = 0.60,
    do_no_harm: bool = True,
    inner_validation_size: float = 0.25,
    min_rule_affected: int = 3,
    min_rule_accuracy_gain: float = 0.0,
    stage_confidence_percentile: float = 10.0,
    move_confidence_percentile: float = 10.0,
    isotonic_reliability_threshold: float = 0.50,
    policy_keep_threshold: float = 0.75,
    policy_correction_max_original_reliability: float = 0.60,
    policy_min_rule_support: int = 5,
    policy_min_rule_error_rate: float = 0.60,
    policy_min_rule_validation_gain: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Rotate a K-fold split over the human consensus data, pooling
    out-of-fold held-out predictions across all folds.

    Returns:
      summary_df  -- one row per method: pooled n, pooled accuracy, and its
                      bootstrap CI (this is the number that goes in the
                      paper -- computed over ~all annotated items, not just
                      one fold's 1/K share).
      pooled_evaluations -- {method: concatenated per-fold evaluation
                      dataframe}, for further analysis (e.g. feeding into
                      coverage_target_report at full pooled n).
    """
    if not tpc.SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for K-fold splitting.")

    from sklearn.model_selection import StratifiedKFold

    base_df = tpc.prepare_auto_df(auto_path)
    base_df, _cm = tpc.compute_confidence_flags(
        base_df, stage_confidence_percentile, move_confidence_percentile
    )

    standard_consensus_df, _sm = tpc.build_consensus(
        annotation_paths, require_same_correction=False, consensus_mode=consensus_mode
    )
    conservative_consensus_df, _cm2 = tpc.build_consensus(
        annotation_paths, require_same_correction=True, consensus_mode=consensus_mode
    )
    base_df, _idc = tpc.configure_auto_id_column(
        base_df, standard_consensus_df["id"].tolist()
    )
    tpc.validate_annotation_ids_against_auto(base_df, standard_consensus_df)

    valid = standard_consensus_df[
        standard_consensus_df["human_timing_correct"].isin(tpc.YES_NO)
    ].copy().reset_index(drop=True)

    if len(valid) < k_folds * 10:
        raise ValueError(
            f"Only {len(valid)} items have valid timing consensus -- too few "
            f"for a reliable {k_folds}-fold split (need at least ~{k_folds*10})."
        )

    # Stratify folds by auto_timing so each fold's held-out portion has a
    # similar label mix, mirroring the stratification used for the single
    # 70/30 split.
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_assignment = np.zeros(len(valid), dtype=int)
    for fold_index, (_train_idx, test_idx) in enumerate(
        skf.split(valid, valid["auto_timing"])
    ):
        fold_assignment[test_idx] = fold_index
    valid["_fold"] = fold_assignment

    pooled_evaluations: Dict[str, List[pd.DataFrame]] = {m: [] for m in methods}

    for fold_index in range(k_folds):
        fold_development = valid[valid["_fold"] != fold_index].reset_index(drop=True)
        fold_heldout = valid[valid["_fold"] == fold_index].reset_index(drop=True)

        fold_dev_ids = set(fold_development["id"])
        fold_conservative_development = (
            conservative_consensus_df[conservative_consensus_df["id"].isin(fold_dev_ids)]
            .copy()
            .reset_index(drop=True)
        )

        if do_no_harm:
            fold_stage_rules, fold_move_rules, _meta1 = tpc.learn_do_no_harm_recompute_rules(
                development_df=fold_development,
                min_support=standard_min_support,
                min_error_rate=standard_min_error_rate,
                validation_size=inner_validation_size,
                random_state=seed + 100 + fold_index,
                min_affected=min_rule_affected,
                min_accuracy_gain=min_rule_accuracy_gain,
            )
            fold_cons_stage_rules, fold_cons_move_rules, _meta2 = (
                tpc.learn_do_no_harm_recompute_rules(
                    development_df=fold_conservative_development,
                    min_support=conservative_min_support,
                    min_error_rate=conservative_min_error_rate,
                    validation_size=inner_validation_size,
                    random_state=seed + 200 + fold_index,
                    min_affected=min_rule_affected,
                    min_accuracy_gain=min_rule_accuracy_gain,
                )
            )
            fold_timing_rules, _meta3 = tpc.learn_do_no_harm_direct_rules(
                development_df=fold_development,
                min_support=standard_min_support,
                min_error_rate=standard_min_error_rate,
                validation_size=inner_validation_size,
                random_state=seed + 300 + fold_index,
                min_affected=min_rule_affected,
                min_accuracy_gain=min_rule_accuracy_gain,
            )
        else:
            fold_stage_rules = tpc.learn_stage_rules(
                fold_development, standard_min_support, standard_min_error_rate
            )
            fold_move_rules = tpc.learn_conditional_move_rules(
                fold_development, standard_min_support, standard_min_error_rate
            )
            fold_timing_rules = tpc.learn_direct_timing_rules(
                fold_development, standard_min_support, standard_min_error_rate
            )
            fold_cons_stage_rules = tpc.learn_stage_rules(
                fold_conservative_development, conservative_min_support, conservative_min_error_rate
            )
            fold_cons_move_rules = tpc.learn_conditional_move_rules(
                fold_conservative_development, conservative_min_support, conservative_min_error_rate
            )

        fold_dev_auto_rows = tpc.make_heldout_auto_rows(base_df, fold_development)
        fold_dev_with_humans = fold_dev_auto_rows.merge(
            fold_development[
                ["id", "human_stage_correct", "human_move_correct", "human_timing_correct"]
            ].rename(columns={"id": "id_for_calibration"}),
            on="id_for_calibration",
            how="inner",
            validate="one_to_one",
        )

        fold_isotonic_base_df, _iso_meta, _iso_models = tpc.add_isotonic_reliability(
            base_df=base_df,
            development_df=fold_dev_with_humans,
            reliability_threshold=isotonic_reliability_threshold,
        )

        for method in methods:
            if method == "human_rules_direct":
                stage_rules, move_rules, timing_rules = (
                    fold_stage_rules, fold_move_rules, fold_timing_rules
                )
            elif method == "human_rules_recompute":
                stage_rules, move_rules, timing_rules = (
                    fold_stage_rules, fold_move_rules, {}
                )
            elif method in {
                "conservative_human_recompute",
                "conservative_recompute_with_isotonic_reliability",
            }:
                stage_rules, move_rules, timing_rules = (
                    fold_cons_stage_rules, fold_cons_move_rules, {}
                )
            else:
                stage_rules, move_rules, timing_rules = {}, {}, {}

            if method == "safe_keep_correct_review":
                method_output = tpc.apply_safe_keep_correct_review_policy(
                    base_df=fold_isotonic_base_df,
                    timing_rules=fold_timing_rules,
                    keep_threshold=policy_keep_threshold,
                    correction_max_original_reliability=policy_correction_max_original_reliability,
                    min_rule_support=policy_min_rule_support,
                    min_rule_error_rate=policy_min_rule_error_rate,
                    min_rule_validation_gain=policy_min_rule_validation_gain,
                )
            else:
                method_output = tpc.apply_calibration_method(
                    base_df=fold_isotonic_base_df,
                    method=method,
                    stage_rules=stage_rules,
                    move_rules=move_rules,
                    timing_rules=timing_rules,
                )

            _fold_summary, fold_evaluation = tpc.evaluate_method_on_heldout(
                method_output=method_output,
                heldout_df=fold_heldout,
                method=method,
            )
            fold_evaluation["_fold"] = fold_index
            pooled_evaluations[method].append(fold_evaluation)

    summary_rows: List[Dict[str, Any]] = []
    pooled_dfs: Dict[str, pd.DataFrame] = {}

    for method in methods:
        pooled = pd.concat(pooled_evaluations[method], ignore_index=True)
        pooled_dfs[method] = pooled

        boot = bootstrap_ci_from_bool_series(
            pooled["calibrated_timing_correct"],
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=seed,
        )

        summary_rows.append(
            {
                "method": method,
                "k_folds": k_folds,
                "n_pooled": boot["n"],
                "pooled_timing_accuracy": boot["point"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "ci_half_width": round((boot["ci_high"] - boot["ci_low"]) / 2, 4),
            }
        )

    return pd.DataFrame(summary_rows), pooled_dfs


# =============================================================================
# 2.6 Paired bootstrap difference test (the correct test for two methods
# evaluated on the SAME held-out items -- more powerful than eyeballing
# whether two separate marginal CIs overlap)
# =============================================================================

def paired_bootstrap_difference(
    pooled_evaluations: Dict[str, pd.DataFrame],
    method_a: str,
    method_b: str,
    id_column: str = "id_for_calibration",
    correct_column: str = "calibrated_timing_correct",
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Test whether method_b's accuracy differs from method_a's, on the same
    set of items, using a paired percentile bootstrap on the difference.

    Two separate marginal CIs (as in run_kfold_pooled_evaluation's summary)
    can look "barely overlapping" while actually understating how confident
    a paired comparison is, because they ignore that both methods were
    scored on the identical items -- some of that per-item variance cancels
    out in the paired difference and does not need to be double-counted.
    This is the more appropriate test once you have two methods' pooled
    K-fold (or any) evaluation on a shared item set.

    Returns point difference (b - a), its bootstrap CI, and whether that CI
    excludes zero (a simple, standard significance check at the given alpha).
    """
    df_a = pooled_evaluations[method_a][[id_column, correct_column]].rename(
        columns={correct_column: "correct_a"}
    )
    df_b = pooled_evaluations[method_b][[id_column, correct_column]].rename(
        columns={correct_column: "correct_b"}
    )

    merged = df_a.merge(df_b, on=id_column, how="inner", validate="one_to_one")

    if len(merged) == 0:
        raise ValueError(
            f"No shared items between '{method_a}' and '{method_b}' pooled "
            f"evaluations -- check that both were run with the same methods "
            f"list and the same K-fold split."
        )

    a = merged["correct_a"].astype(float).to_numpy()
    b = merged["correct_b"].astype(float).to_numpy()
    n = len(merged)

    point_diff = float(b.mean() - a.mean())

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    # Resample ROW indices jointly for both arrays -- this is what makes it
    # a paired bootstrap: each resample keeps (a_i, b_i) pairs intact.
    boot_diffs = b[boot_indices].mean(axis=1) - a[boot_indices].mean(axis=1)

    ci_low, ci_high = np.percentile(
        boot_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )

    significant = bool(ci_low > 0 or ci_high < 0)

    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_paired": int(n),
        "accuracy_a": round(float(a.mean()), 4),
        "accuracy_b": round(float(b.mean()), 4),
        "point_difference": round(point_diff, 4),
        "diff_ci_low": round(float(ci_low), 4),
        "diff_ci_high": round(float(ci_high), 4),
        "significant_at_alpha": significant,
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "note": (
            "significant_at_alpha=True means the paired-bootstrap CI on "
            "(accuracy_b - accuracy_a) excludes zero at the given alpha. "
            "This is a standard, defensible significance statement for two "
            "methods scored on the same held-out items."
        ),
    }



def run_multi_seed(
    auto_path: Path,
    annotation_paths: Sequence[Path],
    methods: Sequence[str],
    seeds: Sequence[int],
    n_bootstrap: int,
    alpha: float,
    **pipeline_kwargs: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat the full pipeline under several seeds.

    Returns:
      per_seed_df   -- one row per (method, seed) with held-out accuracy and
                        that seed's own bootstrap CI.
      aggregate_df  -- one row per method, summarizing across-seed mean/std/
                        min/max of held-out timing accuracy, i.e. how much
                        the reported number moves with the train/held-out
                        split itself (as opposed to sampling noise within one
                        held-out set, which the bootstrap CI already covers).
    """
    per_seed_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        run = run_pipeline_once(
            auto_path=auto_path,
            annotation_paths=annotation_paths,
            methods=methods,
            seed=seed,
            **pipeline_kwargs,
        )

        for method in methods:
            summary = run["heldout_summaries"][method]
            evaluation = run["heldout_evaluations"][method]

            boot = bootstrap_ci_for_heldout_evaluation(
                evaluation_df=evaluation,
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                seed=seed,
            )

            row = {
                "seed": seed,
                "method": method,
                "n_heldout": summary["n_heldout"],
                "calibrated_timing_accuracy": summary["calibrated_timing_accuracy"],
                "timing_accuracy_gain": summary["timing_accuracy_gain"],
                "calibrated_stage_accuracy": summary["calibrated_stage_accuracy"],
                "calibrated_move_accuracy": summary["calibrated_move_accuracy"],
                "timing_ci_low": boot.get("timing", {}).get("ci_low"),
                "timing_ci_high": boot.get("timing", {}).get("ci_high"),
            }

            if "policy_accepted_timing_accuracy" in summary:
                row["policy_accepted_coverage"] = summary.get("policy_accepted_coverage")
                row["policy_accepted_timing_accuracy"] = summary.get(
                    "policy_accepted_timing_accuracy"
                )
                row["policy_review_rate"] = summary.get("policy_review_rate")

            per_seed_rows.append(row)

    per_seed_df = pd.DataFrame(per_seed_rows)

    aggregate_rows: List[Dict[str, Any]] = []
    for method, group in per_seed_df.groupby("method"):
        values = group["calibrated_timing_accuracy"].to_numpy(dtype=float)
        aggregate_rows.append(
            {
                "method": method,
                "n_seeds": int(len(group)),
                "mean_timing_accuracy": round(float(np.mean(values)), 4),
                "std_timing_accuracy": round(float(np.std(values, ddof=1)), 4)
                if len(values) > 1
                else 0.0,
                "min_timing_accuracy": round(float(np.min(values)), 4),
                "max_timing_accuracy": round(float(np.max(values)), 4),
                "range_timing_accuracy": round(
                    float(np.max(values) - np.min(values)), 4
                ),
                "mean_timing_accuracy_gain": round(
                    float(group["timing_accuracy_gain"].mean()), 4
                ),
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows)

    return per_seed_df, aggregate_df


# =============================================================================
# 3.5 Reliability diagnostics and auto-computed threshold grids
# =============================================================================
#
# The isotonic_overall_reliability score is min(stage, move, timing)
# reliability, fit on a modest development set. Its range is data-dependent
# and can sit well below the [0, 1] interval you might naively grid-search
# (e.g. we observed a corpus with p99 = 0.50 -- every "keep_threshold >= 0.5"
# grid point was silently dead). These helpers inspect the *observed*
# distribution and build a threshold grid around it, instead of assuming a
# canonical 0-1 range.

def summarize_reliability_components(
    isotonic_base_df: pd.DataFrame,
    quantiles: Sequence[float] = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
) -> pd.DataFrame:
    """Describe each isotonic reliability column (stage/move/timing/overall).

    Run this before choosing --keep-thresholds / --correction-thresholds by
    hand, or just let --auto-thresholds do it for you (see main()).
    """
    columns = [
        "isotonic_stage_reliability",
        "isotonic_move_reliability",
        "isotonic_timing_reliability",
        "isotonic_overall_reliability",
    ]

    rows: List[Dict[str, Any]] = []
    for column in columns:
        if column not in isotonic_base_df.columns:
            continue

        values = isotonic_base_df[column].dropna()
        if values.empty:
            continue

        row: Dict[str, Any] = {
            "component": column,
            "n": int(len(values)),
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
        }
        for q in quantiles:
            row[f"p{int(q * 100)}"] = round(float(values.quantile(q)), 4)
        rows.append(row)

    return pd.DataFrame(rows)


def compute_auto_threshold_grid(
    scores: pd.Series,
    n_points: int,
    low_quantile: float,
    high_quantile: float,
) -> List[float]:
    """Build a threshold grid from the OBSERVED quantiles of `scores`,
    instead of an assumed 0-1 range. Deduplicates and sorts ascending.

    Falls back to [median] if the scores are degenerate (empty, constant,
    or too few unique values to span the requested quantile range).
    """
    clean = scores.dropna()

    if clean.empty:
        return [0.5]

    if clean.nunique() <= 1:
        return [round(float(clean.iloc[0]), 4)]

    quantile_points = np.linspace(low_quantile, high_quantile, n_points)
    raw_values = clean.quantile(quantile_points).tolist()
    grid = sorted({round(float(v), 4) for v in raw_values})

    if not grid:
        return [round(float(clean.median()), 4)]

    return grid


def compute_auto_keep_and_correction_grids(
    isotonic_base_df: pd.DataFrame,
    n_keep_points: int = 8,
    n_correction_points: int = 5,
    score_column: str = "isotonic_overall_reliability",
) -> Tuple[List[float], List[float]]:
    """Auto-derive sensible keep/correction threshold grids from the
    observed isotonic_overall_reliability distribution.

    Rationale:
      - KEEP should sweep the upper half of the observed distribution
        (p50 -> p99): these are the candidate "confident enough to accept
        automatically" cutoffs.
      - CORRECTION should sweep the lower half (p5 -> p50): a rule should
        only fire on originally low-reliability cases, by policy design
        (correction_max_original_reliability caps how reliable the ORIGINAL
        label can be and still get corrected).

    Because both grids are derived from the same observed scale, this
    self-adjusts if the reliability distribution shifts across re-annotation
    rounds or dataset versions, instead of silently testing thresholds that
    sit outside the data (as a fixed [0.5 ... 0.9] grid would if p99 is
    itself only 0.5, which is what we observed on the 300-item round).
    """
    scores = isotonic_base_df[score_column]

    keep_grid = compute_auto_threshold_grid(
        scores, n_points=n_keep_points, low_quantile=0.50, high_quantile=0.99
    )
    correction_grid = compute_auto_threshold_grid(
        scores, n_points=n_correction_points, low_quantile=0.05, high_quantile=0.50
    )

    return keep_grid, correction_grid


# =============================================================================
# 3.6 Rank-based coverage targeting (fixes threshold-based coverage failure)
# =============================================================================
#
# The safe_keep_correct_review policy's KEEP decision uses an ABSOLUTE
# reliability threshold. That fails to hit a coverage target reliably when
# the reliability distribution is plateaued (many tied scores) and/or the
# held-out set's score distribution doesn't match the full corpus's -- both
# of which we observed. An absolute cutoff of "reliability >= 0.34" can only
# ever accept whatever fraction of THIS SPECIFIC held-out set happens to
# clear 0.34; it cannot be dialed to "accept exactly 60%".
#
# The fix used here is rank-based coverage (identical in spirit to v1's
# original selective-reliability design, and already implemented in v2 as
# heldout_risk_coverage_table): sort the held-out set by reliability score
# and take the top K% BY COUNT. This hits any target coverage exactly,
# regardless of plateaus, ties, or small-sample mismatch between the
# held-out set and the full corpus.

def coverage_target_report(
    auto_path: Path,
    annotation_paths: Sequence[Path],
    seed: int,
    methods: Sequence[str],
    coverage_points: Sequence[float] = (0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20),
    score_column: str = "isotonic_overall_reliability",
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    consensus_mode: str = "auto",
    heldout_size: float = 0.30,
) -> pd.DataFrame:
    """For each method, rank the held-out evaluation set by `score_column`
    and report accuracy at each target coverage, with a bootstrap CI on the
    accepted subset's accuracy at every point. This is the direct,
    honest answer to "how do I get back to 60% coverage": rank-select the
    top 60% by count on the held-out set, then read off its accuracy --
    rather than searching for an absolute threshold that happens to produce
    60% (which may not exist, as we saw).

    Note this reports what accuracy you'd get by trusting the automatic
    label (no rule-based correction) for the top-K% most reliable cases,
    which is the same "keep-only" selective-prediction framing as v1's
    original selective reliability table. It intentionally does not mix in
    the safe_keep_correct_review policy's separate CORRECT branch -- that
    branch is orthogonal and can still be layered on top of whatever is
    sent to review, if desired.
    """
    run = run_pipeline_once(
        auto_path=auto_path,
        annotation_paths=annotation_paths,
        methods=methods,
        seed=seed,
        consensus_mode=consensus_mode,
        heldout_size=heldout_size,
    )

    rows: List[Dict[str, Any]] = []

    for method in methods:
        evaluation = run["heldout_evaluations"][method]

        if score_column not in evaluation.columns:
            continue

        table = tpc.heldout_risk_coverage_table(
            evaluation_df=evaluation,
            score_column=score_column,
            coverage_points=list(coverage_points),
        )

        ranked = evaluation.sort_values(score_column, ascending=False).reset_index(drop=True)
        total = len(ranked)

        for _, table_row in table.iterrows():
            accepted_count = int(table_row["n_accepted"])
            subset = ranked.iloc[:accepted_count]

            boot = bootstrap_ci_from_bool_series(
                subset["calibrated_timing_correct"],
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                seed=seed,
            )

            rows.append(
                {
                    "method": method,
                    "target_coverage": float(table_row["target_coverage"]),
                    "actual_coverage": float(table_row["actual_coverage"]),
                    "n_accepted": accepted_count,
                    "n_total": total,
                    "score_threshold_at_cutoff": float(table_row["threshold"]),
                    "accepted_timing_accuracy": float(
                        table_row["selective_human_validated_accuracy"]
                    ),
                    "accuracy_ci_low": boot["ci_low"],
                    "accuracy_ci_high": boot["ci_high"],
                    "review_rate": round(1.0 - float(table_row["actual_coverage"]), 4),
                }
            )

    return pd.DataFrame(rows)



def threshold_sensitivity_sweep(
    auto_path: Path,
    annotation_paths: Sequence[Path],
    seed: int,
    keep_thresholds: Optional[Sequence[float]] = None,
    correction_thresholds: Optional[Sequence[float]] = None,
    auto_n_keep_points: int = 8,
    auto_n_correction_points: int = 5,
    consensus_mode: str = "auto",
    heldout_size: float = 0.30,
    standard_min_support: int = 3,
    standard_min_error_rate: float = 0.50,
    inner_validation_size: float = 0.25,
    min_rule_affected: int = 3,
    min_rule_accuracy_gain: float = 0.0,
    stage_confidence_percentile: float = 10.0,
    move_confidence_percentile: float = 10.0,
    isotonic_reliability_threshold: float = 0.50,
    policy_min_rule_support: int = 5,
    policy_min_rule_error_rate: float = 0.60,
    policy_min_rule_validation_gain: float = 0.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep policy_keep_threshold x policy_correction_max_original_reliability
    for the safe_keep_correct_review policy, at a fixed seed. Rules and the
    dev/held-out split are computed once (they don't depend on the policy
    thresholds); only the KEEP/CORRECT/REVIEW decision is re-applied per grid
    point, so this is cheap even for a fine grid.

    Combinations where correction_threshold > keep_threshold are skipped:
    that configuration would try to "correct" cases already reliable enough
    to keep automatically, which is not a coherent policy.

    If keep_thresholds / correction_thresholds are left as None (the
    default), grids are auto-derived from the OBSERVED
    isotonic_overall_reliability distribution on this corpus via
    compute_auto_keep_and_correction_grids, rather than assuming a canonical
    [0, 1] range. This avoids the failure mode of testing a grid that sits
    entirely above the data (e.g. all "keep_threshold >= 0.5" points being
    dead because the corpus's p99 reliability is itself only 0.5).

    The returned DataFrame carries the grids actually used in
    `.attrs["keep_thresholds_used"]` / `.attrs["correction_thresholds_used"]`
    for logging/reproducibility.
    """
    base_df = tpc.prepare_auto_df(auto_path)
    base_df, _cm = tpc.compute_confidence_flags(
        base_df, stage_confidence_percentile, move_confidence_percentile
    )

    standard_consensus_df, _sm = tpc.build_consensus(
        annotation_paths, require_same_correction=False, consensus_mode=consensus_mode
    )
    base_df, _idc = tpc.configure_auto_id_column(
        base_df, standard_consensus_df["id"].tolist()
    )
    tpc.validate_annotation_ids_against_auto(base_df, standard_consensus_df)

    standard_development, standard_heldout, _split_meta = tpc.split_consensus_data(
        standard_consensus_df, test_size=heldout_size, random_state=seed
    )

    standard_timing_rules, _direct_meta = tpc.learn_do_no_harm_direct_rules(
        development_df=standard_development,
        min_support=standard_min_support,
        min_error_rate=standard_min_error_rate,
        validation_size=inner_validation_size,
        random_state=seed + 300,
        min_affected=min_rule_affected,
        min_accuracy_gain=min_rule_accuracy_gain,
    )

    development_auto_rows = tpc.make_heldout_auto_rows(base_df, standard_development)
    development_with_humans = development_auto_rows.merge(
        standard_development[
            ["id", "human_stage_correct", "human_move_correct", "human_timing_correct"]
        ].rename(columns={"id": "id_for_calibration"}),
        on="id_for_calibration",
        how="inner",
        validate="one_to_one",
    )

    isotonic_base_df, _iso_meta, _iso_models = tpc.add_isotonic_reliability(
        base_df=base_df,
        development_df=development_with_humans,
        reliability_threshold=isotonic_reliability_threshold,
    )

    if verbose:
        diagnostics = summarize_reliability_components(isotonic_base_df)
        print("Observed isotonic reliability distribution (used to build the grid):")
        print(diagnostics.to_string(index=False))
        print()

    auto_keep_grid, auto_correction_grid = compute_auto_keep_and_correction_grids(
        isotonic_base_df=isotonic_base_df,
        n_keep_points=auto_n_keep_points,
        n_correction_points=auto_n_correction_points,
    )

    resolved_keep_thresholds = (
        list(keep_thresholds) if keep_thresholds is not None else auto_keep_grid
    )
    resolved_correction_thresholds = (
        list(correction_thresholds)
        if correction_thresholds is not None
        else auto_correction_grid
    )

    if verbose:
        source = "user-specified" if keep_thresholds is not None else "auto (observed quantiles)"
        print(f"keep_thresholds ({source}): {resolved_keep_thresholds}")
        source = (
            "user-specified" if correction_thresholds is not None else "auto (observed quantiles)"
        )
        print(f"correction_thresholds ({source}): {resolved_correction_thresholds}")
        print()

    rows: List[Dict[str, Any]] = []

    for keep_threshold in resolved_keep_thresholds:
        for correction_threshold in resolved_correction_thresholds:
            if correction_threshold > keep_threshold:
                continue

            method_output = tpc.apply_safe_keep_correct_review_policy(
                base_df=isotonic_base_df,
                timing_rules=standard_timing_rules,
                keep_threshold=keep_threshold,
                correction_max_original_reliability=correction_threshold,
                min_rule_support=policy_min_rule_support,
                min_rule_error_rate=policy_min_rule_error_rate,
                min_rule_validation_gain=policy_min_rule_validation_gain,
            )

            summary, _evaluation = tpc.evaluate_method_on_heldout(
                method_output=method_output,
                heldout_df=standard_heldout,
                method="safe_keep_correct_review",
            )

            rows.append(
                {
                    "keep_threshold": keep_threshold,
                    "correction_threshold": correction_threshold,
                    "n_heldout": summary["n_heldout"],
                    "policy_accepted_coverage": summary.get("policy_accepted_coverage"),
                    "policy_accepted_timing_accuracy": summary.get(
                        "policy_accepted_timing_accuracy"
                    ),
                    "policy_review_rate": summary.get("policy_review_rate"),
                    "policy_n_keep": summary.get("policy_n_keep"),
                    "policy_n_correct": summary.get("policy_n_correct"),
                    "policy_n_review": summary.get("policy_n_review"),
                    "calibrated_timing_accuracy": summary["calibrated_timing_accuracy"],
                }
            )

    result_df = pd.DataFrame(rows)
    result_df.attrs["keep_thresholds_used"] = resolved_keep_thresholds
    result_df.attrs["correction_thresholds_used"] = resolved_correction_thresholds
    result_df.attrs["reliability_diagnostics"] = summarize_reliability_components(
        isotonic_base_df
    ).to_dict("records")

    if result_df.empty and verbose:
        print(
            "WARNING: threshold sweep produced 0 rows -- every "
            "correction_threshold exceeded every keep_threshold in the grid. "
            "This should not happen with auto-computed grids; check that "
            "isotonic_overall_reliability is not entirely NaN or constant."
        )

    return result_df


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robustness analysis (bootstrap CIs, multi-seed stability, "
        "threshold sweep) for TheraTime post-calibration v2."
    )
    parser.add_argument("--auto", required=True, help="Automatic judgments CSV.")
    parser.add_argument(
        "--ann", nargs="+", required=True, help="Two or more annotation CSV/JSON files."
    )
    parser.add_argument("--out-dir", default="theratime_robustness_outputs")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "conservative_human_recompute", "safe_keep_correct_review"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Random seeds for the multi-seed stability run.",
    )
    parser.add_argument(
        "--primary-seed",
        type=int,
        default=42,
        help="Seed used for the headline bootstrap CI and threshold sweep.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05, help="1 - confidence level.")
    parser.add_argument("--heldout-size", type=float, default=0.30)
    parser.add_argument("--consensus-mode", choices=["auto", "unanimous", "majority"], default="auto")
    parser.add_argument(
        "--keep-thresholds",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Explicit keep-threshold grid. If omitted, a grid is auto-derived "
            "from the observed isotonic_overall_reliability quantiles "
            "(p50 -> p99) on this corpus -- recommended, since the "
            "reliability score's usable range is data-dependent and a fixed "
            "0-1 grid can silently test only dead thresholds."
        ),
    )
    parser.add_argument(
        "--correction-thresholds",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Explicit correction-threshold grid. If omitted, auto-derived "
            "from observed quantiles (p5 -> p50). See --keep-thresholds."
        ),
    )
    parser.add_argument(
        "--n-keep-points",
        type=int,
        default=8,
        help="Number of auto-grid points for keep_threshold when --keep-thresholds is omitted.",
    )
    parser.add_argument(
        "--n-correction-points",
        type=int,
        default=5,
        help="Number of auto-grid points for correction_threshold when --correction-thresholds is omitted.",
    )
    parser.add_argument(
        "--coverage-points",
        nargs="+",
        type=float,
        default=[0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20],
        help=(
            "Target coverage fractions for the rank-based coverage report "
            "(e.g. 0.60 = accept the top 60%% most reliable held-out cases "
            "by count). Unlike --keep-thresholds, this always hits the "
            "requested coverage exactly."
        ),
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help=(
            "Number of folds for the pooled K-fold held-out evaluation. "
            "This tightens the confidence interval by evaluating every "
            "annotated item exactly once (out-of-fold), without requiring "
            "any additional annotation. Set to 0 to skip this step."
        ),
    )
    args = parser.parse_args()

    auto_path = Path(args.auto)
    annotation_paths = [Path(p) for p in args.ann]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("1/6  Headline bootstrap CIs at primary seed", args.primary_seed)
    print("=" * 88)

    primary_run = run_pipeline_once(
        auto_path=auto_path,
        annotation_paths=annotation_paths,
        methods=args.methods,
        seed=args.primary_seed,
        consensus_mode=args.consensus_mode,
        heldout_size=args.heldout_size,
    )

    bootstrap_rows = []
    for method in args.methods:
        evaluation = primary_run["heldout_evaluations"][method]
        boot = bootstrap_ci_for_heldout_evaluation(
            evaluation_df=evaluation,
            n_bootstrap=args.n_bootstrap,
            alpha=args.alpha,
            seed=args.primary_seed,
        )
        for label_type, ci in boot.items():
            bootstrap_rows.append({"method": method, "metric": label_type, **ci})

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_path = out_dir / "theratime_bootstrap_ci.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)
    print(bootstrap_df.to_string(index=False))

    print()
    print("=" * 88)
    print(f"2/6  Multi-seed stability across {len(args.seeds)} seeds")
    print("=" * 88)

    per_seed_df, aggregate_df = run_multi_seed(
        auto_path=auto_path,
        annotation_paths=annotation_paths,
        methods=args.methods,
        seeds=args.seeds,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        consensus_mode=args.consensus_mode,
        heldout_size=args.heldout_size,
    )

    per_seed_path = out_dir / "theratime_multiseed_per_seed.csv"
    aggregate_path = out_dir / "theratime_multiseed_aggregate.csv"
    per_seed_df.to_csv(per_seed_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    print(aggregate_df.to_string(index=False))

    print()
    print("=" * 88)
    print("3/6  Threshold sensitivity sweep (safe_keep_correct_review)")
    print("=" * 88)

    if "safe_keep_correct_review" in args.methods:
        sweep_df = threshold_sensitivity_sweep(
            auto_path=auto_path,
            annotation_paths=annotation_paths,
            seed=args.primary_seed,
            keep_thresholds=args.keep_thresholds,
            correction_thresholds=args.correction_thresholds,
            auto_n_keep_points=args.n_keep_points,
            auto_n_correction_points=args.n_correction_points,
            consensus_mode=args.consensus_mode,
            heldout_size=args.heldout_size,
            verbose=True,
        )
        sweep_path = out_dir / "theratime_threshold_sweep.csv"
        sweep_df.to_csv(sweep_path, index=False)
        print(sweep_df.to_string(index=False))

        reliability_diagnostics = sweep_df.attrs.get("reliability_diagnostics")
        keep_thresholds_used = sweep_df.attrs.get("keep_thresholds_used")
        correction_thresholds_used = sweep_df.attrs.get("correction_thresholds_used")

        diagnostics_path = out_dir / "theratime_reliability_diagnostics.csv"
        if reliability_diagnostics:
            pd.DataFrame(reliability_diagnostics).to_csv(diagnostics_path, index=False)
    else:
        sweep_df = pd.DataFrame()
        sweep_path = None
        reliability_diagnostics = None
        keep_thresholds_used = None
        correction_thresholds_used = None
        diagnostics_path = None
        print("Skipped: 'safe_keep_correct_review' not in --methods.")

    print()
    print("=" * 88)
    print("4/6  Rank-based coverage report (recovers a target coverage, e.g. 60%)")
    print("=" * 88)

    coverage_methods = [m for m in args.methods if m != "safe_keep_correct_review"] or [
        "baseline"
    ]
    coverage_df = coverage_target_report(
        auto_path=auto_path,
        annotation_paths=annotation_paths,
        seed=args.primary_seed,
        methods=coverage_methods,
        coverage_points=args.coverage_points,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        consensus_mode=args.consensus_mode,
        heldout_size=args.heldout_size,
    )
    coverage_path = out_dir / "theratime_coverage_target_report.csv"
    coverage_df.to_csv(coverage_path, index=False)
    print(coverage_df.to_string(index=False))

    print()
    print("=" * 88)
    print(f"5/6  K-fold pooled held-out evaluation (k={args.k_folds}) -- tightens CI, no new annotation")
    print("=" * 88)

    if args.k_folds and args.k_folds >= 2:
        kfold_summary_df, kfold_pooled = run_kfold_pooled_evaluation(
            auto_path=auto_path,
            annotation_paths=annotation_paths,
            methods=args.methods,
            k_folds=args.k_folds,
            seed=args.primary_seed,
            n_bootstrap=args.n_bootstrap,
            alpha=args.alpha,
            consensus_mode=args.consensus_mode,
        )
        kfold_summary_path = out_dir / "theratime_kfold_pooled_summary.csv"
        kfold_summary_df.to_csv(kfold_summary_path, index=False)
        print(kfold_summary_df.to_string(index=False))
        print()
        print(
            "Compare ci_half_width above to the single-split bootstrap CI "
            "half-width in step 1/5 -- pooling across folds uses every "
            "annotated item as a held-out test case exactly once, so this "
            "is normally substantially tighter for the same annotation effort."
        )

        for method, pooled_df in kfold_pooled.items():
            pooled_path = out_dir / f"theratime_kfold_pooled_{method}.csv"
            pooled_df.to_csv(pooled_path, index=False)
    else:
        kfold_summary_df = pd.DataFrame()
        kfold_summary_path = None
        kfold_pooled = {}
        print("Skipped: --k-folds < 2.")

    print()
    print("=" * 88)
    print("6/6  Paired significance test on pooled K-fold results")
    print("=" * 88)

    if kfold_pooled and len(kfold_pooled) >= 2 and "baseline" in kfold_pooled:
        pair_rows = []
        for method in args.methods:
            if method == "baseline" or method not in kfold_pooled:
                continue
            result = paired_bootstrap_difference(
                pooled_evaluations=kfold_pooled,
                method_a="baseline",
                method_b=method,
                n_bootstrap=args.n_bootstrap,
                alpha=args.alpha,
                seed=args.primary_seed,
            )
            pair_rows.append(result)

        pair_df = pd.DataFrame(pair_rows)
        pair_path = out_dir / "theratime_paired_significance.csv"
        if not pair_df.empty:
            pair_df.to_csv(pair_path, index=False)
            print(
                pair_df[
                    [
                        "method_b",
                        "n_paired",
                        "accuracy_a",
                        "accuracy_b",
                        "point_difference",
                        "diff_ci_low",
                        "diff_ci_high",
                        "significant_at_alpha",
                    ]
                ].to_string(index=False)
            )
            print()
            print(
                "significant_at_alpha=True means the paired-bootstrap CI on "
                "(method_b accuracy - baseline accuracy) excludes zero -- a "
                "stronger, more appropriate test than comparing two separate "
                "marginal CIs, since both methods were scored on the same "
                "held-out items."
            )
        else:
            print("No comparable method pairs found.")
    else:
        pair_path = None
        print("Skipped: requires --k-folds >= 2 and 'baseline' plus at least one other method in --methods.")




    report = {
        "purpose": (
            "Robustness analysis layered on theratime_post_calibration.py v2: "
            "bootstrap CIs, multi-seed stability, and threshold sensitivity."
        ),
        "primary_seed": args.primary_seed,
        "seeds_for_stability": args.seeds,
        "n_bootstrap": args.n_bootstrap,
        "alpha": args.alpha,
        "methods": args.methods,
        "output_files": {
            "bootstrap_ci": str(bootstrap_path),
            "multiseed_per_seed": str(per_seed_path),
            "multiseed_aggregate": str(aggregate_path),
            "threshold_sweep": str(sweep_path) if sweep_path else None,
            "reliability_diagnostics": str(diagnostics_path) if diagnostics_path else None,
            "coverage_target_report": str(coverage_path),
            "kfold_pooled_summary": str(kfold_summary_path) if kfold_summary_path else None,
            "paired_significance": str(pair_path) if pair_path else None,
        },
        "threshold_grid": {
            "keep_thresholds_used": keep_thresholds_used,
            "correction_thresholds_used": correction_thresholds_used,
            "grid_source": (
                "user-specified"
                if args.keep_thresholds is not None
                else "auto (observed isotonic_overall_reliability quantiles)"
            ),
        },
        "recommended_paper_wording": (
            "Held-out timing accuracy is reported with 95% percentile "
            "bootstrap confidence intervals (2000 resamples). To assess "
            "sensitivity to the specific development/held-out split, the "
            "full calibration pipeline was repeated under multiple random "
            "seeds; we report the mean and range of held-out timing "
            "accuracy across seeds rather than a single point estimate. "
            "The safe_keep_correct_review policy's keep/correction "
            "thresholds were selected via a sensitivity sweep over a grid "
            "derived from the observed isotonic reliability distribution on "
            "this corpus, rather than an untested fixed default."
        ),
    }
    report_path = out_dir / "theratime_robustness_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 88)
    print("Robustness analysis complete.")
    print(f"Output directory: {out_dir}")
    print(f"Report          : {report_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
