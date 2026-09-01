"""Unified paired statistics for the preregistered decisive controls."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from analyze_protocol_expert_fusion import aligned_metrics, load_npz
from analyze_ta_prism import hac_mean_test, moving_block_ci
from run_master_ext import correlation, summarize_daily, write_daily_csv


def daily_rows(archive: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for date in np.unique(archive["dates"]):
        mask = archive["dates"] == date
        pred, label = archive["predictions"][mask], archive["labels"][mask]
        finite = np.isfinite(pred) & np.isfinite(label)
        rows.append({"date": int(date), "n": int(mask.sum()),
                     "finite_labels": int(finite.sum()),
                     "IC": correlation(pred[finite], label[finite]),
                     "RankIC": correlation(pred[finite], label[finite], rank=True)})
    return rows


def row_map(rows: list[dict]) -> dict[int, dict]:
    return {int(row["date"]): row for row in rows}


def mean_reference(row_sets: list[list[dict]]) -> list[dict]:
    maps = [row_map(rows) for rows in row_sets]
    dates = sorted(set.intersection(*(set(item) for item in maps)))
    return [{"date": date,
             "IC": float(np.mean([item[date]["IC"] for item in maps])),
             "RankIC": float(np.mean([item[date]["RankIC"] for item in maps]))}
            for date in dates]


def holm_two(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted, previous = {}, 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (total - rank) * p_values[key])
        previous = max(previous, value)
        adjusted[key] = previous
    return adjusted


def paired(reference: list[dict], comparator: list[dict], rng) -> dict:
    left, right = row_map(reference), row_map(comparator)
    dates = sorted(set(left) & set(right))
    differences = {
        "S": np.array([0.5 * ((left[d]["IC"] - right[d]["IC"]) +
                              (left[d]["RankIC"] - right[d]["RankIC"]))
                       for d in dates], dtype=np.float64),
        "IC": np.array([left[d]["IC"] - right[d]["IC"] for d in dates],
                       dtype=np.float64),
        "RankIC": np.array([left[d]["RankIC"] - right[d]["RankIC"] for d in dates],
                           dtype=np.float64),
    }
    result = {}
    for metric, values in differences.items():
        finite = values[np.isfinite(values)]
        result[metric] = hac_mean_test(finite, lag=10)
        result[metric]["moving_block_bootstrap"] = moving_block_ci(
            finite, rng, block=20, draws=5000)
    adjusted = holm_two({
        metric: result[metric]["p_two_sided"] for metric in ("IC", "RankIC")})
    for metric in ("IC", "RankIC"):
        result[metric]["holm_adjusted_p"] = adjusted[metric]
    return {"days": len(dates), "reference_minus_comparator": result}


def aggregate_models(items: list[dict]) -> dict:
    result = {}
    for metric in ("IC", "RankIC", "ICIR", "RankICIR"):
        values = np.array([item[metric] for item in items], dtype=np.float64)
        result[metric] = {"mean": float(values.mean()),
                          "sample_std": float(values.std(ddof=1)),
                          "values": values.tolist()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.protocol_dir
    rng = np.random.RandomState(20260828)

    dynafuse_rows, dynafuse_metrics = {}, []
    master_metrics, wide_metrics, xgb_metrics = [], [], []
    for seed in range(3):
        master = load_npz(
            root / "master" / f"master_full_csi300_seed{seed}_predictions.npz")
        expert = load_npz(
            root / "topvenue_components" / "deformable" /
            f"ta_deformable_csi300_seed{seed}_predictions.npz")
        rows, _, fused = aligned_metrics(master, expert, 0.5)
        dynafuse_rows[seed] = rows
        dynafuse_metrics.append(summarize_daily(rows))
        write_daily_csv(args.output_dir / f"dynafuse_seed{seed}_daily.csv", rows)
        master_metrics.append(summarize_daily(daily_rows(master)))
        wide = load_npz(
            root / "widened_master_d320" /
            f"master_full_d320_csi300_seed{seed}_predictions.npz")
        wide_metrics.append(summarize_daily(daily_rows(wide)))
        xgb = load_npz(
            root / "master_xgboost_control" /
            f"master_xgboost_fusion_csi300_seed{seed}_predictions.npz")
        xgb_metrics.append(summarize_daily(daily_rows(xgb)))

    output = {
        "experiment": "decisive controls unified analysis",
        "primary_daily_endpoint": "S=0.5*(IC+RankIC)",
        "statistics": {"newey_west_lag": 10, "block_length": 20,
                       "bootstrap_draws": 5000,
                       "secondary_multiplicity": "Holm within IC/RankIC"},
        "model_aggregates": {
            "MASTER": aggregate_models(master_metrics),
            "DynaFuse": aggregate_models(dynafuse_metrics),
            "MASTER_XGBoost": aggregate_models(xgb_metrics),
            "MASTER_d320": aggregate_models(wide_metrics),
        },
        "matched_comparisons": {},
        "homogeneous_ensemble_comparisons": {},
        "interpretation_guardrail":
            "2022-2025 is a previously inspected development-test period",
    }
    for seed in range(3):
        xgb_rows = daily_rows(load_npz(
            root / "master_xgboost_control" /
            f"master_xgboost_fusion_csi300_seed{seed}_predictions.npz"))
        wide_rows = daily_rows(load_npz(
            root / "widened_master_d320" /
            f"master_full_d320_csi300_seed{seed}_predictions.npz"))
        output["matched_comparisons"][f"dynafuse_vs_xgboost_seed{seed}"] = paired(
            dynafuse_rows[seed], xgb_rows, rng)
        output["matched_comparisons"][f"dynafuse_vs_wide320_seed{seed}"] = paired(
            dynafuse_rows[seed], wide_rows, rng)

    ensemble_dir = root / "master_seed_ensemble_control"
    groups = ((0, 1), (0, 2), (1, 2), (0, 1, 2))
    for group in groups:
        label = "_".join(map(str, group))
        ensemble_rows = daily_rows(load_npz(
            ensemble_dir /
            f"master_seeds_{label}_daily_zscore_mean_csi300_predictions.npz"))
        reference = mean_reference([dynafuse_rows[seed] for seed in group])
        output["homogeneous_ensemble_comparisons"][f"matched_dynafuse_mean_vs_{label}"] = {
            "reference": f"daily metric mean of DynaFuse seeds {list(group)}",
            **paired(reference, ensemble_rows, rng),
        }

    target = args.output_dir / "decisive_controls_statistics_csi300.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
