"""Validation-selected MASTER + XGBoost control under the paper protocol."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import xgboost as xgb
from analyze_protocol_expert_fusion import aligned_metrics, load_npz
from run_master_ext import (ROOT, VALID_END, VALID_START, UniverseStore,
                            correlation, summarize_daily, write_daily_csv)
from run_xgboost_expert import transform


def evaluate_xgboost(model: xgb.Booster, store: UniverseStore, start: int, end: int):
    rows, predictions, labels_all, dates_all, instruments_all = [], [], [], [], []
    dates = store.dates_between(start, end)
    for number, day in enumerate(dates, 1):
        sequence, labels, instruments = store.batch(int(day), training=False)
        pred = model.predict(xgb.DMatrix(transform(sequence), nthread=1))
        finite = np.isfinite(pred) & np.isfinite(labels)
        rows.append({"date": int(day), "n": int(len(labels)),
                     "finite_labels": int(finite.sum()),
                     "IC": correlation(pred[finite], labels[finite]),
                     "RankIC": correlation(pred[finite], labels[finite], rank=True)})
        predictions.append(pred.astype(np.float32))
        labels_all.append(labels.astype(np.float32))
        dates_all.append(np.full(len(labels), int(day), dtype=np.int32))
        instruments_all.append(instruments.astype("S8"))
        if number % 200 == 0 or number == len(dates):
            print(f"evaluate XGBoost {start}-{end}: {number}/{len(dates)}", flush=True)
    return rows, {"dates": np.concatenate(dates_all),
                  "instruments": np.concatenate(instruments_all),
                  "predictions": np.concatenate(predictions),
                  "labels": np.concatenate(labels_all)}


def yearly(rows: list[dict]) -> dict:
    result = {}
    for year in range(2022, 2026):
        subset = [row for row in rows
                  if year * 10000 + 101 <= row["date"] <= year * 10000 + 1231]
        result[str(year)] = summarize_daily(subset)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=("csi300", "csi800"), default="csi300")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--master-validation-dir", type=Path, required=True)
    parser.add_argument("--master-test-dir", type=Path, required=True)
    parser.add_argument("--xgboost-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix_m = f"master_full_{args.universe}_seed{args.seed}"
    prefix_x = f"xgboost_{args.universe}_seed{args.seed}"
    master_valid = load_npz(
        args.master_validation_dir / f"{prefix_m}_validation_predictions.npz")
    master_test = load_npz(args.master_test_dir / f"{prefix_m}_predictions.npz")
    xgb_test = load_npz(args.xgboost_dir / f"{prefix_x}_predictions.npz")

    model = xgb.Booster()
    model.load_model(args.xgboost_dir / f"{prefix_x}.ubj")
    store = UniverseStore(ROOT / "datasets" / "master_ext_clean_v1", args.universe)
    xgb_valid_rows, xgb_valid = evaluate_xgboost(
        model, store, VALID_START, VALID_END)
    np.savez_compressed(
        args.output_dir / f"{prefix_x}_validation_predictions.npz", **xgb_valid)

    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    validation, best_alpha, best_score = {}, None, -float("inf")
    for alpha in grid:
        rows, diagnostics, _ = aligned_metrics(master_valid, xgb_valid, alpha)
        metrics = summarize_daily(rows)
        score = 0.5 * (metrics["IC"] + metrics["RankIC"])
        validation[str(alpha)] = {
            "metrics": metrics, "selection_score": score, **diagnostics}
        if score > best_score:
            best_alpha, best_score = alpha, score

    test_rows, test_diagnostics, fused = aligned_metrics(
        master_test, xgb_test, best_alpha)
    result = {
        "experiment": "validation-selected MASTER + XGBoost fusion control",
        "universe": args.universe, "seed": args.seed,
        "fusion": "daily cross-sectional z-score per expert",
        "alpha_semantics": "XGBoost weight",
        "alpha_grid": list(grid),
        "tie_break": "smallest alpha because grid is iterated in ascending order",
        "validation": validation, "selected_alpha_xgboost": best_alpha,
        "best_validation_score": best_score,
        "xgboost_validation": summarize_daily(xgb_valid_rows),
        "test_all_2022_2025": summarize_daily(test_rows),
        "test_by_year": yearly(test_rows),
        "test_diagnostics": test_diagnostics,
        "test_use": "evaluation only after validation alpha selection",
        "mode": "single-process single-threaded",
    }
    prefix = f"master_xgboost_fusion_{args.universe}_seed{args.seed}"
    (args.output_dir / f"{prefix}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_daily_csv(args.output_dir / f"{prefix}_daily.csv", test_rows)
    np.savez_compressed(args.output_dir / f"{prefix}_predictions.npz",
                        dates=master_test["dates"],
                        instruments=master_test["instruments"],
                        predictions=fused.astype(np.float32),
                        labels=master_test["labels"])
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
