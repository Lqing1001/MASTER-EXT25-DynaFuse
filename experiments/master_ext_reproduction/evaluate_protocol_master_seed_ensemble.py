"""Formal homogeneous MASTER seed-ensemble control for the paper protocol."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch
from run_master_ext import (ROOT, VALID_END, VALID_START, UniverseStore,
                            correlation, evaluate, summarize_daily,
                            write_daily_csv)
from run_master_component_ablation import ComponentMASTER


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.array(archive[key]) for key in archive.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_manifest(args, seeds: list[int]) -> None:
    paths = []
    for seed in seeds:
        prefix = f"master_full_{args.universe}_seed{seed}"
        paths.extend((args.master_dir / f"{prefix}_best.pt",
                      args.master_dir / f"{prefix}_predictions.npz"))
    manifest = {
        "frozen_on": "2026-08-28",
        "status": "development-test controls; 2022-2025 was previously inspected",
        "split": {"train": [20100104, 20191224],
                  "validation": [20200102, 20211224],
                  "development_test": [20220104, 20251231]},
        "selection_score": "0.5 * (mean_daily_IC + mean_daily_RankIC)",
        "seed_ensemble": {
            "members": seeds,
            "two_seed_pairs": [list(pair) for pair in itertools.combinations(seeds, 2)],
            "three_seed_group": seeds,
            "primary": "daily_zscore_mean",
            "secondary": "raw_mean",
            "report_policy": "all pairs and both aggregations; never select on test",
        },
        "master_xgboost": {
            "normalization": "daily cross-sectional z-score per expert",
            "alpha_xgboost_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "tie_break": "smallest alpha within exact floating-point ties",
        },
        "capacity_control": {
            "variant": "MASTER d_model=320 only",
            "seeds": seeds, "epochs": 40, "patience": 40, "learning_rate": 1e-5,
        },
        "statistics": {
            "primary_daily_endpoint": "0.5 * (daily_IC + daily_RankIC)",
            "newey_west_lag": 10,
            "moving_block_length": 20,
            "bootstrap_draws": 5000,
            "secondary_endpoints": ["IC", "RankIC"],
            "multiplicity": "Holm correction for secondary endpoints",
        },
        "input_sha256": {str(path.resolve()): sha256(path) for path in paths},
    }
    target = args.output_dir / "_decisive_controls_frozen_manifest.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(f"frozen manifest mismatch: {target}")
    else:
        target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def assert_aligned(archives: list[dict[str, np.ndarray]]) -> None:
    reference = archives[0]
    for archive in archives[1:]:
        for key in ("dates", "instruments"):
            if not np.array_equal(reference[key], archive[key]):
                raise RuntimeError(f"unaligned prediction arrays: {key}")
        if not np.allclose(reference["labels"], archive["labels"], equal_nan=True):
            raise RuntimeError("unaligned labels")


def daily_zscore(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=np.float64)
    for date in np.unique(dates):
        mask = dates == date
        block = values[mask].astype(np.float64)
        finite = np.isfinite(block)
        if finite.any():
            scale = max(float(np.std(block[finite])), 1e-12)
            normalized = np.full(len(block), np.nan, dtype=np.float64)
            normalized[finite] = (block[finite] - np.mean(block[finite])) / scale
            output[mask] = normalized
    return output


def aggregate(archives: list[dict[str, np.ndarray]], method: str):
    assert_aligned(archives)
    dates, labels = archives[0]["dates"], archives[0]["labels"]
    raw = np.stack([item["predictions"].astype(np.float64) for item in archives])
    if method == "daily_zscore_mean":
        values = np.stack([daily_zscore(row, dates) for row in raw])
    elif method == "raw_mean":
        values = raw
    else:
        raise ValueError(method)
    predictions = np.nanmean(values, axis=0)
    rows, pair_correlations = [], []
    for date in np.unique(dates):
        mask = dates == date
        pred, label = predictions[mask], labels[mask]
        finite = np.isfinite(pred) & np.isfinite(label)
        rows.append({"date": int(date), "n": int(mask.sum()),
                     "finite_labels": int(finite.sum()),
                     "IC": correlation(pred[finite], label[finite]),
                     "RankIC": correlation(pred[finite], label[finite], rank=True)})
        for left, right in itertools.combinations(range(len(archives)), 2):
            a, b = raw[left, mask], raw[right, mask]
            shared = np.isfinite(a) & np.isfinite(b)
            pair_correlations.append(correlation(a[shared], b[shared]))
    packed = {"dates": dates, "instruments": archives[0]["instruments"],
              "predictions": predictions.astype(np.float32), "labels": labels}
    diagnostics = {"mean_daily_member_pair_correlation":
                   float(np.nanmean(pair_correlations)) if pair_correlations else None}
    return rows, packed, diagnostics


def yearly_metrics(rows: list[dict]) -> dict:
    output = {}
    for year in range(2022, 2026):
        subset = [row for row in rows
                  if year * 10000 + 101 <= row["date"] <= year * 10000 + 1231]
        if subset:
            output[str(year)] = summarize_daily(subset)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=("csi300", "csi800"), default="csi300")
    parser.add_argument("--master-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    seeds = sorted(set(args.seeds))
    if len(seeds) != 3:
        raise ValueError("the preregistered control requires exactly three seeds")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for validation inference")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze_manifest(args, seeds)
    store = UniverseStore(ROOT / "datasets" / "master_ext_clean_v1", args.universe)

    validation_archives, test_archives, individual = {}, {}, {}
    for seed in seeds:
        prefix = f"master_full_{args.universe}_seed{seed}"
        model = ComponentMASTER("full", args.universe).to(device)
        model.load_state_dict(torch.load(args.master_dir / f"{prefix}_best.pt",
                                         map_location=device, weights_only=True))
        valid_rows, valid_predictions = evaluate(
            model, store, VALID_START, VALID_END, device, None)
        validation_archives[seed] = valid_predictions
        np.savez_compressed(args.output_dir / f"{prefix}_validation_predictions.npz",
                            **valid_predictions)
        test_archives[seed] = load_npz(args.master_dir / f"{prefix}_predictions.npz")
        test_rows, _, _ = aggregate([test_archives[seed]], "raw_mean")
        individual[str(seed)] = {"validation": summarize_daily(valid_rows),
                                 "test_all_2022_2025": summarize_daily(test_rows)}
        del model
        torch.cuda.empty_cache()

    result = {
        "experiment": "homogeneous MASTER seed ensemble under fixed two-year validation",
        "universe": args.universe, "seeds": seeds,
        "selection_policy": {
            "primary_aggregation":
                "daily cross-sectional z-score per member, then equal mean",
            "secondary_sensitivity": "raw equal-weight mean",
            "pair_policy":
                "report all three two-seed pairs and the three-seed ensemble",
            "test_use": "evaluation only; no test-based selection",
        },
        "individual": individual, "ensembles": {},
        "protocol": {"validation": [VALID_START, VALID_END],
                     "test": [20220104, 20251231]},
        "mode": "single-process single-threaded",
    }
    groups = list(itertools.combinations(seeds, 2)) + [tuple(seeds)]
    for group in groups:
        name = "seeds_" + "_".join(map(str, group))
        result["ensembles"][name] = {}
        for method in ("daily_zscore_mean", "raw_mean"):
            valid_rows, _, valid_diagnostics = aggregate(
                [validation_archives[seed] for seed in group], method)
            test_rows, packed, test_diagnostics = aggregate(
                [test_archives[seed] for seed in group], method)
            valid = summarize_daily(valid_rows)
            result["ensembles"][name][method] = {
                "members": list(group), "validation": valid,
                "validation_selection_score": 0.5 * (valid["IC"] + valid["RankIC"]),
                "validation_diagnostics": valid_diagnostics,
                "test_all_2022_2025": summarize_daily(test_rows),
                "test_by_year": yearly_metrics(test_rows),
                "test_diagnostics": test_diagnostics,
            }
            prefix = f"master_{name}_{method}_{args.universe}"
            write_daily_csv(args.output_dir / f"{prefix}_validation_daily.csv", valid_rows)
            write_daily_csv(args.output_dir / f"{prefix}_daily.csv", test_rows)
            np.savez_compressed(args.output_dir / f"{prefix}_predictions.npz", **packed)
    target = args.output_dir / f"master_seed_ensemble_{args.universe}.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
