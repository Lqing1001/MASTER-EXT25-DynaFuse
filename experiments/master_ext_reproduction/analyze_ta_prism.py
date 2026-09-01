"""Paired significance analysis for TA-PRISM (single-process)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def read_daily(path: Path) -> dict[int, dict[str, float]]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["date"])] = {"IC": float(row["IC"]), "RankIC": float(row["RankIC"])}
    return rows


def hac_mean_test(values: np.ndarray, lag: int = 10) -> dict[str, float]:
    values = values[np.isfinite(values)]
    n = len(values)
    centered = values - values.mean()
    long_run = float(centered @ centered / n)
    for k in range(1, min(lag, n - 1) + 1):
        gamma = float(centered[k:] @ centered[:-k] / n)
        long_run += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    se = math.sqrt(max(long_run, 0.0) / n)
    t = float(values.mean() / se) if se > 0 else float("nan")
    # Normal approximation; conservative enough for >600 daily observations.
    p_two_sided = math.erfc(abs(t) / math.sqrt(2.0)) if np.isfinite(t) else float("nan")
    return {"n": n, "mean_difference": float(values.mean()), "hac_se": se, "hac_t": t, "p_two_sided": p_two_sided}


def moving_block_ci(values: np.ndarray, rng: np.random.RandomState, block: int = 20, draws: int = 5000):
    values = values[np.isfinite(values)]
    n = len(values)
    starts = np.arange(max(1, n - block + 1))
    means = np.empty(draws, dtype=np.float64)
    blocks_needed = int(math.ceil(n / block))
    for i in range(draws):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[s:s + block] for s in chosen])[:n]
        means[i] = sample.mean()
    return {
        "block_length": block, "draws": draws,
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "probability_positive": float(np.mean(means > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "qutr_prism" / "ta_prism_significance.json")
    args = parser.parse_args()

    proposed = read_daily(ROOT / "results" / "qutr_prism" / "temporal_static_csi300_seed0_daily.csv")
    comparators = {
        "PRISM-VQ": ROOT / "results" / "recent_model_screening" / "prism_vq_csi300_seed0_daily.csv",
        "ACT": ROOT / "results" / "recent_model_screening" / "act_csi300_seed0_daily.csv",
        "StockMamba": ROOT / "results" / "recent_model_screening" / "stockmamba_csi300_seed0_daily.csv",
        "MASTER": ROOT / "results" / "master_component_ablation" / "master_full_csi300_seed0_daily.csv",
        "MLP-residual": ROOT / "results" / "qutr_prism" / "mlp_residual_csi300_seed0_daily.csv",
        "QUTR": ROOT / "results" / "qutr_prism" / "qutr_csi300_seed0_daily.csv",
    }
    periods = {
        "paper_comparable_20200701_20221231": (20200701, 20221231),
        "extension_20230101_20251231": (20230101, 20251231),
        "all_20200701_20251231": (20200701, 20251231),
    }
    rng = np.random.RandomState(args.seed)
    output = {"proposed": "TA-PRISM (temporal_static)", "comparisons": {}}
    for name, path in comparators.items():
        baseline = read_daily(path)
        shared = sorted(set(proposed) & set(baseline))
        output["comparisons"][name] = {}
        for period, (start, end) in periods.items():
            dates = [d for d in shared if start <= d <= end]
            output["comparisons"][name][period] = {}
            for metric in ("IC", "RankIC"):
                diff = np.array([proposed[d][metric] - baseline[d][metric] for d in dates], dtype=np.float64)
                finite = diff[np.isfinite(diff)]
                test = hac_mean_test(finite)
                test["moving_block_bootstrap"] = moving_block_ci(finite, rng)
                output["comparisons"][name][period][metric] = test
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
