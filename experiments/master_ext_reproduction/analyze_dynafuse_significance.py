"""Paired daily HAC and moving-block comparisons for DynaFuse."""
from __future__ import annotations
import csv
import json
import math
import os
from pathlib import Path
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from analyze_protocol_expert_fusion import aligned_metrics, load_npz
from analyze_ta_prism import hac_mean_test, moving_block_ci
from run_master_ext import ROOT, write_daily_csv

def read_daily(path):
    rows = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["date"])] = {"IC": float(row["IC"]), "RankIC": float(row["RankIC"])}
    return rows

def main():
    protocol = ROOT / "results" / "protocol_2020_2021_validation"
    master = load_npz(protocol / "master" / "master_full_csi300_seed0_predictions.npz")
    expert = load_npz(protocol / "topvenue_components" / "deformable" /
                      "ta_deformable_csi300_seed0_predictions.npz")
    fused_rows, diagnostics, _ = aligned_metrics(master, expert, 0.5)
    proposed = {int(r["date"]): {"IC": float(r["IC"]), "RankIC": float(r["RankIC"])}
                for r in fused_rows}
    out_dir = protocol / "dynafuse_significance"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_daily_csv(out_dir / "dynafuse_csi300_seed0_daily.csv", fused_rows)
    comparators = {
        "Ridge": protocol / "traditional_baselines" / "ridge_csi300_seed0_daily.csv",
        "Random Forest": protocol / "traditional_baselines" / "random_forest_csi300_seed0_daily.csv",
        "XGBoost": protocol / "traditional_baselines" / "xgboost_csi300_seed0_daily.csv",
        "StockMamba": protocol / "recent_baselines" / "stockmamba_csi300_seed0_daily.csv",
        "ACT": protocol / "recent_baselines" / "act_csi300_seed0_daily.csv",
        "MASTER": protocol / "master" / "master_full_csi300_seed0_daily.csv",
        "PRISM-VQ": protocol / "prism_vq" / "prism_vq_csi300_seed0_daily.csv",
        "Continuous-PRISM": protocol / "continuous_prism" / "no_vq_csi300_seed0_base_daily.csv",
        "TA-PRISM": protocol / "continuous_prism" / "no_vq_csi300_seed0_adapter_daily.csv",
        "Sparse expert": protocol / "topvenue_components" / "deformable" /
                         "ta_deformable_csi300_seed0_daily.csv",
    }
    periods = {
        "early_2022_2023": (20220104, 20231231),
        "extension_2024_2025": (20240101, 20251231),
        "all_2022_2025": (20220104, 20251231),
    }
    rng = np.random.RandomState(20260827)
    output = {"proposed": "DynaFuse seed0 fixed alpha=0.5", "diagnostics": diagnostics,
              "comparisons": {}, "mode": "single-process single-threaded"}
    for name, path in comparators.items():
        baseline = read_daily(path)
        shared = sorted(set(proposed) & set(baseline))
        output["comparisons"][name] = {}
        for period, (start, end) in periods.items():
            dates = [d for d in shared if start <= d <= end]
            output["comparisons"][name][period] = {}
            for metric in ("IC", "RankIC"):
                diff = np.array([proposed[d][metric] - baseline[d][metric]
                                 for d in dates], dtype=np.float64)
                finite = diff[np.isfinite(diff)]
                test = hac_mean_test(finite, lag=10)
                test["moving_block_bootstrap"] = moving_block_ci(
                    finite, rng, block=20, draws=5000)
                output["comparisons"][name][period][metric] = test
    target = out_dir / "dynafuse_paired_significance_csi300_seed0.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
