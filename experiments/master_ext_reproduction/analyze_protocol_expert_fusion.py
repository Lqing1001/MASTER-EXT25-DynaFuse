"""Validation-selected fixed fusion of MASTER and Continuous-PRISM + TA.

Reconstruct validation predictions from frozen checkpoints, choose alpha only on
validation, and apply the locked alpha to test predictions. Single threaded.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch
from run_master_ext import ROOT, VALID_END, VALID_START, UniverseStore, evaluate, summarize_daily
from run_master_component_ablation import ComponentMASTER
from run_prism_backbone_ablation import AblatedPrismRanker, AblatedPrismSpatial, TAPrismAblation


def zscore(x):
    return (x - np.mean(x)) / max(float(np.std(x)), 1e-12)


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        ranks = np.bincount(inverse, weights=ranks)[inverse] / counts[inverse]
    return ranks


def corr(a, b, rank=False):
    if rank:
        a, b = rankdata(a), rankdata(b)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def aligned_metrics(master, ta, alpha):
    for key in ("dates", "instruments"):
        if not np.array_equal(master[key], ta[key]):
            raise RuntimeError(f"unaligned prediction arrays: {key}")
    if not np.allclose(master["labels"], ta["labels"], equal_nan=True):
        raise RuntimeError("unaligned labels")
    dates = master["dates"]
    rows, expert_corrs, residual_ics, residual_rankics = [], [], [], []
    fused = np.full_like(master["predictions"], np.nan, dtype=np.float64)
    for date in np.unique(dates):
        idx = dates == date
        positions = np.flatnonzero(idx)
        mp = master["predictions"][idx].astype(np.float64)
        tp = ta["predictions"][idx].astype(np.float64)
        y = master["labels"][idx].astype(np.float64)
        valid = np.isfinite(mp) & np.isfinite(tp) & np.isfinite(y)
        mpz, tpz = zscore(mp[valid]), zscore(tp[valid])
        fp = (1.0 - alpha) * mpz + alpha * tpz
        fused[positions[valid]] = fp
        residual = tpz - mpz
        expert_corrs.append(corr(mpz, tpz))
        residual_ics.append(corr(residual, y[valid]))
        residual_rankics.append(corr(residual, y[valid], rank=True))
        rows.append({"date": int(date), "n": int(idx.sum()), "finite_labels": int(valid.sum()),
                     "IC": corr(fp, y[valid]), "RankIC": corr(fp, y[valid], rank=True)})
    diagnostics = {
        "mean_daily_expert_correlation": float(np.nanmean(expert_corrs)),
        "mean_residual_IC": float(np.nanmean(residual_ics)),
        "mean_residual_RankIC": float(np.nanmean(residual_rankics)),
    }
    return rows, diagnostics, fused


def load_npz(path):
    with np.load(path) as data:
        return {key: np.array(data[key]) for key in data.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=("csi300", "csi800"), default="csi300")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-dir", type=Path, required=True)
    parser.add_argument("--continuous-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = UniverseStore(ROOT / "datasets" / "master_ext_clean_v1", args.universe)

    master = ComponentMASTER("full", args.universe).to(device)
    master_prefix = f"master_full_{args.universe}_seed{args.seed}"
    master.load_state_dict(torch.load(args.master_dir / f"{master_prefix}_best.pt",
                                      map_location=device, weights_only=True))
    _, master_valid = evaluate(master, store, VALID_START, VALID_END, device, None)

    spatial = AblatedPrismSpatial("no_vq")
    base = AblatedPrismRanker(spatial, "no_vq").to(device)
    continuous_prefix = f"no_vq_{args.universe}_seed{args.seed}"
    base.load_state_dict(torch.load(args.continuous_dir / f"{continuous_prefix}_base_best.pt",
                                    map_location=device, weights_only=True))
    ta = TAPrismAblation(base).to(device)
    adapter_state = torch.load(args.continuous_dir / f"{continuous_prefix}_adapter_best.pt",
                               map_location=device, weights_only=True)
    ta.load_state_dict(adapter_state, strict=False)
    _, ta_valid = evaluate(ta, store, VALID_START, VALID_END, device, None)

    validation = {}
    best_alpha, best_score = None, -float("inf")
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        rows, diagnostics, _ = aligned_metrics(master_valid, ta_valid, alpha)
        metrics = summarize_daily(rows)
        score = 0.5 * (metrics["IC"] + metrics["RankIC"])
        validation[str(alpha)] = {"metrics": metrics, "selection_score": score, **diagnostics}
        if alpha in (0.25, 0.5, 0.75) and score > best_score:
            best_alpha, best_score = alpha, score

    master_test = load_npz(args.master_dir / f"{master_prefix}_predictions.npz")
    ta_test = load_npz(args.continuous_dir / f"{continuous_prefix}_adapter_predictions.npz")
    test_rows, test_diagnostics, fused = aligned_metrics(master_test, ta_test, best_alpha)
    yearly = {}
    for year in range(2022, 2026):
        subset = [row for row in test_rows if year * 10000 + 101 <= row["date"] <= year * 10000 + 1231]
        yearly[str(year)] = summarize_daily(subset)

    prefix = f"master_ta_fixed_fusion_{args.universe}_seed{args.seed}"
    result = {
        "experiment": "validation-selected fixed fusion of MASTER and Continuous-PRISM+TA",
        "universe": args.universe, "seed": args.seed,
        "validation": validation, "selected_alpha_ta": best_alpha,
        "test_all_2022_2025": summarize_daily(test_rows), "test_by_year": yearly,
        "test_diagnostics": test_diagnostics, "mode": "single-process single-threaded",
    }
    (args.output_dir / f"{prefix}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(args.output_dir / f"{prefix}_predictions.npz",
                        dates=master_test["dates"], instruments=master_test["instruments"],
                        predictions=fused, labels=master_test["labels"])
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

