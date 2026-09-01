"""Build DynaFuse progressive, normalization, alpha, and top-k ablations."""
from __future__ import annotations
import json
import os
from pathlib import Path
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import torch
from analyze_protocol_expert_fusion import aligned_metrics, corr, load_npz
from run_master_ext import ROOT, VALID_END, VALID_START, UniverseStore, evaluate, summarize_daily
from run_master_component_ablation import ComponentMASTER
from run_prism_backbone_ablation import AblatedPrismRanker, AblatedPrismSpatial, TAPrismAblation

def single_metrics(arr):
    dates = arr["dates"]
    rows = []
    for date in np.unique(dates):
        idx = dates == date
        p = arr["predictions"][idx].astype(np.float64)
        y = arr["labels"][idx].astype(np.float64)
        valid = np.isfinite(p) & np.isfinite(y)
        rows.append({"date": int(date), "n": int(idx.sum()), "finite_labels": int(valid.sum()),
                     "IC": corr(p[valid], y[valid]), "RankIC": corr(p[valid], y[valid], rank=True)})
    return summarize_daily(rows)

def raw_fusion_metrics(master, expert, alpha):
    for key in ("dates", "instruments"):
        if not np.array_equal(master[key], expert[key]):
            raise RuntimeError(f"unaligned arrays: {key}")
    dates = master["dates"]
    rows = []
    for date in np.unique(dates):
        idx = dates == date
        mp = master["predictions"][idx].astype(np.float64)
        ep = expert["predictions"][idx].astype(np.float64)
        y = master["labels"][idx].astype(np.float64)
        valid = np.isfinite(mp) & np.isfinite(ep) & np.isfinite(y)
        fp = (1.0 - alpha) * mp[valid] + alpha * ep[valid]
        rows.append({"date": int(date), "n": int(idx.sum()), "finite_labels": int(valid.sum()),
                     "IC": corr(fp, y[valid]), "RankIC": corr(fp, y[valid], rank=True)})
    return summarize_daily(rows)

def zfusion_metrics(master, expert, alpha):
    rows, diagnostics, _ = aligned_metrics(master, expert, alpha)
    return {"metrics": summarize_daily(rows), "diagnostics": diagnostics}

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda:0")
    protocol = ROOT / "results" / "protocol_2020_2021_validation"
    master_dir = protocol / "master"
    continuous_dir = protocol / "continuous_prism"
    sparse_dir = protocol / "topvenue_components" / "deformable"
    topk_dir = protocol / "sparse_topk_ablation"
    out_dir = protocol / "dynafuse_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    store = UniverseStore(ROOT / "datasets" / "master_ext_clean_v1", "csi300")

    master = ComponentMASTER("full", "csi300").to(device)
    master.load_state_dict(torch.load(master_dir / "master_full_csi300_seed0_best.pt",
                                      map_location=device, weights_only=True))
    _, master_valid = evaluate(master, store, VALID_START, VALID_END, device, None)
    master_test = load_npz(master_dir / "master_full_csi300_seed0_predictions.npz")

    spatial = AblatedPrismSpatial("no_vq")
    base = AblatedPrismRanker(spatial, "no_vq").to(device)
    base.load_state_dict(torch.load(continuous_dir / "no_vq_csi300_seed0_base_best.pt",
                                    map_location=device, weights_only=True))
    _, base_valid = evaluate(base, store, VALID_START, VALID_END, device, None)
    base_test = load_npz(continuous_dir / "no_vq_csi300_seed0_base_predictions.npz")

    ta = TAPrismAblation(base).to(device)
    ta.load_state_dict(torch.load(continuous_dir / "no_vq_csi300_seed0_adapter_best.pt",
                                  map_location=device, weights_only=True), strict=False)
    _, ta_valid = evaluate(ta, store, VALID_START, VALID_END, device, None)
    ta_test = load_npz(continuous_dir / "no_vq_csi300_seed0_adapter_predictions.npz")

    sparse_valid = load_npz(sparse_dir / "ta_deformable_csi300_seed0_validation_predictions.npz")
    sparse_test = load_npz(sparse_dir / "ta_deformable_csi300_seed0_predictions.npz")

    progressive = {
        "MASTER_only": {"validation": single_metrics(master_valid), "test": single_metrics(master_test)},
        "continuous_expert_only": {"validation": single_metrics(base_valid), "test": single_metrics(base_test)},
        "TA_expert_only": {"validation": single_metrics(ta_valid), "test": single_metrics(ta_test)},
        "sparse_expert_only": {"validation": single_metrics(sparse_valid), "test": single_metrics(sparse_test)},
        "MASTER_plus_continuous": {"validation": zfusion_metrics(master_valid, base_valid, 0.5),
                                   "test": zfusion_metrics(master_test, base_test, 0.5)},
        "MASTER_plus_TA": {"validation": zfusion_metrics(master_valid, ta_valid, 0.5),
                           "test": zfusion_metrics(master_test, ta_test, 0.5)},
        "DynaFuse": {"validation": zfusion_metrics(master_valid, sparse_valid, 0.5),
                     "test": zfusion_metrics(master_test, sparse_test, 0.5)},
        "DynaFuse_without_daily_zscore": {
            "validation": raw_fusion_metrics(master_valid, sparse_valid, 0.5),
            "test": raw_fusion_metrics(master_test, sparse_test, 0.5),
        },
    }
    alpha_sensitivity = {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        alpha_sensitivity[str(alpha)] = {
            "validation": zfusion_metrics(master_valid, sparse_valid, alpha),
            "test": zfusion_metrics(master_test, sparse_test, alpha),
        }
    topk_sensitivity = {}
    paths = {
        1: (topk_dir / "ta_deformable_topk1_csi300_seed0_validation_predictions.npz",
            topk_dir / "ta_deformable_topk1_csi300_seed0_predictions.npz"),
        2: (topk_dir / "ta_deformable_topk2_csi300_seed0_validation_predictions.npz",
            topk_dir / "ta_deformable_topk2_csi300_seed0_predictions.npz"),
        4: (sparse_dir / "ta_deformable_csi300_seed0_validation_predictions.npz",
            sparse_dir / "ta_deformable_csi300_seed0_predictions.npz"),
        8: (topk_dir / "ta_deformable_topk8_csi300_seed0_validation_predictions.npz",
            topk_dir / "ta_deformable_topk8_csi300_seed0_predictions.npz"),
    }
    for k, (vp, tp) in paths.items():
        ev, et = load_npz(vp), load_npz(tp)
        topk_sensitivity[str(k)] = {
            "expert_validation": single_metrics(ev),
            "expert_test": single_metrics(et),
            "fused_validation": zfusion_metrics(master_valid, ev, 0.5),
            "fused_test": zfusion_metrics(master_test, et, 0.5),
        }
    result = {
        "experiment": "DynaFuse component, normalization, alpha, and top-k ablations",
        "protocol": {"train": [20100104, 20191224], "validation": [20200102, 20211224],
                     "test": [20220104, 20251231]},
        "universe": "csi300", "seed": 0,
        "progressive_ablation": progressive,
        "alpha_sensitivity": alpha_sensitivity,
        "topk_sensitivity": topk_sensitivity,
        "mode": "single-process single-threaded; locked checkpoints; no retraining",
    }
    target = out_dir / "dynafuse_ablation_csi300_seed0.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
