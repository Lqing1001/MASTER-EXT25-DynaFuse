"""Analyze the locked Top-1 DynaFuse under strictly training-only normalization."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"

import numpy as np
import torch

from analyze_protocol_expert_fusion import aligned_metrics, corr, load_npz
from run_master_component_ablation import ComponentMASTER
from run_master_ext import (
    DATASET_ROOT, VALID_END, VALID_START, UniverseStore, evaluate, summarize_daily,
)
from run_prism_backbone_ablation import (
    AblatedPrismRanker, AblatedPrismSpatial, TAPrismAblation,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def single_metrics(arr: dict[str, np.ndarray]) -> dict:
    rows = []
    for date in np.unique(arr["dates"]):
        idx = arr["dates"] == date
        pred = arr["predictions"][idx].astype(np.float64)
        label = arr["labels"][idx].astype(np.float64)
        valid = np.isfinite(pred) & np.isfinite(label)
        rows.append({
            "date": int(date), "n": int(idx.sum()), "finite_labels": int(valid.sum()),
            "IC": corr(pred[valid], label[valid]),
            "RankIC": corr(pred[valid], label[valid], rank=True),
        })
    return summarize_daily(rows)


def zfusion_metrics(master: dict[str, np.ndarray], expert: dict[str, np.ndarray], alpha: float) -> dict:
    rows, diagnostics, _ = aligned_metrics(master, expert, alpha)
    return {"metrics": summarize_daily(rows), "diagnostics": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk-sweep-dir", type=Path)
    parser.add_argument("--universe", choices=("csi300", "csi800"), default="csi300")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to reconstruct validation predictions")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest = DATASET_ROOT / "manifest.json"
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    if manifest["status"] != "PASS":
        raise RuntimeError("strict dataset manifest did not pass")
    audit = manifest["temporal_audit"]
    if audit["training_feature_cutoff"] != 20191224 or audit["validation_feature_overlap"]:
        raise RuntimeError(f"invalid strict temporal audit: {audit}")

    universe = args.universe
    store = UniverseStore(DATASET_ROOT, universe)
    master_dir = args.protocol_root / "master"
    continuous_dir = args.protocol_root / "continuous_prism"
    sparse_dir = args.protocol_root / "sparse_top1"

    master = ComponentMASTER("full", universe).to(device)
    master.load_state_dict(torch.load(
        master_dir / f"master_full_{universe}_seed0_best.pt",
        map_location=device, weights_only=True,
    ))
    _, master_valid = evaluate(master, store, VALID_START, VALID_END, device, None)
    master_test = load_npz(master_dir / f"master_full_{universe}_seed0_predictions.npz")

    base = AblatedPrismRanker(AblatedPrismSpatial("no_vq"), "no_vq").to(device)
    base.load_state_dict(torch.load(
        continuous_dir / f"no_vq_{universe}_seed0_base_best.pt",
        map_location=device, weights_only=True,
    ))
    _, base_valid = evaluate(base, store, VALID_START, VALID_END, device, None)
    base_test = load_npz(continuous_dir / f"no_vq_{universe}_seed0_base_predictions.npz")

    ta = TAPrismAblation(base).to(device)
    ta.load_state_dict(torch.load(
        continuous_dir / f"no_vq_{universe}_seed0_adapter_best.pt",
        map_location=device, weights_only=True,
    ), strict=False)
    _, ta_valid = evaluate(ta, store, VALID_START, VALID_END, device, None)
    ta_test = load_npz(continuous_dir / f"no_vq_{universe}_seed0_adapter_predictions.npz")

    sparse_valid = load_npz(
        sparse_dir / f"ta_deformable_topk1_{universe}_seed0_validation_predictions.npz"
    )
    sparse_test = load_npz(
        sparse_dir / f"ta_deformable_topk1_{universe}_seed0_predictions.npz"
    )

    alpha_values = (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)
    validation_alpha_sweep = {
        str(alpha): zfusion_metrics(master_valid, sparse_valid, alpha)
        for alpha in alpha_values
    }
    validation_topk_sweep = {}
    if args.topk_sweep_dir is not None:
        topk_sources = {
            1: sparse_dir / f"ta_deformable_topk1_{universe}_seed0_validation_predictions.npz",
            2: args.topk_sweep_dir / f"ta_deformable_topk2_{universe}_seed0_validation_predictions.npz",
            4: args.topk_sweep_dir / f"ta_deformable_{universe}_seed0_validation_predictions.npz",
            8: args.topk_sweep_dir / f"ta_deformable_topk8_{universe}_seed0_validation_predictions.npz",
        }
        for top_k, source in topk_sources.items():
            expert_valid = load_npz(source)
            validation_topk_sweep[str(top_k)] = {
                "expert": single_metrics(expert_valid),
                "fused_alpha_0.5": zfusion_metrics(master_valid, expert_valid, 0.5),
                "prediction_file": str(source.resolve()),
            }

    progressive = {
        "MASTER": {
            "validation": single_metrics(master_valid),
            "test": single_metrics(master_test),
        },
        "Continuous_PRISM": {
            "validation": single_metrics(base_valid),
            "test": single_metrics(base_test),
        },
        "TA_PRISM": {
            "validation": single_metrics(ta_valid),
            "test": single_metrics(ta_test),
        },
        "Top1_sparse_expert": {
            "validation": single_metrics(sparse_valid),
            "test": single_metrics(sparse_test),
        },
        "DynaFuse_Top1_alpha0.5_daily_zscore": {
            "validation": zfusion_metrics(master_valid, sparse_valid, 0.5),
            "test": zfusion_metrics(master_test, sparse_test, 0.5),
        },
    }
    endpoints = {
        "alpha_0_MASTER": zfusion_metrics(master_test, sparse_test, 0.0),
        "alpha_0.5_DynaFuse": zfusion_metrics(master_test, sparse_test, 0.5),
        "alpha_1_Top1": zfusion_metrics(master_test, sparse_test, 1.0),
    }
    master_metrics = progressive["MASTER"]["test"]
    final_metrics = progressive["DynaFuse_Top1_alpha0.5_daily_zscore"]["test"]["metrics"]
    metric_names = ("IC", "RankIC", "ICIR", "RankICIR")
    deltas = {name: final_metrics[name] - master_metrics[name] for name in metric_names}
    all_four_improve = all(deltas[name] > 0 for name in metric_names)

    result = {
        "experiment": "locked DynaFuse under strictly training-only normalization",
        "universe": universe,
        "dataset": str(DATASET_ROOT),
        "dataset_manifest_sha256": sha256(dataset_manifest),
        "normalization_fit": [20100104, 20191224],
        "protocol": {
            "train": [20100104, 20191224],
            "validation": [20200102, 20211224],
            "test": [20220104, 20251231],
        },
        "locked_configuration": {
            "top_k": 1, "alpha": 0.5, "daily_zscore": True, "seed": 0,
        },
        "progressive": progressive,
        "validation_alpha_sweep": validation_alpha_sweep,
        "validation_topk_sweep": validation_topk_sweep,
        "fixed_alpha_endpoints": endpoints,
        "delta_vs_MASTER": deltas,
        "claim_gate": {
            "criterion": "DynaFuse Test IC, RankIC, ICIR, and RankICIR all exceed MASTER",
            "all_four_improve": all_four_improve,
            "decision": "CONTINUE_BASELINE_REFRESH" if all_four_improve else "STOP_AND_REASSESS",
        },
        "selection_note": "Top-k and alpha were locked before this strict rerun; test data were not used for selection.",
    }
    target = args.output_dir / f"strict_dynafuse_{universe}_seed0.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
