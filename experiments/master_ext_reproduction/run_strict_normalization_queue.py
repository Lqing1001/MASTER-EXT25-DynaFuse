"""Resume-safe serial queue for the strict training-only DynaFuse reproduction."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "run_protocol_2020_2021.py"
DATASET = ROOT / "datasets" / "master_ext_strict_20191224_v1"
RESULTS = ROOT / "results" / "protocol_strict_norm_20191224"
STATUS = RESULTS / "strict_queue_status.json"

ENV = os.environ.copy()
for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    ENV[name] = "1"
ENV["CUDA_VISIBLE_DEVICES"] = "0"
ENV["MASTER_EXT_DATASET_ROOT"] = str(DATASET)
ENV["MASTER_EXT_NORMALIZATION_FIT_END"] = "2019-12-24"


def protocol_command(module: str, *args: str) -> list[str]:
    return [sys.executable, str(LAUNCHER), module, "--", *args]


def write_status(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_dataset() -> None:
    manifest_path = DATASET / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "PASS":
        raise RuntimeError("strict dataset status is not PASS")
    audit = manifest["temporal_audit"]
    if audit["training_feature_cutoff"] != 20191224:
        raise RuntimeError(f"unexpected cutoff: {audit}")
    if audit["validation_feature_overlap"] or audit["labels_used_for_normalization"]:
        raise RuntimeError(f"strict temporal audit failed: {audit}")
    if max(audit["alpha158_fit_max"].values()) > 20191224 or audit["market63_fit_max"] > 20191224:
        raise RuntimeError(f"fit maximum exceeds training boundary: {audit}")


def main() -> None:
    master_dir = RESULTS / "master"
    continuous_dir = RESULTS / "continuous_prism"
    sparse_dir = RESULTS / "sparse_top1"
    final_dir = RESULTS / "final_analysis"
    tasks = [
        (
            "strict_dataset",
            DATASET / "manifest.json",
            [
                sys.executable, str(ROOT / "scripts" / "rebuild_strict_train_only_packages.py"),
                "--source-dataset-root", str(ROOT / "datasets" / "master_ext_clean_v1"),
                "--output-dataset-root", str(DATASET),
                "--audit-output", str(ROOT / "audit_outputs" / "master_ext_strict_20191224_v1_audit.json"),
            ],
        ),
        (
            "master_seed0",
            master_dir / "master_full_csi300_seed0.json",
            protocol_command(
                "run_master_component_ablation", "--variant", "full",
                "--universe", "csi300", "--seed", "0", "--epochs", "40",
                "--patience", "40", "--output-dir", str(master_dir),
            ),
        ),
        (
            "continuous_ta_seed0",
            continuous_dir / "no_vq_csi300_seed0.json",
            protocol_command(
                "run_prism_backbone_ablation", "--variant", "no_vq",
                "--universe", "csi300", "--seed", "0", "--stage1-epochs", "4",
                "--base-epochs", "12", "--adapter-epochs", "10",
                "--patience", "12", "--output-dir", str(continuous_dir),
            ),
        ),
        (
            "top1_sparse_seed0",
            sparse_dir / "ta_deformable_topk1_csi300_seed0.json",
            protocol_command(
                "run_topvenue_ta_components_p2", "--variant", "deformable",
                "--selected", "1", "--universe", "csi300", "--seed", "0",
                "--epochs", "10", "--patience", "10",
                "--continuous-dir", str(continuous_dir), "--output-dir", str(sparse_dir),
            ),
        ),
        (
            "strict_final_analysis",
            final_dir / "strict_dynafuse_csi300_seed0.json",
            protocol_command(
                "analyze_strict_dynafuse", "--protocol-root", str(RESULTS),
                "--output-dir", str(final_dir),
            ),
        ),
    ]

    completed: list[str] = []
    write_status({
        "status": "RUNNING", "started": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(DATASET), "results": str(RESULTS),
        "current": None, "completed": completed, "total": len(tasks),
    })
    for index, (name, expected, command) in enumerate(tasks, 1):
        if expected.exists():
            if name == "strict_dataset":
                verify_dataset()
            print(f"[strict-queue] skip existing {name}: {expected}", flush=True)
            completed.append(name)
            continue
        write_status({
            "status": "RUNNING", "updated": datetime.now().isoformat(timespec="seconds"),
            "current": name, "index": index, "completed": completed,
            "total": len(tasks), "command": command,
        })
        print(f"[strict-queue] start {index}/{len(tasks)} {name}", flush=True)
        subprocess.run(command, cwd=ROOT, env=ENV, check=True)
        if not expected.exists():
            raise RuntimeError(f"{name} completed without expected output: {expected}")
        if name == "strict_dataset":
            verify_dataset()
        completed.append(name)
        print(f"[strict-queue] complete {name}", flush=True)

    final = json.loads((final_dir / "strict_dynafuse_csi300_seed0.json").read_text(encoding="utf-8"))
    write_status({
        "status": "COMPLETED", "finished": datetime.now().isoformat(timespec="seconds"),
        "current": None, "completed": completed, "total": len(tasks),
        "claim_gate": final["claim_gate"], "delta_vs_MASTER": final["delta_vs_MASTER"],
    })
    print(json.dumps(final["claim_gate"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()