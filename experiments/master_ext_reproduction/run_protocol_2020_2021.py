"""Launch an existing experiment under the fixed 2020-2021 validation protocol.

The wrapped experiment remains a single Python process with one CPU thread.
Existing scripts and their historical default splits are left unchanged.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"

TRAIN_START, TRAIN_END = 20100104, 20191224
VALID_START, VALID_END = 20200102, 20211224
TEST_START, TEST_END = 20220104, 20251231


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_protocol_2020_2021.py MODULE [module args]")
    target_name = sys.argv[1]
    target_args = sys.argv[2:]
    if target_args[:1] == ["--"]:
        target_args = target_args[1:]

    base = importlib.import_module("run_master_ext")
    for key, value in {
        "TRAIN_START": TRAIN_START, "TRAIN_END": TRAIN_END,
        "VALID_START": VALID_START, "VALID_END": VALID_END,
        "TEST_START": TEST_START, "TEST_END": TEST_END,
        "PAPER_TEST_END": 20231231, "EXTENSION_START": 20240101,
    }.items():
        setattr(base, key, value)

    def protocol_metrics(daily_rows: list[dict]) -> dict:
        periods = {
            "early_test_20220104_20231231": (TEST_START, 20231231),
            "extension_20240101_20251231": (20240101, TEST_END),
            "all_20220104_20251231": (TEST_START, TEST_END),
        }
        for year in range(2022, 2026):
            periods[str(year)] = (year * 10000 + 101, year * 10000 + 1231)
        result = {}
        for name, (start, end) in periods.items():
            subset = [row for row in daily_rows if start <= row["date"] <= end]
            if subset:
                result[name] = base.summarize_daily(subset)
        return result

    base.metrics_by_period = protocol_metrics
    target = importlib.import_module(target_name)
    for key, value in {
        "TRAIN_START": TRAIN_START, "TRAIN_END": TRAIN_END,
        "VALID_START": VALID_START, "VALID_END": VALID_END,
        "TEST_START": TEST_START, "TEST_END": TEST_END,
        "metrics_by_period": protocol_metrics,
    }.items():
        if hasattr(target, key):
            setattr(target, key, value)

    if "--output-dir" in target_args:
        output_dir = Path(target_args[target_args.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "protocol": "fixed two-year validation",
            "train": [TRAIN_START, TRAIN_END],
            "validation": [VALID_START, VALID_END],
            "test": [TEST_START, TEST_END],
            "checkpoint_selection": "wrapped experiment validation rule",
            "execution": "single-process single-threaded",
            "target_module": target_name,
            "dataset_root": str(Path(os.environ.get("MASTER_EXT_DATASET_ROOT", Path(__file__).resolve().parents[2] / "datasets" / "master_ext_clean_v1")).resolve()),
            "normalization_fit_end": os.environ.get("MASTER_EXT_NORMALIZATION_FIT_END"),
        }
        (output_dir / "_protocol.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    sys.argv = [target_name, *target_args]
    target.main()


if __name__ == "__main__":
    main()
