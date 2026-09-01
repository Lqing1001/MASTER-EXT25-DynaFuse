"""Reproduce official MASTER on MASTER-EXT-clean-v1.

The model architecture and optimization protocol follow SJTU-DMTai/MASTER.
This adapter only replaces Qlib's serialized sampler with a memory-mapped reader
for the cleaned annual NPY packages.  It is deliberately single-process and
single-threaded; one complete trading-day cross-section is one GPU batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.environ.get("MASTER_EXT_DATASET_ROOT", ROOT / "datasets" / "master_ext_clean_v1")).resolve()
OFFICIAL_ROOT = ROOT / "external" / "MASTER-official"
sys.path.insert(0, str(OFFICIAL_ROOT))
from master import MASTER  # noqa: E402


TRAIN_START = 20100104
TRAIN_END = 20200331
VALID_START = 20200401
VALID_END = 20200630
TEST_START = 20200701
TEST_END = 20251231
PAPER_TEST_END = 20221231
EXTENSION_START = 20230101
STEP_LEN = 8
FEATURES = 221


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


@dataclass(frozen=True)
class DayLocation:
    year: int
    start: int
    end: int


class UniverseStore:
    """Memory-mapped daily cross sections with Qlib-equivalent TS padding."""

    def __init__(self, dataset_root: Path, universe: str):
        self.universe = universe
        self.base = dataset_root / "master_input" / universe / "by_year"
        self.arrays: dict[int, dict[str, np.ndarray]] = {}
        self.day_locations: dict[int, DayLocation] = {}
        for year_dir in sorted(self.base.glob("year=*")):
            year = int(year_dir.name.split("=")[1])
            arrays = {
                "features": np.load(year_dir / "features.npy", mmap_mode="r"),
                "labels": np.load(year_dir / "labels.npy", mmap_mode="r"),
                "dates": np.load(year_dir / "dates.npy", mmap_mode="r"),
                "instruments": np.load(year_dir / "instruments.npy", mmap_mode="r"),
            }
            if arrays["features"].shape[1] != FEATURES:
                raise ValueError(f"{year}: expected {FEATURES} features")
            self.arrays[year] = arrays
            dates = arrays["dates"]
            unique, starts, counts = np.unique(dates, return_index=True, return_counts=True)
            for day, start, count in zip(unique, starts, counts):
                self.day_locations[int(day)] = DayLocation(year, int(start), int(start + count))
        self.calendar = np.array(sorted(self.day_locations), dtype=np.int32)
        self.calendar_pos = {int(day): pos for pos, day in enumerate(self.calendar)}
        if len(self.calendar) != 3886:
            raise ValueError(f"unexpected calendar length: {len(self.calendar)}")

    def dates_between(self, start: int, end: int) -> np.ndarray:
        mask = (self.calendar >= start) & (self.calendar <= end)
        return self.calendar[mask]

    def _day_arrays(self, day: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        location = self.day_locations[int(day)]
        arrays = self.arrays[location.year]
        slc = slice(location.start, location.end)
        return arrays["features"][slc], arrays["labels"][slc], arrays["instruments"][slc]

    def batch(self, day: int, training: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        current_x, current_y, current_inst = self._day_arrays(day)
        if training:
            current_keep = np.isfinite(current_y)
            current_y = np.asarray(current_y[current_keep], dtype=np.float32)
            current_inst = np.asarray(current_inst[current_keep])
        else:
            current_y = np.asarray(current_y, dtype=np.float32)
            current_inst = np.asarray(current_inst)

        n = len(current_inst)
        sequence = np.empty((n, STEP_LEN, FEATURES), dtype=np.float32)
        sequence.fill(np.nan)
        valid = np.zeros((n, STEP_LEN), dtype=bool)
        current_pos = self.calendar_pos[int(day)]
        first_pos = max(0, current_pos - STEP_LEN + 1)
        history = self.calendar[first_pos : current_pos + 1]
        offset = STEP_LEN - len(history)

        for slot, history_day in enumerate(history, start=offset):
            day_x, day_y, day_inst = self._day_arrays(int(history_day))
            if training:
                source_keep = np.isfinite(day_y)
                source_inst = np.asarray(day_inst[source_keep])
                source_x = day_x[source_keep]
            else:
                source_inst = np.asarray(day_inst)
                source_x = day_x
            positions = np.searchsorted(source_inst, current_inst)
            bounded = positions < len(source_inst)
            matched = np.zeros(n, dtype=bool)
            matched[bounded] = source_inst[positions[bounded]] == current_inst[bounded]
            if matched.any():
                sequence[matched, slot, :] = source_x[positions[matched]]
                valid[matched, slot] = True

        # Qlib MASTERTSDatasetH uses fillna_type="ffill+bfill" on row indices.
        for slot in range(1, STEP_LEN):
            missing = ~valid[:, slot] & valid[:, slot - 1]
            sequence[missing, slot, :] = sequence[missing, slot - 1, :]
            valid[missing, slot] = True
        for slot in range(STEP_LEN - 2, -1, -1):
            missing = ~valid[:, slot] & valid[:, slot + 1]
            sequence[missing, slot, :] = sequence[missing, slot + 1, :]
            valid[missing, slot] = True
        if not valid.all() or not np.isfinite(sequence).all():
            raise ValueError(f"non-finite sequence for {self.universe} {day}")
        return sequence, current_y, current_inst


def drop_extreme_and_zscore(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Official DropNA -> DropExtremeLabel -> CSZScoreNorm sequence."""
    finite = torch.isfinite(labels)
    original_indices = torch.nonzero(finite, as_tuple=False).squeeze(1)
    labels = labels[finite]
    count = int(0.025 * labels.shape[0])
    if count > 0:
        order = torch.argsort(labels)
        keep_local = order[count:-count]
        original_indices = original_indices[keep_local]
        labels = labels[keep_local]
    labels = (labels - labels.mean()) / labels.std()
    return original_indices, labels


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def correlation(pred: np.ndarray, label: np.ndarray, rank: bool = False) -> float:
    mask = np.isfinite(pred) & np.isfinite(label)
    if mask.sum() < 3:
        return math.nan
    x = pred[mask].astype(np.float64)
    y = label[mask].astype(np.float64)
    if rank:
        x = rank_average(x)
        y = rank_average(y)
    x -= x.mean()
    y -= y.mean()
    denom = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y / denom) if denom > 0 else math.nan


def summarize_daily(rows: list[dict]) -> dict:
    ic = np.array([row["IC"] for row in rows], dtype=np.float64)
    ric = np.array([row["RankIC"] for row in rows], dtype=np.float64)
    ic = ic[np.isfinite(ic)]
    ric = ric[np.isfinite(ric)]
    return {
        "days": len(rows),
        "valid_ic_days": int(len(ic)),
        "IC": float(ic.mean()),
        "IC_std": float(ic.std()),
        "ICIR": float(ic.mean() / ic.std()),
        "RankIC": float(ric.mean()),
        "RankIC_std": float(ric.std()),
        "RankICIR": float(ric.mean() / ric.std()),
    }


def evaluate(
    model: torch.nn.Module,
    store: UniverseStore,
    start: int,
    end: int,
    device: torch.device,
    max_days: int | None = None,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    dates = store.dates_between(start, end)
    if max_days is not None:
        dates = dates[:max_days]
    daily_rows: list[dict] = []
    pred_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    date_chunks: list[np.ndarray] = []
    inst_chunks: list[np.ndarray] = []
    model.eval()
    for number, day in enumerate(dates, 1):
        x, y, instruments = store.batch(int(day), training=False)
        with torch.inference_mode():
            pred = model(torch.from_numpy(x).to(device)).detach().cpu().numpy().reshape(-1)
        ic = correlation(pred, y)
        rank_ic = correlation(pred, y, rank=True)
        daily_rows.append(
            {
                "date": int(day),
                "n": int(len(y)),
                "finite_labels": int(np.isfinite(y).sum()),
                "IC": ic,
                "RankIC": rank_ic,
            }
        )
        pred_chunks.append(pred.astype(np.float32))
        label_chunks.append(y.astype(np.float32))
        date_chunks.append(np.full(len(y), int(day), dtype=np.int32))
        inst_chunks.append(instruments.astype("S8"))
        if number % 100 == 0 or number == len(dates):
            log(f"evaluate {store.universe}: {number}/{len(dates)} days")
    predictions = {
        "dates": np.concatenate(date_chunks),
        "instruments": np.concatenate(inst_chunks),
        "predictions": np.concatenate(pred_chunks),
        "labels": np.concatenate(label_chunks),
    }
    return daily_rows, predictions


def train_one_seed(
    store: UniverseStore,
    universe: str,
    seed: int,
    device: torch.device,
    epochs: int,
    lr: float,
    stop_loss: float,
    output_dir: Path,
    max_train_days: int | None,
    max_eval_days: int | None,
) -> tuple[torch.nn.Module, list[dict]]:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    beta = 5 if universe == "csi300" else 2
    model = MASTER(
        d_feat=158,
        d_model=256,
        t_nhead=4,
        s_nhead=2,
        T_dropout_rate=0.5,
        S_dropout_rate=0.5,
        gate_input_start_index=158,
        gate_input_end_index=221,
        beta=beta,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dates = store.dates_between(TRAIN_START, TRAIN_END)
    if max_train_days is not None:
        train_dates = train_dates[:max_train_days]
    rng = np.random.RandomState(seed)
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        order = train_dates.copy()
        rng.shuffle(order)
        losses = []
        started = time.time()
        for number, day in enumerate(order, 1):
            x, y, _ = store.batch(int(day), training=True)
            feature = torch.from_numpy(x).to(device)
            labels = torch.from_numpy(y).to(device)
            keep, normalized = drop_extreme_and_zscore(labels)
            feature = feature[keep]
            pred = model(feature)
            loss = torch.mean((pred - normalized) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at {day}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if number % 250 == 0 or number == len(order):
                log(f"{universe} seed={seed} epoch={epoch} train {number}/{len(order)} loss={np.mean(losses):.6f}")
        train_loss = float(np.mean(losses))
        valid_rows, _ = evaluate(model, store, VALID_START, VALID_END, device, max_eval_days)
        valid_metrics = summarize_daily(valid_rows)
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "elapsed_seconds": time.time() - started,
            "valid": valid_metrics,
        }
        history.append(epoch_row)
        log(
            f"{universe} seed={seed} epoch={epoch} train_loss={train_loss:.6f} "
            f"valid_IC={valid_metrics['IC']:.6f} valid_RankIC={valid_metrics['RankIC']:.6f}"
        )
        torch.save(model.state_dict(), output_dir / f"{universe}_local_seed{seed}_epoch{epoch}.pt")
        if train_loss <= stop_loss:
            log(f"stop threshold reached: {train_loss:.6f} <= {stop_loss:.6f}")
            break
    return model, history


def metrics_by_period(daily_rows: list[dict]) -> dict:
    periods: dict[str, tuple[int, int]] = {
        "paper_comparable_20200701_20221231": (TEST_START, PAPER_TEST_END),
        "extension_20230101_20251231": (EXTENSION_START, TEST_END),
        "all_20200701_20251231": (TEST_START, TEST_END),
    }
    for year in range(2020, 2026):
        periods[str(year)] = (year * 10000 + 101, year * 10000 + 1231)
    result = {}
    for name, (start, end) in periods.items():
        subset = [row for row in daily_rows if start <= row["date"] <= end]
        if subset:
            result[name] = summarize_daily(subset)
    return result


def write_daily_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "n", "finite_labels", "IC", "RankIC"])
        writer.writeheader()
        writer.writerows(rows)


def load_official_model(universe: str, device: torch.device) -> torch.nn.Module:
    beta = 5 if universe == "csi300" else 2
    model = MASTER(
        d_feat=158,
        d_model=256,
        t_nhead=4,
        s_nhead=2,
        T_dropout_rate=0.5,
        S_dropout_rate=0.5,
        gate_input_start_index=158,
        gate_input_end_index=221,
        beta=beta,
    ).to(device)
    checkpoint = OFFICIAL_ROOT / "model" / f"{universe}_opensource_0.pkl"
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["csi300", "csi800"], required=True)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--stop-loss", type=float, default=0.95)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate-official", action="store_true")
    parser.add_argument("--max-train-days", type=int)
    parser.add_argument("--max-eval-days", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "master_ext_reproduction")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the official MASTER cross-sectional attention")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = UniverseStore(DATASET_ROOT, args.universe)
    log(f"loaded {args.universe}; GPU={torch.cuda.get_device_name(0)}; calendar={len(store.calendar)}")

    passport = {
        "experiment": "MASTER on MASTER-EXT-clean-v1",
        "universe": args.universe,
        "mode": "single-process single-threaded",
        "official_commit": "de8f58557096abde4216a701b35fc4368158d111",
        "dataset": str(DATASET_ROOT),
        "train": [TRAIN_START, TRAIN_END],
        "valid": [VALID_START, VALID_END],
        "test": [TEST_START, TEST_END],
        "step_len": STEP_LEN,
        "features": FEATURES,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "arguments": vars(args) | {"output_dir": str(args.output_dir.resolve())},
    }
    (args.output_dir / f"passport_{args.universe}.json").write_text(
        json.dumps(passport, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.evaluate_official:
        model = load_official_model(args.universe, device)
        rows, predictions = evaluate(model, store, TEST_START, TEST_END, device, args.max_eval_days)
        result = {
            "source": "official opensource seed-0 checkpoint",
            "universe": args.universe,
            "metrics": metrics_by_period(rows),
        }
        (args.output_dir / f"official_checkpoint_{args.universe}_seed0.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_daily_csv(args.output_dir / f"official_checkpoint_{args.universe}_seed0_daily.csv", rows)
        np.savez_compressed(args.output_dir / f"official_checkpoint_{args.universe}_seed0_predictions.npz", **predictions)
        log(json.dumps(result, ensure_ascii=False))

    if args.train:
        for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
            model, history = train_one_seed(
                store=store,
                universe=args.universe,
                seed=seed,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                stop_loss=args.stop_loss,
                output_dir=args.output_dir,
                max_train_days=args.max_train_days,
                max_eval_days=args.max_eval_days,
            )
            rows, predictions = evaluate(model, store, TEST_START, TEST_END, device, args.max_eval_days)
            result = {
                "source": "locally trained official MASTER",
                "universe": args.universe,
                "seed": seed,
                "history": history,
                "metrics": metrics_by_period(rows),
            }
            (args.output_dir / f"local_{args.universe}_seed{seed}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            write_daily_csv(args.output_dir / f"local_{args.universe}_seed{seed}_daily.csv", rows)
            np.savez_compressed(args.output_dir / f"local_{args.universe}_seed{seed}_predictions.npz", **predictions)
            log(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
