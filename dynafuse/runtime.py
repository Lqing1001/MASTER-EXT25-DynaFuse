"""Final MASTER-EXT25 data access and daily evaluation protocol."""
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


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get("MASTER_EXT_DATASET_ROOT", ROOT / "datasets" / "master_ext_strict_20191224_v1")).resolve()

TRAIN_START, TRAIN_END = 20100104, 20191224
VALID_START, VALID_END = 20200102, 20211224
TEST_START, TEST_END = 20220104, 20251231
PAPER_TEST_END, EXTENSION_START = 20231231, 20240101
STEP_LEN, FEATURES = 8, 221

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
        self.base = dataset_root / 'master_input' / universe / 'by_year'
        self.arrays: dict[int, dict[str, np.ndarray]] = {}
        self.day_locations: dict[int, DayLocation] = {}
        for year_dir in sorted(self.base.glob('year=*')):
            year = int(year_dir.name.split('=')[1])
            arrays = {'features': np.load(year_dir / 'features.npy', mmap_mode='r'), 'labels': np.load(year_dir / 'labels.npy', mmap_mode='r'), 'dates': np.load(year_dir / 'dates.npy', mmap_mode='r'), 'instruments': np.load(year_dir / 'instruments.npy', mmap_mode='r')}
            if arrays['features'].shape[1] != FEATURES:
                raise ValueError(f'{year}: expected {FEATURES} features')
            self.arrays[year] = arrays
            dates = arrays['dates']
            unique, starts, counts = np.unique(dates, return_index=True, return_counts=True)
            for day, start, count in zip(unique, starts, counts):
                self.day_locations[int(day)] = DayLocation(year, int(start), int(start + count))
        self.calendar = np.array(sorted(self.day_locations), dtype=np.int32)
        self.calendar_pos = {int(day): pos for pos, day in enumerate(self.calendar)}
        if len(self.calendar) != 3886:
            raise ValueError(f'unexpected calendar length: {len(self.calendar)}')

    def dates_between(self, start: int, end: int) -> np.ndarray:
        mask = (self.calendar >= start) & (self.calendar <= end)
        return self.calendar[mask]

    def _day_arrays(self, day: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        location = self.day_locations[int(day)]
        arrays = self.arrays[location.year]
        slc = slice(location.start, location.end)
        return (arrays['features'][slc], arrays['labels'][slc], arrays['instruments'][slc])

    def batch(self, day: int, training: bool=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        history = self.calendar[first_pos:current_pos + 1]
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
        for slot in range(1, STEP_LEN):
            missing = ~valid[:, slot] & valid[:, slot - 1]
            sequence[missing, slot, :] = sequence[missing, slot - 1, :]
            valid[missing, slot] = True
        for slot in range(STEP_LEN - 2, -1, -1):
            missing = ~valid[:, slot] & valid[:, slot + 1]
            sequence[missing, slot, :] = sequence[missing, slot + 1, :]
            valid[missing, slot] = True
        if not valid.all() or not np.isfinite(sequence).all():
            raise ValueError(f'non-finite sequence for {self.universe} {day}')
        return (sequence, current_y, current_inst)

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
    return (original_indices, labels)

def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='mergesort')
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

def correlation(pred: np.ndarray, label: np.ndarray, rank: bool=False) -> float:
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
    ic = np.array([row['IC'] for row in rows], dtype=np.float64)
    ric = np.array([row['RankIC'] for row in rows], dtype=np.float64)
    ic = ic[np.isfinite(ic)]
    ric = ric[np.isfinite(ric)]
    return {'days': len(rows), 'valid_ic_days': int(len(ic)), 'IC': float(ic.mean()), 'IC_std': float(ic.std()), 'ICIR': float(ic.mean() / ic.std()), 'RankIC': float(ric.mean()), 'RankIC_std': float(ric.std()), 'RankICIR': float(ric.mean() / ric.std())}

def evaluate(model: torch.nn.Module, store: UniverseStore, start: int, end: int, device: torch.device, max_days: int | None=None) -> tuple[list[dict], dict[str, np.ndarray]]:
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
        daily_rows.append({'date': int(day), 'n': int(len(y)), 'finite_labels': int(np.isfinite(y).sum()), 'IC': ic, 'RankIC': rank_ic})
        pred_chunks.append(pred.astype(np.float32))
        label_chunks.append(y.astype(np.float32))
        date_chunks.append(np.full(len(y), int(day), dtype=np.int32))
        inst_chunks.append(instruments.astype('S8'))
        if number % 100 == 0 or number == len(dates):
            log(f'evaluate {store.universe}: {number}/{len(dates)} days')
    predictions = {'dates': np.concatenate(date_chunks), 'instruments': np.concatenate(inst_chunks), 'predictions': np.concatenate(pred_chunks), 'labels': np.concatenate(label_chunks)}
    return (daily_rows, predictions)

def write_daily_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=['date', 'n', 'finite_labels', 'IC', 'RankIC'])
        writer.writeheader()
        writer.writerows(rows)

def metrics_by_period(daily_rows):
    periods = {'early_test_20220104_20231231':(TEST_START,20231231), 'extension_20240101_20251231':(20240101,TEST_END), 'all_20220104_20251231':(TEST_START,TEST_END)}
    periods.update({str(y):(y*10000+101,y*10000+1231) for y in range(2022,2026)})
    return {k:summarize_daily([r for r in daily_rows if a<=r['date']<=b]) for k,(a,b) in periods.items() if any(a<=r['date']<=b for r in daily_rows)}
