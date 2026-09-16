from __future__ import annotations
import argparse
import copy
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import torch
from torch import nn
from dynafuse.runtime import DATASET_ROOT, ROOT, TEST_END, TEST_START, TRAIN_END, TRAIN_START, VALID_END, VALID_START, UniverseStore, drop_extreme_and_zscore, evaluate, metrics_by_period, summarize_daily, write_daily_csv
from dynafuse.vendor.master_components import Gate, PositionalEncoding, SAttention, TAttention, TemporalAttention

def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class MasterExpert(nn.Module):

    def __init__(self, variant: str, universe: str, d_model: int=256):
        super().__init__()
        self.variant = variant
        self.use_gate = True
        self.use_temporal = True
        self.use_spatial = True
        self.mean_pool = False
        if d_model % 4:
            raise ValueError('d_model must be divisible by the four temporal heads')
        beta = 5 if universe == 'csi300' else 2
        self.feature_gate = Gate(63, 158, beta=beta)
        self.projection = nn.Linear(158, d_model)
        self.position = PositionalEncoding(d_model)
        self.temporal = TAttention(d_model=d_model, nhead=4, dropout=0.5)
        self.spatial = SAttention(d_model=d_model, nhead=2, dropout=0.5)
        if not self.mean_pool:
            self.pool = TemporalAttention(d_model=d_model)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stock = x[:, :, :158]
        gate = self.feature_gate(x[:, -1, 158:221])
        stock = stock * gate.unsqueeze(1)
        z = self.position(self.projection(stock))
        z = self.temporal(z)
        z = self.spatial(z)
        z = self.pool(z)
        return self.decoder(z).squeeze(-1)

def train_one(args, store, model, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dates = store.dates_between(TRAIN_START, TRAIN_END)
    if args.max_train_days is not None:
        dates = dates[:args.max_train_days]
    rng = np.random.RandomState(args.seed)
    best_score = -float('inf')
    best_state = None
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = dates.copy()
        rng.shuffle(order)
        losses = []
        started = time.time()
        for number, day in enumerate(order, 1):
            x, y, _ = store.batch(int(day), training=True)
            feature = torch.from_numpy(x).to(device)
            labels = torch.from_numpy(np.array(y, copy=True)).to(device)
            keep, normalized = drop_extreme_and_zscore(labels)
            pred = model(feature[keep])
            loss = torch.mean((pred - normalized) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at {day}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if number % 250 == 0 or number == len(order):
                log(f'{args.variant} epoch={epoch} train {number}/{len(order)} loss={np.mean(losses):.6f}')
        rows, _ = evaluate(model, store, VALID_START, VALID_END, device, args.max_eval_days)
        valid = summarize_daily(rows)
        score = 0.5 * (valid['IC'] + valid['RankIC'])
        improved = score > best_score
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation': valid, 'selection_score': score, 'best': improved, 'elapsed_seconds': time.time() - started})
        log(f"{args.variant} epoch={epoch} valid_IC={valid['IC']:.6f} valid_RankIC={valid['RankIC']:.6f} best={best_epoch}")
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return (history, best_epoch)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.set_defaults(variant='full')
    parser.add_argument('--universe', choices=('csi300', 'csi800'), default='csi300')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--max-train-days', type=int)
    parser.add_argument('--max-eval-days', type=int)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'master')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    seed_all(args.seed)
    device = torch.device('cuda:0')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = UniverseStore(DATASET_ROOT, args.universe)
    model = MasterExpert(args.variant, args.universe, d_model=args.d_model).to(device)
    started = time.time()
    history, best_epoch = train_one(args, store, model, device)
    width_suffix = '' if args.d_model == 256 else f'_d{args.d_model}'
    prefix = f'master_{args.variant}{width_suffix}_{args.universe}_seed{args.seed}'
    torch.save(model.state_dict(), args.output_dir / f'{prefix}_best.pt')
    rows, predictions = evaluate(model, store, TEST_START, TEST_END, device, args.max_eval_days)
    result = {'experiment': 'MASTER expert', 'variant': args.variant, 'universe': args.universe, 'seed': args.seed, 'best_epoch': best_epoch, 'history': history, 'metrics': metrics_by_period(rows), 'parameters': sum((p.numel() for p in model.parameters())), 'd_model': args.d_model, 'elapsed_seconds': time.time() - started, 'mode': 'single-process single-threaded', 'config': vars(args) | {'output_dir': str(args.output_dir.resolve())}}
    (args.output_dir / f'{prefix}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    write_daily_csv(args.output_dir / f'{prefix}_daily.csv', rows)
    np.savez_compressed(args.output_dir / f'{prefix}_predictions.npz', **predictions)
    log(json.dumps({'variant': args.variant, 'best_epoch': best_epoch, 'metrics': result['metrics']}, ensure_ascii=False))
if __name__ == '__main__':
    main()
