from __future__ import annotations
import argparse
import copy
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dynafuse.runtime import DATASET_ROOT, ROOT, TEST_END, TEST_START, TRAIN_END, TRAIN_START, VALID_END, VALID_START, UniverseStore, drop_extreme_and_zscore, evaluate, metrics_by_period, summarize_daily, write_daily_csv
from dynafuse.core import MasterTemporalResidual
from dynafuse.core import pearson_loss

def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class ContinuousEncoder(nn.Module):

    def __init__(self, variant: str, d_model: int=64, codebook_size: int=512):
        super().__init__()
        self.variant = variant
        self.use_vq = False
        self.use_cross_stock = True
        self.use_market_prior = True
        self.gru = nn.GRU(158, d_model, batch_first=True)
        layer = nn.TransformerEncoderLayer(d_model, 2, d_model * 2, 0.1, batch_first=True, norm_first=True)
        self.cross = nn.TransformerEncoder(layer, 2)
        # Registered only for published checkpoint/parameter-count compatibility; never used in forward.
        self.codebook = nn.Embedding(codebook_size, d_model)
        nn.init.normal_(self.codebook.weight, std=0.1)
        self.decoder = nn.Sequential(nn.Linear(d_model + 64, 256), nn.GELU(), nn.Linear(256, 158))
        self.aux = nn.Sequential(nn.Linear(d_model + 64, 128), nn.GELU(), nn.Linear(128, 1))
        self.prior = nn.Linear(63, 64)

    def encode(self, x: torch.Tensor):
        stock, market = (x[:, :, :158], x[:, :, 158:221])
        mean = stock.mean(1, keepdim=True)
        std = stock.std(1, keepdim=True).clamp_min(1e-05)
        norm = (stock - mean) / std
        h = self.gru(norm)[1][-1]
        z = self.cross(h.unsqueeze(0)).squeeze(0)
        indices = torch.full((z.shape[0],), -1, dtype=torch.long, device=z.device)
        quant = z
        straight = z
        prior = self.prior(market[:, -1])
        return (z, quant, straight, indices, prior)

    def forward(self, x: torch.Tensor):
        z, quant, straight, indices, prior = self.encode(x)
        joined = torch.cat([straight, prior], dim=-1)
        reconstruction = self.decoder(joined)
        aux = self.aux(joined).squeeze(-1)
        return (z, quant, indices, reconstruction, aux)

class MixtureRanker(nn.Module):

    def __init__(self, spatial: ContinuousEncoder, variant: str, d_model: int=64, experts: int=2):
        super().__init__()
        self.spatial = spatial
        self.variant = variant
        self.use_temporal_encoder = True
        self.use_conditioned_gate = True
        self.input_proj = nn.Linear(158, d_model)
        layer = nn.TransformerEncoderLayer(d_model, 2, d_model * 2, 0.1, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2)
        self.gate = nn.Linear(d_model, experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model + 64, 128), nn.GELU(), nn.Linear(128, 1)) for _ in range(experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, state, _, _, prior = self.spatial.encode(x)
        seq = self.input_proj(x[:, :, :158])
        context = self.temporal(torch.cat([state.unsqueeze(1), seq], dim=1))[:, 0]
        weights = torch.softmax(self.gate(state), dim=-1)
        joined = torch.cat([context, prior], dim=-1)
        outputs = torch.cat([expert(joined) for expert in self.experts], dim=1)
        return (weights * outputs).sum(1)

class TemporalAdapter(nn.Module):

    def __init__(self, base: MixtureRanker):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.residual = MasterTemporalResidual()
        self.residual_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.base.eval()
        with torch.no_grad():
            base_pred = self.base(x)
        residual = self.residual(x[:, :, :158])
        return base_pred + torch.tanh(self.residual_scale) * residual

def stage1_loss(spatial: ContinuousEncoder, variant: str, x: torch.Tensor, target: torch.Tensor):
    _, _, _, reconstruction, aux = spatial(x)
    terms = {'reconstruction': F.mse_loss(reconstruction, x[:, -1, :158]), 'aux_return': 0.0001 * F.mse_loss(aux, target)}
    return (sum(terms.values()), {name: float(value.detach().cpu()) for name, value in terms.items()})

def evaluate_for_selection(model, store, device, max_eval_days):
    rows, _ = evaluate(model, store, VALID_START, VALID_END, device, max_eval_days)
    metrics = summarize_daily(rows)
    return (metrics, 0.5 * (metrics['IC'] + metrics['RankIC']))

def train_spatial(spatial, variant, store, dates, optimizer, device, epochs, max_train_days):
    if max_train_days is not None:
        dates = dates[:max_train_days]
    history = []
    for epoch in range(epochs):
        spatial.train()
        losses = []
        term_sums: dict[str, list[float]] = {}
        for number, day in enumerate(dates, 1):
            x, y, _ = store.batch(int(day), training=True)
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            keep, target = drop_extreme_and_zscore(y)
            loss, terms = stage1_loss(spatial, variant, x[keep], target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(spatial.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for name, value in terms.items():
                term_sums.setdefault(name, []).append(value)
            if number % 250 == 0 or number == len(dates):
                log(f'{variant} spatial epoch={epoch} {number}/{len(dates)} loss={np.mean(losses):.6f}')
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'terms': {name: float(np.mean(values)) for name, values in term_sums.items()}})
    for parameter in spatial.parameters():
        parameter.requires_grad_(False)
    return history

def train_predictor(model, label, store, dates, optimizer, device, epochs, patience, seed, max_train_days, max_eval_days):
    rng = np.random.RandomState(seed)
    best_score = -float('inf')
    best_state = None
    best_epoch = -1
    stale = 0
    history = []
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for epoch in range(epochs):
        model.train()
        order = dates.copy()
        rng.shuffle(order)
        if max_train_days is not None:
            order = order[:max_train_days]
        losses = []
        for number, day in enumerate(order, 1):
            x, y, _ = store.batch(int(day), training=True)
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            keep, target = drop_extreme_and_zscore(y)
            pred = model(x[keep])
            loss = F.mse_loss(pred, target) + 0.1 * pearson_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if number % 250 == 0 or number == len(order):
                log(f'{label} epoch={epoch} {number}/{len(order)} loss={np.mean(losses):.6f}')
        valid, score = evaluate_for_selection(model, store, device, max_eval_days)
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy({name: value for name, value in model.state_dict().items() if not name.startswith('base.')})
        else:
            stale += 1
        entry = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation': valid, 'selection_score': score, 'best': improved}
        if hasattr(model, 'residual_scale'):
            entry['residual_scale'] = float(torch.tanh(model.residual_scale).detach().cpu())
        history.append(entry)
        log(f"{label} epoch={epoch} valid_IC={valid['IC']:.6f} valid_RankIC={valid['RankIC']:.6f} best={best_epoch}")
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f'{label} did not produce a finite validation checkpoint')
    current = model.state_dict()
    current.update(best_state)
    model.load_state_dict(current)
    return (history, best_epoch, best_state)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.set_defaults(variant='no_vq')
    parser.add_argument('--universe', choices=('csi300', 'csi800'), default='csi300')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stage1-epochs', type=int, default=4)
    parser.add_argument('--base-epochs', type=int, default=12)
    parser.add_argument('--adapter-epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max-train-days', type=int)
    parser.add_argument('--max-eval-days', type=int)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'continuous_prism')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    seed_all(args.seed)
    device = torch.device('cuda:0')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = UniverseStore(DATASET_ROOT, args.universe)
    train_dates = store.dates_between(TRAIN_START, TRAIN_END)
    prefix = f'{args.variant}_{args.universe}_seed{args.seed}'
    started = time.time()
    spatial = ContinuousEncoder(args.variant).to(device)
    optimizer_spatial = torch.optim.AdamW(spatial.parameters(), lr=args.lr, weight_decay=0.0001)
    spatial_history = train_spatial(spatial, args.variant, store, train_dates.copy(), optimizer_spatial, device, args.stage1_epochs, args.max_train_days)
    spatial_source = 'retrained_from_scratch'
    base = MixtureRanker(spatial, args.variant).to(device)
    base_trainable = [p for p in base.parameters() if p.requires_grad]
    optimizer_base = torch.optim.AdamW(base_trainable, lr=args.lr, weight_decay=0.0001)
    base_history, base_best_epoch, base_state = train_predictor(base, f'{args.variant}/base', store, train_dates.copy(), optimizer_base, device, args.base_epochs, args.patience, args.seed, args.max_train_days, args.max_eval_days)
    base_checkpoint = args.output_dir / f'{prefix}_base_best.pt'
    torch.save(base.state_dict(), base_checkpoint)
    base_rows, base_predictions = evaluate(base, store, TEST_START, TEST_END, device, args.max_eval_days)
    write_daily_csv(args.output_dir / f'{prefix}_base_daily.csv', base_rows)
    np.savez_compressed(args.output_dir / f'{prefix}_base_predictions.npz', **base_predictions)
    adapter = TemporalAdapter(base).to(device)
    adapter_trainable = [p for p in adapter.parameters() if p.requires_grad]
    optimizer_adapter = torch.optim.AdamW(adapter_trainable, lr=args.lr, weight_decay=0.0001)
    adapter_history, adapter_best_epoch, adapter_state = train_predictor(adapter, f'{args.variant}/adapter', store, train_dates.copy(), optimizer_adapter, device, args.adapter_epochs, args.patience, args.seed, args.max_train_days, args.max_eval_days)
    torch.save(adapter_state, args.output_dir / f'{prefix}_adapter_best.pt')
    adapter_rows, adapter_predictions = evaluate(adapter, store, TEST_START, TEST_END, device, args.max_eval_days)
    write_daily_csv(args.output_dir / f'{prefix}_adapter_daily.csv', adapter_rows)
    np.savez_compressed(args.output_dir / f'{prefix}_adapter_predictions.npz', **adapter_predictions)
    result = {'experiment': 'Continuous expert: encoder, mixture ranker, temporal residual', 'variant': args.variant, 'what_changed': 'continuous encoder and mixture ranker with temporal residual', 'universe': args.universe, 'seed': args.seed, 'spatial_source': spatial_source, 'spatial_history': spatial_history, 'base': {'best_epoch': base_best_epoch, 'history': base_history, 'metrics': metrics_by_period(base_rows), 'parameters': sum((p.numel() for p in base.parameters())), 'trainable_parameters_stage2': len(base_state) and sum((p.numel() for p in base_trainable)), 'checkpoint': str(base_checkpoint.resolve())}, 'with_temporal_adapter': {'best_epoch': adapter_best_epoch, 'history': adapter_history, 'metrics': metrics_by_period(adapter_rows), 'total_parameters': sum((p.numel() for p in adapter.parameters())), 'adapter_trainable_parameters': sum((p.numel() for p in adapter_trainable)), 'final_residual_scale': float(torch.tanh(adapter.residual_scale).detach().cpu())}, 'elapsed_seconds': time.time() - started, 'mode': 'single-process single-threaded', 'config': vars(args) | {'output_dir': str(args.output_dir.resolve())}}
    (args.output_dir / f'{prefix}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    log(json.dumps({'variant': args.variant, 'base': result['base']['metrics'], 'with_temporal_adapter': result['with_temporal_adapter']['metrics'], 'elapsed_seconds': result['elapsed_seconds']}, ensure_ascii=False))
if __name__ == '__main__':
    main()
