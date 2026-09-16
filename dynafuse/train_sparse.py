from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import torch
from torch import nn
from dynafuse.runtime import DATASET_ROOT, ROOT, TEST_END, TEST_START, TRAIN_END, TRAIN_START, VALID_END, VALID_START, UniverseStore, evaluate, metrics_by_period, write_daily_csv
from dynafuse.train_continuous import MixtureRanker, ContinuousEncoder, TemporalAdapter, train_predictor
from dynafuse.train_continuous import seed_all

class DeformableTemporalResidual(nn.Module):

    def __init__(self, history=8, d_model=64, selected=1):
        super().__init__()
        self.selected = selected
        self.input_proj = nn.Linear(158, d_model)
        self.position = nn.Parameter(torch.randn(1, history, d_model) * 0.02)
        self.selector = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.value = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, stock):
        sequence = self.input_proj(stock) + self.position[:, :stock.shape[1]]
        scores = self.selector(sequence).squeeze(-1)
        top = torch.topk(scores, k=min(self.selected, scores.shape[1]), dim=1).indices
        mask = torch.full_like(scores, -torch.inf)
        mask.scatter_(1, top, scores.gather(1, top))
        weights = torch.softmax(mask, dim=1)
        pooled = torch.einsum('nt,ntd->nd', weights, self.value(sequence))
        return self.head(pooled).squeeze(-1)

class SparseExpert(nn.Module):

    def __init__(self, anchor, variant, selected=1):
        super().__init__()
        self.base = anchor
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.residual = DeformableTemporalResidual(selected=selected)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        self.base.eval()
        with torch.no_grad():
            anchor_pred = self.base(x)
        return anchor_pred + torch.tanh(self.residual_scale) * self.residual(x[:, :, :158])

def main():
    parser = argparse.ArgumentParser()
    parser.set_defaults(variant='deformable')
    parser.add_argument('--universe', choices=('csi300', 'csi800'), default='csi300')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--selected', type=int, default=1, help='number of temporal positions retained by the deformable residual')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max-train-days', type=int)
    parser.add_argument('--max-eval-days', type=int)
    parser.add_argument('--continuous-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
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
    prefix_base = f'no_vq_{args.universe}_seed{args.seed}'
    ranker = MixtureRanker(ContinuousEncoder('no_vq'), 'no_vq').to(device)
    ranker.load_state_dict(torch.load(args.continuous_dir / f'{prefix_base}_base_best.pt', map_location=device, weights_only=True))
    anchor = TemporalAdapter(ranker).to(device)
    anchor.load_state_dict(torch.load(args.continuous_dir / f'{prefix_base}_adapter_best.pt', map_location=device, weights_only=True), strict=False)
    if not 1 <= args.selected <= 8:
        raise ValueError('--selected must lie in [1, 8]')
    model = SparseExpert(anchor, args.variant, selected=args.selected).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0001)
    started = time.time()
    history, best_epoch, best_state = train_predictor(model, f'ta_{args.variant}', store, train_dates.copy(), optimizer, device, args.epochs, args.patience, args.seed, args.max_train_days, args.max_eval_days)
    suffix = f'_topk{args.selected}' if args.variant == 'deformable' and args.selected != 4 else ''
    prefix = f'ta_{args.variant}{suffix}_{args.universe}_seed{args.seed}'
    torch.save(best_state, args.output_dir / f'{prefix}_best.pt')
    valid_rows, valid_predictions = evaluate(model, store, VALID_START, VALID_END, device, args.max_eval_days)
    test_rows, test_predictions = evaluate(model, store, TEST_START, TEST_END, device, args.max_eval_days)
    write_daily_csv(args.output_dir / f'{prefix}_validation_daily.csv', valid_rows)
    write_daily_csv(args.output_dir / f'{prefix}_daily.csv', test_rows)
    np.savez_compressed(args.output_dir / f'{prefix}_validation_predictions.npz', **valid_predictions)
    np.savez_compressed(args.output_dir / f'{prefix}_predictions.npz', **test_predictions)
    result = {'experiment': 'Sparse temporal residual on the frozen temporal expert', 'variant': args.variant, 'universe': args.universe, 'seed': args.seed, 'best_epoch': best_epoch, 'history': history, 'validation_metrics': history[best_epoch]['validation'], 'metrics': metrics_by_period(test_rows), 'trainable_parameters': sum((p.numel() for p in trainable)), 'total_parameters': sum((p.numel() for p in model.parameters())), 'final_residual_scale': float(torch.tanh(model.residual_scale).detach().cpu()), 'elapsed_seconds': time.time() - started, 'mode': 'single-process single-threaded', 'config': vars(args) | {'continuous_dir': str(args.continuous_dir.resolve()), 'output_dir': str(args.output_dir.resolve())}}
    (args.output_dir / f'{prefix}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'variant': args.variant, 'best_epoch': best_epoch, 'validation': result['validation_metrics'], 'metrics': result['metrics'], 'trainable_parameters': result['trainable_parameters']}, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
