"""Single-process compatible neural baselines on MASTER-EXT.

This is a transparent compatibility benchmark, not an official-code reproduction:
* PRISM-VQ: two-stage VQ + code-conditioned MoE; market63 proxies unavailable JKP priors.
* ACT: temporal decomposition + dynamic purification + isolated fluctuation/shock
  branches; static industry/region graphs are unavailable and therefore omitted.
* StockMamba: paper-specified MSS/FGM/TFA/CSA/STD/RPL with a compact pure-PyTorch
  selective state recurrence standing in for the unreleased Mamba-2 source.

All data access, training and evaluation are single-process and single-threaded.
"""

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

for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dynafuse.runtime import (
    DATASET_ROOT,
    ROOT, TEST_END, TEST_START, TRAIN_END, TRAIN_START, VALID_END, VALID_START,
    UniverseStore, drop_extreme_and_zscore, evaluate, metrics_by_period,
    summarize_daily, write_daily_csv,
)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.square().sum().sqrt() * target.square().sum().sqrt()
    return 1.0 - (pred * target).sum() / denom.clamp_min(1e-8)


def rank_position_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(target)
    ranks = torch.empty_like(target)
    ranks[order] = torch.linspace(0.0, 1.0, len(target), device=target.device)
    weights = 1.0 + (2.0 * ranks - 1.0).square()
    wmse = ((weights / weights.mean()) * (pred - target).square()).mean()
    return 0.5 * wmse + 0.5 * pearson_loss(pred, target)


def causal_average(x: torch.Tensor, window: int) -> torch.Tensor:
    outputs = []
    for t in range(x.shape[1]):
        outputs.append(x[:, max(0, t - window + 1):t + 1].mean(dim=1))
    return torch.stack(outputs, dim=1)


class SelectiveMarketScanner(nn.Module):
    """Compact input-dependent state recurrence following the public MSS equations."""

    def __init__(self, input_dim: int = 63, hidden: int = 64):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden, bias=False)
        self.params = nn.Linear(hidden, hidden * 4)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, market: torch.Tensor) -> torch.Tensor:
        u = self.in_proj(market)
        state = torch.zeros_like(u[:, 0])
        outputs = []
        for t in range(u.shape[1]):
            decay, drive, read, gate = self.params(u[:, t]).chunk(4, dim=-1)
            state = torch.sigmoid(decay) * state + torch.sigmoid(drive) * u[:, t]
            outputs.append(self.out(torch.tanh(read) * state * F.silu(gate)))
        return torch.stack(outputs, dim=1)


class StockMambaCompat(nn.Module):
    def __init__(self, d_model: int = 256, dropout: float = 0.5):
        super().__init__()
        self.scanner = SelectiveMarketScanner()
        self.factor_gate = nn.Sequential(nn.Linear(64, 158), nn.Softmax(dim=-1))
        self.factor_proj = nn.Linear(158, d_model)
        self.tfa = nn.MultiheadAttention(d_model, 4, dropout=dropout, batch_first=True)
        self.t_norm = nn.LayerNorm(d_model)
        self.csa = nn.MultiheadAttention(d_model, 2, dropout=dropout, batch_first=True)
        self.s_norm = nn.LayerNorm(d_model)
        self.distill = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stock, market = x[:, :, :158], x[:, :, 158:221]
        regime = self.scanner(market)
        gates = self.factor_gate(regime) * 158.0
        z = self.factor_proj(stock * gates)
        temporal, _ = self.tfa(z, z, z, need_weights=False)
        z = self.t_norm(z + temporal)
        cross, _ = self.csa(z.transpose(0, 1), z.transpose(0, 1), z.transpose(0, 1), need_weights=False)
        z = self.s_norm(z + cross.transpose(0, 1))
        query = z[:, -1:]
        weights = torch.softmax((self.distill(z) * query).sum(-1) / math.sqrt(z.shape[-1]), dim=1)
        pooled = (weights.unsqueeze(-1) * z).sum(1)
        return self.head(pooled).squeeze(-1)


class ACTCompat(nn.Module):
    """Dynamic-relation ACT screen; static industry/region PSPE is unavailable."""

    def __init__(self, d_model: int = 128, dropout: float = 0.2):
        super().__init__()
        self.trend_proj = nn.Sequential(nn.Linear(158, d_model), nn.LayerNorm(d_model))
        self.dynamic_graph = nn.MultiheadAttention(d_model, 4, dropout=dropout, batch_first=True)
        self.backward = nn.Linear(d_model, d_model)
        self.trend_fuse = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LeakyReLU())
        self.fluct_tcn = nn.Sequential(
            nn.Conv1d(158, d_model, 3, padding=1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU(),
        )
        self.shock_proj = nn.Sequential(
            nn.Linear(158 * 2, d_model), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.component_score = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stock = x[:, :, :158]
        trend = causal_average(stock, 4)
        detrended = stock - trend
        fluct = causal_average(detrended, 2)
        shock = stock - trend - fluct

        base = self.trend_proj(trend[:, -1])
        dynamic, _ = self.dynamic_graph(base.unsqueeze(0), base.unsqueeze(0), base.unsqueeze(0), need_weights=False)
        dynamic = dynamic.squeeze(0)
        purified = base - self.backward(dynamic)
        trend_z = self.trend_fuse(torch.cat([dynamic, purified], dim=-1))

        fluct_z = self.fluct_tcn(fluct.transpose(1, 2))[:, :, -1]
        shock_cf = causal_average(shock, 3)[:, -1]
        shock_z = self.shock_proj(torch.cat([shock[:, -1], shock_cf], dim=-1))
        components = torch.stack([trend_z, fluct_z, shock_z], dim=1)
        weights = torch.softmax(self.component_score(components).squeeze(-1), dim=1)
        return self.head((weights.unsqueeze(-1) * components).sum(1)).squeeze(-1)


class PrismSpatial(nn.Module):
    def __init__(self, d_model: int = 64, codebook_size: int = 512):
        super().__init__()
        self.gru = nn.GRU(158, d_model, batch_first=True)
        layer = nn.TransformerEncoderLayer(d_model, 2, d_model * 2, 0.1, batch_first=True, norm_first=True)
        self.cross = nn.TransformerEncoder(layer, 2)
        self.codebook = nn.Embedding(codebook_size, d_model)
        nn.init.normal_(self.codebook.weight, std=0.1)
        self.decoder = nn.Sequential(nn.Linear(d_model + 64, 256), nn.GELU(), nn.Linear(256, 158))
        self.aux = nn.Sequential(nn.Linear(d_model + 64, 128), nn.GELU(), nn.Linear(128, 1))
        self.prior = nn.Linear(63, 64)

    def encode(self, x: torch.Tensor):
        stock, market = x[:, :, :158], x[:, :, 158:221]
        mean = stock.mean(1, keepdim=True)
        std = stock.std(1, keepdim=True).clamp_min(1e-5)
        norm = (stock - mean) / std
        h = self.gru(norm)[1][-1]
        z = self.cross(h.unsqueeze(0)).squeeze(0)
        distances = z.square().sum(1, keepdim=True) + self.codebook.weight.square().sum(1) - 2 * z @ self.codebook.weight.t()
        indices = distances.argmin(1)
        quant = self.codebook(indices)
        straight = z + (quant - z).detach()
        prior = self.prior(market[:, -1])
        return z, quant, straight, indices, prior

    def forward(self, x: torch.Tensor):
        z, quant, straight, indices, prior = self.encode(x)
        joined = torch.cat([straight, prior], dim=-1)
        reconstruction = self.decoder(joined)
        aux = self.aux(joined).squeeze(-1)
        return z, quant, indices, reconstruction, aux


class PrismVQCompat(nn.Module):
    def __init__(self, spatial: PrismSpatial, d_model: int = 64, experts: int = 2):
        super().__init__()
        self.spatial = spatial
        self.input_proj = nn.Linear(158, d_model)
        layer = nn.TransformerEncoderLayer(d_model, 2, d_model * 2, 0.1, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2)
        self.gate = nn.Linear(d_model, experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model + 64, 128), nn.GELU(), nn.Linear(128, 1))
            for _ in range(experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, quant, _, _, prior = self.spatial.encode(x)
        seq = self.input_proj(x[:, :, :158])
        context = self.temporal(torch.cat([quant.unsqueeze(1), seq], dim=1))[:, 0]
        weights = torch.softmax(self.gate(quant), dim=-1)
        joined = torch.cat([context, prior], dim=-1)
        outputs = torch.cat([expert(joined) for expert in self.experts], dim=1)
        return (weights * outputs).sum(1)


def stage1_loss(spatial: PrismSpatial, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    z, quant, _, reconstruction, aux = spatial(x)
    recon = F.mse_loss(reconstruction, x[:, -1, :158])
    vq = F.mse_loss(z.detach(), quant) + 0.25 * F.mse_loss(z, quant.detach())
    distances = torch.cdist(F.normalize(z, dim=-1), F.normalize(spatial.codebook.weight, dim=-1))
    assignments = distances.argmin(1)
    contrast = F.cross_entropy(-distances / 0.07, assignments)
    return recon + vq + contrast + 1e-4 * F.mse_loss(aux, target)


def model_loss(model_name: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if model_name == "stockmamba":
        return rank_position_loss(pred, target)
    if model_name == "act":
        return pearson_loss(pred, target) + 0.1 * F.mse_loss(pred, target)
    return F.mse_loss(pred, target) + 0.1 * pearson_loss(pred, target)


def validation_score(model: nn.Module, store: UniverseStore, device: torch.device, max_eval_days: int | None):
    rows, _ = evaluate(model, store, VALID_START, VALID_END, device, max_eval_days)
    metrics = summarize_daily(rows)
    return metrics, 0.5 * (metrics["IC"] + metrics["RankIC"])


def train_epoch(model_name, model, store, dates, optimizer, device, max_train_days):
    model.train()
    losses = []
    if max_train_days is not None:
        dates = dates[:max_train_days]
    for number, day in enumerate(dates, 1):
        x, y, _ = store.batch(int(day), training=True)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        keep, target = drop_extreme_and_zscore(y)
        pred = model(x[keep])
        loss = model_loss(model_name, pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if number % 250 == 0 or number == len(dates):
            log(f"{model_name} train {number}/{len(dates)} loss={np.mean(losses):.6f}")
    return float(np.mean(losses))


def train_prism_stage1(spatial, store, dates, optimizer, device, epochs, max_train_days):
    if max_train_days is not None:
        dates = dates[:max_train_days]
    history = []
    for epoch in range(epochs):
        spatial.train(); losses = []
        for number, day in enumerate(dates, 1):
            x, y, _ = store.batch(int(day), training=True)
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            keep, target = drop_extreme_and_zscore(y)
            loss = stage1_loss(spatial, x[keep], target)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(spatial.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if number % 250 == 0 or number == len(dates):
                log(f"prism_vq stage1 epoch={epoch} {number}/{len(dates)} loss={np.mean(losses):.6f}")
        history.append(float(np.mean(losses)))
    for parameter in spatial.parameters():
        parameter.requires_grad_(False)
    return history


def build_model(name: str, device: torch.device):
    if name == "stockmamba":
        return StockMambaCompat().to(device), None
    if name == "act":
        return ACTCompat().to(device), None
    spatial = PrismSpatial().to(device)
    return PrismVQCompat(spatial).to(device), spatial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("prism_vq", "act", "stockmamba"), required=True)
    parser.add_argument("--universe", choices=("csi300", "csi800"), default="csi300")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--stage1-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--max-train-days", type=int)
    parser.add_argument("--max-eval-days", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "neural_baselines")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    seed_all(args.seed)
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = UniverseStore(DATASET_ROOT, args.universe)
    model, spatial = build_model(args.model, device)
    train_dates = store.dates_between(TRAIN_START, TRAIN_END)
    rng = np.random.RandomState(args.seed)

    stage1_history = []
    if spatial is not None:
        optimizer1 = torch.optim.AdamW(spatial.parameters(), lr=1e-4, weight_decay=1e-4)
        stage1_history = train_prism_stage1(
            spatial, store, train_dates.copy(), optimizer1, device,
            args.stage1_epochs, args.max_train_days,
        )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    lr = 1e-5 if args.model == "stockmamba" else 1e-4
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    best_score = -float("inf"); best_state = None; best_epoch = -1; stale = 0; history = []
    started = time.time()
    for epoch in range(args.epochs):
        order = train_dates.copy(); rng.shuffle(order)
        train_loss = train_epoch(args.model, model, store, order, optimizer, device, args.max_train_days)
        valid, score = validation_score(model, store, device, args.max_eval_days)
        improved = score > best_score
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": train_loss, "validation": valid, "selection_score": score, "best": improved})
        log(f"{args.model} epoch={epoch} valid_IC={valid['IC']:.6f} valid_RankIC={valid['RankIC']:.6f} best={best_epoch}")
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    prefix = f"{args.model}_{args.universe}_seed{args.seed}"
    torch.save(model.state_dict(), args.output_dir / f"{prefix}_best.pt")
    rows, predictions = evaluate(model, store, TEST_START, TEST_END, device, args.max_eval_days)
    result = {
        "source": "local compatible implementation; not official-code reproduction",
        "model": args.model, "universe": args.universe, "seed": args.seed,
        "best_epoch": best_epoch, "stage1_history": stage1_history, "history": history,
        "metrics": metrics_by_period(rows),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters_stage2": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "elapsed_seconds": time.time() - started,
        "mode": "single-process single-threaded",
        "limitations": {
            "prism_vq": "market63 prior proxy; reduced reconstruction; official repository download unavailable",
            "act": "dynamic relation only; industry and region graphs unavailable",
            "stockmamba": "compact selective recurrence; unreleased exact Mamba-2 source unavailable",
        }[args.model],
        "config": vars(args) | {"output_dir": str(args.output_dir.resolve())},
    }
    (args.output_dir / f"{prefix}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_daily_csv(args.output_dir / f"{prefix}_daily.csv", rows)
    np.savez_compressed(args.output_dir / f"{prefix}_predictions.npz", **predictions)
    log(json.dumps({"model": args.model, "best_epoch": best_epoch, "metrics": result["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
