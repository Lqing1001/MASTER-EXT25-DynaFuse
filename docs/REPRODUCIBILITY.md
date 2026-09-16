# Reproducing the manuscript

Run all commands from the repository root. Full training uses `python -m dynafuse.train` as shown in README. See `configs/protocol.json`; the executable defaults and orchestrator use the same fixed protocol.

## Table-to-code mapping

| Manuscript item | Entry point / artifacts |
|---|---|
| Tables 1–3: data and protocol | `data_tools` reconstruction, strict packaging and semantic checks; generated manifests |
| Tables 4–5: seed-0 broad comparisons | `baselines.*`, `dynafuse.train_master`, then `dynafuse.comparison_tables` |
| Table 6: component comparison | `dynafuse.infer` validation predictions and `dynafuse.comparison_tables` |
| Tables 7–8: paired seeds and tests | `dynafuse.evaluate` |
| Table 9: k and alpha sweeps | `dynafuse.train_sparse`, `dynafuse.sensitivity` |
| Section 5.5.1: model cost | `dynafuse.model_cost` |
| Section 5.5.2: position frequencies | `dynafuse.selection_frequency` |
| Section 5.5.3: membership exclusion | `dynafuse.membership_sensitivity` |

## Validation inference and comparison tables

The MASTER training script preserves the experiment's test-output timing. Generate validation predictions after training:

```bash
python -m dynafuse.infer --dataset-root datasets/master_ext_strict_20191224_v1 --run-dir runs/csi300 --universe csi300 --model master --split validation --output-dir runs/csi300/validation
python -m dynafuse.comparison_tables --results-root runs --paired-summary reports/results.json --master-validation runs/csi300/validation/master_csi300_seed0_validation_predictions.npz --expert-validation runs/csi300/sparse_top1/ta_deformable_topk1_csi300_seed0_validation_predictions.npz --output-dir reports
```

The continuous/sparse training stages already save validation predictions. Broad baselines use seed 0. PRISM-VQ, ACT and StockMamba are local compatible implementations with the qualifications described in their module, not official-code reproductions. Ridge searches alphas 1, 10 and 100 on validation data. Random Forest uses 200 trees, max depth 12 and leaf size 50. XGBoost uses a maximum of 1,200 rounds and validation RMSE early stopping with patience 100. Their feature summary contains 379 values: latest Alpha158, eight-day mean Alpha158, latest Market63.

## Top-k sweep

Only CSI 300 seed 0 is used. Each sweep stage starts from the same trained continuous/TA anchor and runs 10 epochs with patience 10. Set the dataset environment variable before individual stage commands:

PowerShell:

```powershell
$env:MASTER_EXT_DATASET_ROOT = (Resolve-Path datasets/master_ext_strict_20191224_v1).Path
```

Linux/macOS shell:

```bash
export MASTER_EXT_DATASET_ROOT="$PWD/datasets/master_ext_strict_20191224_v1"
```

Run these commands sequentially:

```bash
python -m dynafuse.train_sparse --universe csi300 --seed 0 --selected 2 --continuous-dir runs/csi300/continuous_prism --output-dir runs/csi300/topk_validation_sweep
python -m dynafuse.train_sparse --universe csi300 --seed 0 --selected 4 --continuous-dir runs/csi300/continuous_prism --output-dir runs/csi300/topk_validation_sweep
python -m dynafuse.train_sparse --universe csi300 --seed 0 --selected 8 --continuous-dir runs/csi300/continuous_prism --output-dir runs/csi300/topk_validation_sweep
python -m dynafuse.sensitivity --master-validation runs/csi300/validation/master_csi300_seed0_validation_predictions.npz --top1-validation runs/csi300/sparse_top1/ta_deformable_topk1_csi300_seed0_validation_predictions.npz --sweep-dir runs/csi300/topk_validation_sweep --output-dir reports/sensitivity
```

The alpha grid is 0, 0.25, 0.5, 0.75, 1, with k=1. The k grid is 1, 2, 4, 8, with alpha=0.5. The final method remains k=1 and alpha=0.5; k=2 has the highest validation criterion in the k sweep. No test-set selection is performed.

## Other final analyses

```bash
python -m dynafuse.model_cost --output reports/model_cost.json
python -m dynafuse.selection_frequency --dataset-root datasets/master_ext_strict_20191224_v1 --run-dir runs/csi300 --output reports/selection_frequency.json
python -m dynafuse.membership_sensitivity --raw-features datasets/master_ext_clean_v1/alpha158_raw/by_instrument --master-predictions runs/csi300/master/master_full_csi300_seed0_predictions.npz --expert-predictions runs/csi300/sparse_top1/ta_deformable_topk1_csi300_seed0_predictions.npz --output reports/membership_sensitivity.json
```

Membership sensitivity excludes 890 test date/security pairs, including 866 finite-target observations. Scores are standardized on the complete daily cross section before exclusions. This is a fixed-model test sensitivity check; it does not refit training or validation membership histories.

The model-cost entry counts matrix operations along the active inference path, with two FLOPs per multiply–accumulate. It does not measure runtime or all elementwise operations. DynaFuse contains 1,177,321 registered parameters, versus 1,550,082 for HOM; matrix-FLOP ratios are 61.0% (N=300) and 59.3% (N=800).

## Statistical conventions

- Every seed has the same chronological split and full epoch budget. Patience is at least the stage budget; the best validation checkpoint is selected from all completed epochs.
- Day-level score normalization precedes finite-target filtering. Test metrics use the 964 valid dates.
- ICIR and RankICIR divide the mean by the population standard deviation of daily correlations (ddof=0). Across-seed table SD uses ddof=1.
- Paired inference first averages the three daily seed differences, then applies a lag-10 Newey–West test and a noncircular 20-day moving-block bootstrap with 5,000 draws.
- Bootstrap seeds are in `configs/bootstrap_seeds.json`. MASTER/HOM contrasts use `numpy.default_rng`; MX uses `numpy.RandomState(20260831)` independently for each metric and universe, preserving the reported intervals.
- HOM uses overlapping cyclic pairs; its SD describes dispersion across those pairs.

## Checkpoints and model details

The encoder, mixture ranker, TA residual and sparse residual are fitted in stages. Residual checkpoint files contain only the new stage parameters; use `dynafuse.infer.load_model` to load the full chain. `no_vq_*` and `ta_deformable*` filenames are preserved to load the published experiment artifacts. They are not additional alternative methods.

The continuous encoder retains an unused registered embedding and pretraining heads for checkpoint and parameter-count compatibility. No vector quantization executes in the final continuous model. At k=1, masked softmax gives the selected position weight one, so the hard selection has no score gradient; the value projection and residual head are trained. This behavior is explicitly documented in Section 4.6 and tested here.

Independent stage flags such as `--max-train-days` are for diagnostics. Results from reduced budgets must not be reported as the paper protocol. The default top-level training command checks the strict dataset manifest and uses the full budgets.
