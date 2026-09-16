# MASTER-EXT25 and DynaFuse

Code for **MASTER-EXT25 and DynaFuse: A Point-in-Time Benchmark and Heterogeneous Expert Fusion for Cross-Sectional Stock Ranking**.

[Data preparation](docs/DATA.md) · [Reproduction commands](docs/REPRODUCIBILITY.md) · [Verification](docs/VERIFICATION.md)

DynaFuse combines a market-guided MASTER expert with an independently trained continuous expert: a continuous cross-sectional encoder, mixture ranker, temporal-attention residual and Top-1 sparse temporal residual. Each expert's scores are standardized over the complete daily universe and combined with fixed weights of 0.5/0.5.

## Contents

- `dynafuse/`: final models, sequential training, inference, fixed fusion, paired tests and sensitivity analyses.
- `baselines/`: Ridge, Random Forest, XGBoost, and compatible StockMamba/ACT/PRISM-VQ implementations.
- `data_tools/`: constituent reconstruction, raw features, strictly train-only universe-specific normalization, and data checks.
- `configs/`: final protocol and deterministic bootstrap seeds.
- `reference/`: aggregate experimental results and the tested environment; no security-level inputs or predictions.
- `tests/`: synthetic protocol and staged-training checks.

## Environment

Use Python 3.12. The recorded experiment environment used PyTorch 2.8.0 with CUDA 12.9 on an RTX 5090 (32 GB). Install a PyTorch build appropriate for your GPU and then the pinned dependencies:

```bash
python -m pip install -r requirements.txt
python -m tests.test_protocol
```

`requirements.txt` pins the PyTorch release; `reference/environment.json` records the CUDA wheel build actually tested. GPU kernels and library versions can affect retraining numerics. Training requires CUDA; inference and synthetic tests can run on CPU. All stages run serially, with one CPU numerical thread and one daily cross section per batch.

## Train and evaluate

Obtain the licensed inputs and build the strict dataset following [DATA.md](docs/DATA.md). From this repository root:

```bash
python -m dynafuse.train --dataset-root datasets/master_ext_strict_20191224_v1 --output-root runs --include-baselines
python -m dynafuse.evaluate --results-root runs --output-dir reports
```

The training command runs CSI 300 and CSI 800, seeds 0/1/2, the final DynaFuse components, three XGBoost seeds per universe for MX, and the broad seed-0 baselines. It requires a new empty output directory and a PASS dataset manifest. Add `--dry-run` to inspect the exact commands without training. Individual stage CLIs support deliberate reruns.

The evaluator produces Table 7, Table 8, per-seed metrics and daily metric archives. HOM uses cyclic MASTER pairs `(0,1), (1,2), (2,0)`. MX uses same-seed MASTER/XGBoost pairs with fixed equal weights. It does not tune fusion weights on test data.

| Protocol | Value |
|---|---|
| Training and normalization fit | 2010-01-04 to 2019-12-24 |
| Validation | 2020-01-02 to 2021-12-24 |
| Test | 2022-01-04 to 2025-12-31 |
| Input | 8 days × (158 stock + 63 market features) |
| MASTER budget / patience | 40 / 40 |
| Continuous stages | encoder 4, ranker 12, TA 10, sparse 10 epochs |
| Ranker/TA patience; sparse patience | 12; 10 |
| Neural checkpoint selection | mean validation IC and RankIC |

## Main aggregate results

| Universe | MASTER RankIC | DynaFuse RankIC | Relative gain |
|---|---:|---:|---:|
| CSI 300 | 0.06547 | 0.07046 | 7.62% |
| CSI 800 | 0.05911 | 0.06394 | 8.17% |

Values are means across three seeds. `reference/results.json` preserves full precision. DynaFuse's RankIC gains over MASTER pass both the paired Newey–West and moving-block bootstrap checks. IC and comparisons with HOM/MX have different significance outcomes; see the complete reference tables.

The CSI 800 DynaFuse RankIC sample standard deviation is **0.0009806719853848503**, displayed as **0.00098**. The supplied manuscript's Table 7 has a transcription error (`0.00096`) in that cell. The release uses the recomputed value.

## Data and checkpoint access

CSMAR workbooks, licensed membership inputs, feature arrays, per-security predictions and trained checkpoints are not distributed in this package. Users with authorized source access can construct the data and train the models. The scripts, schemas, protocol and aggregate reference results are provided here; this is not a bundled-data reproduction.

## Citation and license

Use `CITATION.cff` for the manuscript citation; no publication DOI is assigned in this release. Code is MIT licensed. The upstream MASTER MIT notice is retained; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
