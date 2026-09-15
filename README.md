# MASTER-EXT25-DynaFuse

## Layout

- scripts/: data construction and validation toolkit.
- experiments/master_ext_reproduction/: final DynaFuse training and analysis.
- external/MASTER-official/: minimum upstream MASTER dependency and license.
- CODE_MANIFEST.json: exact file list and SHA-256 digests.

## Environment

Python 3.10 or newer is recommended. Install packages with:

    pip install -r requirements.txt

Install CUDA-enabled PyTorch separately for the training host.

## Data

CSMAR exports are license restricted and are not included. Expected inputs are placed under datasets/master_extension_raw/. Build the clean and strict packages with:

    python scripts/build_master_ext_clean_v1.py --stage all
    python scripts/rebuild_strict_train_only_packages.py

The strict package must have a passing manifest.json and a normalization cutoff of 2019-12-24.

## Locked seed-0 reproduction

CSI 300:

    python experiments/master_ext_reproduction/run_strict_normalization_queue.py

CSI 800:

    python experiments/master_ext_reproduction/run_strict_csi800_core_queue.py

The queues train MASTER, the continuous expert and the Top-1 sparse residual, then evaluate the fixed fusion:

    0.5 * Z_t(MASTER) + 0.5 * Z_t(continuous_sparse_expert)
