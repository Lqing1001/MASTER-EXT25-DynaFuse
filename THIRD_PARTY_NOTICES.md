# Third-party notices

`dynafuse/vendor/master_components.py` contains the positional encoding, temporal attention, cross-stock attention, temporal pooling and feature gate from SJTU-DMTai/MASTER. The source was reduced to these components, removing the Qlib training wrapper. Their calculations and parameter names are preserved.

Upstream: https://github.com/SJTU-DMTai/MASTER

Copyright (c) 2025 Data Management Technology and AI. MIT license retained in `dynafuse/vendor/LICENSE-MASTER.txt`.

The StockMamba, ACT and PRISM-VQ baselines are local compatible implementations of the architectures described in the cited papers. They are not their official code releases. Their approximations are documented in `baselines/neural.py` and in the manuscript.

NumPy, PyTorch, pandas, SciPy, scikit-learn, XGBoost, openpyxl, joblib and threadpoolctl are external dependencies and retain their respective licenses. They are installed separately, not vendored here.

The repository MIT license covers the released code and documentation. It does not grant access to, or redistribution rights for, CSMAR data or externally acquired constituent snapshots. Obtain source materials under their applicable licenses.
