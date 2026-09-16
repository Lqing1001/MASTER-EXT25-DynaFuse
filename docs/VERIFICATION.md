# Release verification

The independent release was checked against the experiment artifacts supporting the September 16, 2026 manuscript.

- All six model/component constructions have exactly the same initialized tensors, train/eval forward outputs and sampled gradients as the original final implementations.
- Both universes and all three seeds load their actual MASTER and continuous-expert checkpoints. On the inspected complete cross section (2022-01-04), all twelve model/seed/universe comparisons reproduce stored predictions with maximum absolute error **0**.
- Recomputing the full saved test predictions reproduces all paired means, sample SDs, Newey–West statistics, moving-block bootstrap intervals and score-correlation summaries exactly.
- The Table 9 validation sweep, eight position frequencies, matrix-FLOP counts and membership exclusion are recomputed by the release entry points. Membership exclusion identifies 4,814 overall differing pairs, 890 test pairs and 866 finite-target test pairs.
- Four synthetic tests pass, covering full-universe score standardization before target filtering, universe-specific training-only normalization, Top-1 selector gradients and staged optimizer/checkpoint wiring.
- Ten critical raw-feature, adjustment, membership and strict-packaging function bodies match the original functions' Python abstract syntax trees.

See `reference/verification.json` and the aggregate reference files. Full raw workbook extraction, full dataset rebuilding, and complete model retraining were not repeated during packaging. Checkpoint comparisons sample one date; full-date statistical comparisons use the existing saved prediction archives. Hardware/library differences or changes in source data can affect retraining results.

The only detected numeric transcription error in the complete Table 7 comparison is the CSI 800 DynaFuse RankIC SD: the correct five-decimal value is **0.00098**, not 0.00096. Reference files contain the corrected value.
