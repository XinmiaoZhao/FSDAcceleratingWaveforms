# Numerical data

These machine-readable tables support the six article figures. They are
generated from `configs/paper.yaml` by
`scripts/reproduce_all.py --mode full` and licensed under CC BY 4.0.

- `fig01_phase_shifts.csv.gz`: signed phase shifts for six binary systems and
  the three public waveform methods.
- `fig02_mismatch_aligo_a1e-4.csv`: aLIGO mismatch comparison at
  `a = 1e-4 s^-1`.
- `fig03_mismatch_et_a1e-4.csv`: ET mismatch comparison at `a = 1e-4 s^-1`.
- `fig04_mismatch_et_a1e-5.csv`: ET mismatch comparison at `a = 1e-5 s^-1`.
- `fig05_fsd_order.csv`: non-spinning FSD truncation-order comparison.
- `fig06_fisher_uncertainty.csv`: one-parameter acceleration uncertainty at
  SNR 1000.
- `validation_phase_sample_rate.csv`: 4096 Hz versus 2048 Hz TDS phase check.

See `docs/data_dictionary.md` for fields and units. `SHA256SUMS.json` records
the exact release file hashes.
