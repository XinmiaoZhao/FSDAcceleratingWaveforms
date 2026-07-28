# Data dictionary

All acceleration quantities use `s^-1`, masses use solar masses, frequencies
use Hz, and phase shifts use radians.

## Phase table

| Field | Meaning |
|---|---|
| `system_index` | Stable order of the six systems in `paper.yaml` |
| `mass1_msun`, `mass2_msun` | Source-frame component masses |
| `chi1`, `chi2` | Aligned dimensionless spins |
| `acceleration_s_inv` | Effective line-of-sight acceleration |
| `sample_rate_hz` | TDS sampling rate used for the comparison |
| `frequency_hz` | Frequency at which the signed phase shift is recorded |
| `method` | `TDS`, `FSD`, or `SPA+PN` |
| `phase_shift_rad` | Signed phase shift after anchoring at the ISCO frequency |

The dense calculation grid is deterministically reduced to logarithmically
spaced plotting points. The archived points are the sole input to the figure.

## Mismatch tables

| Field | Meaning |
|---|---|
| `primary_mass_msun`, `secondary_mass_msun`, `total_mass_msun` | Binary masses |
| `chi1`, `chi2` | Aligned spins |
| `psd` | PyCBC PSD factory name |
| `f_low_hz`, `f_isco_hz` | Lower cutoff and inspiral boundary |
| `stage` | `inspiral`, `merger-ringdown`, or `full` |
| `method` | Model compared against TDS |
| `fsd_order` | FSD truncation order; blank for SPA+PN |
| `mismatch` | PSD-weighted, time/phase-maximized mismatch |

## Fisher table

`acceleration_uncertainty_s_inv` is the one-standard-deviation uncertainty from
a single-parameter Fisher matrix after normalizing each waveform family to the
configured network SNR.
