# Provenance and third-party notices

## Project code

The FSD, TDS, SPA+PN, Fisher, table-generation, and plotting code was developed
for the associated article by Xinmiao Zhao and Han Yan. Public interfaces use
the method names `FSD`, `TDS`, and `SPA+PN`, together with acceleration \(a\)
in \(\mathrm{s}^{-1}\).

## IMRPhenomD

`src/fsd_accelerating_waveforms/phenomd.py`, `phenomd_deriv.py`, and
`phenomd_utils.py` are Python adaptations of IMRPhenomD equations and
implementation structure distributed through LALSuite/LALSimulation. The
analytic derivative extension was developed for this work. These source files
are distributed under `GPL-3.0-or-later`.

- LALSuite: https://lscsoft.docs.ligo.org/lalsuite/lalsuite/index.html
- LALSimulation source: https://git.ligo.org/lscsoft/lalsuite

## PyCBC

PyCBC supplies waveform utilities, detector noise power spectral densities,
Fourier transforms, and matched-filter overlaps used by the validation and
figure pipeline:

- PyCBC: https://github.com/gwastro/pycbc

LALSuite and PyCBC are not bundled with this repository. Their names and marks
belong to their respective copyright holders.
