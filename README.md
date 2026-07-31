# FSDAcceleratingWaveforms

Code, machine-readable numerical data, and figures supporting:

> Xinmiao Zhao, Han Yan, and Xian Chen, “A Novel Method to Construct
> Frequency-Domain Gravitational Waveform for Accelerating Sources,”
> [arXiv:2604.00253](https://doi.org/10.48550/arXiv.2604.00253).

The implementation compares three waveform prescriptions:

- **FSD** — frequency-domain spectral differentiation, through third order;
- **TDS** — a numerical time-domain-stretching benchmark;
- **SPA+PN** — a stationary-phase, post-Newtonian comparison model.

The public acceleration parameter is `acceleration_s_inv`, denoted \(a\) in the
article and measured in \(\mathrm{s}^{-1}\). `configs/paper.yaml` is the single
source of truth for masses, spins, acceleration values, frequency grids, power
spectral densities, FSD orders, and the Fisher-matrix SNR.

## Reproduce the results

Create the tested Conda environment from the platform-specific explicit lock:

```bash
conda create --name fsd-accelerating-waveforms \
  --file environment-locks/linux-64.conda.lock
# Apple Silicon: use environment-locks/osx-arm64.conda.lock
conda activate fsd-accelerating-waveforms
python -m pip install --no-deps -e .
```

`environment.yml` remains the human-readable source specification. The
explicit locks pin every Conda package URL and SHA256; PyCBC and the required
LAL components are installed from Conda Forge, while pip installs only this
repository's source tree.

Run the lightweight calculation used by continuous integration:

```bash
FSD_REQUIRE_EXACT_LOCK=1 \
  python scripts/reproduce_all.py --mode quick --output-root build/quick
```

Recompute all six article datasets and figures:

```bash
FSD_REQUIRE_EXACT_LOCK=1 python scripts/reproduce_all.py --mode full
```

Regenerate figures directly from the archived numerical tables:

```bash
python scripts/plot_all.py
```

Both entry points accept `--figure fig01` through `--figure fig06`.
`notebooks/paper_figures_walkthrough.ipynb` reads, checks, and replots all six
tables without duplicating the waveform algorithms. Its outputs are written
under the ignored `build/` directory.

## Repository contents

- `src/fsd_accelerating_waveforms/`: waveform and analysis modules;
- `scripts/`: command-line reproduction entry points;
- `configs/paper.yaml`: the complete paper configuration;
- `data/`: one tidy CSV or CSV.GZ table per article figure, plus the sampling
  convergence table;
- `figures/`: the six article figures rendered from those tables;
- `notebooks/`: a runnable walkthrough of all six figures;
- `tests/`: waveform, regression, metadata, and reproducibility checks;
- `docs/`: data definitions, validation criteria, and validation results.

`reproduction_record.json` records the exact configuration hash, dependency
versions and Conda build strings, exact-lock verification, CPU and numerical
library identity, runtime, peak memory use, generated-file hashes, and
scientific validation results for the archived full calculation.
`SHA256SUMS.json` provides a release-wide file manifest.

## Contributions

Xinmiao Zhao developed the software implementation and performed the numerical
calculations. Xinmiao Zhao and Han Yan jointly designed the FSD algorithm and
analyzed the results. Xian Chen supervised the project and contributed to the
analysis and interpretation.

## Citation

Version 1.0.0 is archived at
[10.5281/zenodo.21643294](https://doi.org/10.5281/zenodo.21643294).
Citation metadata for the software and associated article are provided in
`CITATION.cff` and `.zenodo.json`.

> X. Zhao and H. Yan, *FSDAcceleratingWaveforms: Code and numerical data for
> “A Novel Method to Construct Frequency-Domain Gravitational Waveform for
> Accelerating Sources”*, Version 1.0.0, [Software], Zenodo (2026),
> https://doi.org/10.5281/zenodo.21643294.

Please also cite the associated
[article](https://doi.org/10.48550/arXiv.2604.00253).

## Licenses and third-party attribution

Source code is licensed under `GPL-3.0-or-later`. Numerical data and generated
figures are licensed under `CC-BY-4.0`. See `LICENSE`,
`LICENSES/CC-BY-4.0.txt`, and `NOTICE.md`.

The IMRPhenomD implementation adapts algorithms and implementation structure
from LALSuite/LALSimulation and is tested against PyCBC/LALSuite. LALSuite and
PyCBC remain independent projects; see `NOTICE.md` for links and attribution.
