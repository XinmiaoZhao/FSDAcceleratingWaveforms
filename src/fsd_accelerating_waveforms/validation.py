"""Scientific regression and artifact-integrity checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .tables import read_rows
from .waveforms import METHOD_FSD, METHOD_SPA_PN


FIG05_LOW_MASS_BASELINE = {
    ("1", METHOD_FSD): 1.981042757914775e-05,
    ("2", METHOD_FSD): 1.0250461079941431e-07,
    ("3", METHOD_FSD): 9.077733009732469e-10,
    ("", METHOD_SPA_PN): 3.888990685041449e-08,
}


def _finite_column(rows: list[dict[str, str]], key: str) -> bool:
    return bool(rows) and bool(
        np.all(np.isfinite([float(row[key]) for row in rows]))
    )


def validate_generated_data(
    config: dict,
    data_dir: Path,
    figure_dir: Path,
    *,
    require_all: bool,
) -> dict[str, object]:
    checks: dict[str, object] = {}
    failures: list[str] = []

    figure_paths = [
        figure_dir / config["figures"]["fig01"]["figure_file"],
        *[
            figure_dir / config["mismatch"]["cases"][figure_id]["figure_file"]
            for figure_id in ("fig02", "fig03", "fig04")
        ],
        figure_dir / config["fsd_order"]["figure_file"],
        figure_dir / config["fisher"]["figure_file"],
    ]
    existing_figures = [path for path in figure_paths if path.exists()]
    checks["figure_count"] = len(existing_figures)
    if require_all and len(existing_figures) != 6:
        failures.append(f"expected 6 figures, found {len(existing_figures)}")
    for path in existing_figures:
        if path.stat().st_size < 10_000:
            failures.append(f"figure is unexpectedly small: {path}")

    phase_path = data_dir / config["figures"]["fig01"]["data_file"]
    if phase_path.exists():
        phase_rows = read_rows(phase_path)
        checks["phase_rows"] = len(phase_rows)
        checks["phase_finite"] = _finite_column(phase_rows, "phase_shift_rad")
        if not checks["phase_finite"]:
            failures.append("phase table contains non-finite values")

    for figure_id in ("fig02", "fig03", "fig04"):
        path = data_dir / config["mismatch"]["cases"][figure_id]["data_file"]
        if not path.exists():
            continue
        rows = read_rows(path)
        checks[f"{figure_id}_rows"] = len(rows)
        if not _finite_column(rows, "mismatch"):
            failures.append(f"{figure_id} contains non-finite mismatch values")

    fsd_order_path = data_dir / config["fsd_order"]["data_file"]
    if fsd_order_path.exists():
        rows = read_rows(fsd_order_path)
        checks["fig05_rows"] = len(rows)
        if not _finite_column(rows, "mismatch"):
            failures.append("fig05 contains non-finite mismatch values")
        spins = {
            (float(row["chi1"]), float(row["chi2"])) for row in rows
        }
        checks["fig05_spins"] = [list(spin) for spin in sorted(spins)]
        if spins != {(0.0, 0.0)}:
            failures.append(f"fig05 must be non-spinning, found {spins}")
        inspiral = [row for row in rows if row["stage"] == "inspiral"]
        masses = sorted({float(row["total_mass_msun"]) for row in inspiral})
        checks["fig05_mass_points"] = len(masses)
        if config["_meta"]["mode"] == "full" and len(masses) != 9:
            failures.append(f"fig05 expected 9 mass points, found {len(masses)}")

        low_mass = min(masses)
        low_rows = [
            row
            for row in inspiral
            if float(row["total_mass_msun"]) == low_mass
        ]
        baseline_results: dict[str, float] = {}
        for row in low_rows:
            key = (row["fsd_order"], row["method"])
            if key not in FIG05_LOW_MASS_BASELINE:
                continue
            actual = float(row["mismatch"])
            expected = FIG05_LOW_MASS_BASELINE[key]
            relative = abs(actual / expected - 1.0)
            baseline_results[f"{row['method']}-order-{row['fsd_order'] or 'na'}"] = (
                relative
            )
            if config["_meta"]["mode"] == "full" and relative > 5.0e-7:
                failures.append(
                    f"fig05 low-mass baseline changed for {key}: "
                    f"{actual} vs {expected}"
                )
        checks["fig05_low_mass_relative_errors"] = baseline_results

    fisher_path = data_dir / config["fisher"]["data_file"]
    if fisher_path.exists():
        fisher_rows = read_rows(fisher_path)
        checks["fig06_rows"] = len(fisher_rows)
        if not _finite_column(
            fisher_rows, "acceleration_uncertainty_s_inv"
        ):
            failures.append("Fisher table contains non-finite uncertainties")

    convergence_path = data_dir / "validation_phase_sample_rate.csv"
    if convergence_path.exists():
        convergence_rows = read_rows(convergence_path)
        differences = np.array(
            [float(row["absolute_difference_rad"]) for row in convergence_rows]
        )
        checks["phase_convergence_rows"] = len(convergence_rows)
        checks["phase_convergence_max_abs_rad"] = float(np.max(differences))
        checks["phase_convergence_median_abs_rad"] = float(
            np.median(differences)
        )
        phase_config = config["phase"]
        mass_threshold = float(
            phase_config["display_mass_threshold_msun"]
        )
        display_differences = np.array(
            [
                float(row["absolute_difference_rad"])
                for row in convergence_rows
                if float(row["frequency_hz"])
                <= (
                    float(phase_config["display_f_max_low_mass_hz"])
                    if float(row["mass1_msun"]) < mass_threshold
                    else float(phase_config["display_f_max_high_mass_hz"])
                )
            ]
        )
        display_max = float(np.max(display_differences))
        display_limit = float(
            phase_config["convergence_max_abs_rad"]
        )
        checks["phase_convergence_display_rows"] = int(
            display_differences.size
        )
        checks["phase_convergence_display_max_abs_rad"] = display_max
        checks["phase_convergence_display_median_abs_rad"] = float(
            np.median(display_differences)
        )
        checks["phase_convergence_display_limit_abs_rad"] = display_limit
        if config["_meta"]["mode"] == "full" and display_max > display_limit:
            failures.append(
                "4096/2048 Hz phase convergence exceeds the configured "
                f"display-band limit: {display_max} > {display_limit}"
            )

    checks["status"] = "passed" if not failures else "failed"
    checks["failures"] = failures
    return checks


def write_validation_status(path: Path, status: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
