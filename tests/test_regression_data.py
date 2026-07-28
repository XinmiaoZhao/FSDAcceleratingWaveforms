from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from fsd_accelerating_waveforms.tables import read_rows
from fsd_accelerating_waveforms.analysis import write_csv


ROOT = Path(__file__).resolve().parents[1]
BASELINE = {
    ("FSD", "1"): 1.981042757914775e-05,
    ("FSD", "2"): 1.0250461079941431e-07,
    ("FSD", "3"): 9.077733009732469e-10,
    ("SPA+PN", ""): 3.888990685041449e-08,
}


def test_nonspinning_nine_point_regression() -> None:
    rows = read_rows(ROOT / "data" / "fig05_fsd_order.csv")
    inspiral = [row for row in rows if row["stage"] == "inspiral"]
    masses = sorted({float(row["total_mass_msun"]) for row in inspiral})
    assert len(masses) == 9
    assert {
        (float(row["chi1"]), float(row["chi2"])) for row in rows
    } == {(0.0, 0.0)}

    low_mass_values = {
        (row["method"], row["fsd_order"]): float(row["mismatch"])
        for row in inspiral
        if float(row["total_mass_msun"]) == masses[0]
    }
    for key, expected in BASELINE.items():
        assert np.isclose(low_mass_values[key], expected, rtol=5.0e-7)


def test_fisher_table_is_finite() -> None:
    fisher = read_rows(ROOT / "data" / "fig06_fisher_uncertainty.csv")
    assert fisher
    assert all(
        np.isfinite(float(row["acceleration_uncertainty_s_inv"]))
        and float(row["acceleration_uncertainty_s_inv"]) > 0.0
        for row in fisher
    )


def test_compressed_tables_are_byte_reproducible(tmp_path: Path) -> None:
    path = tmp_path / "table.csv.gz"
    rows = [{"frequency_hz": 10.0, "value": 1.25}]
    write_csv(path, ("frequency_hz", "value"), rows)
    first = path.read_bytes()
    write_csv(path, ("frequency_hz", "value"), rows)
    assert path.read_bytes() == first


def test_full_reproduction_record_and_phase_convergence() -> None:
    record = json.loads((ROOT / "reproduction_record.json").read_text())
    status = json.loads(
        (ROOT / "docs" / "validation_status.json").read_text()
    )
    assert record["mode"] == "full"
    assert record["data_generation"] == "computed"
    assert record["selected_figures"] == [
        "fig01",
        "fig02",
        "fig03",
        "fig04",
        "fig05",
        "fig06",
    ]
    assert record["wall_time_seconds"] > 0.0
    assert record["peak_rss_bytes"] > 0
    assert record["validation"]["status"] == "passed"
    assert status["status"] == "passed"
    assert status["config_sha256"] == record["config_sha256"]

    convergence = record["validation"]
    assert (
        convergence["phase_convergence_max_abs_rad"]
        <= convergence["phase_convergence_display_limit_abs_rad"]
    )
    assert convergence["phase_convergence_rows"] > 0

    for relative, expected in record["generated_files"].items():
        path = ROOT / relative
        assert path.stat().st_size == expected["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
