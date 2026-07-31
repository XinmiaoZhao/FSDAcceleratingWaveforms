from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from fsd_accelerating_waveforms import waveforms
from fsd_accelerating_waveforms.config import (
    FIGURE_IDS,
    load_config,
    normalize_figure_ids,
)
from fsd_accelerating_waveforms.reproduction import (
    _environment_lock_verification,
)


ROOT = Path(__file__).resolve().parents[1]


def test_paper_yaml_is_the_complete_figure_source() -> None:
    config = load_config(ROOT / "configs" / "paper.yaml", mode="full")
    assert tuple(config["figures"]) == FIGURE_IDS
    assert FIGURE_IDS == ("fig01", "fig02", "fig03", "fig04", "fig05", "fig06")
    assert config["common"]["sample_rate_hz"] == 4096.0
    assert config["fsd_order"]["chi1"] == 0.0
    assert config["fsd_order"]["chi2"] == 0.0
    assert config["mismatch"]["spins"] == [0.0, 0.5, 0.99]
    assert config["fisher"]["snr"] == 1000.0
    assert config["fisher"]["fsd_order"] == 3
    with pytest.raises(ValueError):
        normalize_figure_ids("fig" + "S01")


def test_public_waveform_signatures_use_acceleration() -> None:
    public_functions = (
        waveforms.fsd_waveform,
        waveforms.tds_waveform,
        waveforms.spa_pn_waveform,
        waveforms.fisher_acceleration_uncertainty,
    )
    for function in public_functions:
        signature = inspect.signature(function)
        assert "acceleration_s_inv" in signature.parameters
        assert "gamma" not in signature.parameters
    assert waveforms.METHODS == ("FSD", "TDS", "SPA+PN")


def test_explicit_environment_verification_is_exact(tmp_path: Path) -> None:
    first = (
        "https://conda.anaconda.org/conda-forge/noarch/a-1-0.conda#"
        + "1" * 64
    )
    second = (
        "https://conda.anaconda.org/conda-forge/osx-arm64/b-2-0.conda#"
        + "2" * 64
    )
    lock = tmp_path / "osx-arm64.conda.lock"
    lock.write_text(
        "# platform: osx-arm64\n@EXPLICIT\n"
        + first
        + "\n"
        + second
        + "\n",
        encoding="utf-8",
    )
    verified = _environment_lock_verification(
        lock,
        "@EXPLICIT\n" + second + "\n" + first + "\n",
    )
    assert verified["status"] == "verified"
    assert verified["exact_match"] is True
    assert verified["expected_package_count"] == 2
    mismatched = _environment_lock_verification(
        lock,
        "@EXPLICIT\n" + first + "\n",
    )
    assert mismatched["status"] == "mismatch"
    assert mismatched["exact_match"] is False
    assert mismatched["missing_package_count"] == 1
