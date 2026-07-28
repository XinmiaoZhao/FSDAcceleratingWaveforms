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
