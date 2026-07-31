from __future__ import annotations

import numpy as np
from pycbc.filter import match
from pycbc.psd import aLIGOZeroDetHighPower

from fsd_accelerating_waveforms.waveforms import (
    Binary,
    fsd_waveform,
    luminosity_distance_mpc,
    mismatch_stages,
    project_to_grid,
    spa_pn_waveform,
    tds_waveform,
    vacuum_waveform,
)


def small_parameters() -> tuple[Binary, dict[str, float]]:
    binary = Binary(30.0, 20.0, 0.0, 0.0)
    common = {
        "delta_f_hz": 1.0 / 32.0,
        "f_start_hz": 10.0,
        "f_final_hz": 512.0,
        "distance_mpc": luminosity_distance_mpc(0.2),
    }
    return binary, common


def test_zero_acceleration_recovers_vacuum() -> None:
    binary, common = small_parameters()
    vacuum = vacuum_waveform(binary, **common)
    fsd = fsd_waveform(
        binary, acceleration_s_inv=0.0, order=3, **common
    )
    spa = spa_pn_waveform(binary, acceleration_s_inv=0.0, **common)
    np.testing.assert_allclose(
        fsd.numpy(), vacuum.numpy(), rtol=2.0e-15, atol=1.0e-60
    )
    np.testing.assert_allclose(spa.numpy(), vacuum.numpy(), rtol=0.0, atol=0.0)


def test_tds_zero_acceleration_recovers_vacuum_across_grids() -> None:
    binary = Binary(30.0, 20.0, 0.0, 0.0)
    distance = luminosity_distance_mpc(0.2)
    for delta_f_hz in (1.0 / 16.0, 1.0 / 32.0):
        for sample_rate_hz in (1024.0, 2048.0):
            common = {
                "delta_f_hz": delta_f_hz,
                "f_start_hz": 10.0,
                "f_final_hz": 512.0,
                "distance_mpc": distance,
            }
            tds = tds_waveform(
                binary,
                acceleration_s_inv=0.0,
                sample_rate_hz=sample_rate_hz,
                **common,
            )
            vacuum = project_to_grid(
                vacuum_waveform(
                    binary,
                    delta_f_hz=float(tds.delta_f),
                    f_start_hz=common["f_start_hz"],
                    f_final_hz=common["f_final_hz"],
                    distance_mpc=distance,
                ),
                tds,
            )
            psd = aLIGOZeroDetHighPower(len(tds), float(tds.delta_f), 10.0)
            overlap, _ = match(
                tds,
                vacuum,
                psd=psd,
                low_frequency_cutoff=10.0,
                high_frequency_cutoff=500.0,
            )
            assert 1.0 - float(overlap) <= 1.0e-12


def test_public_methods_have_finite_outputs_and_mismatches() -> None:
    binary, common = small_parameters()
    tds = tds_waveform(
        binary,
        acceleration_s_inv=1.0e-5,
        sample_rate_hz=1024.0,
        **common,
    )
    assert np.all(np.isfinite(tds.numpy()))
    for order in (1, 2, 3):
        fsd = fsd_waveform(
            binary,
            acceleration_s_inv=1.0e-5,
            order=order,
            **common,
        )
        assert np.all(np.isfinite(fsd.numpy()))
    spa = spa_pn_waveform(
        binary, acceleration_s_inv=1.0e-5, **common
    )
    assert np.all(np.isfinite(spa.numpy()))
    values = mismatch_stages(
        tds,
        fsd,
        binary=binary,
        psd_name="EinsteinTelescopeP1600143",
        f_low_hz=10.0,
        f_end_margin_hz=5.0,
    )
    assert set(values) == {"inspiral", "merger-ringdown", "full"}
    assert all(np.isfinite(value) and value >= 0.0 for value in values.values())
