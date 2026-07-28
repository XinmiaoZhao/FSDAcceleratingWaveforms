from __future__ import annotations

import numpy as np

from fsd_accelerating_waveforms.waveforms import (
    Binary,
    fsd_waveform,
    luminosity_distance_mpc,
    mismatch_stages,
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
