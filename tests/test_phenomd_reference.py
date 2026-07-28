from __future__ import annotations

from pycbc.filter import optimized_match
from pycbc.waveform import get_fd_waveform

from fsd_accelerating_waveforms.waveforms import Binary, vacuum_waveform


def mismatch_against_pycbc(chi: float) -> float:
    binary = Binary(30.0, 20.0, chi, chi)
    ours = vacuum_waveform(
        binary,
        delta_f_hz=1.0 / 8.0,
        f_start_hz=20.0,
        f_final_hz=512.0,
        distance_mpc=1000.0,
    )
    pycbc, _ = get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=30.0,
        mass2=20.0,
        spin1z=chi,
        spin2z=chi,
        distance=1000.0,
        delta_f=1.0 / 8.0,
        f_lower=20.0,
        f_final=512.0,
        f_ref=20.0,
    )
    return float(
        1.0
        - optimized_match(
            ours,
            pycbc,
            low_frequency_cutoff=20.0,
            high_frequency_cutoff=512.0,
        )[0]
    )


def test_nonspinning_phenomd_matches_pycbc_to_roundoff() -> None:
    assert abs(mismatch_against_pycbc(0.0)) < 1.0e-12


def test_high_spin_phenomd_stays_within_reference_tolerance() -> None:
    # The Python transcription and current LALSuite differ slightly in the
    # calibrated high-spin sector but remain well below the paper comparisons.
    assert abs(mismatch_against_pycbc(0.99)) < 5.0e-5
