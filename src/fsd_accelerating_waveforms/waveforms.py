"""Clean FSD, TDS, and SPA+PN waveform interfaces used by the paper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.cosmology import Planck18
from pycbc import pnutils
from pycbc.conversions import get_final_from_initial, tau_from_final_mass_spin
from pycbc.filter import optimized_match, overlap, sigma
from pycbc.psd import EinsteinTelescopeP1600143, aLIGOZeroDetHighPower
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.waveform import get_waveform_filter_length_in_time
from pycbc.waveform import utils as waveform_utils
from pycbc.waveform.waveform import props

from .phenomd import XLALSimIMRPhenomDFrequencySequence
from .phenomd_deriv import XLALSimIMRPhenomDAccFrequencySequence
from .phenomd_utils import LAL_MTSUN_SI


METHOD_FSD = "FSD"
METHOD_TDS = "TDS"
METHOD_SPA_PN = "SPA+PN"
METHODS = (METHOD_FSD, METHOD_TDS, METHOD_SPA_PN)

SPEED_OF_LIGHT_M_S = 299_792_458.0
NEWTON_G_SI = 6.67408e-11
SOLAR_MASS_KG = 1.98855e30
LEGACY_SOLAR_TIME_S = NEWTON_G_SI * SOLAR_MASS_KG / SPEED_OF_LIGHT_M_S**3

PSD_FACTORIES = {
    "aLIGOZeroDetHighPower": aLIGOZeroDetHighPower,
    "EinsteinTelescopeP1600143": EinsteinTelescopeP1600143,
}


@dataclass(frozen=True)
class Binary:
    mass1_msun: float
    mass2_msun: float
    chi1: float = 0.0
    chi2: float = 0.0

    @property
    def total_mass_msun(self) -> float:
        return self.mass1_msun + self.mass2_msun

    @property
    def chirp_mass_msun(self) -> float:
        m1 = self.mass1_msun
        m2 = self.mass2_msun
        return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


def luminosity_distance_mpc(redshift: float) -> float:
    return float(Planck18.luminosity_distance(redshift).value)


def _frequency_series(array: np.ndarray, delta_f_hz: float) -> FrequencySeries:
    duration_s = int(1.0 / delta_f_hz)
    return FrequencySeries(
        np.asarray(array, dtype=np.complex128),
        delta_f=delta_f_hz,
        epoch=-duration_s,
        copy=False,
    )


def vacuum_waveform(
    binary: Binary,
    *,
    delta_f_hz: float,
    f_start_hz: float,
    f_final_hz: float,
    distance_mpc: float,
) -> FrequencySeries:
    """Return the non-accelerating IMRPhenomD plus polarization."""

    strain = XLALSimIMRPhenomDFrequencySequence(
        phi0=0.0,
        fRef_in=f_start_hz,
        deltaF=delta_f_hz,
        m1=binary.mass1_msun,
        m2=binary.mass2_msun,
        chi1=binary.chi1,
        chi2=binary.chi2,
        distance_mpc=distance_mpc,
        freqs=[f_start_hz, f_final_hz],
        extraParams={},
        NRTidal_version="None",
    )
    return _frequency_series(strain, delta_f_hz)


def fsd_waveform(
    binary: Binary,
    *,
    acceleration_s_inv: float,
    order: int,
    delta_f_hz: float,
    f_start_hz: float,
    f_final_hz: float,
    distance_mpc: float,
    t0_s: float = 0.0,
) -> FrequencySeries:
    """Return the analytic FSD waveform through order 1, 2, or 3."""

    if order not in {1, 2, 3}:
        raise ValueError("FSD order must be 1, 2, or 3")
    strain = XLALSimIMRPhenomDAccFrequencySequence(
        phi0=0.0,
        fRef_in=f_start_hz,
        deltaF=delta_f_hz,
        m1=binary.mass1_msun,
        m2=binary.mass2_msun,
        chi1=binary.chi1,
        chi2=binary.chi2,
        distance_mpc=distance_mpc,
        acc=acceleration_s_inv,
        t0=t0_s,
        order=order,
        freqs=[f_start_hz, f_final_hz],
        extraParams={},
        NRTidal_version="None",
    )
    return _frequency_series(strain, delta_f_hz)


def _linear_interp_extrapolate(
    x_new: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    if np.iscomplexobj(y):
        return _linear_interp_extrapolate(
            x_new, x, np.real(y)
        ) + 1j * _linear_interp_extrapolate(x_new, x, np.imag(y))
    values = np.interp(x_new, x, y)
    left = x_new < x[0]
    if np.any(left):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        values[left] = y[0] + slope * (x_new[left] - x[0])
    right = x_new > x[-1]
    if np.any(right):
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        values[right] = y[-1] + slope * (x_new[right] - x[-1])
    return values


def _time_domain_stretching(
    waveform: TimeSeries,
    acceleration_s_inv: float,
    t0_s: float,
) -> TimeSeries:
    sample_times = np.asarray(waveform.sample_times)
    centered_time = sample_times - t0_s
    observed_time = centered_time - acceleration_s_inv * centered_time**2 + t0_s
    values = _linear_interp_extrapolate(
        sample_times, observed_time, np.asarray(waveform)
    )
    return TimeSeries(
        values,
        delta_t=float(sample_times[1] - sample_times[0]),
        epoch=float(sample_times[0]),
    )


def tds_waveform(
    binary: Binary,
    *,
    acceleration_s_inv: float,
    delta_f_hz: float,
    sample_rate_hz: float,
    f_start_hz: float,
    f_final_hz: float,
    distance_mpc: float,
    t0_s: float = 0.0,
) -> FrequencySeries:
    """Return the numerical finite-window TDS reference waveform."""

    params = {
        "approximant": "IMRPhenomD",
        "mass1": binary.mass1_msun,
        "mass2": binary.mass2_msun,
        "spin1z": binary.chi1,
        "spin2z": binary.chi2,
        "distance": distance_mpc,
        "delta_t": 1.0 / sample_rate_hz,
        "delta_f": delta_f_hz,
        "f_lower": f_start_hz,
        "f_final": f_final_hz,
    }
    working = props(None, **params)
    final_mass, final_spin = get_final_from_initial(
        mass1=binary.mass1_msun,
        mass2=binary.mass2_msun,
        spin1z=binary.chi1,
        spin2z=binary.chi2,
    )
    wrap_s = max(5.0, float(tau_from_final_mass_spin(final_mass, final_spin) * 10.0))

    duration_s = float(get_waveform_filter_length_in_time(**working))
    full_duration_s = duration_s
    while full_duration_s < duration_s * 1.5:
        full_duration_s = float(get_waveform_filter_length_in_time(**working))
        working["f_lower"] *= 0.99

    working.setdefault("f_ref", params["f_lower"])
    padded_duration_s = (max(0.0, full_duration_s) + 0.1 + wrap_s) * 1.5
    sample_count = int(padded_duration_s / params["delta_t"])
    fft_size = int(pnutils.nearest_larger_binary_number(sample_count))
    padded_duration_s = fft_size * params["delta_t"]
    working["delta_f"] = 1.0 / padded_duration_s
    frequency_size = fft_size // 2 + 1

    base = vacuum_waveform(
        binary,
        delta_f_hz=working["delta_f"],
        f_start_hz=working["f_lower"],
        f_final_hz=f_final_hz,
        distance_mpc=distance_mpc,
    )
    base.resize(frequency_size)
    base = base.cyclic_time_shift(-wrap_s)
    base_td = waveform_utils.fd_to_td(
        base,
        delta_t=params["delta_t"],
        left_window=(working["f_lower"], params["f_lower"]),
    )
    stretched = _time_domain_stretching(base_td, acceleration_s_inv, t0_s)
    result = stretched.to_frequencyseries().cyclic_time_shift(wrap_s)
    return result


def spa_pn_phase_shift(
    frequencies_hz: np.ndarray,
    binary: Binary,
    acceleration_s_inv: float,
) -> np.ndarray:
    """Return the 3PN spin-aligned SPA+PN acceleration phase shift."""

    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    phase = np.zeros_like(frequencies_hz)
    positive = frequencies_hz > 0.0
    if not np.any(positive):
        return phase

    velocity = (
        np.pi
        * frequencies_hz[positive]
        * binary.total_mass_msun
        * LEGACY_SOLAR_TIME_S
    ) ** (1.0 / 3.0)
    m1 = binary.mass1_msun
    m2 = binary.mass2_msun
    total_mass = binary.total_mass_msun
    symmetric_mass_ratio = m1 * m2 / total_mass**2
    mass_difference = (m1 - m2) / total_mass
    dimensionless_acceleration = acceleration_s_inv * LEGACY_SOLAR_TIME_S

    a0 = 25.0 * total_mass / (32768.0 * symmetric_mass_ratio**2)
    a2 = (
        25.0
        * total_mass
        * (743.0 + 924.0 * symmetric_mass_ratio)
        / (4128768.0 * symmetric_mass_ratio**2)
    )
    a3 = (
        -5.0 * total_mass * np.pi / (512.0 * symmetric_mass_ratio**2)
        + 5.0
        / (24576.0 * total_mass * symmetric_mass_ratio**2)
        * (
            (-75.0 * total_mass * mass_difference + 188.0 * m1**2)
            * binary.chi1
            + (75.0 * total_mass * mass_difference + 188.0 * m2**2)
            * binary.chi2
        )
    )
    a4 = (
        25.0
        * total_mass
        * (
            1755623.0
            + 112.0
            * symmetric_mass_ratio
            * (32633.0 + 23121.0 * symmetric_mass_ratio)
        )
        / (2774532096.0 * symmetric_mass_ratio**2)
    )
    a5 = (
        -5.0
        * total_mass
        * np.pi
        * (20807.0 + 8036.0 * symmetric_mass_ratio)
        / (1376256.0 * symmetric_mass_ratio**2)
        + 5.0
        / (4128768.0 * total_mass * symmetric_mass_ratio**2)
        * (
            (
                -5.0
                * total_mass
                * mass_difference
                * (32477.0 + 8736.0 * symmetric_mass_ratio)
                + 14.0 * m1**2 * (33049.0 + 8932.0 * symmetric_mass_ratio)
            )
            * binary.chi1
            + (
                5.0
                * total_mass
                * mass_difference
                * (32477.0 + 8736.0 * symmetric_mass_ratio)
                + 14.0 * m2**2 * (33049.0 + 8932.0 * symmetric_mass_ratio)
            )
            * binary.chi2
        )
    )
    a6 = (
        total_mass
        / (46146017820672.0 * symmetric_mass_ratio**2)
        * (
            23100.0
            * symmetric_mass_ratio
            * (
                3311653861.0
                + 84.0
                * symmetric_mass_ratio
                * (2030687.0 + 1856036.0 * symmetric_mass_ratio)
            )
            - 234710784.0
            * np.pi**2
            * (-18944.0 + 11275.0 * symmetric_mass_ratio)
            - 28907482848623.0
            + 4592284139520.0 * np.euler_gamma
            + 9184568279040.0 * np.log(2.0)
            + 2296142069760.0 * np.log(velocity**2)
        )
        + np.pi
        / (6144.0 * total_mass * symmetric_mass_ratio**2)
        * (
            (1725.0 * total_mass * mass_difference - 4454.0 * m1**2)
            * binary.chi1
            - (1725.0 * total_mass * mass_difference + 4454.0 * m2**2)
            * binary.chi2
        )
    )
    phase[positive] = dimensionless_acceleration * velocity ** (-13.0) * (
        a0
        + a2 * velocity**2
        + a3 * velocity**3
        + a4 * velocity**4
        + a5 * velocity**5
        + a6 * velocity**6
    )
    return phase


def spa_pn_waveform(
    binary: Binary,
    *,
    acceleration_s_inv: float,
    delta_f_hz: float,
    f_start_hz: float,
    f_final_hz: float,
    distance_mpc: float,
) -> FrequencySeries:
    base = vacuum_waveform(
        binary,
        delta_f_hz=delta_f_hz,
        f_start_hz=f_start_hz,
        f_final_hz=f_final_hz,
        distance_mpc=distance_mpc,
    )
    phase = spa_pn_phase_shift(
        np.asarray(base.sample_frequencies), binary, acceleration_s_inv
    )
    return FrequencySeries(
        np.asarray(base) * np.exp(1j * phase),
        delta_f=base.delta_f,
        epoch=base.epoch,
        copy=False,
    )


def project_to_grid(
    source: FrequencySeries, target: FrequencySeries
) -> FrequencySeries:
    target_frequencies = np.asarray(target.sample_frequencies)
    if source.delta_f > target.delta_f:
        source_frequencies = np.asarray(source.sample_frequencies)
        source_values = np.asarray(source)
        values = np.interp(
            target_frequencies, source_frequencies, source_values.real
        ) + 1j * np.interp(
            target_frequencies, source_frequencies, source_values.imag
        )
        return FrequencySeries(
            values,
            delta_f=target.delta_f,
            epoch=source.epoch,
            copy=False,
        )
    source_indices = np.rint(target_frequencies / source.delta_f).astype(int)
    valid = source_indices < len(source)
    if not np.allclose(
        target_frequencies[valid],
        source_indices[valid] * source.delta_f,
        rtol=0.0,
        atol=source.delta_f * 1.0e-7,
    ):
        raise ValueError("Source and target frequency grids are not commensurate")
    values = np.zeros(len(target), dtype=np.complex128)
    values[valid] = np.asarray(source)[source_indices[valid]]
    return FrequencySeries(
        values,
        delta_f=target.delta_f,
        epoch=source.epoch,
        copy=False,
    )


def isco_frequency_hz(binary: Binary) -> float:
    velocity = 1.0 / np.sqrt(6.0)
    return float(
        velocity**3
        * SPEED_OF_LIGHT_M_S**3
        / (
            NEWTON_G_SI
            * SOLAR_MASS_KG
            * binary.total_mass_msun
            * np.pi
        )
    )


def last_nonzero_frequency_hz(waveform: FrequencySeries) -> float:
    nonzero = np.flatnonzero(np.asarray(waveform) != 0.0)
    if nonzero.size == 0:
        raise ValueError("Waveform contains no non-zero samples")
    return float(nonzero[-1] * waveform.delta_f)


def psd_for(
    name: str,
    *,
    length: int,
    delta_f_hz: float,
    f_low_hz: float,
) -> FrequencySeries:
    try:
        factory = PSD_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported PSD {name!r}") from exc
    return factory(
        length=length,
        delta_f=delta_f_hz,
        low_freq_cutoff=f_low_hz,
    )


def mismatch(
    reference: FrequencySeries,
    model: FrequencySeries,
    *,
    psd_name: str,
    f_low_hz: float,
    f_high_hz: float,
) -> float:
    if len(reference) != len(model) or reference.delta_f != model.delta_f:
        raise ValueError("Mismatch inputs must share length and frequency spacing")
    psd = psd_for(
        psd_name,
        length=len(reference),
        delta_f_hz=reference.delta_f,
        f_low_hz=f_low_hz,
    )
    value = 1.0 - optimized_match(
        reference,
        model,
        psd=psd,
        low_frequency_cutoff=f_low_hz,
        high_frequency_cutoff=f_high_hz,
    )[0]
    return float(abs(value))


def mismatch_stages(
    reference_tds: FrequencySeries,
    model: FrequencySeries,
    *,
    binary: Binary,
    psd_name: str,
    f_low_hz: float,
    f_end_margin_hz: float,
) -> dict[str, float]:
    model_on_tds_grid = project_to_grid(model, reference_tds)
    isco_hz = isco_frequency_hz(binary)
    f_end_hz = min(
        last_nonzero_frequency_hz(model_on_tds_grid) - f_end_margin_hz,
        float(reference_tds.sample_frequencies[-1]),
    )
    if not f_low_hz < isco_hz < f_end_hz:
        raise ValueError(
            f"Invalid mismatch bands: {f_low_hz=}, {isco_hz=}, {f_end_hz=}"
        )
    return {
        "inspiral": mismatch(
            reference_tds,
            model_on_tds_grid,
            psd_name=psd_name,
            f_low_hz=f_low_hz,
            f_high_hz=isco_hz,
        ),
        "merger-ringdown": mismatch(
            reference_tds,
            model_on_tds_grid,
            psd_name=psd_name,
            f_low_hz=isco_hz,
            f_high_hz=f_end_hz,
        ),
        "full": mismatch(
            reference_tds,
            model_on_tds_grid,
            psd_name=psd_name,
            f_low_hz=f_low_hz,
            f_high_hz=f_end_hz,
        ),
    }


def phase_shift(
    accelerated: FrequencySeries,
    reference: FrequencySeries,
    *,
    anchor_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(accelerated) != len(reference) or accelerated.delta_f != reference.delta_f:
        raise ValueError("Phase-shift inputs must share a frequency grid")
    reference_values = np.asarray(reference)
    difference = np.asarray(accelerated) - reference_values
    valid = reference_values != 0.0
    ratio = np.zeros(len(reference_values), dtype=np.complex128)
    ratio[valid] = difference[valid] / reference_values[valid]
    shift = np.full(len(reference_values), np.nan)
    shift[valid] = ratio[valid].imag / (1.0 + ratio[valid].real)
    frequencies = np.asarray(reference.sample_frequencies)
    anchor_index = int(np.argmin(np.abs(frequencies - anchor_frequency_hz)))
    if not np.isfinite(shift[anchor_index]):
        finite_indices = np.flatnonzero(np.isfinite(shift))
        anchor_index = int(
            finite_indices[
                np.argmin(np.abs(frequencies[finite_indices] - anchor_frequency_hz))
            ]
        )
    shift -= shift[anchor_index]
    return frequencies, shift


def fisher_acceleration_uncertainty(
    method: str,
    binary: Binary,
    *,
    acceleration_s_inv: float,
    relative_step: float,
    target_snr: float,
    psd_name: str,
    f_low_hz: float,
    f_high_hz: float,
    delta_f_hz: float,
    sample_rate_hz: float,
    f_start_hz: float,
    f_final_hz: float,
    distance_mpc: float,
    fsd_order: int,
    t0_s: float = 0.0,
) -> float:
    if acceleration_s_inv == 0.0:
        raise ValueError("Fisher finite difference requires non-zero acceleration")
    step = abs(acceleration_s_inv) * relative_step

    def generate(value: float) -> FrequencySeries:
        if method == METHOD_FSD:
            return fsd_waveform(
                binary,
                acceleration_s_inv=value,
                order=fsd_order,
                delta_f_hz=delta_f_hz,
                f_start_hz=f_start_hz,
                f_final_hz=f_final_hz,
                distance_mpc=distance_mpc,
                t0_s=t0_s,
            )
        if method == METHOD_TDS:
            return tds_waveform(
                binary,
                acceleration_s_inv=value,
                delta_f_hz=delta_f_hz,
                sample_rate_hz=sample_rate_hz,
                f_start_hz=f_start_hz,
                f_final_hz=f_final_hz,
                distance_mpc=distance_mpc,
                t0_s=t0_s,
            )
        if method == METHOD_SPA_PN:
            return spa_pn_waveform(
                binary,
                acceleration_s_inv=value,
                delta_f_hz=delta_f_hz,
                f_start_hz=f_start_hz,
                f_final_hz=f_final_hz,
                distance_mpc=distance_mpc,
            )
        raise ValueError(f"Unsupported method {method!r}")

    center = generate(acceleration_s_inv)
    plus = generate(acceleration_s_inv + step)
    minus = generate(acceleration_s_inv - step)
    if len(plus) != len(minus) or plus.delta_f != minus.delta_f:
        raise ValueError("Fisher finite-difference grids differ")
    derivative = FrequencySeries(
        (np.asarray(plus) - np.asarray(minus)) / (2.0 * step),
        delta_f=plus.delta_f,
        epoch=plus.epoch,
        copy=False,
    )
    psd = psd_for(
        psd_name,
        length=len(center),
        delta_f_hz=center.delta_f,
        f_low_hz=f_low_hz,
    )
    center_snr = sigma(
        center,
        psd=psd,
        low_frequency_cutoff=f_low_hz,
        high_frequency_cutoff=f_high_hz,
    )
    fisher = overlap(
        derivative,
        derivative,
        psd=psd,
        low_frequency_cutoff=f_low_hz,
        high_frequency_cutoff=f_high_hz,
        normalized=False,
    )
    scaled_fisher = float(target_snr**2 / center_snr**2 * fisher)
    if not np.isfinite(scaled_fisher) or scaled_fisher <= 0.0:
        raise ValueError(f"Invalid Fisher information {scaled_fisher}")
    return float(1.0 / np.sqrt(scaled_fisher))
