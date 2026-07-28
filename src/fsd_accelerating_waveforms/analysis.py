"""Generate the machine-readable tables supporting each paper figure."""

from __future__ import annotations

import csv
import gc
import gzip
import io
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .waveforms import (
    METHOD_FSD,
    METHOD_SPA_PN,
    METHOD_TDS,
    Binary,
    fisher_acceleration_uncertainty,
    fsd_waveform,
    isco_frequency_hz,
    luminosity_distance_mpc,
    mismatch_stages,
    phase_shift,
    spa_pn_waveform,
    tds_waveform,
    vacuum_waveform,
)


PHASE_FIELDS = (
    "system_index",
    "mass1_msun",
    "mass2_msun",
    "chi1",
    "chi2",
    "acceleration_s_inv",
    "sample_rate_hz",
    "frequency_hz",
    "method",
    "phase_shift_rad",
)
MISMATCH_FIELDS = (
    "primary_mass_msun",
    "secondary_mass_msun",
    "total_mass_msun",
    "chi1",
    "chi2",
    "acceleration_s_inv",
    "psd",
    "f_low_hz",
    "f_isco_hz",
    "stage",
    "method",
    "fsd_order",
    "mismatch",
)
FISHER_FIELDS = (
    "primary_mass_msun",
    "secondary_mass_msun",
    "total_mass_msun",
    "chirp_mass_msun",
    "chi1",
    "chi2",
    "acceleration_s_inv",
    "psd",
    "snr",
    "method",
    "fsd_order",
    "acceleration_uncertainty_s_inv",
)


@contextmanager
def _open_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with path.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                mtime=0,
            ) as compressed:
                with io.TextIOWrapper(
                    compressed,
                    encoding="utf-8",
                    newline="",
                ) as text:
                    yield text
        return
    with path.open("w", encoding="utf-8", newline="") as text:
        yield text


def write_csv(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    with _open_csv(path) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _sample_indices(
    frequencies: np.ndarray,
    values: np.ndarray,
    *,
    f_min_hz: float,
    f_max_hz: float,
    count: int,
) -> np.ndarray:
    valid = (
        np.isfinite(values)
        & np.isfinite(frequencies)
        & (frequencies >= f_min_hz)
        & (frequencies <= f_max_hz)
    )
    indices = np.flatnonzero(valid)
    if indices.size <= count:
        return indices
    targets = np.geomspace(
        max(f_min_hz, float(frequencies[indices[0]])),
        min(f_max_hz, float(frequencies[indices[-1]])),
        count,
    )
    selected = np.searchsorted(frequencies, targets)
    selected = np.clip(selected, indices[0], indices[-1])
    return np.unique(selected[valid[selected]])


def _phase_rows_for_method(
    *,
    system_index: int,
    binary: Binary,
    acceleration_s_inv: float,
    sample_rate_hz: float,
    method: str,
    frequencies: np.ndarray,
    shifts: np.ndarray,
    f_start_hz: float,
    f_final_hz: float,
    output_points: int,
) -> list[dict[str, object]]:
    indices = _sample_indices(
        frequencies,
        shifts,
        f_min_hz=f_start_hz,
        f_max_hz=f_final_hz,
        count=output_points,
    )
    return [
        {
            "system_index": system_index,
            "mass1_msun": binary.mass1_msun,
            "mass2_msun": binary.mass2_msun,
            "chi1": binary.chi1,
            "chi2": binary.chi2,
            "acceleration_s_inv": acceleration_s_inv,
            "sample_rate_hz": sample_rate_hz,
            "frequency_hz": float(frequencies[index]),
            "method": method,
            "phase_shift_rad": float(shifts[index]),
        }
        for index in indices
    ]


def generate_phase_table(
    config: dict,
    data_dir: Path,
) -> list[Path]:
    common = config["common"]
    phase_config = config["phase"]
    acceleration = float(phase_config["acceleration_s_inv"])
    sample_rate = float(common["sample_rate_hz"])
    f_start = float(common["f_start_hz"])
    f_final = float(common["f_final_hz"])
    delta_f = float(phase_config["delta_f_hz"])
    t0 = float(common["t0_s"])
    distance = luminosity_distance_mpc(float(common["redshift"]))
    output_points = int(phase_config["output_points_per_curve"])
    selected = set(
        phase_config.get(
            "system_indices", range(len(phase_config["systems"]))
        )
    )
    rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []

    for system_index, values in enumerate(phase_config["systems"]):
        if system_index not in selected:
            continue
        binary = Binary(
            float(values["mass1_msun"]),
            float(values["mass2_msun"]),
            float(values["chi1"]),
            float(values["chi2"]),
        )
        common_waveform = {
            "delta_f_hz": delta_f,
            "f_start_hz": f_start,
            "f_final_hz": f_final,
            "distance_mpc": distance,
        }
        anchor = isco_frequency_hz(binary)

        tds_accelerated = tds_waveform(
            binary,
            acceleration_s_inv=acceleration,
            sample_rate_hz=sample_rate,
            t0_s=t0,
            **common_waveform,
        )
        tds_zero = tds_waveform(
            binary,
            acceleration_s_inv=0.0,
            sample_rate_hz=sample_rate,
            t0_s=t0,
            **common_waveform,
        )
        tds_frequencies, tds_shift = phase_shift(
            tds_accelerated, tds_zero, anchor_frequency_hz=anchor
        )
        rows.extend(
            _phase_rows_for_method(
                system_index=system_index,
                binary=binary,
                acceleration_s_inv=acceleration,
                sample_rate_hz=sample_rate,
                method=METHOD_TDS,
                frequencies=tds_frequencies,
                shifts=tds_shift,
                f_start_hz=f_start,
                f_final_hz=f_final,
                output_points=output_points,
            )
        )

        vacuum = vacuum_waveform(binary, **common_waveform)
        fsd = fsd_waveform(
            binary,
            acceleration_s_inv=acceleration,
            order=1,
            t0_s=t0,
            **common_waveform,
        )
        fsd_frequencies, fsd_shift = phase_shift(
            fsd, vacuum, anchor_frequency_hz=anchor
        )
        rows.extend(
            _phase_rows_for_method(
                system_index=system_index,
                binary=binary,
                acceleration_s_inv=acceleration,
                sample_rate_hz=sample_rate,
                method=METHOD_FSD,
                frequencies=fsd_frequencies,
                shifts=fsd_shift,
                f_start_hz=f_start,
                f_final_hz=f_final,
                output_points=output_points,
            )
        )

        spa = spa_pn_waveform(
            binary,
            acceleration_s_inv=acceleration,
            **common_waveform,
        )
        spa_frequencies, spa_shift = phase_shift(
            spa, vacuum, anchor_frequency_hz=anchor
        )
        rows.extend(
            _phase_rows_for_method(
                system_index=system_index,
                binary=binary,
                acceleration_s_inv=acceleration,
                sample_rate_hz=sample_rate,
                method=METHOD_SPA_PN,
                frequencies=spa_frequencies,
                shifts=spa_shift,
                f_start_hz=f_start,
                f_final_hz=f_final,
                output_points=output_points,
            )
        )

        if config["_meta"]["mode"] == "full":
            convergence_rate = float(
                phase_config["convergence_sample_rate_hz"]
            )
            convergence_final = float(
                phase_config["convergence_f_final_hz"]
            )
            convergence_waveform = dict(common_waveform)
            convergence_waveform["f_final_hz"] = convergence_final
            lower_accelerated = tds_waveform(
                binary,
                acceleration_s_inv=acceleration,
                sample_rate_hz=convergence_rate,
                t0_s=t0,
                **convergence_waveform,
            )
            lower_zero = tds_waveform(
                binary,
                acceleration_s_inv=0.0,
                sample_rate_hz=convergence_rate,
                t0_s=t0,
                **convergence_waveform,
            )
            lower_frequencies, lower_shift = phase_shift(
                lower_accelerated,
                lower_zero,
                anchor_frequency_hz=anchor,
            )
            mass_threshold = float(
                phase_config["display_mass_threshold_msun"]
            )
            display_f_max = (
                float(phase_config["display_f_max_low_mass_hz"])
                if binary.mass1_msun < mass_threshold
                else float(phase_config["display_f_max_high_mass_hz"])
            )
            convergence_indices = _sample_indices(
                lower_frequencies,
                lower_shift,
                f_min_hz=f_start,
                f_max_hz=min(convergence_final, display_f_max),
                count=output_points,
            )
            high_interp = np.interp(
                lower_frequencies[convergence_indices],
                tds_frequencies,
                tds_shift,
            )
            for index, high_value in zip(convergence_indices, high_interp):
                low_value = float(lower_shift[index])
                convergence_rows.append(
                    {
                        "system_index": system_index,
                        "mass1_msun": binary.mass1_msun,
                        "mass2_msun": binary.mass2_msun,
                        "chi1": binary.chi1,
                        "chi2": binary.chi2,
                        "frequency_hz": float(lower_frequencies[index]),
                        "phase_shift_4096_rad": float(high_value),
                        "phase_shift_2048_rad": low_value,
                        "absolute_difference_rad": abs(
                            float(high_value) - low_value
                        ),
                    }
                )
            del lower_accelerated, lower_zero

        del tds_accelerated, tds_zero, vacuum, fsd, spa
        gc.collect()

    data_path = data_dir / config["figures"]["fig01"]["data_file"]
    write_csv(data_path, PHASE_FIELDS, rows)
    generated = [data_path]
    if convergence_rows:
        convergence_path = data_dir / "validation_phase_sample_rate.csv"
        write_csv(
            convergence_path,
            (
                "system_index",
                "mass1_msun",
                "mass2_msun",
                "chi1",
                "chi2",
                "frequency_hz",
                "phase_shift_4096_rad",
                "phase_shift_2048_rad",
                "absolute_difference_rad",
            ),
            convergence_rows,
        )
        generated.append(convergence_path)
    return generated


def _mass_grid(config: dict, section: dict) -> list[float]:
    return [
        float(value)
        for value in section.get(
            "primary_masses_msun",
            config["mismatch"]["primary_masses_msun"],
        )
    ]


def _binary_from_primary(
    primary_mass: float,
    mass_ratio: float,
    spin: float,
) -> Binary:
    return Binary(primary_mass, primary_mass / mass_ratio, spin, spin)


def generate_mismatch_table(
    config: dict,
    figure_id: str,
    data_dir: Path,
) -> Path:
    common = config["common"]
    mismatch_config = config["mismatch"]
    case = mismatch_config["cases"][figure_id]
    distance = luminosity_distance_mpc(float(common["redshift"]))
    rows: list[dict[str, object]] = []
    acceleration = float(case["acceleration_s_inv"])
    delta_f = float(mismatch_config["delta_f_hz"])
    common_waveform = {
        "delta_f_hz": delta_f,
        "f_start_hz": float(common["f_start_hz"]),
        "f_final_hz": float(common["f_final_hz"]),
        "distance_mpc": distance,
    }

    for spin in (float(value) for value in mismatch_config["spins"]):
        for primary_mass in _mass_grid(config, mismatch_config):
            binary = _binary_from_primary(
                primary_mass,
                float(mismatch_config["mass_ratio_m1_over_m2"]),
                spin,
            )
            tds = tds_waveform(
                binary,
                acceleration_s_inv=acceleration,
                sample_rate_hz=float(common["sample_rate_hz"]),
                t0_s=float(common["t0_s"]),
                **common_waveform,
            )
            def iter_models():
                yield (
                    METHOD_FSD,
                    1,
                    fsd_waveform(
                        binary,
                        acceleration_s_inv=acceleration,
                        order=1,
                        t0_s=float(common["t0_s"]),
                        **common_waveform,
                    ),
                )
                yield (
                    METHOD_SPA_PN,
                    "",
                    spa_pn_waveform(
                        binary,
                        acceleration_s_inv=acceleration,
                        **common_waveform,
                    ),
                )

            for method, order, model in iter_models():
                values = mismatch_stages(
                    tds,
                    model,
                    binary=binary,
                    psd_name=str(case["psd"]),
                    f_low_hz=float(case["f_low_hz"]),
                    f_end_margin_hz=float(mismatch_config["f_end_margin_hz"]),
                )
                for stage, value in values.items():
                    rows.append(
                        {
                            "primary_mass_msun": binary.mass1_msun,
                            "secondary_mass_msun": binary.mass2_msun,
                            "total_mass_msun": binary.total_mass_msun,
                            "chi1": binary.chi1,
                            "chi2": binary.chi2,
                            "acceleration_s_inv": acceleration,
                            "psd": case["psd"],
                            "f_low_hz": case["f_low_hz"],
                            "f_isco_hz": isco_frequency_hz(binary),
                            "stage": stage,
                            "method": method,
                            "fsd_order": order,
                            "mismatch": value,
                        }
                    )
                del model
                gc.collect()
            del tds
            gc.collect()

    path = data_dir / case["data_file"]
    write_csv(path, MISMATCH_FIELDS, rows)
    return path


def generate_fsd_order_table(
    config: dict,
    data_dir: Path,
) -> Path:
    common = config["common"]
    mismatch_config = config["mismatch"]
    fsd_order = config["fsd_order"]
    acceleration = float(fsd_order["acceleration_s_inv"])
    distance = luminosity_distance_mpc(float(common["redshift"]))
    delta_f = float(mismatch_config["delta_f_hz"])
    common_waveform = {
        "delta_f_hz": delta_f,
        "f_start_hz": float(common["f_start_hz"]),
        "f_final_hz": float(common["f_final_hz"]),
        "distance_mpc": distance,
    }
    rows: list[dict[str, object]] = []

    for primary_mass in _mass_grid(config, fsd_order):
        binary = Binary(
            primary_mass,
            primary_mass
            / float(mismatch_config["mass_ratio_m1_over_m2"]),
            float(fsd_order["chi1"]),
            float(fsd_order["chi2"]),
        )
        tds = tds_waveform(
            binary,
            acceleration_s_inv=acceleration,
            sample_rate_hz=float(common["sample_rate_hz"]),
            t0_s=float(common["t0_s"]),
            **common_waveform,
        )
        for order in (int(value) for value in fsd_order["orders"]):
            model = fsd_waveform(
                binary,
                acceleration_s_inv=acceleration,
                order=order,
                t0_s=float(common["t0_s"]),
                **common_waveform,
            )
            values = mismatch_stages(
                tds,
                model,
                binary=binary,
                psd_name=str(fsd_order["psd"]),
                f_low_hz=float(fsd_order["f_low_hz"]),
                f_end_margin_hz=float(mismatch_config["f_end_margin_hz"]),
            )
            for stage, value in values.items():
                rows.append(
                    {
                        "primary_mass_msun": binary.mass1_msun,
                        "secondary_mass_msun": binary.mass2_msun,
                        "total_mass_msun": binary.total_mass_msun,
                        "chi1": binary.chi1,
                        "chi2": binary.chi2,
                        "acceleration_s_inv": acceleration,
                        "psd": fsd_order["psd"],
                        "f_low_hz": fsd_order["f_low_hz"],
                        "f_isco_hz": isco_frequency_hz(binary),
                        "stage": stage,
                        "method": METHOD_FSD,
                        "fsd_order": order,
                        "mismatch": value,
                    }
                )
            del model
            gc.collect()

        model = spa_pn_waveform(
            binary,
            acceleration_s_inv=acceleration,
            **common_waveform,
        )
        values = mismatch_stages(
            tds,
            model,
            binary=binary,
            psd_name=str(fsd_order["psd"]),
            f_low_hz=float(fsd_order["f_low_hz"]),
            f_end_margin_hz=float(mismatch_config["f_end_margin_hz"]),
        )
        for stage, value in values.items():
            rows.append(
                {
                    "primary_mass_msun": binary.mass1_msun,
                    "secondary_mass_msun": binary.mass2_msun,
                    "total_mass_msun": binary.total_mass_msun,
                    "chi1": binary.chi1,
                    "chi2": binary.chi2,
                    "acceleration_s_inv": acceleration,
                    "psd": fsd_order["psd"],
                    "f_low_hz": fsd_order["f_low_hz"],
                    "f_isco_hz": isco_frequency_hz(binary),
                    "stage": stage,
                    "method": METHOD_SPA_PN,
                    "fsd_order": "",
                    "mismatch": value,
                }
            )
        del tds, model
        gc.collect()

    path = data_dir / fsd_order["data_file"]
    write_csv(path, MISMATCH_FIELDS, rows)
    return path


def generate_fisher_table(config: dict, data_dir: Path) -> Path:
    common = config["common"]
    fisher = config["fisher"]
    distance = luminosity_distance_mpc(float(common["redshift"]))
    rows: list[dict[str, object]] = []

    for primary_mass in (float(value) for value in fisher["primary_masses_msun"]):
        binary = _binary_from_primary(
            primary_mass,
            float(fisher["mass_ratio_m1_over_m2"]),
            float(fisher["chi1"]),
        )
        if binary.chi2 != float(fisher["chi2"]):
            binary = Binary(
                binary.mass1_msun,
                binary.mass2_msun,
                float(fisher["chi1"]),
                float(fisher["chi2"]),
            )
        for method in (METHOD_TDS, METHOD_FSD, METHOD_SPA_PN):
            uncertainty = fisher_acceleration_uncertainty(
                method,
                binary,
                acceleration_s_inv=float(fisher["acceleration_s_inv"]),
                relative_step=float(fisher["relative_step"]),
                target_snr=float(fisher["snr"]),
                psd_name=str(fisher["psd"]),
                f_low_hz=float(fisher["f_low_hz"]),
                f_high_hz=float(fisher["f_final_hz"]),
                delta_f_hz=float(fisher["delta_f_hz"]),
                sample_rate_hz=float(fisher["sample_rate_hz"]),
                f_start_hz=float(common["f_start_hz"]),
                f_final_hz=float(fisher["f_final_hz"]),
                distance_mpc=distance,
                fsd_order=int(fisher["fsd_order"]),
                t0_s=float(common["t0_s"]),
            )
            rows.append(
                {
                    "primary_mass_msun": binary.mass1_msun,
                    "secondary_mass_msun": binary.mass2_msun,
                    "total_mass_msun": binary.total_mass_msun,
                    "chirp_mass_msun": binary.chirp_mass_msun,
                    "chi1": binary.chi1,
                    "chi2": binary.chi2,
                    "acceleration_s_inv": fisher["acceleration_s_inv"],
                    "psd": fisher["psd"],
                    "snr": fisher["snr"],
                    "method": method,
                    "fsd_order": (
                        fisher["fsd_order"] if method == METHOD_FSD else ""
                    ),
                    "acceleration_uncertainty_s_inv": uncertainty,
                }
            )
            gc.collect()

    path = data_dir / fisher["data_file"]
    write_csv(path, FISHER_FIELDS, rows)
    return path


def generate_figure_data(
    config: dict,
    figure_id: str,
    data_dir: Path,
) -> list[Path]:
    if figure_id == "fig01":
        return generate_phase_table(config, data_dir)
    if figure_id in {"fig02", "fig03", "fig04"}:
        return [generate_mismatch_table(config, figure_id, data_dir)]
    if figure_id == "fig05":
        return [generate_fsd_order_table(config, data_dir)]
    if figure_id == "fig06":
        return [generate_fisher_table(config, data_dir)]
    raise ValueError(f"Unsupported figure {figure_id}")
