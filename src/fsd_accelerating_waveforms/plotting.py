"""Render the six article figures from archived machine-readable tables."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from .tables import read_rows
from .waveforms import METHOD_FSD, METHOD_SPA_PN, METHOD_TDS


DISPLAY_LABELS = {
    METHOD_TDS: "TDS",
    METHOD_FSD: "FSD",
    METHOD_SPA_PN: "SPA & PN",
}
PHASE_STYLES = {
    METHOD_TDS: ("C0", "-"),
    METHOD_FSD: ("C1", "--"),
    METHOD_SPA_PN: ("C2", "-."),
}
MARKERS = {0.0: "o", 0.5: "x", 0.99: "^"}
STAGE_TITLES = {
    "inspiral": "Inspiral",
    "merger-ringdown": "Merger–ringdown",
    "full": "Full",
}


def configure_matplotlib() -> None:
    """Apply the visual settings used by the article's original notebook."""

    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.grid": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def plot_phase(config: dict, data_dir: Path, figure_dir: Path) -> Path:
    data_path = data_dir / config["figures"]["fig01"]["data_file"]
    rows = read_rows(data_path)
    systems = sorted({int(row["system_index"]) for row in rows})
    columns = 3 if len(systems) > 1 else 1
    row_count = int(np.ceil(len(systems) / columns))
    fig, axes = plt.subplots(
        row_count,
        columns,
        figsize=(7.0 * columns, 7.0 * row_count),
        squeeze=False,
    )
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["system_index"]), row["method"])].append(row)

    for axis, system_index in zip(axes.flat, systems):
        system_rows = [
            row for row in rows if int(row["system_index"]) == system_index
        ]
        first = system_rows[0]
        for method in (METHOD_TDS, METHOD_FSD, METHOD_SPA_PN):
            method_rows = sorted(
                grouped[(system_index, method)],
                key=lambda row: float(row["frequency_hz"]),
            )
            frequencies = np.array(
                [float(row["frequency_hz"]) for row in method_rows]
            )
            values = np.abs(
                np.array([float(row["phase_shift_rad"]) for row in method_rows])
            )
            positive = values > 0.0
            color, linestyle = PHASE_STYLES[method]
            axis.loglog(
                frequencies[positive],
                values[positive],
                label=DISPLAY_LABELS[method],
                color=color,
                linestyle=linestyle,
            )
        mass1 = float(first["mass1_msun"])
        mass2 = float(first["mass2_msun"])
        chi = float(first["chi1"])
        mass_threshold = float(
            config["phase"]["display_mass_threshold_msun"]
        )
        f_max = (
            float(config["phase"]["display_f_max_low_mass_hz"])
            if mass1 < mass_threshold
            else float(config["phase"]["display_f_max_high_mass_hz"])
        )
        axis.set_xlim(float(config["common"]["f_start_hz"]), f_max)
        axis.set_ylim(1.0e-10, 1.0e2)
        axis.set_xlabel(r"$f$ (Hz)", fontsize=16)
        axis.set_ylabel(r"$|\Delta \Psi|$", fontsize=16)
        axis.legend(fontsize=14)
        axis.tick_params(axis="both", which="both", labelsize=13)
        axis.set_title(
            rf"$m_1 = {mass1:g}M_\odot, m_2 = {mass2:g}M_\odot, "
            rf"\chi_1 = \chi_2 = {chi:g}$",
            fontsize=20,
        )
    for axis in list(axes.flat)[len(systems) :]:
        axis.set_visible(False)
    path = figure_dir / config["figures"]["fig01"]["figure_file"]
    return _save(fig, path)


def _plot_mismatch_panels(
    rows: list[dict[str, str]],
    *,
    stages: tuple[str, ...],
    output: Path,
) -> Path:
    fig, axes = plt.subplots(
        1,
        len(stages),
        figsize=(7.0 * len(stages), 7.0),
        squeeze=False,
    )
    spins = sorted({float(row["chi1"]) for row in rows})
    for axis, stage in zip(axes.flat, stages):
        for spin in spins:
            for method in (METHOD_FSD, METHOD_SPA_PN):
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["stage"] == stage
                        and row["method"] == method
                        and float(row["chi1"]) == spin
                    ),
                    key=lambda row: float(row["total_mass_msun"]),
                )
                if not selected:
                    continue
                method_label = "FSD" if method == METHOD_FSD else "SPA"
                axis.plot(
                    [float(row["total_mass_msun"]) for row in selected],
                    [float(row["mismatch"]) for row in selected],
                    color="C0" if method == METHOD_FSD else "C1",
                    linestyle="--" if method == METHOD_FSD else "-",
                    marker=MARKERS.get(spin, "s"),
                    label=rf"$\chi={spin:g}$, TDS vs. {method_label}",
                )
        axis.set_yscale("log")
        axis.set_xlabel(r"$M\,(M_\odot)$", fontsize=20)
        axis.set_ylabel("Mismatch", fontsize=18)
        axis.legend(fontsize=14, framealpha=0.5)
        axis.tick_params(axis="both", which="both", labelsize=13)
        axis.set_title(STAGE_TITLES[stage], fontsize=20)
    return _save(fig, output)


def plot_mismatch(
    config: dict,
    figure_id: str,
    data_dir: Path,
    figure_dir: Path,
) -> Path:
    case = config["mismatch"]["cases"][figure_id]
    rows = read_rows(data_dir / case["data_file"])
    return _plot_mismatch_panels(
        rows,
        stages=("inspiral", "merger-ringdown", "full"),
        output=figure_dir / case["figure_file"],
    )


def plot_fsd_order(config: dict, data_dir: Path, figure_dir: Path) -> Path:
    fsd_order = config["fsd_order"]
    rows = read_rows(data_dir / fsd_order["data_file"])
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.0))
    series = (
        (METHOD_FSD, "1", "1st order", "C0", "--", "o"),
        (METHOD_FSD, "2", "2nd order", "C3", "--", "x"),
        (METHOD_FSD, "3", "3rd order", "C2", "--", "^"),
        (METHOD_SPA_PN, "", "SPA & PN", "C1", "-", "s"),
    )
    for axis, stage in zip(axes, ("inspiral", "merger-ringdown")):
        for method, order, label, color, linestyle, marker in series:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["stage"] == stage
                    and row["method"] == method
                    and row["fsd_order"] == order
                ),
                key=lambda row: float(row["total_mass_msun"]),
            )
            axis.plot(
                [float(row["total_mass_msun"]) for row in selected],
                [float(row["mismatch"]) for row in selected],
                color=color,
                linestyle=linestyle,
                marker=marker,
                label=label,
            )
        axis.set_yscale("log")
        axis.set_xlabel(r"$M\,(M_\odot)$", fontsize=20)
        axis.set_ylabel("Mismatch", fontsize=18)
        axis.legend(fontsize=14, framealpha=0.5)
        axis.set_title(STAGE_TITLES[stage], fontsize=20)
        axis.tick_params(axis="both", which="both", labelsize=13)
    return _save(fig, figure_dir / fsd_order["figure_file"])


def plot_fisher(config: dict, data_dir: Path, figure_dir: Path) -> Path:
    fisher = config["fisher"]
    rows = read_rows(data_dir / fisher["data_file"])
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    for method in grouped:
        grouped[method].sort(key=lambda row: float(row["chirp_mass_msun"]))

    fig, (axis, difference_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8.0, 8.0),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
    )
    styles = {
        METHOD_TDS: ("C0", "-", "o"),
        METHOD_FSD: ("C1", "--", "x"),
        METHOD_SPA_PN: ("C2", "--", "^"),
    }
    for method in (METHOD_TDS, METHOD_FSD, METHOD_SPA_PN):
        selected = grouped[method]
        color, linestyle, marker = styles[method]
        axis.plot(
            [float(row["chirp_mass_msun"]) for row in selected],
            [float(row["acceleration_uncertainty_s_inv"]) for row in selected],
            color=color,
            linestyle=linestyle,
            marker=marker,
            label=DISPLAY_LABELS[method],
        )
    axis.set_ylabel(r"$\Delta a\,$(s$^{-1}$)", fontsize=18)
    axis.set_yscale("log")
    axis.set_xlim(10, 60)
    axis.legend(fontsize=14)
    axis.tick_params(axis="y", which="both", labelsize=13)

    inset = inset_axes(axis, width="30%", height="30%", loc="lower right")
    for method in (METHOD_TDS, METHOD_FSD, METHOD_SPA_PN):
        selected = grouped[method]
        color, linestyle, marker = styles[method]
        inset.plot(
            [float(row["chirp_mass_msun"]) for row in selected],
            [float(row["acceleration_uncertainty_s_inv"]) for row in selected],
            color=color,
            linestyle=linestyle,
            marker=marker,
        )
    inset.set_xlim(49, 50)
    inset.set_ylim(1.63e-5, 1.77e-5)
    inset.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    inset.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    inset.set_yscale("log")
    mark_inset(axis, inset, loc1=1, loc2=2, fc="none", ec="0.5")

    tds_values = np.array(
        [
            float(row["acceleration_uncertainty_s_inv"])
            for row in grouped[METHOD_TDS]
        ]
    )
    chirp_masses = np.array(
        [float(row["chirp_mass_msun"]) for row in grouped[METHOD_TDS]]
    )
    difference_labels = {
        METHOD_FSD: (
            r"$|\Delta a_\mathrm{FSD}/\Delta a_\mathrm{TDS} - 1|$"
        ),
        METHOD_SPA_PN: (
            r"$|\Delta a_\mathrm{SPA}/\Delta a_\mathrm{TDS} - 1|$"
        ),
    }
    for method in (METHOD_FSD, METHOD_SPA_PN):
        values = np.array(
            [
                float(row["acceleration_uncertainty_s_inv"])
                for row in grouped[method]
            ]
        )
        color, linestyle, marker = styles[method]
        difference_axis.plot(
            chirp_masses,
            np.abs(values / tds_values - 1.0),
            color=color,
            linestyle=linestyle,
            marker=marker,
            label=difference_labels[method],
        )
    difference_axis.set_xlabel(r"$\mathcal{M}\,(M_\odot)$", fontsize=18)
    difference_axis.set_ylabel(
        r"$|\Delta a_x/\Delta a_\mathrm{TDS} - 1|$", fontsize=18
    )
    difference_axis.tick_params(axis="both", which="both", labelsize=13)
    difference_axis.legend(fontsize=14)
    difference_axis.set_ylim(0, 0.0145)
    return _save(fig, figure_dir / fisher["figure_file"])


def plot_figure(
    config: dict,
    figure_id: str,
    data_dir: Path,
    figure_dir: Path,
) -> Path:
    configure_matplotlib()
    if figure_id == "fig01":
        return plot_phase(config, data_dir, figure_dir)
    if figure_id in {"fig02", "fig03", "fig04"}:
        return plot_mismatch(config, figure_id, data_dir, figure_dir)
    if figure_id == "fig05":
        return plot_fsd_order(config, data_dir, figure_dir)
    if figure_id == "fig06":
        return plot_fisher(config, data_dir, figure_dir)
    raise ValueError(f"Unsupported figure {figure_id}")
