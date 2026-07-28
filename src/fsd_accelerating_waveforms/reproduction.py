"""Orchestrate data generation, plotting, and reproducibility records."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .analysis import generate_figure_data
from .config import normalize_figure_ids
from .plotting import plot_figure
from .validation import validate_generated_data, write_validation_status


DEPENDENCIES = (
    "astropy",
    "lalsuite",
    "matplotlib",
    "numpy",
    "pycbc",
    "PyYAML",
    "scipy",
    "setuptools",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def run_reproduction(
    config: dict,
    *,
    config_path: Path,
    output_root: Path,
    figure: str | None,
    reuse_data: bool = False,
) -> dict[str, object]:
    start = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    data_dir = output_root / "data"
    figure_dir = output_root / "figures"
    selected = normalize_figure_ids(figure)
    generated: list[Path] = []

    for figure_id in selected:
        if reuse_data:
            print(
                f"[reproduce] reusing archived data for {figure_id}",
                flush=True,
            )
        else:
            print(f"[reproduce] computing {figure_id}", flush=True)
            generated.extend(generate_figure_data(config, figure_id, data_dir))
        print(f"[reproduce] plotting {figure_id}", flush=True)
        generated.append(plot_figure(config, figure_id, data_dir, figure_dir))

    require_all = set(selected) == set(normalize_figure_ids(None))
    if reuse_data:
        generated.extend(path for path in data_dir.iterdir() if path.is_file())
    validation = validate_generated_data(
        config, data_dir, figure_dir, require_all=require_all
    )
    elapsed = time.perf_counter() - start
    try:
        recorded_config_path = str(config_path.resolve().relative_to(output_root))
    except ValueError:
        recorded_config_path = str(config_path.resolve())
    record = {
        "schema_version": 1,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "mode": config["_meta"]["mode"],
        "data_generation": "reused" if reuse_data else "computed",
        "selected_figures": list(selected),
        "config_path": recorded_config_path,
        "config_sha256": sha256(config_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
        "wall_time_seconds": elapsed,
        "peak_rss_bytes": peak_rss_bytes(),
        "generated_files": {
            str(path.relative_to(output_root)): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(set(generated))
            if path.exists()
        },
        "validation": validation,
    }
    record_path = (
        output_root / "build" / "reuse" / "reproduction_record.json"
        if reuse_data
        else output_root / "reproduction_record.json"
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if (
        require_all
        and config["_meta"]["mode"] == "full"
        and not reuse_data
    ):
        write_validation_status(
            output_root / "docs" / "validation_status.json",
            {
                "schema_version": 1,
                "status": validation["status"],
                "config_sha256": record["config_sha256"],
                "reproduction_record": "reproduction_record.json",
                "checks": validation,
            },
        )
    if validation["status"] != "passed":
        raise RuntimeError(
            "Scientific validation failed: "
            + "; ".join(str(item) for item in validation["failures"])
        )
    print(
        f"[reproduce] completed in {elapsed:.2f} s; "
        f"peak RSS {record['peak_rss_bytes'] / 1024**2:.1f} MiB",
        flush=True,
    )
    return record
