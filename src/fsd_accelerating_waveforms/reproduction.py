"""Orchestrate data generation, plotting, and reproducibility records."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
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
    "lal",
    "lalsimulation",
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


def _command_output(*command: str, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            list(command),
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _explicit_records(text: str) -> list[str]:
    return sorted(
        {
            line.strip()
            for line in text.splitlines()
            if line.strip()
            and not line.lstrip().startswith(("#", "@"))
        }
    )


def _record_set_sha256(records: list[str]) -> str:
    payload = json.dumps(
        sorted(records),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _environment_lock_verification(
    lock_path: Path,
    actual_explicit: str | None,
) -> dict[str, object]:
    if not lock_path.is_file():
        return {
            "status": "lock_missing",
            "exact_match": False,
            "expected_package_count": 0,
            "observed_package_count": 0,
        }
    lock_text = lock_path.read_text(encoding="utf-8")
    if "@EXPLICIT" not in lock_text:
        return {
            "status": "not_explicit",
            "exact_match": False,
            "expected_package_count": 0,
            "observed_package_count": 0,
        }
    expected = _explicit_records(lock_text)
    actual = _explicit_records(actual_explicit or "")
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    exact_match = bool(expected and actual and not missing and not unexpected)
    return {
        "status": (
            "verified"
            if exact_match
            else "unavailable"
            if actual_explicit is None
            else "mismatch"
        ),
        "exact_match": exact_match,
        "expected_package_count": len(expected),
        "observed_package_count": len(actual),
        "expected_records_sha256": _record_set_sha256(expected),
        "observed_records_sha256": _record_set_sha256(actual),
        "missing_package_count": len(missing),
        "unexpected_package_count": len(unexpected),
        "missing_records_sha256": _record_set_sha256(missing),
        "unexpected_records_sha256": _record_set_sha256(unexpected),
    }


def build_provenance(project_root: Path) -> dict[str, object]:
    import numpy as np

    commit = os.environ.get("FSD_SOURCE_COMMIT") or _command_output(
        "git", "rev-parse", "HEAD", cwd=project_root
    )
    tag = _command_output(
        "git", "describe", "--tags", "--exact-match", "HEAD", cwd=project_root
    )
    status = _command_output(
        "git", "status", "--porcelain=v1", cwd=project_root
    )
    raw_lock = os.environ.get("FSD_ENVIRONMENT_LOCK")
    if raw_lock:
        lock_path = Path(raw_lock)
        if not lock_path.is_absolute():
            lock_path = (project_root / lock_path).resolve()
    else:
        if sys.platform == "darwin" and platform.machine() == "arm64":
            platform_lock = (
                project_root / "environment-locks/osx-arm64.conda.lock"
            )
        elif sys.platform == "linux" and platform.machine() in {
            "x86_64",
            "amd64",
        }:
            platform_lock = (
                project_root / "environment-locks/linux-64.conda.lock"
            )
        else:
            platform_lock = project_root / "environment.yml"
        lock_path = platform_lock
    numpy_build = np.__config__.show(mode="dicts").get(
        "Build Dependencies", {}
    )
    numerical_libraries = {
        name: {
            key: value
            for key, value in details.items()
            if key in {"name", "version", "found", "openblas configuration"}
        }
        for name, details in numpy_build.items()
        if name in {"blas", "lapack"}
    }
    conda_explicit = _command_output(
        "conda",
        "list",
        "--explicit",
        "--sha256",
    )
    lock_verification = _environment_lock_verification(
        lock_path,
        conda_explicit,
    )
    conda_packages = []
    conda_json = _command_output("conda", "list", "--json")
    if conda_json:
        selected = {
            "python",
            "numpy",
            "scipy",
            "fftw",
            "libblas",
            "liblapack",
            "liblal",
            "liblalsimulation",
            "lalsuite",
            "pycbc",
            "python-lal",
            "python-lalsimulation",
        }
        conda_packages = [
            {
                key: row.get(key)
                for key in ("name", "version", "build_string", "channel")
            }
            for row in json.loads(conda_json)
            if row.get("name") in selected
        ]
    return {
        "source": {
            "commit": commit,
            "exact_tag": tag,
            "dirty": None if status is None else bool(status),
        },
        "environment_lock": {
            "path": lock_path.name,
            "sha256": sha256(lock_path) if lock_path.is_file() else None,
            "verification": lock_verification,
        },
        "cpu": {
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "numerical_libraries": numerical_libraries,
        "conda_builds": conda_packages,
        "fft_backend": "numpy-pocketfft/PyCBC",
    }


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
    project_root = Path(__file__).resolve().parents[2]
    provenance = build_provenance(project_root)
    require_exact_lock = os.environ.get(
        "FSD_REQUIRE_EXACT_LOCK", ""
    ).strip().lower() in {"1", "true", "yes"}
    lock_verification = provenance["environment_lock"]["verification"]
    if require_exact_lock and not lock_verification["exact_match"]:
        raise RuntimeError(
            "FSD_REQUIRE_EXACT_LOCK is enabled but the active Conda "
            f"environment is {lock_verification['status']}"
        )

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
        "schema_version": 2,
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
        "build_provenance": provenance,
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
                "schema_version": 2,
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
