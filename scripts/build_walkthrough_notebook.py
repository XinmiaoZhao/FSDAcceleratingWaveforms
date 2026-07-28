#!/usr/bin/env python3
"""Build the deterministic paper-figures walkthrough notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "paper_figures_walkthrough.ipynb"


def cell_id(cell_type: str, source: str) -> str:
    content = f"{cell_type}\0{source}".encode()
    return hashlib.sha256(content).hexdigest()[:12]


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id("markdown", source),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id("code", source),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    cells = [
        markdown(
            """# Paper figures walkthrough

## Goal

Inspect the six archived numerical tables, apply the public validation checks,
and reproduce every article figure with the same plotting functions used by
`scripts/plot_all.py`. The notebook contains no waveform implementation or
hidden calculation state.
"""
        ),
        markdown(
            """## Setup

Locate the repository root, import the table, plotting, and validation helpers,
and load the single parameter source `configs/paper.yaml`.
"""
        ),
        code(
            """from pathlib import Path
import sys

root = Path.cwd().resolve()
if not (root / "configs" / "paper.yaml").exists():
    root = root.parent
if not (root / "configs" / "paper.yaml").exists():
    raise FileNotFoundError("Run this notebook from the repository root or notebooks/")

sys.path.insert(0, str(root / "src"))

from fsd_accelerating_waveforms.config import FIGURE_IDS, load_config
from fsd_accelerating_waveforms.plotting import plot_figure
from fsd_accelerating_waveforms.tables import read_rows
from fsd_accelerating_waveforms.validation import validate_generated_data

config = load_config(root / "configs" / "paper.yaml", mode="full")
data_dir = root / "data"
figure_dir = root / "build" / "notebook_figures"
"""
        ),
        markdown(
            """## Steps

### 1. Inspect all six figure tables

The catalog below is derived from the configuration. Each table is read in the
same way and summarized by filename, row count, and field names.
"""
        ),
        code(
            """table_files = {
    "fig01": config["figures"]["fig01"]["data_file"],
    "fig02": config["mismatch"]["cases"]["fig02"]["data_file"],
    "fig03": config["mismatch"]["cases"]["fig03"]["data_file"],
    "fig04": config["mismatch"]["cases"]["fig04"]["data_file"],
    "fig05": config["fsd_order"]["data_file"],
    "fig06": config["fisher"]["data_file"],
}

table_inventory = []
for figure_id in FIGURE_IDS:
    path = data_dir / table_files[figure_id]
    rows = read_rows(path)
    table_inventory.append(
        {
            "figure": figure_id,
            "file": path.name,
            "rows": len(rows),
            "fields": tuple(rows[0]) if rows else (),
        }
    )

table_inventory
"""
        ),
        markdown(
            """### 2. Replot all six figures

Outputs are written under ignored `build/notebook_figures/`. The archived
release figures and numerical tables are not modified.
"""
        ),
        code(
            """replotted = {
    figure_id: plot_figure(config, figure_id, data_dir, figure_dir)
    for figure_id in FIGURE_IDS
}
replotted
"""
        ),
        markdown(
            """## Checks

Run the same artifact and scientific checks used by the command-line
reproduction workflow, then confirm that every configured figure was produced.
"""
        ),
        code(
            """validation = validate_generated_data(
    config,
    data_dir,
    figure_dir,
    require_all=True,
)
assert validation["status"] == "passed", validation["failures"]
assert set(replotted) == set(FIGURE_IDS)
assert all(path.exists() for path in replotted.values())
validation
"""
        ),
        markdown(
            """## Next Steps

Use `python scripts/plot_all.py` for a direct table-to-figure run, or
`python scripts/reproduce_all.py --mode full` to recompute the tables before
plotting. Release provenance, dependency versions, runtime, memory use, and
hashes are recorded in `reproduction_record.json`.
"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
