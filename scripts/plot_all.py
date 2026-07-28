#!/usr/bin/env python3
"""Regenerate figures from the archived numerical tables only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fsd_accelerating_waveforms.config import load_config, normalize_figure_ids
from fsd_accelerating_waveforms.plotting import plot_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "paper.yaml"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "figures")
    args = parser.parse_args()

    config = load_config(args.config, mode="full")
    for figure_id in normalize_figure_ids(args.figure):
        path = plot_figure(
            config,
            figure_id,
            args.data_root.resolve(),
            args.output_root.resolve(),
        )
        print(path)


if __name__ == "__main__":
    main()
