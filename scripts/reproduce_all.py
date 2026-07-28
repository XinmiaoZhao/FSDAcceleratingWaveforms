#!/usr/bin/env python3
"""Compute machine-readable tables and render paper figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fsd_accelerating_waveforms.config import load_config
from fsd_accelerating_waveforms.reproduction import run_reproduction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "quick"), default="full")
    parser.add_argument("--figure")
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help="validate archived tables and replot without recomputing them",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "paper.yaml"
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    output_root = args.output_root
    if output_root is None:
        output_root = ROOT if args.mode == "full" else ROOT / "build" / "quick"
    config = load_config(args.config, mode=args.mode)
    run_reproduction(
        config,
        config_path=args.config,
        output_root=output_root.resolve(),
        figure=args.figure,
        reuse_data=args.reuse_data,
    )


if __name__ == "__main__":
    main()
