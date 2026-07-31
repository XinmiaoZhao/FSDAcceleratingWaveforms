#!/usr/bin/env python3
"""Verify that an installed Conda environment exactly matches an explicit lock."""

from __future__ import annotations

import argparse
from pathlib import Path
import string


def package_records(path: Path) -> set[str]:
    lines = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "@"))
    }
    if not lines:
        raise ValueError(f"explicit environment has no package records: {path}")
    for line in lines:
        if "#" not in line:
            raise ValueError(f"explicit environment lacks SHA256 records: {path}")
        url, digest = line.rsplit("#", 1)
        if (
            not url.startswith("https://")
            or len(digest) != 64
            or any(character not in string.hexdigits for character in digest)
        ):
            raise ValueError(f"invalid explicit package record in {path}: {line}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()
    expected = package_records(args.expected)
    actual = package_records(args.actual)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = [
            *(f"missing: {record}" for record in missing),
            *(f"unexpected: {record}" for record in unexpected),
        ]
        raise SystemExit("explicit environment mismatch:\n" + "\n".join(details))
    print(f"verified {len(expected)} exact Conda package records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
