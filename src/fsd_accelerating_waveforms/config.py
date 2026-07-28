"""Configuration loading and figure selection."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


FIGURE_IDS = ("fig01", "fig02", "fig03", "fig04", "fig05", "fig06")


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def load_config(path: str | Path, mode: str = "full") -> dict[str, Any]:
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    if mode not in {"full", "quick"}:
        raise ValueError("mode must be 'full' or 'quick'")
    if mode == "quick":
        quick = config.get("quick", {})
        for section, update in quick.items():
            if section not in config:
                config[section] = {}
            _deep_update(config[section], update)
    config["_meta"] = {"mode": mode, "config_path": str(path.resolve())}
    return config


def normalize_figure_ids(figure: str | None) -> tuple[str, ...]:
    if figure is None:
        return FIGURE_IDS
    if figure not in FIGURE_IDS:
        raise ValueError(f"Unknown figure {figure!r}; choose from {', '.join(FIGURE_IDS)}")
    return (figure,)
