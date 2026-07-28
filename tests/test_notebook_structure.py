from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_walkthrough_has_no_duplicate_waveform_implementation() -> None:
    notebook = json.loads(
        (ROOT / "notebooks" / "paper_figures_walkthrough.ipynb").read_text()
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert notebook["nbformat"] == 4
    cell_ids = [cell["id"] for cell in notebook["cells"]]
    assert len(cell_ids) == len(set(cell_ids))
    assert all(
        cell.get("execution_count") is None and cell.get("outputs") == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "## Goal" in source
    assert "## Setup" in source
    assert "## Steps" in source
    assert "## Checks" in source
    assert "## Next Steps" in source
    assert "fsd_waveform(" not in source
    assert "tds_waveform(" not in source
    assert "spa_pn_waveform(" not in source
    assert "build\" / \"notebook_figures" in source
    assert "fig" + "S01" not in source
    assert "higher" + "-order" not in source.lower()
    for figure_id in ("fig01", "fig02", "fig03", "fig04", "fig05", "fig06"):
        assert figure_id in source
    macos_home_prefix = "/".join(("", "Users", ""))
    assert macos_home_prefix not in source
