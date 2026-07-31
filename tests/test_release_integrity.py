from __future__ import annotations

import gzip
import hashlib
import json
import re
import tomllib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.json"
EXCLUDED_PARTS = {
    ".git",
    ".ipynb_checkpoints",
    ".pytest_cache",
    "__pycache__",
    "build",
}
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(
            part in EXCLUDED_PARTS or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.is_symlink():
            raise AssertionError(f"release contains a symbolic link: {relative}")
        if path.is_file() and relative.as_posix() != MANIFEST.name:
            files[relative.as_posix()] = path
    return files


def text_content(path: Path) -> str | None:
    if path.name.endswith(".csv.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name == "LICENSE":
        return path.read_text(encoding="utf-8")
    return None


def test_manifest_matches_the_complete_release_tree() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in payload["files"]}
    actual = release_files()
    assert set(actual) == set(expected)
    assert payload["file_count"] == len(expected)
    for relative, path in actual.items():
        assert path.stat().st_size == expected[relative]["size_bytes"]
        assert sha256(path) == expected[relative]["sha256"]


def test_release_text_is_public_and_utf8() -> None:
    forbidden = (
        "/" + "Users/",
        "/" + "home/",
        "\\" + "Users\\",
        "GW19" + "0814",
        "GW" + "AccInference",
        "git." + "over" + "leaf.com",
        "Over" + "leaf",
        "docs/" + "reports",
        "run_" + "records",
        "results" + "/",
        "runs" + "/",
        "fig" + "S01",
        "chi" + "099",
        "supplement_" + "fsd_order",
        "[" + "XX]",
        "XXX" + "XXXX",
    )
    secret_patterns = (
        re.compile(
            "-----BEGIN "
            + "(?:RSA |OPENSSH |EC |DSA )?"
            + "PRIVATE "
            + "KEY-----"
        ),
        re.compile(r"\b" + "ghp" + r"_[A-Za-z0-9]{20,}\b"),
        re.compile(r"\b" + "github" + r"_pat_[A-Za-z0-9_]{20,}\b"),
        re.compile(r"\b" + "AKIA" + r"[0-9A-Z]{16}\b"),
    )
    for relative, path in release_files().items():
        text = text_content(path)
        if text is None:
            continue
        for token in forbidden:
            assert token not in text, (relative, token)
        for pattern in secret_patterns:
            assert pattern.search(text) is None, (relative, pattern.pattern)


def test_citation_and_zenodo_metadata_are_consistent() -> None:
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    cff_creators = [
        (
            f"{author['family-names']}, {author['given-names']}",
            author["orcid"].removeprefix("https://orcid.org/"),
        )
        for author in cff["authors"]
    ]
    zenodo_creators = [
        (creator["name"], creator["orcid"]) for creator in zenodo["creators"]
    ]
    assert cff_creators == zenodo_creators
    assert cff["version"] == zenodo["version"] == "1.0.0"
    assert cff["license"] == zenodo["license"] == "GPL-3.0-or-later"
    software_doi = "10.5281/zenodo.21643294"
    assert cff["doi"] == software_doi
    assert str(cff["date-released"]) == "2026-07-28"
    assert any(
        identifier["type"] == "doi"
        and identifier["value"] == software_doi
        for identifier in cff["identifiers"]
    )
    assert zenodo["language"] == "eng"
    assert any(
        contributor["name"] == "Chen, Xian"
        and contributor["type"] == "Supervisor"
        and contributor["orcid"] == "0000-0003-3950-9317"
        for contributor in zenodo["contributors"]
    )
    article_doi = "10.48550/arXiv.2604.00253"
    assert any(
        identifier["type"] == "doi" and identifier["value"] == article_doi
        for identifier in cff["identifiers"]
    )
    assert any(
        related["identifier"] == f"https://doi.org/{article_doi}"
        and related["relation"] == "isSupplementTo"
        and related["scheme"] == "doi"
        and related["resource_type"] == "publication-article"
        for related in zenodo["related_identifiers"]
    )


def test_code_data_and_figure_licenses_are_present() -> None:
    assert (ROOT / "LICENSE").read_bytes() == (
        ROOT / "LICENSES" / "GPL-3.0-or-later.txt"
    ).read_bytes()
    cc_by = (
        ROOT / "LICENSES" / "CC-BY-4.0.txt"
    ).read_text(encoding="utf-8")
    assert "Creative Commons Attribution 4.0" in cc_by
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "GPL-3.0-or-later" in readme
    assert "CC-BY-4.0" in readme


def test_pycbc_pkg_resources_compatibility_is_pinned() -> None:
    environment = yaml.safe_load(
        (ROOT / "environment.yml").read_text(encoding="utf-8")
    )
    assert "setuptools=78.1.1" in environment["dependencies"]
    assert "pycbc=2.9.0" in environment["dependencies"]
    pip_dependencies = next(
        item["pip"]
        for item in environment["dependencies"]
        if isinstance(item, dict) and "pip" in item
    )
    assert pip_dependencies == ["-e ."]
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert "setuptools>=78.1.1,<82" in project["project"]["dependencies"]


def test_platform_explicit_locks_are_complete_and_scientific() -> None:
    for platform_name, filename in (
        ("linux-64", "linux-64.conda.lock"),
        ("osx-arm64", "osx-arm64.conda.lock"),
    ):
        text = (ROOT / "environment-locks" / filename).read_text(
            encoding="utf-8"
        )
        records = [
            line
            for line in text.splitlines()
            if line and not line.startswith(("#", "@"))
        ]
        assert f"# platform: {platform_name}" in text
        assert len(records) > 200
        assert len(records) == len(set(records))
        assert all(line.startswith("https://conda.anaconda.org/conda-forge/") for line in records)
        assert all(len(line.rsplit("#", 1)[-1]) == 64 for line in records)
        assert any("/pycbc-2.9.0-" in line for line in records)
        assert any("/python-lalsimulation-6.2.0-" in line for line in records)


def test_notebook_execution_dependency_is_declared() -> None:
    environment = yaml.safe_load(
        (ROOT / "environment.yml").read_text(encoding="utf-8")
    )
    assert "nbconvert=7.17.1" in environment["dependencies"]
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert (
        "nbconvert>=7.17,<8"
        in project["project"]["optional-dependencies"]["notebook"]
    )


def test_phenomd_semantic_audit_passed() -> None:
    audit = json.loads(
        (ROOT / "docs" / "phenomd_semantic_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["status"] == "passed"
    assert audit["all_importable_library_asts_equal"] is True
    assert set(audit["files"]) == {
        "phenomd.py",
        "phenomd_deriv.py",
        "phenomd_utils.py",
    }
    for result in audit["files"].values():
        assert (
            result["before_library_ast_sha256"]
            == result["after_library_ast_sha256"]
        )
