from __future__ import annotations

from pathlib import Path

import yaml

from gy_doc_search.config import load_config
from gy_doc_search.indexer import run_index
from gy_doc_search.searcher import get_file_content, list_sources, search


def _write_project_config(project_root: Path) -> None:
    config = {
        "sources": [{"path": "./docs"}],
        "embedding": {"provider": "simple"},
    }
    (project_root / ".gy-doc-search" / "config.yaml").write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )


def test_basic_search(tmp_project: Path) -> None:
    docs_dir = tmp_project / "docs"
    (docs_dir / "payments.md").write_text(
        "# Payments\n\nAuthorization requests validate funds before capture.",
        encoding="utf-8",
    )
    (docs_dir / "settlement.md").write_text(
        "# Settlement\n\nBatch settlement closes the day and posts cleared records.",
        encoding="utf-8",
    )
    _write_project_config(tmp_project)
    config = load_config(cwd=tmp_project)
    run_index(config)

    results = search("authorization funds", config, top_k=2, threshold=0.0)
    assert results
    assert results[0].source_file == "docs/payments.md"


def test_path_filtering(tmp_project: Path) -> None:
    api_dir = tmp_project / "docs" / "api"
    api_dir.mkdir(parents=True)
    (api_dir / "one.md").write_text("# API\n\nError handling reference.", encoding="utf-8")
    (tmp_project / "docs" / "guide.md").write_text("# Guide\n\nGeneral onboarding.", encoding="utf-8")
    _write_project_config(tmp_project)
    config = load_config(cwd=tmp_project)
    run_index(config)

    results = search("error handling", config, top_k=5, threshold=0.0, filter_path="docs/api/")
    assert results
    assert all(result.source_file.startswith("docs/api/") for result in results)


def test_threshold_filtering(tmp_project: Path) -> None:
    (tmp_project / "docs" / "guide.md").write_text("# Guide\n\nGeneral onboarding only.", encoding="utf-8")
    _write_project_config(tmp_project)
    config = load_config(cwd=tmp_project)
    run_index(config)

    results = search("unrelated concept", config, top_k=5, threshold=0.99)
    assert results == []


def test_get_file_content_and_list_sources(tmp_project: Path) -> None:
    target = tmp_project / "docs" / "guide.md"
    target.write_text("# Guide\n\nGeneral onboarding only.", encoding="utf-8")
    _write_project_config(tmp_project)
    config = load_config(cwd=tmp_project)
    run_index(config)

    assert "General onboarding" in get_file_content("docs/guide.md", str(tmp_project))
    assert list_sources(config) == ["docs/guide.md"]


def test_metadata_filtering(tmp_project: Path) -> None:
    (tmp_project / "docs" / "guide.md").write_text(
        "---\ndomain: payments\n---\n# Guide\n\nCard authorization details.",
        encoding="utf-8",
    )
    _write_project_config(tmp_project)
    config = load_config(cwd=tmp_project)
    run_index(config)

    results = search("authorization", config, threshold=0.0, filter_metadata={"fm_domain": "payments"})
    assert results
    assert results[0].metadata["front_matter"]["domain"] == "payments"
