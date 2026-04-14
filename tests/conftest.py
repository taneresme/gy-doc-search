from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sample_docs_dir() -> Path:
    return Path(__file__).parent / "sample_docs"


@pytest.fixture
def default_profile() -> dict:
    return {
        "min_chunk_tokens": 100,
        "max_chunk_tokens": 800,
        "target_chunk_tokens": 600,
        "overlap_tokens": 100,
        "heading_levels": [1, 2, 3],
    }


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    ds_dir = tmp_path / ".doc-search"
    ds_dir.mkdir()
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    return tmp_path
