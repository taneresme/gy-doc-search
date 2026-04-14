from __future__ import annotations

from pathlib import Path

from gy_doc_search.chunker import (
    apply_sizing,
    chunk_file,
    chunk_sources,
    extract_front_matter,
    flatten_with_ancestry,
    parse_headings,
)


def test_extract_front_matter(sample_docs_dir: Path) -> None:
    text = (sample_docs_dir / "with_frontmatter.md").read_text(encoding="utf-8")
    front_matter, body = extract_front_matter(text)
    assert front_matter["domain"] == "payments"
    assert "# Settlement" in body


def test_parse_headings_ignores_code_blocks(sample_docs_dir: Path) -> None:
    text = (sample_docs_dir / "with_code_blocks.md").read_text(encoding="utf-8")
    sections = parse_headings(text, [1, 2, 3])
    flat = flatten_with_ancestry(sections)
    assert len(flat) == 2
    assert flat[0]["heading_path"] == "Code Samples"
    assert flat[1]["heading_path"] == "Code Samples > Real Section"


def test_basic_heading_splitting(sample_docs_dir: Path, default_profile: dict) -> None:
    profile = dict(default_profile)
    profile["min_chunk_tokens"] = 5
    chunks = chunk_file(
        str(sample_docs_dir / "simple.md"),
        str(sample_docs_dir),
        profile,
    )
    assert len(chunks) >= 2
    assert chunks[0].source_file == "simple.md"
    assert chunks[0].heading_path.startswith("Payment API")


def test_heading_ancestry(sample_docs_dir: Path, default_profile: dict) -> None:
    profile = dict(default_profile)
    profile["min_chunk_tokens"] = 10
    chunks = chunk_file(
        str(sample_docs_dir / "nested_headings.md"),
        str(sample_docs_dir),
        profile,
    )
    assert any("Root Topic > Secondary Topic > Tertiary Topic" in c.heading_path for c in chunks)


def test_small_chunk_merging(default_profile: dict) -> None:
    profile = dict(default_profile)
    profile["min_chunk_tokens"] = 30
    sections = [
        {"heading_path": "One", "content": "word " * 20, "deepest_level": 2},
        {"heading_path": "Two", "content": "word " * 30, "deepest_level": 2},
        {"heading_path": "Three", "content": "word " * 120, "deepest_level": 2},
    ]
    merged = apply_sizing(sections, profile)
    assert len(merged) == 2


def test_large_chunk_splitting(sample_docs_dir: Path) -> None:
    profile = {
        "min_chunk_tokens": 20,
        "max_chunk_tokens": 80,
        "target_chunk_tokens": 60,
        "overlap_tokens": 10,
        "heading_levels": [1, 2, 3],
    }
    chunks = chunk_file(
        str(sample_docs_dir / "large_sections.md"),
        str(sample_docs_dir),
        profile,
    )
    assert len(chunks) >= 3


def test_no_headings(sample_docs_dir: Path, default_profile: dict) -> None:
    chunks = chunk_file(
        str(sample_docs_dir / "no_headings.md"),
        str(sample_docs_dir),
        default_profile,
    )
    assert len(chunks) == 1
    assert chunks[0].heading_path == "no_headings.md"


def test_empty_file(sample_docs_dir: Path, default_profile: dict) -> None:
    chunks = chunk_file(
        str(sample_docs_dir / "empty.md"),
        str(sample_docs_dir),
        default_profile,
    )
    assert chunks == []


def test_unicode_content(sample_docs_dir: Path, default_profile: dict) -> None:
    chunks = chunk_file(
        str(sample_docs_dir / "unicode_content.md"),
        str(sample_docs_dir),
        default_profile,
    )
    assert len(chunks) == 1
    assert "日本語" in chunks[0].content


def test_source_metadata_applied(tmp_project: Path) -> None:
    docs_dir = tmp_project / "docs"
    (docs_dir / "a.md").write_text("# A\n\ncontent " * 50, encoding="utf-8")
    sources = [
        {
            "path": "./docs",
            "metadata_defaults": {"domain": "payments"},
            "profile": "default",
        }
    ]
    profiles = {
        "default": {
            "min_chunk_tokens": 10,
            "max_chunk_tokens": 500,
            "target_chunk_tokens": 100,
            "overlap_tokens": 10,
            "heading_levels": [1, 2, 3],
        }
    }
    chunks = chunk_sources(sources, str(tmp_project), profiles)
    assert len(chunks) == 1
    assert chunks[0].source_metadata["domain"] == "payments"
