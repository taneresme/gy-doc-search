from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from gy_doc_search.cli import main


def test_init_creates_project_files(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        result = runner.invoke(main, ["init"], catch_exceptions=False)
        assert result.exit_code == 0
        assert Path(".gy-doc-search/config.yaml").exists()
        config = Path(".gy-doc-search/config.yaml").read_text(encoding="utf-8")
        assert "./docs" in config


def test_init_with_claude_code(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        result = runner.invoke(main, ["init", "--with-claude-code"], catch_exceptions=False)
        assert result.exit_code == 0
        assert Path(".claude/mcp_servers.json").exists()


def test_init_with_sources(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        result = runner.invoke(main, ["init", "--sources", "./docs", "--sources", "./specs"], catch_exceptions=False)
        assert result.exit_code == 0
        config = Path(".gy-doc-search/config.yaml").read_text(encoding="utf-8")
        assert './docs' in config
        assert './specs' in config


def test_status_outside_project(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        result = runner.invoke(main, ["status"])
        assert result.exit_code != 0
        assert "No gy-doc-search project found" in result.output


def test_index_dry_run(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path(".gy-doc-search").mkdir()
        Path("docs").mkdir()
        Path("docs/guide.md").write_text("# Guide\n\nBody", encoding="utf-8")
        Path(".gy-doc-search/config.yaml").write_text(
            yaml.safe_dump({"sources": [{"path": "./docs"}], "embedding": {"provider": "simple"}}),
            encoding="utf-8",
        )
        result = runner.invoke(main, ["index", "--dry-run"], catch_exceptions=False)
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["new"] == ["docs/guide.md"]


def test_status_uses_legacy_project_dir(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path(".doc-search").mkdir()
        Path("docs").mkdir()
        Path(".doc-search/config.yaml").write_text(
            yaml.safe_dump({"sources": [{"path": "./docs"}], "embedding": {"provider": "simple"}}),
            encoding="utf-8",
        )
        result = runner.invoke(main, ["status"], catch_exceptions=False)
        assert result.exit_code == 0
