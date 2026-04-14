from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from gy_doc_search.cli import main
from gy_doc_search.eval import evaluate_cases, load_eval_cases


def _write_project(tmp_path: Path) -> Path:
    (tmp_path / ".gy-doc-search").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "payments.md").write_text(
        "# Payments\n\nAuthorization requests validate funds before capture.",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "settlement.md").write_text(
        "# Settlement\n\nBatch settlement closes the day and posts cleared records.",
        encoding="utf-8",
    )
    (tmp_path / ".gy-doc-search" / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "sources": [{"path": "./docs"}],
                "embedding": {"provider": "simple"},
                "storage": {"backend": "local"},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_load_eval_cases(sample_docs_dir: Path) -> None:
    cases = load_eval_cases(Path(__file__).parent / "sample_eval_cases.yaml")
    assert len(cases) == 2
    assert cases[0]["query"] == "authorization funds"


def test_evaluate_cases(tmp_path: Path) -> None:
    project = _write_project(tmp_path)
    cases_path = project / "eval_cases.yaml"
    cases_path.write_text((Path(__file__).parent / "sample_eval_cases.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    from gy_doc_search.config import load_config

    config = load_config(cwd=project, user_config_path=project / "missing-user-config.yaml")
    report = evaluate_cases(config, cases_path, rebuild_index=True)
    assert report["retrieval"]["cases"] == 2
    assert report["retrieval"]["primary_hit_rate"] >= 0.5


def test_eval_cli_json(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        project = _write_project(Path("."))
        cases_path = project / "eval_cases.yaml"
        cases_path.write_text(
            (Path(__file__).parent / "sample_eval_cases.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        result = runner.invoke(main, ["eval", "--cases", str(cases_path), "--json"], catch_exceptions=False)
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["retrieval"]["cases"] == 2
