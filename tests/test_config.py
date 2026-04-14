from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gy_doc_search.config import ConfigError, find_project_root, load_config, load_defaults


def test_default_loading(tmp_path: Path) -> None:
    config = load_config(cwd=tmp_path, user_config_path=tmp_path / "missing-user-config.yaml")
    assert config["chunking"]["default_profile"] == "default"
    assert config["sources"] == [{"path": "./docs", "recursive": True, "filter": "*.md"}]


def test_user_config_merge(tmp_path: Path) -> None:
    user_config = tmp_path / "user-config.yaml"
    user_config.write_text(
        yaml.safe_dump({"retrieval": {"default_top_k": 9}}),
        encoding="utf-8",
    )
    config = load_config(cwd=tmp_path, user_config_path=user_config)
    assert config["retrieval"]["default_top_k"] == 9


def test_project_config_merge(tmp_project: Path) -> None:
    (tmp_project / ".doc-search" / "config.yaml").write_text(
        yaml.safe_dump({"sources": [{"path": "./docs"}], "retrieval": {"default_top_k": 3}}),
        encoding="utf-8",
    )
    config = load_config(cwd=tmp_project, user_config_path=tmp_project / "missing-user-config.yaml")
    assert config["retrieval"]["default_top_k"] == 3
    assert config["_project_root"] == str(tmp_project)


def test_project_root_discovery(tmp_project: Path) -> None:
    nested = tmp_project / "docs" / "nested"
    nested.mkdir()
    (tmp_project / ".doc-search" / "config.yaml").write_text(
        yaml.safe_dump({"sources": [{"path": "./docs"}]}),
        encoding="utf-8",
    )
    assert find_project_root(nested) == tmp_project


def test_missing_project_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(cwd=tmp_path, require_project=True, user_config_path=tmp_path / "missing-user-config.yaml")


def test_invalid_source_path_raises(tmp_project: Path) -> None:
    (tmp_project / ".doc-search" / "config.yaml").write_text(
        yaml.safe_dump({"sources": [{"path": "./missing"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(cwd=tmp_project, user_config_path=tmp_project / "missing-user-config.yaml")


def test_profile_resolution(tmp_project: Path) -> None:
    config_data = load_defaults()
    config_data["sources"] = [{"path": "./docs", "profile": "default"}]
    (tmp_project / ".doc-search" / "config.yaml").write_text(
        yaml.safe_dump(config_data),
        encoding="utf-8",
    )
    config = load_config(cwd=tmp_project, user_config_path=tmp_project / "missing-user-config.yaml")
    assert config["sources"][0]["profile"] == "default"
