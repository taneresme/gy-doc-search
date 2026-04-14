"""Configuration loading and project discovery."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from gy_doc_search.defaults import DEFAULTS


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping: {path}")
    return data


def load_defaults() -> dict:
    return copy.deepcopy(DEFAULTS)


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk up from the given path looking for .doc-search/config.yaml."""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    while current != current.parent:
        if (current / ".doc-search" / "config.yaml").exists():
            return current
        current = current.parent
    if (current / ".doc-search" / "config.yaml").exists():
        return current
    return None


def _user_config_path() -> Path:
    return Path.home() / ".config" / "gy-doc-search" / "config.yaml"


def _storage_dir(project_root: Path, config: dict) -> Path:
    dirname = config.get("storage", {}).get("index_dirname", ".index")
    return project_root / ".doc-search" / dirname


def load_config(
    cwd: Path | None = None,
    require_project: bool = False,
    user_config_path: Path | None = None,
) -> dict:
    """Merge defaults, user config, and project config."""
    config = load_defaults()

    user_path = user_config_path or _user_config_path()
    if user_path.exists():
        deep_merge(config, load_yaml(user_path))

    project_root = find_project_root(cwd)
    if project_root is not None:
        project_config_path = project_root / ".doc-search" / "config.yaml"
        deep_merge(config, load_yaml(project_config_path))
        config["_project_root"] = str(project_root)
        config["_doc_search_dir"] = str(project_root / ".doc-search")
        config["_storage_dir"] = str(_storage_dir(project_root, config))
        config["_chroma_dir"] = str(project_root / ".doc-search" / ".chroma")
        config["_index_state_path"] = str(
            project_root
            / ".doc-search"
            / config["storage"]["state_file"]
        )
    elif require_project:
        raise ConfigError(
            "No gy-doc-search project found. Run from within a directory that contains "
            ".doc-search/config.yaml, or initialize one with `gy-doc-search init`."
        )

    validate_config(config)
    return config


def resolve_source_entry(
    source: dict,
    project_root: Path,
    config: dict,
) -> dict:
    """Resolve a source entry to an absolute path and effective profile."""
    if "path" not in source or not source["path"]:
        raise ConfigError("Each source entry must define a non-empty `path`.")

    profile_name = source.get(
        "profile",
        config["chunking"].get("default_profile", "default"),
    )
    profiles = config["chunking"].get("profiles", {})
    if profile_name not in profiles:
        raise ConfigError(f"Unknown chunking profile `{profile_name}`.")

    abs_path = (project_root / source["path"]).resolve()
    return {
        **source,
        "recursive": source.get("recursive", True),
        "filter": source.get("filter", "*.md"),
        "profile": profile_name,
        "metadata_defaults": source.get("metadata_defaults", {}),
        "_abs_path": abs_path,
        "_profile_config": profiles[profile_name],
    }


def validate_config(config: dict) -> None:
    sources = config.get("sources", [])
    if sources is None:
        raise ConfigError("`sources` must be a list.")
    if not isinstance(sources, list):
        raise ConfigError("`sources` must be a list.")

    profiles = config.get("chunking", {}).get("profiles", {})
    default_profile = config.get("chunking", {}).get("default_profile", "default")
    if default_profile not in profiles:
        raise ConfigError(
            f"Default chunking profile `{default_profile}` is not defined."
        )

    workers = config.get("performance", {}).get("workers", 0)
    if not isinstance(workers, int) or workers < 0:
        raise ConfigError("`performance.workers` must be a non-negative integer (0 = auto).")

    if "_project_root" in config:
        project_root = Path(config["_project_root"])
        for source in sources:
            resolved = resolve_source_entry(source, project_root, config)
            if not resolved["_abs_path"].exists():
                raise ConfigError(
                    f"Configured source path does not exist: {source['path']}"
                )
