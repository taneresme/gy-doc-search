"""CLI entrypoint for gy-doc-search."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click
from jinja2 import Environment, FileSystemLoader

from gy_doc_search import __version__
from gy_doc_search.config import ConfigError, load_config
from gy_doc_search.eval import evaluate_cases, format_eval_report, report_to_json
from gy_doc_search.indexer import clean_index, get_stats, inspect_index_changes, run_index, verify_index
from gy_doc_search.searcher import format_results, get_file_content, list_sources, results_to_json, search


def _template_env() -> Environment:
    template_dir = Path(__file__).parent / "templates"
    return Environment(loader=FileSystemLoader(str(template_dir)), autoescape=False)


def render_template(name: str, **context: object) -> str:
    return _template_env().get_template(name).render(**context)


def _ensure_project_loaded() -> dict:
    return load_config(require_project=True)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def setup_claude_code(project_root: Path) -> None:
    claude_dir = project_root / ".claude"
    claude_dir.mkdir(exist_ok=True)
    config_path = claude_dir / "mcp_servers.json"
    payload = {}
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["gy-doc-search"] = {"command": "gy-doc-search", "args": ["serve"]}
    _write_text(config_path, json.dumps(payload, indent=2))


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Semantic search over project documentation."""


@main.command()
@click.option("--sources", multiple=True, help="Source directories or files to index.")
@click.option("--with-claude-code", is_flag=True, help="Set up Claude Code MCP integration.")
def init(sources: tuple[str, ...], with_claude_code: bool) -> None:
    """Initialize gy-doc-search in the current project."""
    project_root = Path.cwd()
    doc_search_dir = project_root / ".doc-search"
    doc_search_dir.mkdir(exist_ok=True)

    source_entries = [{"path": path} for path in (sources or ("./docs",))]
    config_text = render_template("config.yaml.j2", sources=source_entries)
    _write_text(doc_search_dir / "config.yaml", config_text)
    _write_text(doc_search_dir / ".gitignore", render_template("gitignore.j2"))
    click.echo("Created .doc-search/config.yaml")
    click.echo("Created .doc-search/.gitignore")

    if with_claude_code:
        setup_claude_code(project_root)
        click.echo("Created .claude/mcp_servers.json")
        claude_md = render_template("claude_md.j2", sources=source_entries)
        _write_text(project_root / "CLAUDE.md", claude_md)
        click.echo("Created CLAUDE.md")

    click.echo("Run `gy-doc-search index` to build the search index.")


@main.command()
@click.option("--incremental", is_flag=True, help="Only process changed and new files.")
@click.option("--dry-run", is_flag=True, help="Show what would be indexed.")
@click.option("--force", is_flag=True, help="Force a full reindex.")
def index(incremental: bool, dry_run: bool, force: bool) -> None:
    """Index configured documentation sources."""
    try:
        config = _ensure_project_loaded()
    except ConfigError as exc:
        raise click.ClickException(str(exc))

    if dry_run:
        changes = inspect_index_changes(config)
        click.echo(json.dumps(changes, indent=2))
        return

    click.echo("loading config")
    mode = "incremental" if incremental and not force else "full"
    click.echo(f"starting {mode} index")
    stats = run_index(
        config,
        incremental=incremental and not force,
        reporter=click.echo,
    )
    click.echo(
        f"Indexed {stats.total_files} files into {stats.total_chunks} chunks "
        f"using {stats.embedding_model}."
    )


@main.command()
@click.argument("query")
@click.option("--top-k", type=int, default=None, help="Number of results.")
@click.option("--threshold", type=float, default=None, help="Minimum score.")
@click.option("--path", "filter_path", default=None, help="Path prefix filter.")
@click.option("--files-only", is_flag=True, help="Show matching file paths only.")
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON.")
def query(query: str, top_k: int | None, threshold: float | None, filter_path: str | None, files_only: bool, as_json: bool) -> None:
    """Search indexed documentation."""
    try:
        config = _ensure_project_loaded()
    except ConfigError as exc:
        raise click.ClickException(str(exc))

    results = search(
        query=query,
        config=config,
        top_k=top_k,
        threshold=threshold,
        filter_path=filter_path,
    )
    effective_top_k = top_k or config["retrieval"]["default_top_k"]
    effective_threshold = threshold if threshold is not None else config["retrieval"]["similarity_threshold"]

    if files_only:
        for path in sorted({result.source_file for result in results}):
            click.echo(path)
        return
    if as_json:
        click.echo(results_to_json(results))
        return
    click.echo(format_results(results, query, int(effective_top_k), float(effective_threshold)))


@main.command(name="get")
@click.argument("file_path")
def get_cmd(file_path: str) -> None:
    """Read a full source file from disk."""
    try:
        config = _ensure_project_loaded()
    except ConfigError as exc:
        raise click.ClickException(str(exc))
    click.echo(get_file_content(file_path, config["_project_root"]))


@main.command(name="list")
@click.option("--prefix", default=None, help="Optional path prefix filter.")
def list_cmd(prefix: str | None) -> None:
    """List indexed source files."""
    try:
        config = _ensure_project_loaded()
    except ConfigError as exc:
        raise click.ClickException(str(exc))
    for path in list_sources(config, prefix):
        click.echo(path)


@main.command()
def status() -> None:
    """Show project index status."""
    try:
        config = _ensure_project_loaded()
    except ConfigError as exc:
        raise click.ClickException(str(exc))
    stats = get_stats(config)
    click.echo(
        json.dumps(
            stats.__dict__,
            indent=2,
            ensure_ascii=False,
        )
    )


@main.command()
def clean() -> None:
    """Remove stored index data but keep configuration."""
    try:
        config = _ensure_project_loaded()
    except ConfigError as exc:
        raise click.ClickException(str(exc))

    clean_index(config)
    storage_dir = Path(config["_storage_dir"])
    if storage_dir.exists():
        shutil.rmtree(storage_dir, ignore_errors=True)
    click.echo("Index data removed.")


@main.command()
def verify() -> None:
    """Verify that indexed source files still exist."""
    try:
        config = _ensure_project_loaded()
    except ConfigError as exc:
        raise click.ClickException(str(exc))

    missing = verify_index(config)
    if missing:
        for path in missing:
            click.echo(path)
        raise click.ClickException(f"{len(missing)} indexed files are missing.")
    click.echo("Index is valid.")


@main.command()
@click.option("--cases", "cases_path", required=True, help="Path to YAML eval cases.")
@click.option("--top-k", type=int, default=None, help="Override top-k for all eval queries.")
@click.option("--skip-index", is_flag=True, help="Reuse the current index instead of rebuilding.")
@click.option("--json", "as_json", is_flag=True, help="Output evaluation report as JSON.")
def eval(cases_path: str, top_k: int | None, skip_index: bool, as_json: bool) -> None:
    """Run an offline retrieval evaluation against labeled queries."""
    progress = None if as_json else click.echo
    if progress is not None:
        progress("loading config")
    try:
        config = _ensure_project_loaded()
    except ConfigError as exc:
        raise click.ClickException(str(exc))

    report = evaluate_cases(
        config=config,
        cases_path=cases_path,
        top_k=top_k,
        rebuild_index=not skip_index,
        reporter=progress,
    )
    if as_json:
        click.echo(report_to_json(report))
        return
    click.echo(format_eval_report(report))


@main.command()
@click.option("--transport", type=click.Choice(["stdio", "sse"]), default="stdio")
@click.option("--port", type=int, default=8080)
def serve(transport: str, port: int) -> None:
    """Start the MCP server."""
    from gy_doc_search.mcp_server import run_server

    run_server(transport=transport, port=port)
