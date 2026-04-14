"""MCP server entrypoint."""

from __future__ import annotations

import json

from gy_doc_search.config import load_config
from gy_doc_search.indexer import get_stats
from gy_doc_search.searcher import format_results, get_file_content, list_sources, search


def create_app():
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "MCP support is not installed. Install `gy-doc-search[mcp]` to use `gy-doc-search serve`."
        ) from exc

    config = load_config(require_project=True)
    app = FastMCP(config["mcp"]["server_name"])

    @app.tool()
    def search_docs(query: str, top_k: int = 5, filter_path: str | None = None) -> str:
        results = search(query, config, top_k=top_k, filter_path=filter_path)
        threshold = config["retrieval"]["similarity_threshold"]
        return format_results(results, query, top_k, threshold)

    @app.tool()
    def get_full_doc(file_path: str) -> str:
        return get_file_content(file_path, config["_project_root"])

    @app.tool()
    def list_doc_sources(prefix: str | None = None) -> str:
        return "\n".join(list_sources(config, prefix))

    @app.tool()
    def doc_index_stats() -> str:
        stats = get_stats(config)
        return json.dumps(stats.__dict__, indent=2, ensure_ascii=False)

    return app


def run_server(transport: str = "stdio", port: int = 8080) -> None:
    app = create_app()
    if transport == "stdio":
        app.run(transport="stdio")
    else:
        app.run(transport="sse", port=port)
