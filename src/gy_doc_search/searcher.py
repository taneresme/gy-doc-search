"""Search functionality for indexed documentation."""

from __future__ import annotations

import json
from pathlib import Path

from gy_doc_search.embedder import create_embedder
from gy_doc_search.models import SearchResult
from gy_doc_search.storage import open_store


def _build_where_filter(filter_path: str | None, filter_metadata: dict | None) -> dict | None:
    where: dict = {}
    if filter_path:
        where["__source_file_prefix"] = filter_path
    for key, value in (filter_metadata or {}).items():
        where[key] = value
    return where or None


def _metadata_from_record(metadata: dict) -> dict:
    front_matter = {
        key[3:]: value for key, value in metadata.items() if key.startswith("fm_")
    }
    source_metadata = {
        key[3:]: value for key, value in metadata.items() if key.startswith("sm_")
    }
    other = {
        key: value
        for key, value in metadata.items()
        if not key.startswith("fm_") and not key.startswith("sm_")
    }
    return {
        **other,
        "front_matter": front_matter,
        "source_metadata": source_metadata,
    }


def _parse_vector_results(raw_results: dict) -> list[SearchResult]:
    ids = raw_results.get("ids", [[]])[0]
    docs = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]
    results: list[SearchResult] = []
    for chunk_id, document, metadata, distance in zip(ids, docs, metadatas, distances):
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        results.append(
            SearchResult(
                chunk_id=chunk_id,
                source_file=metadata.get("source_file", ""),
                heading_path=metadata.get("heading_path", ""),
                content=document,
                score=score,
                word_count=int(metadata.get("word_count", 0)),
                metadata=_metadata_from_record(metadata),
            )
        )
    return results


_bm25_cache: dict[tuple, tuple] = {}


def _maybe_hybrid_search(
    query: str,
    config: dict,
    store,
    where_filter: dict | None,
    vector_results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    if not config["retrieval"].get("hybrid_search"):
        return vector_results
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        return vector_results

    cache_key = (str(getattr(store, "documents_path", id(store))), repr(where_filter), store.count())
    if cache_key not in _bm25_cache:
        records = store.get(where=where_filter, include=["documents", "metadatas"])
        tokenized = [document.lower().split() for document in records.get("documents", [])]
        if not tokenized:
            return vector_results
        _bm25_cache[cache_key] = (BM25Okapi(tokenized), records)
        if len(_bm25_cache) > 10:
            _bm25_cache.clear()
    bm25, records = _bm25_cache[cache_key]
    scores = bm25.get_scores(query.lower().split())
    bm25_ranked = []
    for index, score in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[: top_k * 4]:
        metadata = records["metadatas"][index]
        bm25_ranked.append(
            SearchResult(
                chunk_id=records["ids"][index],
                source_file=metadata.get("source_file", ""),
                heading_path=metadata.get("heading_path", ""),
                content=records["documents"][index],
                score=float(score),
                word_count=int(metadata.get("word_count", 0)),
                metadata=_metadata_from_record(metadata),
            )
        )

    fused: dict[str, float] = {}
    by_id: dict[str, SearchResult] = {}
    for rank, result in enumerate(vector_results):
        fused[result.chunk_id] = fused.get(result.chunk_id, 0.0) + 1.0 / (60 + rank + 1)
        by_id[result.chunk_id] = result
    for rank, result in enumerate(bm25_ranked):
        fused[result.chunk_id] = fused.get(result.chunk_id, 0.0) + 1.0 / (60 + rank + 1)
        by_id.setdefault(result.chunk_id, result)

    merged = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    results = []
    for chunk_id, fused_score in merged[: top_k * 2]:
        result = by_id[chunk_id]
        result.score = max(result.score, min(1.0, fused_score * 10))
        results.append(result)
    return results


def _maybe_rerank(query: str, config: dict, results: list[SearchResult], top_k: int) -> list[SearchResult]:
    if not config["retrieval"].get("reranking") or not results:
        return results
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return results
    model = CrossEncoder(config["retrieval"]["rerank_model"])
    scores = model.predict([(query, result.content) for result in results])
    ranked = sorted(zip(results, scores), key=lambda item: item[1], reverse=True)
    return [result for result, _ in ranked[:top_k]]


def search(
    query: str,
    config: dict,
    top_k: int | None = None,
    threshold: float | None = None,
    filter_path: str | None = None,
    filter_metadata: dict | None = None,
) -> list[SearchResult]:
    top_k = top_k or int(config["retrieval"]["default_top_k"])
    top_k = min(top_k, int(config["retrieval"]["max_top_k"]))
    threshold = (
        float(threshold)
        if threshold is not None
        else float(config["retrieval"]["similarity_threshold"])
    )

    embedder = create_embedder(config)
    store = open_store(config)
    where_filter = _build_where_filter(filter_path, filter_metadata)
    query_embedding = [embedder.embed_query(query)]
    vector_raw = store.query(
        query_embeddings=query_embedding,
        n_results=max(top_k * 4, top_k),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )
    vector_results = _parse_vector_results(vector_raw)
    results = _maybe_hybrid_search(query, config, store, where_filter, vector_results, top_k)
    results = _maybe_rerank(query, config, results, top_k)
    return [result for result in results if result.score >= threshold][:top_k]


def get_file_content(file_path: str, project_root: str) -> str:
    path = (Path(project_root) / file_path).resolve()
    return path.read_text(encoding="utf-8")


def list_sources(config: dict, prefix: str | None = None) -> list[str]:
    store = open_store(config)
    records = store.get(include=["metadatas"])
    paths = sorted({metadata.get("source_file", "") for metadata in records.get("metadatas", []) if metadata.get("source_file")})
    if prefix:
        return [path for path in paths if path.startswith(prefix)]
    return paths


def format_results(
    results: list[SearchResult],
    query: str,
    top_k: int,
    threshold: float,
) -> str:
    if not results:
        return f'SEARCH RESULTS for: "{query}"\nNo results matched the threshold of {threshold:.2f}.'

    lines = [
        "══════════════════════════════════════════════════════════",
        f'SEARCH RESULTS for: "{query}"',
        f"Top {min(top_k, len(results))} results | Threshold: {threshold:.2f}",
        "══════════════════════════════════════════════════════════",
        "",
    ]
    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                f"── Result {index} ─── Score: {result.score:.2f} ──────────────────────────────",
                f"Source: {result.source_file}",
                f"Section: {result.heading_path}",
                f"Words: {result.word_count}",
                "",
                result.content,
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def results_to_json(results: list[SearchResult]) -> str:
    return json.dumps([result.__dict__ for result in results], indent=2, ensure_ascii=False)
