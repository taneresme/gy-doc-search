"""Offline evaluation helpers for retrieval and chunking tuning."""

from __future__ import annotations

import json
import math
import resource
import sys
import time
from pathlib import Path

import yaml

from gy_doc_search.indexer import get_stats, run_index
from gy_doc_search.searcher import search
from gy_doc_search.storage import open_store


def _emit(reporter, message: str) -> None:
    if reporter is not None:
        reporter(message)


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(value / (1024 * 1024), 2)
    return round(value / 1024, 2)


def _safe_round(value: float) -> float:
    return round(value, 4)


def load_eval_cases(path: str | Path) -> list[dict]:
    eval_path = Path(path)
    data = yaml.safe_load(eval_path.read_text(encoding="utf-8")) or []
    if isinstance(data, dict):
        data = data.get("cases", [])
    if not isinstance(data, list):
        raise ValueError("Evaluation cases file must contain a list or a `cases:` mapping.")

    cases: list[dict] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Eval case {index} must be a mapping.")
        query = item.get("query")
        if not query or not isinstance(query, str):
            raise ValueError(f"Eval case {index} is missing a string `query`.")
        relevant_files = item.get("relevant_files", []) or []
        relevant_headings = item.get("relevant_headings", []) or []
        if not isinstance(relevant_files, list) or not all(
            isinstance(value, str) for value in relevant_files
        ):
            raise ValueError(f"Eval case {index} has invalid `relevant_files`.")
        if not isinstance(relevant_headings, list) or not all(
            isinstance(value, str) for value in relevant_headings
        ):
            raise ValueError(f"Eval case {index} has invalid `relevant_headings`.")
        cases.append(
            {
                "id": item.get("id", f"case-{index}"),
                "query": query,
                "relevant_files": relevant_files,
                "relevant_headings": relevant_headings,
                "filter_path": item.get("filter_path"),
                "top_k": item.get("top_k"),
                "threshold": item.get("threshold"),
                "notes": item.get("notes", ""),
            }
        )
    return cases


def _reciprocal_rank(results, candidates: set[str], attr: str) -> tuple[float, str | None]:
    if not candidates:
        return 0.0, None
    for index, result in enumerate(results, start=1):
        if getattr(result, attr) in candidates:
            return 1.0 / index, getattr(result, attr)
    return 0.0, None


def _dcg(results, candidates: set[str], attr: str, k: int) -> float:
    value = 0.0
    for index, result in enumerate(results[:k], start=1):
        if getattr(result, attr) in candidates:
            value += 1.0 / math.log2(index + 1)
    return value


def evaluate_cases(
    config: dict,
    cases_path: str | Path,
    top_k: int | None = None,
    rebuild_index: bool = True,
    reporter=None,
) -> dict:
    _emit(reporter, "loading evaluation cases")
    cases = load_eval_cases(cases_path)
    if not cases:
        raise ValueError("No evaluation cases were found.")

    index_duration_sec = 0.0
    index_stats = get_stats(config)
    if rebuild_index:
        _emit(reporter, "rebuilding index for evaluation")
        index_started = time.perf_counter()
        index_stats = run_index(config, incremental=False, reporter=reporter)
        index_duration_sec = time.perf_counter() - index_started

    store = open_store(config)
    _emit(reporter, f"evaluating {len(cases)} queries")

    case_results: list[dict] = []
    total_file_hits = 0
    total_heading_hits = 0
    total_primary_hits = 0
    total_file_rr = 0.0
    total_heading_rr = 0.0
    total_primary_rr = 0.0
    total_ndcg = 0.0
    total_latency_ms = 0.0

    for case in cases:
        case_top_k = int(case["top_k"] or top_k or config["retrieval"]["default_top_k"])
        query_started = time.perf_counter()
        results = search(
            query=case["query"],
            config=config,
            top_k=case_top_k,
            threshold=case["threshold"],
            filter_path=case["filter_path"],
        )
        latency_ms = (time.perf_counter() - query_started) * 1000
        total_latency_ms += latency_ms

        relevant_files = set(case["relevant_files"])
        relevant_headings = set(case["relevant_headings"])
        file_rr, matched_file = _reciprocal_rank(results, relevant_files, "source_file")
        heading_rr, matched_heading = _reciprocal_rank(results, relevant_headings, "heading_path")
        primary_rr = heading_rr if relevant_headings else file_rr
        primary_match_type = "heading" if relevant_headings else "file"
        primary_match_value = matched_heading if relevant_headings else matched_file
        dcg = _dcg(results, relevant_headings or relevant_files, "heading_path" if relevant_headings else "source_file", case_top_k)
        ideal_hits = len(relevant_headings or relevant_files)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(ideal_hits, case_top_k) + 1)) or 1.0
        ndcg = dcg / ideal_dcg

        total_file_hits += int(file_rr > 0)
        total_heading_hits += int(heading_rr > 0)
        total_primary_hits += int(primary_rr > 0)
        total_file_rr += file_rr
        total_heading_rr += heading_rr
        total_primary_rr += primary_rr
        total_ndcg += ndcg

        case_results.append(
            {
                "id": case["id"],
                "query": case["query"],
                "top_k": case_top_k,
                "latency_ms": _safe_round(latency_ms),
                "file_hit": file_rr > 0,
                "heading_hit": heading_rr > 0,
                "primary_hit": primary_rr > 0,
                "file_reciprocal_rank": _safe_round(file_rr),
                "heading_reciprocal_rank": _safe_round(heading_rr),
                "primary_reciprocal_rank": _safe_round(primary_rr),
                "ndcg_at_k": _safe_round(ndcg),
                "matched_file": matched_file,
                "matched_heading": matched_heading,
                "primary_match_type": primary_match_type,
                "primary_match_value": primary_match_value,
                "result_files": [result.source_file for result in results],
                "result_headings": [result.heading_path for result in results],
                "notes": case["notes"],
            }
        )

    total_cases = len(cases)
    summary = {
        "cases": total_cases,
        "top_k": int(top_k or config["retrieval"]["default_top_k"]),
        "primary_hit_rate": _safe_round(total_primary_hits / total_cases),
        "file_hit_rate": _safe_round(total_file_hits / total_cases),
        "heading_hit_rate": _safe_round(total_heading_hits / total_cases),
        "mrr": _safe_round(total_primary_rr / total_cases),
        "file_mrr": _safe_round(total_file_rr / total_cases),
        "heading_mrr": _safe_round(total_heading_rr / total_cases),
        "ndcg_at_k": _safe_round(total_ndcg / total_cases),
        "avg_latency_ms": _safe_round(total_latency_ms / total_cases),
    }

    return {
        "index": {
            "rebuild_index": rebuild_index,
            "duration_sec": _safe_round(index_duration_sec),
            "peak_rss_mb": _peak_rss_mb(),
            "storage_backend": store.backend_name,
            "total_files": index_stats.total_files,
            "total_chunks": index_stats.total_chunks,
            "avg_chunk_words": index_stats.avg_chunk_words,
            "embedding_model": index_stats.embedding_model,
        },
        "retrieval": summary,
        "cases": case_results,
    }


def format_eval_report(report: dict) -> str:
    index = report["index"]
    retrieval = report["retrieval"]
    lines = [
        "══════════════════════════════════════════════════════════",
        "EVALUATION SUMMARY",
        "══════════════════════════════════════════════════════════",
        f"Index backend: {index['storage_backend']}",
        f"Embedding model: {index['embedding_model']}",
        f"Index files/chunks: {index['total_files']} files / {index['total_chunks']} chunks",
        f"Average chunk words: {index['avg_chunk_words']}",
        f"Index duration: {index['duration_sec']:.4f}s",
        f"Peak RSS: {index['peak_rss_mb']:.2f} MB",
        "",
        f"Cases: {retrieval['cases']}",
        f"Primary hit rate: {retrieval['primary_hit_rate']:.4f}",
        f"File hit rate: {retrieval['file_hit_rate']:.4f}",
        f"Heading hit rate: {retrieval['heading_hit_rate']:.4f}",
        f"MRR: {retrieval['mrr']:.4f}",
        f"nDCG@k: {retrieval['ndcg_at_k']:.4f}",
        f"Average latency: {retrieval['avg_latency_ms']:.4f} ms",
        "",
    ]
    for case in report["cases"]:
        lines.extend(
            [
                f"[{case['id']}] {case['query']}",
                (
                    f"primary_hit={case['primary_hit']} "
                    f"rr={case['primary_reciprocal_rank']:.4f} "
                    f"latency_ms={case['latency_ms']:.4f}"
                ),
                f"matched={case['primary_match_type']}:{case['primary_match_value']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def report_to_json(report: dict) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)
