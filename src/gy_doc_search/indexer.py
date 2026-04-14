"""Indexing pipeline for documentation chunks."""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys

from gy_doc_search.chunker import _iter_source_files, chunk_file
from gy_doc_search.config import resolve_source_entry
from gy_doc_search.embedder import Embedder, create_embedder, embedding_model_label
from gy_doc_search.models import IndexStats
from gy_doc_search.storage import open_store

try:
    from rich.progress import Progress
except Exception:  # pragma: no cover - fallback for minimal environments
    Progress = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chunk_one_file(args: tuple) -> tuple:
    """Module-level worker for ProcessPoolExecutor (must be picklable on macOS spawn)."""
    relative_path, abs_path, base_dir, profile_config, metadata_defaults, profile_name, mtime, md5 = args
    from gy_doc_search.chunker import chunk_file as _chunk_file  # local import keeps workers lean
    file_chunks = _chunk_file(str(abs_path), base_dir, profile_config, metadata_defaults, profile_name=profile_name)
    file_state = {
        "mtime": mtime,
        "md5": md5,
        "chunk_ids": [chunk.chunk_id for chunk in file_chunks],
    }
    return relative_path, file_state, file_chunks


def _flatten_metadata(chunk) -> dict:
    metadata = {
        "source_file": chunk.source_file,
        "heading_path": chunk.heading_path,
        "heading_level": chunk.heading_level,
        "word_count": chunk.word_count,
        "chunk_index": chunk.chunk_index,
        "total_chunks_in_file": chunk.total_chunks_in_file,
        "profile": chunk.profile,
    }
    for key, value in (chunk.front_matter or {}).items():
        metadata[f"fm_{key}"] = value
    for key, value in (chunk.source_metadata or {}).items():
        metadata[f"sm_{key}"] = value
    return metadata


def _state_path(config: dict) -> Path:
    return Path(config["_index_state_path"])


def _load_state(config: dict) -> dict:
    path = _state_path(config)
    if not path.exists():
        return {"files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(config: dict, state: dict) -> None:
    path = _state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _scan_project_files(config: dict) -> dict[str, dict]:
    project_root = Path(config["_project_root"])
    manifests: dict[str, dict] = {}
    seen: set[Path] = set()
    profiles = config["chunking"]["profiles"]
    for source in config["sources"]:
        resolved = resolve_source_entry(source, project_root, config)
        for path in _iter_source_files(resolved):
            if path in seen:
                continue
            seen.add(path)
            relative_path = str(path.resolve().relative_to(project_root.resolve()))
            manifests[relative_path] = {
                "abs_path": path,
                "md5": _md5(path),
                "mtime": path.stat().st_mtime,
                "profile": resolved["profile"],
                "profile_config": profiles[resolved["profile"]],
                "metadata_defaults": resolved.get("metadata_defaults", {}),
            }
    return manifests


def _progress():
    if Progress is None:
        return None
    return Progress()


def _build_stats(config: dict, store, state: dict) -> IndexStats:
    docs = store.get(include=["documents", "metadatas"])
    total_chunks = len(docs["ids"])
    metadatas = docs.get("metadatas", [])
    documents = docs.get("documents", [])
    total_words = sum(int(metadata.get("word_count", 0)) for metadata in metadatas)
    counts = Counter(metadata.get("source_file", "") for metadata in metadatas)
    sources = [{"path": path, "files": 1, "chunks": chunk_count} for path, chunk_count in sorted(counts.items())]
    avg_chunk_words = int(total_words / total_chunks) if total_chunks else 0
    return IndexStats(
        total_files=len({metadata.get("source_file", "") for metadata in metadatas}),
        total_chunks=total_chunks,
        total_words=total_words,
        avg_chunk_words=avg_chunk_words,
        last_indexed=state.get("last_indexed", ""),
        sources=sources,
        embedding_model=state.get("embedding_model", embedding_model_label(config)),
        chroma_collection=config["chroma"]["collection_name"],
    )


def _emit(reporter, message: str) -> None:
    if reporter is not None:
        reporter(message)


def _is_embedding_memory_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "out of memory",
        "invalid buffer size",
        "not enough memory",
        "mps backend out of memory",
        "cuda out of memory",
    )
    return any(marker in message for marker in markers)


def _clear_torch_memory() -> None:
    try:
        import torch

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        return None


def _upsert_chunks(config: dict, store, embedder: Embedder, chunks: list, reporter=None) -> None:
    if not chunks:
        return
    from gy_doc_search.storage import LocalVectorStore
    configured_batch_size = int(config["embedding"].get("batch_size", 32))
    batch_size = max(1, configured_batch_size)
    start = 0
    batch_counter = 0
    total_batches = max(1, (len(chunks) + batch_size - 1) // batch_size)
    ctx = store.deferred_persist() if isinstance(store, LocalVectorStore) else contextlib.nullcontext()
    with ctx:
        while start < len(chunks):
            batch = chunks[start : start + batch_size]
            try:
                batch_counter += 1
                _emit(
                    reporter,
                    (
                        f"embedding chunks {start + 1}-{start + len(batch)} of {len(chunks)} "
                        f"(batch {batch_counter}/{total_batches}, size={batch_size})"
                    ),
                )
                embedder.set_batch_size(batch_size)
                embeddings = embedder.embed_documents([chunk.content for chunk in batch])
            except RuntimeError as exc:
                if _is_embedding_memory_error(exc) and batch_size > 1:
                    next_batch_size = max(1, batch_size // 2)
                    print(
                        (
                            f"Embedding batch of size {batch_size} ran out of memory. "
                            f"Retrying with batch size {next_batch_size}."
                        ),
                        file=sys.stderr,
                    )
                    _clear_torch_memory()
                    batch_size = next_batch_size
                    batch_counter -= 1
                    total_batches = max(1, (len(chunks) - start + batch_size - 1) // batch_size)
                    continue
                if _is_embedding_memory_error(exc):
                    raise RuntimeError(
                        "Embedding failed due to insufficient memory even at batch size 1. "
                        "Lower chunk size, switch embedding provider, or use a smaller model."
                    ) from exc
                raise

            _emit(
                reporter,
                f"writing batch {batch_counter} to {store.backend_name} store",
            )
            store.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.content for chunk in batch],
                embeddings=embeddings,
                metadatas=[_flatten_metadata(chunk) for chunk in batch],
            )
            start += len(batch)


def inspect_index_changes(config: dict) -> dict:
    state = _load_state(config)
    manifests = _scan_project_files(config)
    current_model = embedding_model_label(config)
    if state.get("embedding_model") and state.get("embedding_model") != current_model:
        return {
            "mode": "full",
            "reason": "embedding model changed",
            "new": sorted(manifests),
            "changed": [],
            "deleted": [],
            "unchanged": [],
        }

    new: list[str] = []
    changed: list[str] = []
    unchanged: list[str] = []
    deleted = sorted(set(state.get("files", {})) - set(manifests))
    for relative_path, manifest in manifests.items():
        previous = state.get("files", {}).get(relative_path)
        if previous is None:
            new.append(relative_path)
        elif (
            previous.get("mtime") != manifest["mtime"]
            or previous.get("md5") != manifest["md5"]
        ):
            changed.append(relative_path)
        else:
            unchanged.append(relative_path)
    return {
        "mode": "incremental",
        "reason": "",
        "new": sorted(new),
        "changed": sorted(changed),
        "deleted": deleted,
        "unchanged": sorted(unchanged),
    }


def full_index(config: dict, reporter=None) -> IndexStats:
    _emit(reporter, "opening vector store")
    store = open_store(config)
    _emit(reporter, f"using storage backend: {store.backend_name}")
    _emit(reporter, "resetting index")
    store.reset()
    _emit(reporter, "loading embedding model")
    embedder = create_embedder(config)
    _emit(reporter, f"using embedding model: {embedding_model_label(config)}")
    state = {
        "last_indexed": _utc_now(),
        "embedding_model": embedding_model_label(config),
        "files": {},
    }
    _emit(reporter, "scanning source files")
    manifests = _scan_project_files(config)
    _emit(reporter, f"found {len(manifests)} files to index")
    chunks = []
    _emit(reporter, "chunking files")
    workers = int(config.get("performance", {}).get("workers", 0)) or None
    args_list = [
        (
            relative_path,
            manifest["abs_path"],
            config["_project_root"],
            manifest["profile_config"],
            manifest["metadata_defaults"],
            manifest["profile"],
            manifest["mtime"],
            manifest["md5"],
        )
        for relative_path, manifest in manifests.items()
    ]
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for relative_path, file_state, file_chunks in pool.map(_chunk_one_file, args_list):
                _emit(reporter, f"chunked {relative_path} ({len(file_chunks)} chunks)")
                state["files"][relative_path] = file_state
                chunks.extend(file_chunks)
    except Exception:
        for args in args_list:
            relative_path, file_state, file_chunks = _chunk_one_file(args)
            _emit(reporter, f"chunked {relative_path} ({len(file_chunks)} chunks)")
            state["files"][relative_path] = file_state
            chunks.extend(file_chunks)
    _emit(reporter, f"generated {len(chunks)} chunks")
    _upsert_chunks(config, store, embedder, chunks, reporter=reporter)
    _emit(reporter, "saving index state")
    _save_state(config, state)
    return _build_stats(config, store, state)


def incremental_index(config: dict, reporter=None) -> IndexStats:
    state = _load_state(config)
    if state.get("embedding_model") and state.get("embedding_model") != embedding_model_label(config):
        _emit(reporter, "embedding model changed, switching to full reindex")
        return full_index(config, reporter=reporter)

    _emit(reporter, "opening vector store")
    store = open_store(config)
    _emit(reporter, f"using storage backend: {store.backend_name}")
    _emit(reporter, "loading embedding model")
    embedder = create_embedder(config)
    _emit(reporter, f"using embedding model: {embedding_model_label(config)}")
    _emit(reporter, "scanning source files")
    manifests = _scan_project_files(config)
    state.setdefault("files", {})
    state["embedding_model"] = embedding_model_label(config)

    deleted_files = sorted(set(state["files"]) - set(manifests))
    if deleted_files:
        _emit(reporter, f"removing {len(deleted_files)} deleted files from the index")
    for relative_path in deleted_files:
        store.delete(state["files"][relative_path].get("chunk_ids", []))
        state["files"].pop(relative_path, None)

    changed_or_new = []
    for relative_path, manifest in manifests.items():
        previous = state["files"].get(relative_path)
        if previous is None or previous.get("mtime") != manifest["mtime"] or previous.get("md5") != manifest["md5"]:
            changed_or_new.append((relative_path, manifest, previous))

    _emit(
        reporter,
        (
            f"incremental scan complete: {len(changed_or_new)} files to process, "
            f"{len(manifests) - len(changed_or_new)} unchanged"
        ),
    )
    # Chunk all changed/new files in parallel, then delete-old + upsert per file
    workers = int(config.get("performance", {}).get("workers", 0)) or None
    chunk_args = [
        (
            relative_path,
            manifest["abs_path"],
            config["_project_root"],
            manifest["profile_config"],
            manifest["metadata_defaults"],
            manifest["profile"],
            manifest["mtime"],
            manifest["md5"],
        )
        for relative_path, manifest, _ in changed_or_new
    ]
    chunked: dict[str, tuple] = {}
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for relative_path, file_state, file_chunks in pool.map(_chunk_one_file, chunk_args):
                chunked[relative_path] = (file_state, file_chunks)
    except Exception:
        for args in chunk_args:
            relative_path, file_state, file_chunks = _chunk_one_file(args)
            chunked[relative_path] = (file_state, file_chunks)

    for relative_path, manifest, previous in changed_or_new:
        file_state, file_chunks = chunked[relative_path]
        if previous:
            _emit(reporter, f"re-indexing changed file: {relative_path}")
            store.delete(previous.get("chunk_ids", []))
        else:
            _emit(reporter, f"indexing new file: {relative_path}")
        _emit(reporter, f"generated {len(file_chunks)} chunks for {relative_path}")
        _upsert_chunks(config, store, embedder, file_chunks, reporter=reporter)
        state["files"][relative_path] = file_state

    state["last_indexed"] = _utc_now()
    _emit(reporter, "saving index state")
    _save_state(config, state)
    return _build_stats(config, store, state)


def run_index(config: dict, incremental: bool = False, reporter=None) -> IndexStats:
    return (
        incremental_index(config, reporter=reporter)
        if incremental
        else full_index(config, reporter=reporter)
    )


def get_stats(config: dict) -> IndexStats:
    store = open_store(config)
    state = _load_state(config)
    return _build_stats(config, store, state)


def verify_index(config: dict) -> list[str]:
    state = _load_state(config)
    project_root = Path(config["_project_root"])
    missing: list[str] = []
    for relative_path in sorted(state.get("files", {})):
        if not (project_root / relative_path).exists():
            missing.append(relative_path)
    return missing


def clean_index(config: dict) -> None:
    store = open_store(config)
    store.reset()
    state_path = _state_path(config)
    if state_path.exists():
        state_path.unlink()
