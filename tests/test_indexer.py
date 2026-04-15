from __future__ import annotations

import contextlib
from pathlib import Path

from gy_doc_search.indexer import _build_stats, _iter_chunk_results, full_index, incremental_index
from gy_doc_search.models import Chunk, IndexStats


def _chunk(chunk_id: str, source_file: str, heading: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_file=source_file,
        heading_path=heading,
        heading_level=1,
        content=f"[Source: {source_file}]\n[Section: {heading}]\n\ncontent",
        word_count=10,
        chunk_index=0,
        total_chunks_in_file=1,
    )


def _config(tmp_path: Path) -> dict:
    return {
        "_project_root": str(tmp_path),
        "_index_state_path": str(tmp_path / ".gy-doc-search" / ".index_state.json"),
        "embedding": {"batch_size": 2},
        "performance": {"workers": 0},
        "chunking": {"profiles": {}},
        "sources": [],
        "chroma": {"collection_name": "project_docs"},
    }


class _FakeStore:
    backend_name = "fake"

    def __init__(self) -> None:
        self.deleted: list[list[str]] = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def delete(self, ids: list[str]) -> None:
        self.deleted.append(ids)

    def count(self) -> int:
        return 0


def test_full_index_upserts_each_file_as_soon_as_it_is_chunked(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = _FakeStore()
    stats = IndexStats(2, 3, 30, 10, "2026-01-01T00:00:00+00:00", [], "simple-hash", "project_docs")
    upserted: list[list[str]] = []
    saved_states: list[dict] = []

    manifests = {
        "docs/a.md": {
            "abs_path": tmp_path / "docs" / "a.md",
            "profile_config": {},
            "metadata_defaults": {},
            "profile": "default",
            "mtime": 1.0,
            "md5": "a",
        },
        "docs/b.md": {
            "abs_path": tmp_path / "docs" / "b.md",
            "profile_config": {},
            "metadata_defaults": {},
            "profile": "default",
            "mtime": 2.0,
            "md5": "b",
        },
    }
    chunk_results = [
        ("docs/a.md", {"mtime": 1.0, "md5": "a", "chunk_ids": ["a1"]}, [_chunk("a1", "docs/a.md", "A")]),
        (
            "docs/b.md",
            {"mtime": 2.0, "md5": "b", "chunk_ids": ["b1", "b2"]},
            [_chunk("b1", "docs/b.md", "B1"), _chunk("b2", "docs/b.md", "B2")],
        ),
    ]

    monkeypatch.setattr("gy_doc_search.indexer.open_store", lambda cfg: store)
    monkeypatch.setattr("gy_doc_search.indexer.create_embedder", lambda cfg: object())
    monkeypatch.setattr("gy_doc_search.indexer.embedding_model_label", lambda cfg: "simple-hash")
    monkeypatch.setattr("gy_doc_search.indexer._scan_project_files", lambda cfg: manifests)
    monkeypatch.setattr("gy_doc_search.indexer._chunk_args_from_manifests", lambda cfg, mf: [("unused",)])
    monkeypatch.setattr("gy_doc_search.indexer._iter_chunk_results", lambda args, workers: iter(chunk_results))
    monkeypatch.setattr("gy_doc_search.indexer._persist_context", lambda current_store: contextlib.nullcontext())
    monkeypatch.setattr(
        "gy_doc_search.indexer._upsert_chunks",
        lambda cfg, current_store, embedder, chunks, reporter=None: upserted.append([chunk.chunk_id for chunk in chunks]),
    )
    monkeypatch.setattr("gy_doc_search.indexer._save_state", lambda cfg, state: saved_states.append(state))
    monkeypatch.setattr("gy_doc_search.indexer._build_stats", lambda cfg, current_store, state: stats)

    result = full_index(config)

    assert result is stats
    assert store.reset_calls == 1
    assert upserted == [["a1"], ["b1", "b2"]]
    assert saved_states[0]["files"]["docs/a.md"]["chunk_ids"] == ["a1"]
    assert saved_states[0]["files"]["docs/b.md"]["chunk_ids"] == ["b1", "b2"]


def test_incremental_index_reindexes_each_file_in_stream_order(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = _FakeStore()
    stats = IndexStats(2, 2, 20, 10, "2026-01-01T00:00:00+00:00", [], "simple-hash", "project_docs")
    events: list[tuple[str, list[str]]] = []

    state = {
        "last_indexed": "2026-01-01T00:00:00+00:00",
        "embedding_model": "simple-hash",
        "files": {
            "docs/a.md": {"mtime": 1.0, "md5": "old-a", "chunk_ids": ["old-a1"]},
            "docs/untouched.md": {"mtime": 5.0, "md5": "same", "chunk_ids": ["u1"]},
        },
    }
    manifests = {
        "docs/a.md": {
            "abs_path": tmp_path / "docs" / "a.md",
            "profile_config": {},
            "metadata_defaults": {},
            "profile": "default",
            "mtime": 2.0,
            "md5": "new-a",
        },
        "docs/b.md": {
            "abs_path": tmp_path / "docs" / "b.md",
            "profile_config": {},
            "metadata_defaults": {},
            "profile": "default",
            "mtime": 3.0,
            "md5": "new-b",
        },
        "docs/untouched.md": {
            "abs_path": tmp_path / "docs" / "untouched.md",
            "profile_config": {},
            "metadata_defaults": {},
            "profile": "default",
            "mtime": 5.0,
            "md5": "same",
        },
    }
    chunk_results = [
        ("docs/a.md", {"mtime": 2.0, "md5": "new-a", "chunk_ids": ["new-a1"]}, [_chunk("new-a1", "docs/a.md", "A")]),
        ("docs/b.md", {"mtime": 3.0, "md5": "new-b", "chunk_ids": ["new-b1"]}, [_chunk("new-b1", "docs/b.md", "B")]),
    ]

    monkeypatch.setattr("gy_doc_search.indexer._load_state", lambda cfg: state)
    monkeypatch.setattr("gy_doc_search.indexer.open_store", lambda cfg: store)
    monkeypatch.setattr("gy_doc_search.indexer.create_embedder", lambda cfg: object())
    monkeypatch.setattr("gy_doc_search.indexer.embedding_model_label", lambda cfg: "simple-hash")
    monkeypatch.setattr("gy_doc_search.indexer._scan_project_files", lambda cfg: manifests)
    monkeypatch.setattr("gy_doc_search.indexer._chunk_args_from_manifests", lambda cfg, mf: [("unused",)])
    monkeypatch.setattr("gy_doc_search.indexer._iter_chunk_results", lambda args, workers: iter(chunk_results))
    monkeypatch.setattr("gy_doc_search.indexer._persist_context", lambda current_store: contextlib.nullcontext())
    monkeypatch.setattr(
        "gy_doc_search.indexer._upsert_chunks",
        lambda cfg, current_store, embedder, chunks, reporter=None: events.append(("upsert", [chunk.chunk_id for chunk in chunks])),
    )
    monkeypatch.setattr(
        store,
        "delete",
        lambda ids: events.append(("delete", ids)),
    )
    monkeypatch.setattr("gy_doc_search.indexer._save_state", lambda cfg, current_state: None)
    monkeypatch.setattr("gy_doc_search.indexer._build_stats", lambda cfg, current_store, current_state: stats)

    result = incremental_index(config)

    assert result is stats
    assert events == [
        ("delete", ["old-a1"]),
        ("upsert", ["new-a1"]),
        ("upsert", ["new-b1"]),
    ]
    assert state["files"]["docs/a.md"]["chunk_ids"] == ["new-a1"]
    assert state["files"]["docs/b.md"]["chunk_ids"] == ["new-b1"]
    assert state["files"]["docs/untouched.md"]["chunk_ids"] == ["u1"]


def test_build_stats_only_requests_metadata() -> None:
    class Store:
        def count(self) -> int:
            return 2

        def get(self, include: list[str]) -> dict:
            assert include == ["metadatas"]
            return {
                "metadatas": [
                    {"source_file": "docs/a.md", "word_count": 10},
                    {"source_file": "docs/b.md", "word_count": 20},
                ]
            }

    stats = _build_stats(
        {"chroma": {"collection_name": "project_docs"}, "embedding": {"provider": "simple"}},
        Store(),
        {"last_indexed": "2026-01-01T00:00:00+00:00", "embedding_model": "simple-hash"},
    )

    assert stats.total_files == 2
    assert stats.total_chunks == 2
    assert stats.total_words == 30
    assert stats.avg_chunk_words == 15


def test_iter_chunk_results_bounds_in_flight_futures(monkeypatch) -> None:
    class FakeFuture:
        def __init__(self, value, order: int) -> None:
            self._value = value
            self.order = order

        def result(self):
            return self._value

    class FakePool:
        def __init__(self, max_workers=None) -> None:
            self.max_workers = max_workers
            self.submissions = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def submit(self, func, args):
            future = FakeFuture(func(args), self.submissions)
            self.submissions += 1
            return future

    seen_pending_sizes: list[int] = []

    def fake_wait(futures, return_when=None):
        assert return_when is not None
        seen_pending_sizes.append(len(futures))
        chosen = min(futures, key=lambda future: future.order)
        remaining = set(futures)
        remaining.remove(chosen)
        return {chosen}, remaining

    monkeypatch.setattr("gy_doc_search.indexer.ProcessPoolExecutor", FakePool)
    monkeypatch.setattr("gy_doc_search.indexer.wait", fake_wait)
    monkeypatch.setattr("gy_doc_search.indexer._chunk_one_file", lambda args: (args[0], {"chunk_ids": [args[0]]}, []))

    args_list = [("a",), ("b",), ("c",), ("d",), ("e",)]
    results = list(_iter_chunk_results(args_list, workers=2))

    assert [item[0] for item in results] == ["a", "b", "c", "d", "e"]
    assert seen_pending_sizes
    assert max(seen_pending_sizes) <= 2
