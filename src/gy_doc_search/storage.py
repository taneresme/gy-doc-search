"""Persistent local storage for chunk records."""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Any

try:
    import numpy as _np
    _NUMPY = True
except ImportError:
    _NUMPY = False


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    if _NUMPY:
        a = _np.asarray(left, dtype=_np.float32)
        b = _np.asarray(right, dtype=_np.float32)
        dot = float(_np.dot(a, b))
        left_norm = float(_np.linalg.norm(a)) or 1.0
        right_norm = float(_np.linalg.norm(b)) or 1.0
    else:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
        right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    score = dot / (left_norm * right_norm)
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _split_where_filter(where: dict | None) -> tuple[dict | None, str | None]:
    if not where:
        return None, None
    exact: dict[str, Any] = {}
    prefix: str | None = None
    for key, value in where.items():
        if key == "__source_file_prefix":
            prefix = str(value)
        else:
            exact[key] = value
    return (exact or None), prefix


def _match_metadata(metadata: dict, exact_where: dict | None, prefix: str | None) -> bool:
    if exact_where:
        for key, value in exact_where.items():
            if metadata.get(key) != value:
                return False
    if prefix and not str(metadata.get("source_file", "")).startswith(prefix):
        return False
    return True


def _records_to_query_result(
    ranked: list[tuple[dict, float]],
    include: list[str],
) -> dict:
    result = {"ids": [[record["id"] for record, _ in ranked]]}
    if "documents" in include:
        result["documents"] = [[record["document"] for record, _ in ranked]]
    if "metadatas" in include:
        result["metadatas"] = [[record["metadata"] for record, _ in ranked]]
    if "distances" in include:
        result["distances"] = [[1.0 - score for _, score in ranked]]
    if "embeddings" in include:
        result["embeddings"] = [[record["embedding"] for record, _ in ranked]]
    return result


class VectorStore:
    """Common protocol for local and Chroma-backed stores."""

    backend_name = "unknown"

    def persist(self) -> None:
        """Flush pending changes if the backend needs it."""

    def reset(self) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        raise NotImplementedError

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        raise NotImplementedError

    def get(
        self,
        ids: list[str] | None = None,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        raise NotImplementedError


class LocalVectorStore(VectorStore):
    """Simple JSON-backed vector store used when Chroma is unavailable."""

    backend_name = "local"

    def __init__(self, storage_dir: str, collection_name: str):
        self.storage_dir = Path(storage_dir)
        self.collection_name = collection_name
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.documents_path = self.storage_dir / "documents.json"
        self.records: dict[str, dict] = {}
        self._auto_persist: bool = True
        self._load()

    def _load(self) -> None:
        if not self.documents_path.exists():
            self.records = {}
            return
        data = json.loads(self.documents_path.read_text(encoding="utf-8"))
        self.records = {record["id"]: record for record in data.get("records", [])}

    def persist(self) -> None:
        payload = {
            "collection_name": self.collection_name,
            "records": sorted(self.records.values(), key=lambda item: item["id"]),
        }
        self.documents_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @contextlib.contextmanager
    def deferred_persist(self):
        """Batch all writes and persist exactly once on exit."""
        self._auto_persist = False
        try:
            yield self
        finally:
            self._auto_persist = True
            self.persist()

    def reset(self) -> None:
        self.records = {}
        self.persist()

    def count(self) -> int:
        return len(self.records)

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        for chunk_id, document, embedding, metadata in zip(
            ids, documents, embeddings, metadatas
        ):
            self.records[chunk_id] = {
                "id": chunk_id,
                "document": document,
                "embedding": embedding,
                "metadata": metadata,
            }
        if self._auto_persist:
            self.persist()

    def delete(self, ids: list[str]) -> None:
        for chunk_id in ids:
            self.records.pop(chunk_id, None)
        if self._auto_persist:
            self.persist()

    def filtered_records(self, where: dict | None = None) -> list[dict]:
        exact_where, prefix = _split_where_filter(where)
        return [
            record
            for record in self.records.values()
            if _match_metadata(record["metadata"], exact_where, prefix)
        ]

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        include = include or ["documents", "metadatas", "distances"]
        records = self.filtered_records(where)
        ids_batch: list[list[str]] = []
        docs_batch: list[list[str]] = []
        metas_batch: list[list[dict]] = []
        distances_batch: list[list[float]] = []
        embeddings_batch: list[list[list[float]]] = []
        for query_embedding in query_embeddings:
            if _NUMPY and records:
                matrix = _np.asarray([r["embedding"] for r in records], dtype=_np.float32)
                q = _np.asarray(query_embedding, dtype=_np.float32)
                dots = matrix @ q
                q_norm = float(_np.linalg.norm(q)) or 1.0
                row_norms = _np.linalg.norm(matrix, axis=1)
                row_norms[row_norms == 0] = 1.0
                raw = dots / (row_norms * q_norm)
                scores = _np.clip((raw + 1.0) / 2.0, 0.0, 1.0)
                top_idx = _np.argsort(scores)[::-1][:n_results]
                ranked = [(records[int(i)], float(scores[i])) for i in top_idx]
            else:
                ranked = sorted(
                    (
                        (record, cosine_similarity(query_embedding, record["embedding"]))
                        for record in records
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )[:n_results]
            batch_result = _records_to_query_result(ranked, include)
            ids_batch.extend(batch_result["ids"])
            if "documents" in include:
                docs_batch.extend(batch_result["documents"])
            if "metadatas" in include:
                metas_batch.extend(batch_result["metadatas"])
            if "distances" in include:
                distances_batch.extend(batch_result["distances"])
            if "embeddings" in include:
                embeddings_batch.extend(batch_result["embeddings"])

        result = {"ids": ids_batch}
        if "documents" in include:
            result["documents"] = docs_batch
        if "metadatas" in include:
            result["metadatas"] = metas_batch
        if "distances" in include:
            result["distances"] = distances_batch
        if "embeddings" in include:
            result["embeddings"] = embeddings_batch
        return result

    def get(
        self,
        ids: list[str] | None = None,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        include = include or ["documents", "metadatas"]
        if ids is not None:
            records = [self.records[chunk_id] for chunk_id in ids if chunk_id in self.records]
        else:
            records = self.filtered_records(where)

        result = {"ids": [record["id"] for record in records]}
        if "documents" in include:
            result["documents"] = [record["document"] for record in records]
        if "metadatas" in include:
            result["metadatas"] = [record["metadata"] for record in records]
        if "embeddings" in include:
            result["embeddings"] = [record["embedding"] for record in records]
        return result


class ChromaVectorStore(VectorStore):
    """Thin wrapper over ChromaDB with a compatible API."""

    backend_name = "chroma"
    _GET_PAGE_SIZE = 1000

    def __init__(self, chroma_dir: str, collection_name: str):
        import chromadb

        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def persist(self) -> None:
        return None

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return int(self.collection.count())

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    def _get_records(
        self,
        ids: list[str] | None = None,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> list[dict]:
        include = include or ["documents", "metadatas"]
        exact_where, prefix = _split_where_filter(where)
        records = []
        if ids is not None:
            raw_batches = [
                self.collection.get(
                    ids=ids,
                    where=exact_where,
                    include=include,
                )
            ]
        else:
            raw_batches = []
            offset = 0
            while True:
                raw = self.collection.get(
                    where=exact_where,
                    limit=self._GET_PAGE_SIZE,
                    offset=offset,
                    include=include,
                )
                raw_batches.append(raw)
                batch_size = len(raw.get("ids", []))
                if batch_size < self._GET_PAGE_SIZE:
                    break
                offset += batch_size

        for raw in raw_batches:
            documents = raw.get("documents") or [None] * len(raw.get("ids", []))
            metadatas = raw.get("metadatas") or [None] * len(raw.get("ids", []))
            embeddings = raw.get("embeddings") or [None] * len(raw.get("ids", []))
            for chunk_id, document, metadata, embedding in zip(
                raw.get("ids", []),
                documents,
                metadatas,
                embeddings,
            ):
                metadata = metadata or {}
                if not _match_metadata(metadata, exact_where, prefix):
                    continue
                records.append(
                    {
                        "id": chunk_id,
                        "document": document,
                        "metadata": metadata,
                        "embedding": embedding,
                    }
                )
        return records

    def get(
        self,
        ids: list[str] | None = None,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        include = include or ["documents", "metadatas"]
        records = self._get_records(ids=ids, where=where, include=include)
        result = {"ids": [record["id"] for record in records]}
        if "documents" in include:
            result["documents"] = [record["document"] for record in records]
        if "metadatas" in include:
            result["metadatas"] = [record["metadata"] for record in records]
        if "embeddings" in include:
            result["embeddings"] = [record["embedding"] for record in records]
        return result

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        include = include or ["documents", "metadatas", "distances"]
        exact_where, prefix = _split_where_filter(where)
        if prefix:
            records = self._get_records(
                where=where,
                include=["documents", "metadatas", "embeddings"],
            )
            ids_batch: list[list[str]] = []
            docs_batch: list[list[str]] = []
            metas_batch: list[list[dict]] = []
            distances_batch: list[list[float]] = []
            embeddings_batch: list[list[list[float]]] = []
            for query_embedding in query_embeddings:
                ranked = sorted(
                    (
                        (record, cosine_similarity(query_embedding, record["embedding"]))
                        for record in records
                        if record["embedding"] is not None
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )[:n_results]
                batch_result = _records_to_query_result(ranked, include)
                ids_batch.extend(batch_result["ids"])
                if "documents" in include:
                    docs_batch.extend(batch_result["documents"])
                if "metadatas" in include:
                    metas_batch.extend(batch_result["metadatas"])
                if "distances" in include:
                    distances_batch.extend(batch_result["distances"])
                if "embeddings" in include:
                    embeddings_batch.extend(batch_result["embeddings"])
            result = {"ids": ids_batch}
            if "documents" in include:
                result["documents"] = docs_batch
            if "metadatas" in include:
                result["metadatas"] = metas_batch
            if "distances" in include:
                result["distances"] = distances_batch
            if "embeddings" in include:
                result["embeddings"] = embeddings_batch
            return result

        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=exact_where,
            include=include,
        )


def open_store(config: dict) -> VectorStore:
    backend = config.get("storage", {}).get("backend", "auto")
    if backend in {"auto", "chroma"}:
        try:
            __import__("chromadb")
            return ChromaVectorStore(
                config["_chroma_dir"],
                config["chroma"]["collection_name"],
            )
        except Exception:
            if backend == "chroma":
                raise
    return LocalVectorStore(
        config["_storage_dir"],
        config["chroma"]["collection_name"],
    )
