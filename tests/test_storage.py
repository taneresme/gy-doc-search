from __future__ import annotations

from gy_doc_search.storage import ChromaVectorStore


def test_chroma_get_records_pages_large_collection() -> None:
    class FakeCollection:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get(self, **kwargs) -> dict:
            self.calls.append(kwargs)
            offset = kwargs.get("offset", 0) or 0
            limit = kwargs.get("limit")
            assert limit == 1000
            if offset == 0:
                size = 1000
            elif offset == 1000:
                size = 5
            else:
                size = 0
            ids = [f"id-{offset + index}" for index in range(size)]
            return {
                "ids": ids,
                "metadatas": [{"source_file": f"docs/{chunk_id}.md"} for chunk_id in ids],
            }

    store = object.__new__(ChromaVectorStore)
    store.collection = FakeCollection()

    records = store._get_records(include=["metadatas"])

    assert len(records) == 1005
    assert len(store.collection.calls) == 2
    assert store.collection.calls[0]["offset"] == 0
    assert store.collection.calls[1]["offset"] == 1000
    assert all("metadata" in record for record in records)


def test_chroma_get_records_pages_large_id_list() -> None:
    class FakeCollection:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get(self, **kwargs) -> dict:
            self.calls.append(kwargs)
            ids = kwargs["ids"]
            return {
                "ids": ids,
                "metadatas": [{"source_file": f"docs/{chunk_id}.md"} for chunk_id in ids],
            }

    store = object.__new__(ChromaVectorStore)
    store.collection = FakeCollection()
    ids = [f"id-{index}" for index in range(1005)]

    records = store._get_records(ids=ids, include=["metadatas"])

    assert len(records) == 1005
    assert len(store.collection.calls) == 2
    assert len(store.collection.calls[0]["ids"]) == 1000
    assert len(store.collection.calls[1]["ids"]) == 5
    assert all("metadata" in record for record in records)
