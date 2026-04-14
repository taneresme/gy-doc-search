"""Data models for indexing and search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    heading_path: str
    heading_level: int
    content: str
    word_count: int
    chunk_index: int
    total_chunks_in_file: int
    front_matter: dict = field(default_factory=dict)
    source_metadata: dict = field(default_factory=dict)
    profile: str = "default"

    def to_record(self) -> dict:
        return asdict(self)


@dataclass
class SearchResult:
    chunk_id: str
    source_file: str
    heading_path: str
    content: str
    score: float
    word_count: int
    metadata: dict = field(default_factory=dict)


@dataclass
class IndexStats:
    total_files: int
    total_chunks: int
    total_words: int
    avg_chunk_words: int
    last_indexed: str
    sources: list[dict]
    embedding_model: str
    chroma_collection: str
