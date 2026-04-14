"""Embedding provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
import hashlib
import math


class Embedder(ABC):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Backward-compatible alias for document embeddings."""
        return self.embed_documents(texts)

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for index documents."""

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a search query."""
        return self.embed_documents([text])[0]

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""

    def set_batch_size(self, batch_size: int) -> None:
        """Allow runtime batch-size tuning for embedders that support it."""
        return None


class LexicalHashEmbedder(Embedder):
    """Dependency-free fallback embedder based on token hashing."""

    def __init__(self, dims: int = 256):
        self.dims = dims

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            counts = Counter(token.lower() for token in text.split())
            vector = [0.0] * self.dims
            for token, count in counts.items():
                index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dims
                vector[index] += float(count)
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    def dimension(self) -> int:
        return self.dims


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        ).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"{self.document_prefix}{text}" for text in texts]
        return self._encode(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"{self.query_prefix}{text}"])[0]

    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = max(1, int(batch_size))


class OpenAIEmbedder(Embedder):
    def __init__(self, model_name: str):
        import openai

        self.client = openai.OpenAI()
        self.model_name = model_name
        self._dimension: int | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_documents(["dimension probe"])[0])
        return self._dimension


class OllamaEmbedder(Embedder):
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", max_concurrent: int = 4):
        import requests

        self._requests = requests
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._dimension: int | None = None
        self._max_concurrent = max(1, int(max_concurrent))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from concurrent.futures import ThreadPoolExecutor

        def _embed_one(text: str) -> list[float]:
            response = self._requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["embedding"]

        max_w = min(len(texts), self._max_concurrent)
        if max_w <= 1:
            return [_embed_one(t) for t in texts]
        with ThreadPoolExecutor(max_workers=max_w) as pool:
            return list(pool.map(_embed_one, texts))

    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_documents(["dimension probe"])[0])
        return self._dimension


def create_embedder(config: dict) -> Embedder:
    provider = config["embedding"]["provider"]
    if provider == "sentence-transformers":
        try:
            model_name = config["embedding"]["model_name"]
            query_prefix = config["embedding"].get("query_prefix")
            document_prefix = config["embedding"].get("document_prefix")
            if query_prefix is None and "nomic-embed" in model_name:
                query_prefix = "search_query: "
            if document_prefix is None and "nomic-embed" in model_name:
                document_prefix = "search_document: "
            return SentenceTransformerEmbedder(
                model_name,
                batch_size=int(config["embedding"].get("batch_size", 64)),
                normalize_embeddings=bool(
                    config["embedding"].get("normalize_embeddings", True)
                ),
                query_prefix=query_prefix or "",
                document_prefix=document_prefix or "",
            )
        except Exception:
            return LexicalHashEmbedder()
    if provider == "openai":
        return OpenAIEmbedder(config["embedding"]["openai_model"])
    if provider == "ollama":
        return OllamaEmbedder(
            config["embedding"]["ollama_model"],
            config["embedding"].get("ollama_base_url", "http://localhost:11434"),
            max_concurrent=int(config["embedding"].get("ollama_max_concurrent", 4)),
        )
    if provider == "simple":
        return LexicalHashEmbedder()
    raise ValueError(f"Unknown embedding provider: {provider}")


def embedding_model_label(config: dict) -> str:
    provider = config["embedding"]["provider"]
    if provider == "sentence-transformers":
        try:
            __import__("sentence_transformers")
            return config["embedding"]["model_name"]
        except Exception:
            return "simple-hash-fallback"
    if provider == "openai":
        return config["embedding"]["openai_model"]
    if provider == "ollama":
        return config["embedding"]["ollama_model"]
    if provider == "simple":
        return "simple-hash"
    return provider
