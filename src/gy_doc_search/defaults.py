"""Built-in configuration defaults."""

from __future__ import annotations

DEFAULTS: dict = {
    "sources": [
        {
            "path": "./docs",
            "recursive": True,
            "filter": "*.md",
        }
    ],
    "chunking": {
        "default_profile": "default",
        "profiles": {
            "default": {
                "min_chunk_tokens": 60,
                "max_chunk_tokens": 240,
                "target_chunk_tokens": 160,
                "overlap_tokens": 30,
                "heading_levels": [1, 2, 3, 4],
            }
        },
    },
    "embedding": {
        "provider": "simple",
        "model_name": "nomic-ai/nomic-embed-text-v1.5",
        "openai_model": "text-embedding-3-small",
        "ollama_model": "nomic-embed-text",
        "ollama_base_url": "http://localhost:11434",
        "batch_size": 32,
        "normalize_embeddings": True,
        "query_prefix": None,
        "document_prefix": None,
    },
    "retrieval": {
        "default_top_k": 10,
        "max_top_k": 20,
        "similarity_threshold": 0.1,
        "hybrid_search": True,
        "reranking": False,
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    },
    "chroma": {
        "collection_name": "project_docs",
    },
    "storage": {
        "backend": "local",
        "index_dirname": ".index",
        "documents_file": "documents.json",
        "state_file": ".index_state.json",
    },
    "mcp": {
        "server_name": "gy-doc-search",
        "server_version": "1.0.0",
    },
    "performance": {
        "workers": 0,  # 0 = os.cpu_count(); 1 = serial (useful for debugging)
    },
}
