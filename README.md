# gy-doc-search

`gy-doc-search` is a reusable CLI and MCP server for building a local retrieval layer over project documentation and other UTF-8 text sources.

It is designed for AI coding workflows where "search the docs" needs to mean more than keyword grep. `gy-doc-search` turns documentation into stable, source-aware chunks, keeps them incrementally indexed, and serves the most relevant sections back through either a CLI or an MCP interface. That makes it suitable both as a developer tool you run in a terminal and as infrastructure behind coding agents that need grounded answers.

A few characteristics make it distinct:

- local-first operation with a simple project-level config and no required external service
- dual interface: direct CLI for humans, MCP server mode for agent integrations
- resilient retrieval stack with semantic search, optional hybrid BM25 fusion, and optional reranking
- per-source chunking profiles, so API references, ADRs, runbooks, and specs can be indexed differently
- incremental indexing with file change detection, plus a fallback local store when Chroma is not installed
- evaluation tooling for measuring retrieval quality before changing chunking or embedding settings

Markdown is the primary target because headings, sections, and front matter are preserved as metadata. In practice, the indexer can ingest any UTF-8 text file that matches a configured glob. Files without Markdown headings are still indexed as plain-text documents, but Markdown-shaped sources produce the best chunk boundaries and retrieval quality.

## Quickstart

```bash
pip install gy-doc-search
gy-doc-search init --sources ./docs
gy-doc-search index
gy-doc-search query "how authentication works"
```

## Supported File Types

`gy-doc-search` does not enforce a fixed list of extensions. Each source entry chooses which files are indexed through its `filter` glob:

- best-supported: `.md`, `.mdx`, and other Markdown-like text with `#` headings
- also workable: `.txt`, `.rst`, `.adoc`, or project-specific text formats, as long as they are UTF-8
- skipped at chunking time: non-UTF-8 files

Examples:

```yaml
sources:
  - path: "./docs"
    filter: "*.md"
```

```yaml
sources:
  - path: "./docs"
    filter: "*.mdx"
```

```yaml
sources:
  - path: "./handbook"
    filter: "*.txt"
```

```yaml
sources:
  - path: "./specs"
    filter: "*.rst"
```

You can mix source types by declaring multiple entries:

```yaml
sources:
  - path: "./docs"
    filter: "*.md"
    profile: "docs"

  - path: "./guides"
    filter: "*.mdx"
    profile: "docs"

  - path: "./runbooks"
    filter: "*.txt"
    profile: "plain_text"

  - path: "./specs"
    filter: "*.rst"
    profile: "plain_text"

chunking:
  default_profile: "docs"
  profiles:
    docs:
      min_chunk_tokens: 60
      max_chunk_tokens: 240
      target_chunk_tokens: 160
      overlap_tokens: 30
      heading_levels: [1, 2, 3, 4]

    plain_text:
      min_chunk_tokens: 80
      max_chunk_tokens: 260
      target_chunk_tokens: 180
      overlap_tokens: 40
      heading_levels: [1, 2]
```

If you want to include multiple extensions from the same directory, declare multiple source entries with different `filter` values:

```yaml
sources:
  - path: "./content"
    filter: "*.md"
  - path: "./content"
    filter: "*.mdx"
  - path: "./content"
    filter: "*.txt"
```

## CLI Interface

```bash
# Initialize in a new project
gy-doc-search init                         # Creates .doc-search/ with default config
gy-doc-search init --sources ./docs ./specs  # Pre-fill source paths

# Indexing
gy-doc-search index                        # Full reindex
gy-doc-search index --incremental          # Only changed files
gy-doc-search index --dry-run              # Show what would change without indexing

# Querying
gy-doc-search query "payment auth flow"
gy-doc-search query "error codes" --top-k 10 --path "api/"
gy-doc-search query "deployment" --files-only

# Read a specific file
gy-doc-search get docs/api/payments.md

# List indexed sources
gy-doc-search list
gy-doc-search list --prefix "api/"

# MCP server mode (for Claude Code)
gy-doc-search serve                        # Starts MCP server on stdio
gy-doc-search serve --transport sse --port 8080  # SSE transport

# Maintenance
gy-doc-search status                       # Shows config, index freshness, stats
gy-doc-search clean                        # Wipes the vector store
gy-doc-search verify                       # Checks that indexed files still exist

# Evaluation
gy-doc-search eval --cases eval_cases.yaml
gy-doc-search eval --cases eval_cases.yaml --json
gy-doc-search eval --cases eval_cases.yaml --skip-index
```

## Notes

- Project configuration lives in `.doc-search/config.yaml`.
- `gy-doc-search` auto-selects a storage backend:
  - `chromadb` installed: persistent Chroma collection in `.doc-search/.chroma/`
  - no `chromadb`: local JSON-backed store in `.doc-search/.index/`
- If `sentence-transformers` is installed, semantic embeddings use the configured transformer model. Without it, `gy-doc-search` falls back to a built-in lexical hash embedder.
- If `mcp` is installed, `gy-doc-search serve` starts the MCP server for Claude Code or other MCP clients.
- `gy-doc-search index` now prints explicit phases during indexing, including config loading, file scanning, chunking, embedding, and index writes.

## Evaluation

`gy-doc-search eval` is an offline benchmark command for tuning chunking and retrieval settings against labeled queries.

It exists because retrieval quality is highly project-specific. A documentation set full of API references behaves differently from architecture decision records, onboarding guides, incident runbooks, or product specs. The same chunk sizes, overlap, heading boundaries, embedding model, and retrieval settings will not perform equally well across all repositories.

The evaluation feature was built to address those differences directly. Instead of tuning `gy-doc-search` by intuition, you can define a representative set of queries for your own project, label the files and headings that should be retrieved, and measure how configuration changes affect actual outcomes. This makes the retrieval layer more defensible and much easier to adapt as a codebase grows or documentation habits change.

That is especially useful for coding agents. An agent can be prompted to use `gy-doc-search eval` as a feedback loop:

- establish a baseline with the current configuration
- adjust one part of the config, such as chunk sizing, heading levels, hybrid retrieval, or reranking
- rerun the evaluation
- compare hit rates, ranking quality, chunk counts, and latency
- keep the change only if the results improve for the project’s real query set

In other words, `eval` turns configuration tuning into an iterative workflow instead of a one-shot guess. That helps agents and humans converge on settings that are specific to the repository they are working in.

It measures:

- retrieval quality: primary hit rate, file hit rate, heading hit rate, MRR, and nDCG@k
- runtime behavior: index duration, average query latency, and peak RSS
- index shape: total files, total chunks, average chunk words, storage backend, and embedding model

The evaluation command uses a YAML file that defines expected files and headings for each query.

Example:

```yaml
cases:
  - id: payments-auth
    query: "authorization funds"
    relevant_files:
      - "docs/payments.md"
    relevant_headings:
      - "Payments"

  - id: settlement-close
    query: "day closing settlement"
    relevant_files:
      - "docs/settlement.md"
    relevant_headings:
      - "Settlement"
```

Run it like this:

```bash
gy-doc-search eval --cases eval_cases.yaml
gy-doc-search eval --cases eval_cases.yaml --json
gy-doc-search eval --cases eval_cases.yaml --skip-index
```

Options:

- `--cases`: path to the YAML evaluation file
- `--top-k`: override the retrieval depth for all evaluation queries
- `--skip-index`: reuse the current index instead of rebuilding it before evaluation
- `--json`: emit the full structured evaluation report as JSON

Recommended tuning workflow:

1. Create 20-50 representative queries with labeled `relevant_files` and, when possible, `relevant_headings`.
2. Run `gy-doc-search eval` with the current configuration to establish a baseline.
3. Change one variable at a time, such as `min_chunk_tokens`, `max_chunk_tokens`, `target_chunk_tokens`, `overlap_tokens`, `heading_levels`, `hybrid_search`, or `reranking`.
4. Rerun the evaluation and compare hit rate, MRR, nDCG, chunk counts, and latency across runs.
5. Keep only changes that improve retrieval for the project’s labeled queries.
6. Tune `embedding.batch_size` separately for runtime stability rather than retrieval quality.

Example agent prompt:

```text
Use gy-doc-search eval to tune the retrieval config for this repository.
Start by running a baseline evaluation against the existing labeled cases.
Then iteratively adjust one retrieval or chunking parameter at a time.
After each run, compare the metrics and explain whether the change improved project-specific retrieval quality.
Stop when further changes no longer produce a clear improvement, and summarize the recommended config updates.
```

For lower-memory machines, start conservatively:

```yaml
embedding:
  provider: "sentence-transformers"
  model_name: "all-MiniLM-L6-v2"
  batch_size: 1
```

Co-authored by Taner Esme, Claude Code, and Codex.
