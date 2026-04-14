"""Markdown-aware chunking for documentation sources."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import yaml

from gy_doc_search.config import resolve_source_entry
from gy_doc_search.models import Chunk

LOGGER = logging.getLogger(__name__)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^```")


def token_count(text: str) -> int:
    return len(text.split())


def deterministic_hash(source_file: str, heading_path: str, chunk_index: int) -> str:
    value = f"{source_file}|{heading_path}|{chunk_index}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def extract_front_matter(text: str) -> tuple[dict, str]:
    """Separate YAML front matter from the body."""
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            fm_text = "\n".join(lines[1:index]).strip()
            body = "\n".join(lines[index + 1 :]).lstrip("\n")
            if not fm_text:
                return {}, body
            parsed = yaml.safe_load(fm_text) or {}
            if not isinstance(parsed, dict):
                raise ValueError("YAML front matter must be a mapping.")
            return parsed, body

    return {}, text


def parse_headings(text: str, levels: list[int]) -> list[dict]:
    """Parse markdown text into a tree of heading sections."""
    root: list[dict] = []
    stack: list[dict] = []
    in_code_block = False

    def current_target() -> dict | None:
        return stack[-1] if stack else None

    for line in text.splitlines():
        if FENCE_RE.match(line.strip()):
            target = current_target()
            if target is not None:
                target["content_parts"].append(line)
            in_code_block = not in_code_block
            continue

        match = HEADING_RE.match(line)
        if match and not in_code_block:
            level = len(match.group(1))
            title = match.group(2).strip()
            if level in levels:
                node = {
                    "title": title,
                    "level": level,
                    "content_parts": [],
                    "children": [],
                }
                while stack and stack[-1]["level"] >= level:
                    stack.pop()
                if stack:
                    stack[-1]["children"].append(node)
                else:
                    root.append(node)
                stack.append(node)
                continue

        target = current_target()
        if target is not None:
            target["content_parts"].append(line)

    def finalize(nodes: list[dict]) -> list[dict]:
        finalized: list[dict] = []
        for node in nodes:
            finalized.append(
                {
                    "title": node["title"],
                    "level": node["level"],
                    "content": "\n".join(node["content_parts"]).strip(),
                    "children": finalize(node["children"]),
                }
            )
        return finalized

    return finalize(root)


def flatten_with_ancestry(
    sections: list[dict],
    ancestors: list[str] | None = None,
) -> list[dict]:
    """Flatten heading tree and prepend ancestry to each section."""
    ancestry = ancestors or []
    flattened: list[dict] = []
    for section in sections:
        heading_path_parts = [*ancestry, section["title"]]
        if section["content"] or not section["children"]:
            flattened.append(
                {
                    "heading_path": " > ".join(heading_path_parts),
                    "content": section["content"].strip(),
                    "deepest_level": section["level"],
                }
            )
        flattened.extend(
            flatten_with_ancestry(section["children"], heading_path_parts)
        )
    return flattened


def _split_large_section(section: dict, profile_config: dict) -> list[dict]:
    max_tokens = profile_config["max_chunk_tokens"]
    target_tokens = profile_config.get("target_chunk_tokens", max_tokens)
    overlap_tokens = profile_config.get("overlap_tokens", 0)

    if token_count(section["content"]) <= max_tokens:
        return [section]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", section["content"])
        if paragraph.strip()
    ]
    if not paragraphs:
        return [section]

    chunks: list[dict] = []
    current_paragraphs: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = token_count(paragraph)
        if current_paragraphs and current_tokens + paragraph_tokens > max_tokens:
            chunks.append(
                {
                    **section,
                    "content": "\n\n".join(current_paragraphs).strip(),
                }
            )
            if overlap_tokens > 0:
                overlap_parts: list[str] = []
                overlap_seen = 0
                for existing in reversed(current_paragraphs):
                    overlap_parts.insert(0, existing)
                    overlap_seen += token_count(existing)
                    if overlap_seen >= overlap_tokens:
                        break
                current_paragraphs = overlap_parts[:]
                current_tokens = token_count("\n\n".join(current_paragraphs))
            else:
                current_paragraphs = []
                current_tokens = 0

        current_paragraphs.append(paragraph)
        current_tokens = token_count("\n\n".join(current_paragraphs))

        if current_tokens >= target_tokens and current_tokens >= max_tokens:
            chunks.append(
                {
                    **section,
                    "content": "\n\n".join(current_paragraphs).strip(),
                }
            )
            current_paragraphs = []
            current_tokens = 0

    if current_paragraphs:
        chunks.append(
            {
                **section,
                "content": "\n\n".join(current_paragraphs).strip(),
            }
        )

    return [chunk for chunk in chunks if chunk["content"].strip()]


def apply_sizing(sections: list[dict], profile_config: dict) -> list[dict]:
    """Merge small sections and split large sections."""
    split_sections: list[dict] = []
    for section in sections:
        split_sections.extend(_split_large_section(section, profile_config))

    min_tokens = profile_config["min_chunk_tokens"]
    merged: list[dict] = []
    pending: dict | None = None

    for section in split_sections:
        if not section["content"].strip():
            continue

        if pending is None:
            pending = dict(section)
            continue

        if token_count(pending["content"]) < min_tokens:
            pending = {
                "heading_path": pending["heading_path"],
                "content": (
                    pending["content"].rstrip() + "\n\n" + section["content"].lstrip()
                ).strip(),
                "deepest_level": max(
                    pending["deepest_level"], section["deepest_level"]
                ),
            }
            continue

        merged.append(pending)
        pending = dict(section)

    if pending is not None:
        if merged and token_count(pending["content"]) < min_tokens:
            merged[-1] = {
                "heading_path": merged[-1]["heading_path"],
                "content": (
                    merged[-1]["content"].rstrip() + "\n\n" + pending["content"].lstrip()
                ).strip(),
                "deepest_level": max(
                    merged[-1]["deepest_level"], pending["deepest_level"]
                ),
            }
        else:
            merged.append(pending)

    return merged


def chunk_file(
    filepath: str,
    base_dir: str,
    profile_config: dict,
    source_metadata: dict | None = None,
    profile_name: str = "default",
) -> list[Chunk]:
    """Process a single markdown file into chunks using the given profile."""
    source_metadata = source_metadata or {}
    path = Path(filepath)
    base_path = Path(base_dir)

    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        LOGGER.warning("Skipping non-UTF8 file: %s", path)
        return []

    if not raw_text.strip():
        return []
    if path.stat().st_size > 1024 * 1024:
        LOGGER.warning("Processing large file over 1MB: %s", path)

    front_matter, body = extract_front_matter(raw_text)
    body = body.strip()
    if not body:
        return []

    relative_path = str(path.resolve().relative_to(base_path.resolve()))
    sections = parse_headings(body, profile_config.get("heading_levels", [1, 2, 3]))

    if sections:
        flat_sections = flatten_with_ancestry(sections)
    else:
        flat_sections = [
            {
                "heading_path": path.name,
                "content": body,
                "deepest_level": 0,
            }
        ]

    sized_chunks = apply_sizing(flat_sections, profile_config)
    chunks: list[Chunk] = []
    total_chunks = len(sized_chunks)
    for index, section in enumerate(sized_chunks):
        content = (
            f"[Source: {relative_path}]\n"
            f"[Section: {section['heading_path']}]\n\n"
            f"{section['content'].strip()}"
        )
        chunks.append(
            Chunk(
                chunk_id=deterministic_hash(relative_path, section["heading_path"], index),
                source_file=relative_path,
                heading_path=section["heading_path"],
                heading_level=section["deepest_level"],
                content=content,
                word_count=token_count(content),
                chunk_index=index,
                total_chunks_in_file=total_chunks,
                front_matter=front_matter,
                source_metadata=source_metadata,
                profile=profile_name,
            )
        )

    return chunks


def _iter_source_files(source: dict) -> list[Path]:
    abs_path = Path(source["_abs_path"])
    pattern = source.get("filter", "*.md")
    if abs_path.is_file():
        return [abs_path]
    if abs_path.is_dir():
        iterator = abs_path.rglob(pattern) if source.get("recursive", True) else abs_path.glob(pattern)
        return sorted(path for path in iterator if path.is_file())
    return []


def chunk_sources(sources: list[dict], project_root: str, profiles: dict) -> list[Chunk]:
    """Process all configured sources and return their chunks."""
    base_dir = Path(project_root)
    config = {
        "chunking": {
            "default_profile": "default",
            "profiles": profiles,
        }
    }
    chunks: list[Chunk] = []
    seen: set[Path] = set()
    for source in sources:
        resolved = resolve_source_entry(source, base_dir, config)
        for path in _iter_source_files(resolved):
            if path in seen:
                continue
            seen.add(path)
            chunks.extend(
                chunk_file(
                    str(path),
                    str(base_dir),
                    resolved["_profile_config"],
                    resolved.get("metadata_defaults"),
                    profile_name=resolved["profile"],
                )
            )
    return chunks
