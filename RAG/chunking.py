#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from RAG.schema import ChunkMetadata, Document

try:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
except ModuleNotFoundError:
    enc = None


def count_tokens(text: str) -> int:
    if enc is None:
        return len(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]|\S", text.lower()))
    return len(enc.encode(text))


def trim_to_tokens(text: str, max_tokens: int) -> str:
    if enc is None:
        tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]|\S", text)
        return "".join(tokens[:max_tokens]) if len(tokens) > max_tokens else text
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def token_overlap(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    if enc is None:
        tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]|\S", text)
        return "".join(tokens[-overlap_tokens:])
    tokens = enc.encode(text)
    return enc.decode(tokens[-overlap_tokens:])


@dataclass
class ChunkingConfig:
    max_tokens: int = 600
    overlap_tokens: int = 120
    min_chunk_tokens: int = 20


class MarkdownAwareChunker:
    """Heading-aware and code-fence-aware chunker with section metadata."""

    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    fence_pattern = re.compile(r"^\s*(```|~~~)")

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(
        self,
        text: str,
        source: str,
        doc_id: Optional[str] = None,
        page: Optional[int] = None,
    ) -> List[Document]:
        doc_id = doc_id or self._build_doc_id(source, text)
        created_at = datetime.now(timezone.utc).isoformat()
        chunks: List[Document] = []
        current_lines: List[str] = []
        current_tokens = 0
        section_stack: List[str] = []
        in_code_block = False

        for block in self._iter_blocks(text):
            stripped = block.strip("\n")
            heading = self.heading_pattern.match(stripped)
            starts_fence = self.fence_pattern.match(stripped) is not None

            if heading and not in_code_block:
                self._flush(
                    chunks=chunks,
                    lines=current_lines,
                    source=source,
                    doc_id=doc_id,
                    section=self._section_path(section_stack),
                    page=page,
                    created_at=created_at,
                )
                current_lines = []
                current_tokens = 0
                level = len(heading.group(1))
                title = heading.group(2).strip()
                section_stack = section_stack[: level - 1] + [title]

            block_tokens = count_tokens(block)
            if (
                current_lines
                and current_tokens + block_tokens > self.config.max_tokens
                and not in_code_block
            ):
                previous = "\n".join(current_lines)
                self._flush(
                    chunks=chunks,
                    lines=current_lines,
                    source=source,
                    doc_id=doc_id,
                    section=self._section_path(section_stack),
                    page=page,
                    created_at=created_at,
                )
                overlap = token_overlap(previous, self.config.overlap_tokens)
                current_lines = [overlap] if overlap else []
                current_tokens = count_tokens(overlap)

            if block_tokens > self.config.max_tokens and not starts_fence:
                for part in self._split_long_block(block):
                    current_lines.append(part)
                    current_tokens += count_tokens(part)
                    self._flush(
                        chunks=chunks,
                        lines=current_lines,
                        source=source,
                        doc_id=doc_id,
                        section=self._section_path(section_stack),
                        page=page,
                        created_at=created_at,
                    )
                    current_lines = []
                    current_tokens = 0
                continue

            current_lines.append(block)
            current_tokens += block_tokens

            if starts_fence:
                in_code_block = not in_code_block

        self._flush(
            chunks=chunks,
            lines=current_lines,
            source=source,
            doc_id=doc_id,
            section=self._section_path(section_stack),
            page=page,
            created_at=created_at,
        )
        return chunks

    def _flush(
        self,
        chunks: List[Document],
        lines: List[str],
        source: str,
        doc_id: str,
        section: str,
        page: Optional[int],
        created_at: str,
    ) -> None:
        text = "\n".join(line for line in lines if line).strip()
        if not text:
            return
        token_count = count_tokens(text)
        if token_count < self.config.min_chunk_tokens and chunks:
            chunks[-1].text = f"{chunks[-1].text}\n\n{text}"
            chunks[-1].metadata["token_count"] = count_tokens(chunks[-1].text)
            return
        metadata = ChunkMetadata(
            doc_id=doc_id,
            source=source,
            chunk_id=len(chunks),
            section=section or "Document",
            page=page,
            created_at=created_at,
            token_count=token_count,
        )
        chunks.append(Document(text=text, metadata=metadata.to_dict()))

    def _iter_blocks(self, text: str) -> Iterable[str]:
        block: List[str] = []
        for line in text.splitlines():
            if line.strip() == "":
                if block:
                    yield "\n".join(block)
                    block = []
                continue
            if self.heading_pattern.match(line) and block:
                yield "\n".join(block)
                block = []
            block.append(line.rstrip())
        if block:
            yield "\n".join(block)

    def _split_long_block(self, text: str) -> List[str]:
        if enc is None:
            tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]|\S", text)
            step = max(1, self.config.max_tokens - self.config.overlap_tokens)
            return ["".join(tokens[start : start + self.config.max_tokens]) for start in range(0, len(tokens), step)]
        tokens = enc.encode(text)
        step = max(1, self.config.max_tokens - self.config.overlap_tokens)
        parts = []
        for start in range(0, len(tokens), step):
            parts.append(enc.decode(tokens[start : start + self.config.max_tokens]))
        return parts

    def _section_path(self, section_stack: List[str]) -> str:
        return " > ".join(section_stack) if section_stack else "Document"

    def _build_doc_id(self, source: str, text: str) -> str:
        normalized = os.path.normpath(source)
        payload = f"{normalized}\n{text[:2048]}".encode("utf-8", errors="ignore")
        return hashlib.sha1(payload).hexdigest()[:16]


class ParentChildChunker:
    """
    Two-pass chunker: small child chunks for precise vector retrieval,
    full-section parent chunks for richer LLM context.

    Workflow:
        child_docs, parent_map = chunker.chunk(text, source)
        # Index child_docs into vector index
        # Pass parent_map to ContextBuilder for parent expansion
    """

    def __init__(
        self,
        child_max_tokens: int = 150,
        child_overlap_tokens: int = 30,
        parent_max_tokens: int = 2000,
    ) -> None:
        self._child_chunker = MarkdownAwareChunker(
            ChunkingConfig(max_tokens=child_max_tokens, overlap_tokens=child_overlap_tokens)
        )
        self._parent_chunker = MarkdownAwareChunker(
            ChunkingConfig(max_tokens=parent_max_tokens, overlap_tokens=0)
        )

    def chunk(
        self,
        text: str,
        source: str,
        doc_id: Optional[str] = None,
        page: Optional[int] = None,
    ) -> tuple[List[Document], dict[str, str]]:
        """
        Returns:
            child_docs  – small chunks to be embedded and indexed
            parent_map  – {parent_id: parent_text} for context expansion
        """
        parent_docs = self._parent_chunker.chunk(text, source, doc_id, page)
        parent_map: dict[str, str] = {}
        child_docs: List[Document] = []

        for parent_doc in parent_docs:
            pdoc_id = parent_doc.metadata.get("doc_id", "")
            pchunk_id = parent_doc.metadata.get("chunk_id", 0)
            parent_id = f"{pdoc_id}::{pchunk_id}"
            parent_map[parent_id] = parent_doc.text

            children = self._child_chunker.chunk(
                parent_doc.text,
                source,
                doc_id=pdoc_id,
                page=page,
            )
            for child in children:
                child.metadata["parent_id"] = parent_id
                child_docs.append(child)

        return child_docs, parent_map
