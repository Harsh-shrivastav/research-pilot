"""Context-aware chunking with parent-child relationships: steps 2 and 3 of
the Phase 3 pipeline.

Strategy ("small-to-big" / parent-document chunking)
-----------------------------------------------------
1. Walk the document's already-structured `DocumentElement` rows (headings,
   paragraphs, tables, figures, captions, list items, formulas — Phase 2's
   output) in document order, grouping by section so a chunk never silently
   mixes text from two different sections.
2. Pack consecutive elements into **parent** chunks up to ~`chunk_parent_tokens`
   tokens. A single element (e.g. one large table) is never split across
   parents, so a table always stays intact somewhere.
3. Split each parent's text into smaller, sentence-aligned **child** chunks
   (~`chunk_child_tokens` tokens, `chunk_child_overlap_tokens` overlap). Only
   children are embedded and searched; a hit expands back out to its parent's
   full text before being handed to the LLM, which gives precise retrieval
   (small vectors, less topic dilution) without starving the model of
   surrounding context.
4. A table/figure element is kept as a single parent *and* single child (no
   sub-splitting) so its structure is never torn apart by chunk boundaries.

Every chunk keeps document_id, page_number, section_id, parent_chunk_id and a
metadata dict (section title, contributing element ids/types, best-effort
bbox) so retrieval hits can always be traced back to an exact page.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.models import DocumentElement, DocumentSection
from app.processing.schema import FIGURE, TABLE
from app.rag.text_clean import approx_token_count, clean_text, split_sentences


@dataclass
class ChunkDraft:
    id: str
    document_id: str
    page_number: int
    section_id: Optional[str]
    parent_chunk_id: Optional[str]
    chunk_level: str  # "parent" | "child"
    order_index: int
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return approx_token_count(self.text)


def _section_title(section: Optional[DocumentSection]) -> Optional[str]:
    return section.title if section is not None else None


def _element_bbox(element: DocumentElement) -> Optional[dict]:
    if element.bbox_x0 is None or element.bbox_y0 is None:
        return None
    return {
        "x0": element.bbox_x0,
        "y0": element.bbox_y0,
        "x1": element.bbox_x1 if element.bbox_x1 is not None else element.bbox_x0,
        "y1": element.bbox_y1 if element.bbox_y1 is not None else element.bbox_y0,
    }


def _split_children(
    parent: ChunkDraft,
    document_id: str,
    child_tokens: int,
    overlap_tokens: int,
    order_start: int,
) -> list[ChunkDraft]:
    """Split one parent chunk's text into overlapping, sentence-aligned
    children. Falls back to the whole parent as a single child when the
    parent is already small (the common case for tables/figures/short
    sections), or when it has no sentence-shaped structure to split on."""
    sentences = split_sentences(parent.text)
    if not sentences or approx_token_count(parent.text) <= child_tokens:
        return [
            ChunkDraft(
                id=str(uuid.uuid4()),
                document_id=document_id,
                page_number=parent.page_number,
                section_id=parent.section_id,
                parent_chunk_id=parent.id,
                chunk_level="child",
                order_index=order_start,
                text=parent.text,
                metadata=dict(parent.metadata),
            )
        ]

    children: list[ChunkDraft] = []
    current: list[str] = []
    current_tokens = 0
    order = order_start

    def flush() -> None:
        nonlocal current, current_tokens, order
        if not current:
            return
        text = " ".join(current)
        children.append(
            ChunkDraft(
                id=str(uuid.uuid4()),
                document_id=document_id,
                page_number=parent.page_number,
                section_id=parent.section_id,
                parent_chunk_id=parent.id,
                chunk_level="child",
                order_index=order,
                text=text,
                metadata=dict(parent.metadata),
            )
        )
        order += 1

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        stoks = approx_token_count(sentence)
        if current and current_tokens + stoks > child_tokens:
            flush()
            # Build the overlap tail (trailing sentences up to overlap_tokens)
            # so the next child re-includes a bit of context and doesn't lose
            # a fact that fell right on a boundary.
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(current):
                ptoks = approx_token_count(prev)
                if tail_tokens + ptoks > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_tokens += ptoks
            current = tail
            current_tokens = tail_tokens
        current.append(sentence)
        current_tokens += stoks
        i += 1
    flush()

    return children or [
        ChunkDraft(
            id=str(uuid.uuid4()),
            document_id=document_id,
            page_number=parent.page_number,
            section_id=parent.section_id,
            parent_chunk_id=parent.id,
            chunk_level="child",
            order_index=order_start,
            text=parent.text,
            metadata=dict(parent.metadata),
        )
    ]


def build_chunks(
    document_id: str,
    elements: list[DocumentElement],
    sections_by_id: dict[str, DocumentSection],
    *,
    parent_tokens: int = 1400,
    child_tokens: int = 320,
    child_overlap_tokens: int = 60,
) -> list[ChunkDraft]:
    """Build the full parent+child chunk set for one document's elements.

    `elements` must be in document order (order_index ascending); that is how
    `documents.py` / `DocumentElement.document.elements` already returns them.
    """
    parents: list[ChunkDraft] = []
    all_chunks: list[ChunkDraft] = []
    order_index = 0

    # Group into runs that share a section, preserving document order.
    runs: list[list[DocumentElement]] = []
    current_run: list[DocumentElement] = []
    current_section = "__unset__"
    for element in elements:
        content = clean_text(element.content or "")
        if not content:
            continue
        key = element.section_id or "__none__"
        if key != current_section and current_run:
            runs.append(current_run)
            current_run = []
        current_section = key
        current_run.append(element)
    if current_run:
        runs.append(current_run)

    for run in runs:
        section = sections_by_id.get(run[0].section_id) if run[0].section_id else None
        section_title = _section_title(section)

        buffer: list[DocumentElement] = []
        buffer_tokens = 0

        def close_parent(buf: list[DocumentElement]) -> None:
            nonlocal order_index
            if not buf:
                return
            text = "\n\n".join(clean_text(e.content or "") for e in buf if (e.content or "").strip())
            if not text.strip():
                return
            element_types = sorted({e.element_type for e in buf})
            parent = ChunkDraft(
                id=str(uuid.uuid4()),
                document_id=document_id,
                page_number=buf[0].page_number,
                section_id=buf[0].section_id,
                parent_chunk_id=None,
                chunk_level="parent",
                order_index=order_index,
                text=text,
                metadata={
                    "section_title": section_title,
                    "start_page": buf[0].page_number,
                    "end_page": buf[-1].page_number,
                    "element_ids": [e.id for e in buf],
                    "element_types": element_types,
                    "bbox": _element_bbox(buf[0]),
                },
            )
            order_index += 1
            parents.append(parent)
            all_chunks.append(parent)
            children = _split_children(
                parent, document_id, child_tokens, child_overlap_tokens, order_index
            )
            order_index += len(children)
            all_chunks.extend(children)

        for element in run:
            content = clean_text(element.content or "")
            if not content:
                continue
            tokens = approx_token_count(content)

            # Tables and figures are never merged with neighbouring prose —
            # keep them as their own intact parent+child so structure is
            # never torn across a chunk boundary.
            if element.element_type in (TABLE, FIGURE):
                close_parent(buffer)
                buffer = []
                buffer_tokens = 0
                close_parent([element])
                continue

            if buffer and buffer_tokens + tokens > parent_tokens:
                close_parent(buffer)
                buffer = []
                buffer_tokens = 0

            buffer.append(element)
            buffer_tokens += tokens

        close_parent(buffer)

    return all_chunks
