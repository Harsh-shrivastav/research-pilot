"""Persist a `ParsedDocument` into the Phase 2 tables.

This is the only place that knows how the parser's dataclasses map onto ORM
rows. Writing is idempotent per document: an existing extraction is deleted
first, so reprocessing never leaves duplicate pages/sections/elements behind.
"""
from __future__ import annotations

import logging

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models import (
    Document,
    DocumentElement,
    DocumentPage,
    DocumentReference,
    DocumentSection,
)
from app.processing.schema import FIGURE, TABLE, ParsedDocument

logger = logging.getLogger(__name__)

# Guard rails so one pathological PDF cannot exhaust the database.
MAX_ELEMENTS = 20_000
MAX_SECTIONS = 2_000
MAX_REFERENCES = 2_000
MAX_PAGE_TEXT_CHARS = 200_000
MAX_ELEMENT_CHARS = 100_000


def clear_extraction(db: Session, document_id: str) -> None:
    """Remove any previously extracted structure for this document."""
    # Order matters: elements reference pages and sections.
    db.execute(delete(DocumentElement).where(DocumentElement.document_id == document_id))
    db.execute(delete(DocumentReference).where(DocumentReference.document_id == document_id))
    db.execute(delete(DocumentSection).where(DocumentSection.document_id == document_id))
    db.execute(delete(DocumentPage).where(DocumentPage.document_id == document_id))
    db.flush()


def persist(db: Session, document: Document, parsed: ParsedDocument) -> dict[str, int]:
    """Write the parsed structure and update the document's metadata columns.

    Returns a count summary. The caller owns the transaction.
    """
    clear_extraction(db, document.id)

    truncated: list[str] = []

    # -- pages ------------------------------------------------------------
    page_rows: dict[int, DocumentPage] = {}
    for page in parsed.pages:
        text = page.text or ""
        if len(text) > MAX_PAGE_TEXT_CHARS:
            text = text[:MAX_PAGE_TEXT_CHARS]
            truncated.append(f"page {page.page_number} text")
        row = DocumentPage(
            document_id=document.id,
            page_number=page.page_number,
            width=page.width or None,
            height=page.height or None,
            rotation=page.rotation or 0,
            text=text,
            char_count=len(text),
            has_text_layer=bool(page.has_text_layer),
            ocr_applied=bool(page.ocr_applied),
            image_count=page.image_count or 0,
        )
        db.add(row)
        page_rows[page.page_number] = row

    # -- sections ---------------------------------------------------------
    sections = parsed.sections[:MAX_SECTIONS]
    if len(parsed.sections) > MAX_SECTIONS:
        truncated.append(f"sections ({len(parsed.sections)} -> {MAX_SECTIONS})")

    section_rows: dict[int, DocumentSection] = {}
    for section in sections:
        row = DocumentSection(
            document_id=document.id,
            title=(section.title or "Untitled section")[:1000],
            level=max(1, min(section.level, 6)),
            order_index=section.order_index,
            start_page=section.start_page,
            end_page=max(section.end_page, section.start_page),
            text=section.text or "",
            char_count=len(section.text or ""),
        )
        db.add(row)
        section_rows[section.index] = row

    # Flush so every row has its generated id before we wire up foreign keys.
    db.flush()

    for section in sections:
        if section.parent_index is None:
            continue
        child = section_rows.get(section.index)
        parent = section_rows.get(section.parent_index)
        if child is not None and parent is not None:
            child.parent_id = parent.id

    # -- elements ---------------------------------------------------------
    elements = parsed.elements[:MAX_ELEMENTS]
    if len(parsed.elements) > MAX_ELEMENTS:
        truncated.append(f"elements ({len(parsed.elements)} -> {MAX_ELEMENTS})")

    table_count = 0
    figure_count = 0

    for element in elements:
        content = element.content or ""
        if len(content) > MAX_ELEMENT_CHARS:
            content = content[:MAX_ELEMENT_CHARS]
            truncated.append("an element's content")

        bbox = element.bbox
        page_row = page_rows.get(element.page_number)
        section_row = (
            section_rows.get(element.section_index)
            if element.section_index is not None
            else None
        )

        db.add(
            DocumentElement(
                document_id=document.id,
                page_id=page_row.id if page_row is not None else None,
                section_id=section_row.id if section_row is not None else None,
                page_number=element.page_number,
                element_type=element.element_type,
                content=content,
                order_index=element.order_index,
                level=element.level,
                bbox_x0=bbox.x0 if bbox else None,
                bbox_y0=bbox.y0 if bbox else None,
                bbox_x1=bbox.x1 if bbox else None,
                bbox_y1=bbox.y1 if bbox else None,
                element_metadata=element.meta or None,
            )
        )
        if element.element_type == TABLE:
            table_count += 1
        elif element.element_type == FIGURE:
            figure_count += 1

    # -- references -------------------------------------------------------
    references = parsed.references[:MAX_REFERENCES]
    if len(parsed.references) > MAX_REFERENCES:
        truncated.append(f"references ({len(parsed.references)} -> {MAX_REFERENCES})")

    for reference in references:
        db.add(
            DocumentReference(
                document_id=document.id,
                order_index=reference.order_index,
                raw_text=reference.raw_text,
                authors=_clip(reference.authors, 1000),
                title=_clip(reference.title, 1000),
                year=reference.year,
                venue=_clip(reference.venue, 1000),
                doi=_clip(reference.doi, 255),
                url=_clip(reference.url, 1000),
                page_number=reference.page_number,
            )
        )

    # -- document-level metadata -----------------------------------------
    metadata = dict(parsed.metadata or {})
    if truncated:
        metadata["truncated"] = truncated
        logger.warning("document %s: truncated %s", document.id, ", ".join(truncated))

    document.title = _clip(parsed.title, 1000)
    document.authors = _clip(parsed.authors, 1000)
    document.page_count = parsed.page_count
    document.word_count = parsed.word_count
    document.parser_backend = parsed.backend
    document.ocr_used = bool(parsed.ocr_used)
    document.doc_metadata = _jsonable(metadata)
    document.section_count = len(sections)
    document.element_count = len(elements)
    document.table_count = table_count
    document.figure_count = figure_count
    document.reference_count = len(references)

    return {
        "pages": len(parsed.pages),
        "sections": len(sections),
        "elements": len(elements),
        "references": len(references),
        "tables": table_count,
        "figures": figure_count,
    }


def _clip(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def _jsonable(value):
    """Make sure whatever we hand the JSON column can actually serialise."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
