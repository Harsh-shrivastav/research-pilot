"""Orchestrates steps 1-6 of the Phase 3 pipeline for one document: text
cleaning happens inside chunking, then context-aware chunking, embedding
generation, and PostgreSQL/SQLite storage.

Called by the worker right after a document's Phase 2 parse reaches READY,
and by the manual `/reindex` endpoint (e.g. after switching
`EMBEDDING_PROVIDER`, since a document's chunks were embedded with whatever
provider was active when they were last indexed).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import Document, DocumentElement, DocumentSection, RagStatus
from app.rag.chunking import build_chunks
from app.rag.factory import get_embedding_provider
from app.rag.persistence import clear_chunks, persist_chunks

logger = logging.getLogger(__name__)


class IndexingError(RuntimeError):
    pass


def stale_embedding_filter(active_provider: str):
    """SQL filter for documents indexed with a different embedding provider
    than `active_provider` — their vectors are not comparable with new query
    embeddings. Never-indexed documents (NULL) are not "stale"; the worker /
    reindex endpoint handles those. Shared by /api/system/rag's
    `stale_documents` count and scripts/reindex_all.py so both agree."""
    return Document.embedding_provider.is_not(None) & (
        Document.embedding_provider != active_provider
    )


def index_document(db: Session, document: Document, settings: Settings | None = None) -> dict:
    """Chunk, embed and persist one document. Raises IndexingError on
    failure; never leaves the document mid-indexed (chunks are only written
    once embedding succeeds for all of them)."""
    settings = settings or get_settings()

    elements = db.scalars(
        select(DocumentElement)
        .where(DocumentElement.document_id == document.id)
        .order_by(DocumentElement.order_index)
    ).all()
    if not elements:
        raise IndexingError("document has no extracted elements to index yet")

    sections = db.scalars(
        select(DocumentSection).where(DocumentSection.document_id == document.id)
    ).all()
    sections_by_id = {s.id: s for s in sections}

    drafts = build_chunks(
        document.id,
        list(elements),
        sections_by_id,
        parent_tokens=settings.chunk_parent_tokens,
        child_tokens=settings.chunk_child_tokens,
        child_overlap_tokens=settings.chunk_child_overlap_tokens,
    )
    children = [d for d in drafts if d.chunk_level == "child"]
    if not children:
        raise IndexingError("chunking produced no searchable (child) chunks")

    provider = get_embedding_provider(settings)
    try:
        vectors = provider.embed_documents([c.text for c in children])
    except Exception as exc:
        detail = " ".join(str(exc).split())
        if "RESOURCE_EXHAUSTED" in detail or "429" in detail[:40]:
            detail = "the embedding quota is used up for now (free tiers reset daily) - retry indexing later"
        raise IndexingError(f"embedding generation failed ({provider.name}): {detail[:240]}") from exc

    if len(vectors) != len(children):
        raise IndexingError(
            f"embedding provider returned {len(vectors)} vectors for {len(children)} chunks"
        )

    embeddings_by_id = {child.id: vector for child, vector in zip(children, vectors)}

    child_count = persist_chunks(db, document.id, drafts, embeddings_by_id)

    document.rag_status = RagStatus.INDEXED
    document.rag_error = None
    document.rag_indexed_at = datetime.now(timezone.utc)
    document.chunk_count = child_count
    document.embedding_provider = provider.name
    document.embedding_model = getattr(provider, "model", None)

    return {"parents": len(drafts) - child_count, "children": child_count, "provider": provider.name}


def mark_indexing_failed(db: Session, document: Document, message: str) -> None:
    clear_chunks(db, document.id)
    document.rag_status = RagStatus.FAILED
    document.rag_error = message[:2000]
    document.chunk_count = 0
