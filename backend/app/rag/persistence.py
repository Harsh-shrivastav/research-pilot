"""Persist chunk drafts (with their computed embeddings) into
`document_chunks`. Mirrors the idempotent-reindex shape of
`app/persistence.py`: any previous chunk set for a document is deleted first,
so re-indexing (a manual reindex, or a provider change) never leaves stale
chunks behind.
"""
from __future__ import annotations

import logging

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models import DocumentChunk
from app.rag.chunking import ChunkDraft

logger = logging.getLogger(__name__)

MAX_CHUNKS_PER_DOCUMENT = 20_000


def clear_chunks(db: Session, document_id: str) -> None:
    db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    db.flush()


def persist_chunks(
    db: Session,
    document_id: str,
    drafts: list[ChunkDraft],
    embeddings_by_child_id: dict[str, list[float]],
) -> int:
    """Write parent chunks first (no embedding), then children (embedded).
    Returns the number of child (searchable) chunks written."""
    clear_chunks(db, document_id)

    if len(drafts) > MAX_CHUNKS_PER_DOCUMENT:
        logger.warning(
            "document %s: chunk set truncated %s -> %s", document_id, len(drafts), MAX_CHUNKS_PER_DOCUMENT
        )
    drafts = drafts[:MAX_CHUNKS_PER_DOCUMENT]

    # Parents before children so a child's parent_chunk_id FK always resolves
    # (SQLite enforces FKs when PRAGMA foreign_keys is on; Postgres always does).
    parents = [d for d in drafts if d.chunk_level == "parent"]
    children = [d for d in drafts if d.chunk_level == "child"]

    child_count = 0
    for draft in parents + children:
        embedding = embeddings_by_child_id.get(draft.id) if draft.chunk_level == "child" else None
        db.add(
            DocumentChunk(
                id=draft.id,
                document_id=document_id,
                page_number=draft.page_number,
                section_id=draft.section_id,
                parent_chunk_id=draft.parent_chunk_id,
                chunk_level=draft.chunk_level,
                order_index=draft.order_index,
                text=draft.text,
                char_count=len(draft.text),
                token_count=draft.token_count,
                embedding=embedding,
                chunk_metadata=draft.metadata or None,
            )
        )
        if draft.chunk_level == "child":
            child_count += 1

    db.flush()
    return child_count
