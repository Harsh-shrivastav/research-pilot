"""Phase 6A — Workspace retrieval service.

A thin composition layer that reuses the existing HybridRetriever (and
VectorRetriever fallback) to retrieve across multiple documents belonging
to one workspace.  The architecture:

    Workspace
       ↓
    Selected Documents
       ↓
    Existing Hybrid Retrieval (per-document)
       ↓
    Workspace-filtered RetrievedChunks (merged + re-ranked)
       ↓
    Evidence
       ↓
    Context Construction
       ↓
    LLM
       ↓
    Answer + Citations

This module NEVER modifies the HybridRetriever algorithm.  It calls
``retriever.retrieve(document_id, query, top_k)`` once per workspace
document and merges the results, preserving every field on RetrievedChunk
(document_id, chunk_id, page_number, section, scores, etc.).
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    Document,
    RagStatus,
    ResearchWorkspace,
    WorkspaceDocument,
)
from app.rag.interfaces import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)


class WorkspaceRetrievalError(RuntimeError):
    """User-facing error (no documents, not indexed, ownership, etc.)."""


def resolve_workspace_documents(
    db: Session, workspace_id: str, owner_id: str
) -> list[Document]:
    """Return the Document objects belonging to this workspace,
    after verifying the workspace is owned by the given user."""
    workspace = db.scalar(
        select(ResearchWorkspace).where(
            ResearchWorkspace.id == workspace_id,
            ResearchWorkspace.owner_id == owner_id,
        )
    )
    if workspace is None:
        raise WorkspaceRetrievalError("Workspace not found.")

    links = db.scalars(
        select(WorkspaceDocument).where(
            WorkspaceDocument.workspace_id == workspace_id
        )
    ).all()
    if not links:
        raise WorkspaceRetrievalError(
            "This workspace has no documents. Add at least one document before asking questions."
        )

    doc_ids = [link.document_id for link in links]
    documents = list(
        db.scalars(
            select(Document).where(
                Document.id.in_(doc_ids),
                Document.owner_id == owner_id,
            )
        ).all()
    )
    return documents


def indexed_workspace_documents(db: Session, workspace_id: str, owner_id: str) -> list[Document]:
    """The workspace's documents that are indexed for chat. Raises
    WorkspaceRetrievalError (user-facing) if there are none."""
    documents = resolve_workspace_documents(db, workspace_id, owner_id)

    indexed_docs = [d for d in documents if d.rag_status == RagStatus.INDEXED and d.chunk_count > 0]
    if not indexed_docs:
        not_ready = [
            f"{d.title or d.original_filename} ({d.rag_status})"
            for d in documents
        ]
        raise WorkspaceRetrievalError(
            "None of the workspace documents have been indexed for chat yet. "
            f"Documents: {', '.join(not_ready)}. "
            "Index your documents first (they are indexed automatically after upload processing completes)."
        )
    return indexed_docs


def merge_with_quota(
    chunks_by_doc: dict[str, list[RetrievedChunk]], top_k: int, min_per_paper: int
) -> list[RetrievedChunk]:
    """Merge per-paper results so every paper contributes its best
    min(min_per_paper, available) chunks before any paper gets more.

    Quota chunks come first (round-robin by per-paper rank, so context
    trimming at the end of the list never removes a paper's quota), then the
    remaining slots up to `top_k` go to the best leftover chunks by score.
    The result can exceed `top_k` when the quotas alone do.
    """
    ranked = {doc_id: sorted(chunks, key=lambda c: c.score, reverse=True) for doc_id, chunks in chunks_by_doc.items()}
    quota = max(0, min_per_paper)

    selected: list[RetrievedChunk] = []
    for rank in range(quota):
        for chunks in ranked.values():
            if rank < len(chunks):
                selected.append(chunks[rank])

    leftovers = [c for chunks in ranked.values() for c in chunks[quota:]]
    leftovers.sort(key=lambda c: c.score, reverse=True)
    selected.extend(leftovers[: max(0, top_k - len(selected))])
    return selected


def retrieve_for_workspace(
    retriever: Retriever,
    db: Session,
    workspace_id: str,
    owner_id: str,
    query: str,
    top_k: int,
    *,
    queries_by_doc: dict[str, str] | None = None,
    min_per_paper: int = 2,
    min_score: float = 0.0,
    documents: list[Document] | None = None,
) -> list[RetrievedChunk]:
    """Multi-document retrieval: run the existing retrieval pipeline once per
    workspace document (with that paper's rewritten sub-query when given),
    then merge with a per-paper quota (see `merge_with_quota`).

    `min_score` > 0 drops a paper entirely when its best final score is
    below it (the optional pre-LLM gate; 0 = off).

    Every RetrievedChunk retains its original document_id, chunk_id,
    page_number, section, snippet, and all retrieval scores.
    """
    indexed_docs = documents if documents is not None else indexed_workspace_documents(db, workspace_id, owner_id)
    queries_by_doc = queries_by_doc or {}

    chunks_by_doc: dict[str, list[RetrievedChunk]] = {}
    for doc in indexed_docs:
        try:
            chunks = retriever.retrieve(doc.id, queries_by_doc.get(doc.id, query), top_k)
        except Exception:
            logger.exception(
                "retrieval failed for workspace document %s (%s); skipping",
                doc.id,
                doc.title or doc.original_filename,
            )
            # Don't fail the whole workspace query if one document errors
            continue
        if not chunks:
            continue
        best = max(c.score for c in chunks)
        if min_score > 0 and best < min_score:
            logger.info("workspace doc %s below retrieval_min_score (%.3f < %.3f); excluded", doc.id, best, min_score)
            continue
        chunks_by_doc[doc.id] = chunks

    return merge_with_quota(chunks_by_doc, top_k, min_per_paper)
