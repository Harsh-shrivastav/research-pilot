"""Evidence service — Task 5 of the Evidence & Research Object Layer.

Responsible for:
  - creating evidence records
  - retrieving evidence
  - attaching evidence to claims
  - validating evidence references (chunk/page/section must belong to the
    same document as the evidence row)
  - converting retrieved chunks into Evidence objects

This module never runs its own retrieval. It only ever turns the *existing*
hybrid retrieval pipeline's `RetrievedChunk` results (see
`app/rag/interfaces.py` and `app/rag/hybrid_retriever.py`) into Evidence rows
— see `evidence_from_retrieved_chunk` / `persist_evidence_from_retrieved_chunks`.
"""
from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    Claim,
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
    Evidence,
    EvidenceSourceType,
    SupportStatus,
)

# `RetrievedChunk` is only needed for type hints; imported lazily inside the
# function that uses it to avoid `app.rag` <-> `app.evidence` import cycles
# (app/rag/service.py imports from app.evidence to attach evidence to chat
# answers — see Task 6).
try:  # pragma: no cover - typing only
    from app.rag.interfaces import RetrievedChunk
except Exception:  # pragma: no cover
    RetrievedChunk = Any  # type: ignore


class EvidenceNotFoundError(LookupError):
    pass


class EvidenceValidationError(ValueError):
    """An evidence reference (chunk/page/section) does not belong to the
    document the evidence row claims to be for."""


def _snippet(text: str, max_chars: int = 500) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "\u2026"


def _validate_reference(db: Session, document_id: str, chunk_id: Optional[str],
                         page_id: Optional[str], section_id: Optional[str]) -> None:
    """Every optional reference, if given, must belong to `document_id`.
    Mirrors the ownership-check pattern used elsewhere in the app
    (`routers/documents.py:_get_owned_document`), just scoped to a document
    instead of a user."""
    if chunk_id is not None:
        chunk = db.get(DocumentChunk, chunk_id)
        if chunk is None or chunk.document_id != document_id:
            raise EvidenceValidationError(f"chunk_id {chunk_id!r} does not belong to document {document_id!r}")
    if page_id is not None:
        page = db.get(DocumentPage, page_id)
        if page is None or page.document_id != document_id:
            raise EvidenceValidationError(f"page_id {page_id!r} does not belong to document {document_id!r}")
    if section_id is not None:
        section = db.get(DocumentSection, section_id)
        if section is None or section.document_id != document_id:
            raise EvidenceValidationError(f"section_id {section_id!r} does not belong to document {document_id!r}")


def create_evidence(
    db: Session,
    *,
    document_id: str,
    snippet: str,
    source_type: str = EvidenceSourceType.PARAGRAPH,
    chunk_id: str | None = None,
    page_id: str | None = None,
    section_id: str | None = None,
    page_number: int | None = None,
    section_title: str | None = None,
    subsection_title: str | None = None,
    bbox: dict[str, float] | None = None,
    retrieval_score: float | None = None,
    confidence: float | None = None,
    support_status: str = SupportStatus.UNCERTAIN,
    claim_id: str | None = None,
    evidence_metadata: dict | None = None,
    commit: bool = True,
) -> Evidence:
    """Create one Evidence row. Validates every reference belongs to
    `document_id` before writing anything (Task 5: "validating evidence
    references")."""
    document = db.get(Document, document_id)
    if document is None:
        raise EvidenceValidationError(f"document {document_id!r} does not exist")

    _validate_reference(db, document_id, chunk_id, page_id, section_id)

    if claim_id is not None:
        claim = db.get(Claim, claim_id)
        if claim is None or claim.document_id != document_id:
            raise EvidenceValidationError(f"claim_id {claim_id!r} does not belong to document {document_id!r}")

    if source_type not in EvidenceSourceType.ALL:
        raise EvidenceValidationError(f"unknown source_type {source_type!r}")
    if support_status not in SupportStatus.ALL:
        raise EvidenceValidationError(f"unknown support_status {support_status!r}")

    bbox = bbox or {}
    evidence = Evidence(
        document_id=document_id,
        claim_id=claim_id,
        chunk_id=chunk_id,
        page_id=page_id,
        section_id=section_id,
        page_number=page_number,
        section_title=section_title,
        subsection_title=subsection_title,
        snippet=_snippet(snippet),
        bbox_x0=bbox.get("x0"),
        bbox_y0=bbox.get("y0"),
        bbox_x1=bbox.get("x1"),
        bbox_y1=bbox.get("y1"),
        retrieval_score=retrieval_score,
        source_type=source_type,
        confidence=confidence,
        support_status=support_status,
        evidence_metadata=evidence_metadata,
    )
    db.add(evidence)
    if commit:
        db.commit()
        db.refresh(evidence)
    else:
        db.flush()
    return evidence


def get_evidence(db: Session, evidence_id: str) -> Evidence:
    evidence = db.get(Evidence, evidence_id)
    if evidence is None:
        raise EvidenceNotFoundError(f"evidence {evidence_id!r} not found")
    return evidence


def list_evidence_for_document(
    db: Session, document_id: str, limit: int = 100, offset: int = 0
) -> list[Evidence]:
    return list(
        db.scalars(
            select(Evidence)
            .where(Evidence.document_id == document_id)
            .order_by(Evidence.created_at.desc())
            .offset(offset)
            .limit(limit)
        ).all()
    )


def attach_evidence_to_claim(db: Session, evidence_id: str, claim_id: str, commit: bool = True) -> Evidence:
    """Attach an existing (possibly standalone) Evidence row to a Claim."""
    evidence = get_evidence(db, evidence_id)
    claim = db.get(Claim, claim_id)
    if claim is None:
        raise EvidenceValidationError(f"claim {claim_id!r} does not exist")
    if claim.document_id != evidence.document_id:
        raise EvidenceValidationError(
            "cannot attach evidence to a claim from a different document "
            f"(evidence.document_id={evidence.document_id!r}, claim.document_id={claim.document_id!r})"
        )
    evidence.claim_id = claim.id
    if commit:
        db.commit()
        db.refresh(evidence)
    return evidence


def evidence_from_retrieved_chunk(chunk: "RetrievedChunk") -> dict:
    """Pure conversion: one `RetrievedChunk` (existing hybrid retrieval
    output — see `app/rag/interfaces.py`) -> the kwargs `create_evidence`
    needs. Does not touch the database, so it can be used to preview evidence
    before persisting it.
    """
    evidence_metadata: dict[str, Any] = {}
    for field in ("vector_score", "keyword_score", "fusion_score", "rerank_score"):
        value = getattr(chunk, field, None)
        if value is not None:
            evidence_metadata[field] = value
    if getattr(chunk, "element_types", None):
        evidence_metadata["element_types"] = list(chunk.element_types)
    if getattr(chunk, "parent_chunk_id", None):
        evidence_metadata["parent_chunk_id"] = chunk.parent_chunk_id

    return dict(
        document_id=chunk.document_id,
        snippet=chunk.child_text,
        source_type=EvidenceSourceType.PARAGRAPH,
        chunk_id=chunk.chunk_id,
        section_id=chunk.section_id,
        page_number=chunk.page_number,
        section_title=chunk.section_title,
        bbox=chunk.bbox,
        retrieval_score=chunk.score,
        evidence_metadata=evidence_metadata or None,
    )


def persist_evidence_from_retrieved_chunks(
    db: Session,
    document_id: str,
    chunks: list["RetrievedChunk"],
    *,
    claim_id: str | None = None,
    support_status: str = SupportStatus.UNCERTAIN,
    commit: bool = True,
) -> list[Evidence]:
    """Batch version of `create_evidence` + `evidence_from_retrieved_chunk`
    for a set of retrieved chunks that all belong to `document_id`. Used by
    `RAGChatService` (Task 6) to ground a chat answer's citations in Evidence
    rows without running a second retrieval pass."""
    created: list[Evidence] = []
    for chunk in chunks:
        if chunk.document_id != document_id:
            continue  # defensive: retrieval is always scoped to one document
        kwargs = evidence_from_retrieved_chunk(chunk)
        kwargs["claim_id"] = claim_id
        kwargs["support_status"] = support_status
        created.append(create_evidence(db, commit=False, **kwargs))
    if commit:
        db.commit()
        for evidence in created:
            db.refresh(evidence)
    return created
