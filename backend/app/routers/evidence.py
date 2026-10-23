"""Evidence & Research Object Layer API — Task 7.

Minimal surface for the frontend: fetch one piece of evidence, list a
document's evidence, fetch one claim (with its evidence), list a document's
claims, and the paper-level metadata a citation/evidence card needs for its
DOI/arXiv links (Task 9). Follows the existing router conventions
(`app/routers/documents.py`): ownership is always resolved through the
document a row belongs to, via `_get_owned_document`.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.evidence import claim_service, evidence_service, paper_service
from app.models import Claim, Document, Evidence
from app.routers.documents import _get_owned_document
from app.schemas import (
    BBoxOut,
    ClaimOut,
    EvidenceOut,
    ResearchPaperOut,
)

router = APIRouter(tags=["evidence"])


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _paper_out(db: Session, document: Document) -> ResearchPaperOut:
    paper = paper_service.get_or_create_paper(db, document)
    return ResearchPaperOut.model_validate(paper)


def _evidence_out(db: Session, evidence: Evidence, document: Document) -> EvidenceOut:
    bbox = None
    if evidence.bbox_x0 is not None and evidence.bbox_y0 is not None:
        bbox = BBoxOut(
            x0=evidence.bbox_x0,
            y0=evidence.bbox_y0,
            x1=evidence.bbox_x1 if evidence.bbox_x1 is not None else evidence.bbox_x0,
            y1=evidence.bbox_y1 if evidence.bbox_y1 is not None else evidence.bbox_y0,
        )
    return EvidenceOut(
        id=evidence.id,
        document_id=evidence.document_id,
        claim_id=evidence.claim_id,
        chunk_id=evidence.chunk_id,
        page_id=evidence.page_id,
        section_id=evidence.section_id,
        page_number=evidence.page_number,
        section_title=evidence.section_title,
        subsection_title=evidence.subsection_title,
        snippet=evidence.snippet,
        bbox=bbox,
        retrieval_score=evidence.retrieval_score,
        source_type=evidence.source_type,
        confidence=evidence.confidence,
        support_status=evidence.support_status,
        support_reason=((evidence.evidence_metadata or {}).get("verification") or {}).get("reason"),
        support_checked=bool((evidence.evidence_metadata or {}).get("verification")),
        created_at=evidence.created_at,
        document_title=document.title,
        paper=_paper_out(db, document),
    )


def _claim_out(claim: Claim) -> ClaimOut:
    return ClaimOut.model_validate(claim)


# ---------------------------------------------------------------------------
# Ownership helpers
# ---------------------------------------------------------------------------


def _get_owned_evidence(db: Session, evidence_id: str) -> tuple[Evidence, Document]:
    try:
        evidence = evidence_service.get_evidence(db, evidence_id)
    except evidence_service.EvidenceNotFoundError:
        raise HTTPException(status_code=404, detail="Evidence not found.")
    document = _get_owned_document(db, evidence.document_id)  # 404s on ownership mismatch too
    return evidence, document


def _get_owned_claim(db: Session, claim_id: str) -> tuple[Claim, Document]:
    try:
        claim = claim_service.get_claim(db, claim_id)
    except claim_service.ClaimNotFoundError:
        raise HTTPException(status_code=404, detail="Claim not found.")
    document = _get_owned_document(db, claim.document_id)
    return claim, document


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


@router.get("/api/evidence/{evidence_id}", response_model=EvidenceOut)
def get_evidence(evidence_id: str, db: Session = Depends(get_db)) -> EvidenceOut:
    evidence, document = _get_owned_evidence(db, evidence_id)
    return _evidence_out(db, evidence, document)


@router.get("/api/documents/{document_id}/evidence", response_model=list[EvidenceOut])
def list_document_evidence(
    document_id: str, limit: int = 100, offset: int = 0, db: Session = Depends(get_db)
) -> list[EvidenceOut]:
    document = _get_owned_document(db, document_id)
    items = evidence_service.list_evidence_for_document(db, document.id, limit=limit, offset=offset)
    return [_evidence_out(db, e, document) for e in items]


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


@router.get("/api/claims/{claim_id}", response_model=ClaimOut)
def get_claim(claim_id: str, db: Session = Depends(get_db)) -> ClaimOut:
    claim, _ = _get_owned_claim(db, claim_id)
    return _claim_out(claim)


@router.get("/api/claims/{claim_id}/evidence", response_model=list[EvidenceOut])
def list_claim_evidence(claim_id: str, db: Session = Depends(get_db)) -> list[EvidenceOut]:
    claim, document = _get_owned_claim(db, claim_id)
    return [_evidence_out(db, e, document) for e in claim.evidence_items]


@router.get("/api/documents/{document_id}/claims", response_model=list[ClaimOut])
def list_document_claims(
    document_id: str, limit: int = 100, offset: int = 0, db: Session = Depends(get_db)
) -> list[ClaimOut]:
    document = _get_owned_document(db, document_id)
    claims = claim_service.list_claims_for_document(db, document.id, limit=limit, offset=offset)
    return [_claim_out(c) for c in claims]


# ---------------------------------------------------------------------------
# Paper metadata (Task 3 / Task 9)
# ---------------------------------------------------------------------------


@router.get("/api/documents/{document_id}/paper", response_model=ResearchPaperOut)
def get_document_paper(document_id: str, db: Session = Depends(get_db)) -> ResearchPaperOut:
    document = _get_owned_document(db, document_id)
    return _paper_out(db, document)
