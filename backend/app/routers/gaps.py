"""Phase 8 — Research Gap Detection API router.

Endpoints:
    POST /api/workspaces/{workspace_id}/gaps/analyze  → Run gap detection
    GET  /api/workspaces/{workspace_id}/gaps           → List gaps
    GET  /api/gaps/{gap_id}                            → Get gap detail
    GET  /api/gaps/{gap_id}/evidence                   → Get gap evidence
    GET  /api/gaps/{gap_id}/claims                     → Get gap claims (with evidence)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.gaps import gap_service
from app.models import GapType, User
from app.rag.workspace_retrieval import WorkspaceRetrievalError
from app.schemas import (
    EvidenceOut,
    GapAnalysisResult,
    GapClaimOut,
    GapEvidenceOut,
    ResearchGapOut,
    ResearchGapSummaryOut,
)

router = APIRouter(tags=["gaps"])


def _gap_summary(gap) -> ResearchGapSummaryOut:
    return ResearchGapSummaryOut(
        id=gap.id,
        workspace_id=gap.workspace_id,
        title=gap.title,
        description=gap.description,
        gap_type=gap.gap_type,
        gap_type_label=GapType.LABELS.get(gap.gap_type, gap.gap_type),
        confidence=gap.confidence,
        status=gap.status,
        supporting_paper_count=gap.supporting_paper_count,
        evidence_count=len(gap.gap_evidence_items) if gap.gap_evidence_items else 0,
        research_direction=gap.research_direction,
        created_at=gap.created_at,
        updated_at=gap.updated_at,
    )


def _evidence_out(ev) -> EvidenceOut:
    """Build EvidenceOut, inlining the paper and document title."""
    bbox = None
    if ev.bbox_x0 is not None:
        bbox = {"x0": ev.bbox_x0, "y0": ev.bbox_y0, "x1": ev.bbox_x1, "y1": ev.bbox_y1}

    paper_out = None
    if hasattr(ev, "document") and ev.document and hasattr(ev.document, "paper") and ev.document.paper:
        p = ev.document.paper
        paper_out = {
            "id": p.id,
            "document_id": p.document_id,
            "title": p.title,
            "authors": p.authors,
            "abstract": p.abstract,
            "publication_year": p.publication_year,
            "venue": p.venue,
            "doi": p.doi,
            "doi_url": p.doi_url,
            "arxiv_id": p.arxiv_id,
            "arxiv_url": p.arxiv_url,
            "publisher_url": p.publisher_url,
            "source_url": p.source_url,
            "metadata_source": p.metadata_source,
            "metadata_confidence": p.metadata_confidence,
            "created_at": p.created_at,
            "updated_at": p.updated_at,
        }

    doc_title = None
    if hasattr(ev, "document") and ev.document:
        doc_title = ev.document.title or ev.document.original_filename

    return EvidenceOut(
        id=ev.id,
        document_id=ev.document_id,
        claim_id=ev.claim_id,
        chunk_id=ev.chunk_id,
        page_id=ev.page_id,
        section_id=ev.section_id,
        page_number=ev.page_number,
        section_title=ev.section_title,
        subsection_title=ev.subsection_title,
        snippet=ev.snippet,
        bbox=bbox,
        retrieval_score=ev.retrieval_score,
        source_type=ev.source_type,
        confidence=ev.confidence,
        support_status=ev.support_status,
        created_at=ev.created_at,
        document_title=doc_title,
        paper=paper_out,
    )


def _claim_out(gc) -> GapClaimOut:
    claim = gc.claim
    doc = claim.document
    return GapClaimOut(
        id=gc.id,
        claim_id=claim.id,
        document_id=claim.document_id,
        document_title=(doc.title or doc.original_filename) if doc else None,
        claim_text=claim.claim_text,
        claim_type=claim.claim_type,
        support_status=claim.support_status,
        evidence=[_evidence_out(ev) for ev in claim.evidence_items],
    )


def _gap_detail(gap) -> ResearchGapOut:
    evidence_items = []
    for ge in (gap.gap_evidence_items or []):
        ev_out = None
        if ge.evidence:
            ev_out = _evidence_out(ge.evidence)
        evidence_items.append(GapEvidenceOut(
            id=ge.id,
            evidence_id=ge.evidence_id,
            role=ge.role,
            evidence=ev_out,
        ))

    return ResearchGapOut(
        claims=[_claim_out(gc) for gc in (gap.gap_claim_items or [])],
        id=gap.id,
        workspace_id=gap.workspace_id,
        title=gap.title,
        description=gap.description,
        gap_type=gap.gap_type,
        gap_type_label=GapType.LABELS.get(gap.gap_type, gap.gap_type),
        confidence=gap.confidence,
        confidence_reasoning=gap.confidence_reasoning,
        status=gap.status,
        research_direction=gap.research_direction,
        source_paper_ids=gap.source_paper_ids or [],
        supporting_paper_count=gap.supporting_paper_count,
        evidence_items=evidence_items,
        created_at=gap.created_at,
        updated_at=gap.updated_at,
    )


@router.post("/api/workspaces/{workspace_id}/gaps/analyze")
def analyze_gaps(
    workspace_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> GapAnalysisResult:
    try:
        outcome = gap_service.analyze_gaps(db, workspace_id, user.id)
    except WorkspaceRetrievalError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except gap_service.GapDetectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    gaps = outcome.gaps
    type_counts: dict[str, int] = {}
    for g in gaps:
        label = GapType.LABELS.get(g.gap_type, g.gap_type)
        type_counts[label] = type_counts.get(label, 0) + 1

    return GapAnalysisResult(
        papers_analyzed=outcome.papers_analyzed,
        gaps_found=len(gaps),
        candidates_rejected=outcome.candidates_rejected,
        gaps=[_gap_summary(g) for g in gaps],
        gap_type_counts=type_counts,
    )


@router.get("/api/workspaces/{workspace_id}/gaps")
def list_gaps(
    workspace_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[ResearchGapSummaryOut]:
    try:
        gaps = gap_service.list_gaps(db, workspace_id, user.id)
    except WorkspaceRetrievalError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return [_gap_summary(g) for g in gaps]


@router.get("/api/gaps/{gap_id}")
def get_gap(
    gap_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResearchGapOut:
    try:
        gap = gap_service.get_gap(db, gap_id, user.id)
    except gap_service.GapNotFoundError:
        raise HTTPException(status_code=404, detail="Gap not found")
    return _gap_detail(gap)


@router.get("/api/gaps/{gap_id}/evidence")
def get_gap_evidence(
    gap_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[GapEvidenceOut]:
    try:
        items = gap_service.get_gap_evidence(db, gap_id, user.id)
    except gap_service.GapNotFoundError:
        raise HTTPException(status_code=404, detail="Gap not found")

    result = []
    for ge in items:
        ev_out = None
        if ge.evidence:
            ev_out = _evidence_out(ge.evidence)
        result.append(GapEvidenceOut(
            id=ge.id,
            evidence_id=ge.evidence_id,
            role=ge.role,
            evidence=ev_out,
        ))
    return result


@router.get("/api/gaps/{gap_id}/claims")
def get_gap_claims(
    gap_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[GapClaimOut]:
    try:
        items = gap_service.get_gap_claims(db, gap_id, user.id)
    except gap_service.GapNotFoundError:
        raise HTTPException(status_code=404, detail="Gap not found")
    return [_claim_out(gc) for gc in items]
