"""Phase 6B — Comparison endpoints.

    POST   /api/workspaces/{workspace_id}/comparisons        create + run
    POST   /api/comparisons/{comparison_id}/comparability     re-run the comparability check
    GET    /api/workspaces/{workspace_id}/comparisons         list (summary)
    GET    /api/comparisons/{comparison_id}                   full detail
    DELETE /api/comparisons/{comparison_id}                   delete
    GET    /api/comparisons/default-dimensions                canonical dimensions

Ownership is always resolved the same way the rest of the app does it: through
`get_current_user` + an owner_id check (`app/comparison/comparison_service.py`
delegates workspace ownership to `resolve_workspace_documents`, the same
helper `app/rag/workspace_retrieval.py` uses for workspace chat).

The comparison run itself (`create_and_run_comparison`) is synchronous: a
matrix is at most 5 papers x a handful of dimensions, each cell a single
retrieval + a single non-streaming LLM call, so — unlike Phase 2 document
parsing — there's no need for a background worker/polling here. It returns
once every cell (including any that individually failed) is persisted.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.comparison import comparison_service
from app.rag.factory import get_llm_provider
from app.writing import review_writer
from app.evidence.verifier import VerifierUnavailableError
from app.evidence import verification_service
from app.comparison.dimensions import DEFAULT_DIMENSIONS
from app.config import get_settings
from app.database import get_db
from app.deps import get_current_user
from app.models import ComparisonCell, ComparisonMatrix, Document, LiteratureDraft
from app.rag.workspace_context import get_document_titles
from app.rag.workspace_retrieval import WorkspaceRetrievalError
from app.routers.documents import _to_out as document_to_out
from app.schemas import (
    ComparabilityOut,
    ComparisonCellOut,
    ComparisonCreateIn,
    ComparisonDimensionOut,
    ComparisonMatrixOut,
    ComparisonSummaryOut,
    DraftEditIn,
    DraftVerificationOut,
    LiteratureDraftOut,
    VerificationSummaryOut,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["comparison"])
settings = get_settings()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _cell_support(cell: ComparisonCell, support: dict[str, dict]) -> dict:
    """Citation-verification fields of a cell, from its first grounded citation
    (a cell has one Claim, so all its evidence carries the same verdict)."""
    for citation in cell.citations or []:
        info = support.get(citation.get("evidence_id") or "")
        if info:
            return {
                "support_status": info["status"],
                "support_reason": info["reason"],
                "support_checked": info["checked"],
            }
    return {}


def _cell_out(cell: ComparisonCell, doc_titles: dict[str, str], support: dict[str, dict] | None = None) -> ComparisonCellOut:
    support = support or {}
    citations = [
        {**c, **({"support_status": support[c["evidence_id"]]["status"],
                  "support_reason": support[c["evidence_id"]]["reason"],
                  "support_checked": support[c["evidence_id"]]["checked"]}
                 if c.get("evidence_id") in support else {})}
        for c in (cell.citations or [])
    ]
    return ComparisonCellOut(
        id=cell.id,
        document_id=cell.document_id,
        document_title=doc_titles.get(cell.document_id),
        dimension_key=cell.dimension_key,
        dimension_label=cell.dimension_label,
        value_text=cell.value_text,
        insufficient_evidence=cell.insufficient_evidence,
        citations=citations,
        llm_provider=cell.llm_provider,
        llm_model=cell.llm_model,
        error=cell.error,
        quote_verified=cell.quote_verified,
        **_cell_support(cell, support),
        created_at=cell.created_at,
    )


def _comparability_out(raw: dict | None) -> ComparabilityOut | None:
    """The stored comparability dict, or None if absent / not in the expected
    shape (a bad stored value must never break loading the matrix)."""
    if not raw:
        return None
    try:
        return ComparabilityOut(**raw)
    except Exception:
        logger.warning("ignoring a malformed stored comparability value")
        return None


def _matrix_out(db: Session, matrix: ComparisonMatrix) -> ComparisonMatrixOut:
    documents = list(
        db.query(Document).filter(Document.id.in_(matrix.document_ids)).all()
    )
    # Preserve the matrix's own column order (documents.query has no
    # guaranteed order matching document_ids).
    docs_by_id = {doc.id: doc for doc in documents}
    ordered_documents = [docs_by_id[doc_id] for doc_id in matrix.document_ids if doc_id in docs_by_id]
    doc_titles = get_document_titles(db, matrix.document_ids)
    support = verification_service.load_support_info(
        db,
        [c.get("evidence_id") for cell in matrix.cells for c in (cell.citations or []) if c.get("evidence_id")],
    )

    return ComparisonMatrixOut(
        id=matrix.id,
        workspace_id=matrix.workspace_id,
        title=matrix.title,
        status=matrix.status,
        error=matrix.error,
        dimensions=[ComparisonDimensionOut(**dim) for dim in matrix.dimensions],
        documents=[document_to_out(doc) for doc in ordered_documents],
        cells=[_cell_out(cell, doc_titles, support) for cell in matrix.cells],
        comparability=_comparability_out(matrix.comparability),
        created_at=matrix.created_at,
        updated_at=matrix.updated_at,
    )


def _summary_out(matrix: ComparisonMatrix) -> ComparisonSummaryOut:
    return ComparisonSummaryOut(
        id=matrix.id,
        workspace_id=matrix.workspace_id,
        title=matrix.title,
        status=matrix.status,
        document_count=len(matrix.document_ids or []),
        dimension_count=len(matrix.dimensions or []),
        created_at=matrix.created_at,
        updated_at=matrix.updated_at,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/api/comparisons/default-dimensions", response_model=list[ComparisonDimensionOut])
def get_default_dimensions() -> list[ComparisonDimensionOut]:
    """The canonical comparison criteria, for the frontend to offer as
    pre-selected chips (a researcher can still type their own)."""
    return [ComparisonDimensionOut(**dim.to_dict()) for dim in DEFAULT_DIMENSIONS]


@router.post("/api/workspaces/{workspace_id}/comparisons", response_model=ComparisonMatrixOut)
def create_comparison(
    workspace_id: str, payload: ComparisonCreateIn, db: Session = Depends(get_db)
) -> ComparisonMatrixOut:
    user = get_current_user(db)
    if not payload.document_ids:
        raise HTTPException(status_code=400, detail="Select at least two papers to compare.")
    try:
        matrix = comparison_service.create_and_run_comparison(
            db,
            workspace_id=workspace_id,
            owner_id=user.id,
            document_ids=payload.document_ids,
            dimensions_raw=payload.dimensions,
            title=payload.title,
            settings=settings,
        )
    except WorkspaceRetrievalError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except comparison_service.ComparisonError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _matrix_out(db, matrix)


@router.post("/api/comparisons/{comparison_id}/comparability", response_model=ComparisonMatrixOut)
def rerun_comparability(comparison_id: str, db: Session = Depends(get_db)) -> ComparisonMatrixOut:
    """Re-run the "are these papers directly comparable?" check on a stored
    comparison and return the updated matrix."""
    user = get_current_user(db)
    try:
        matrix = comparison_service.rerun_comparability(db, comparison_id, user.id, settings=settings)
    except comparison_service.ComparisonNotFoundError:
        raise HTTPException(status_code=404, detail="Comparison not found.")
    except comparison_service.ComparisonError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception:
        logger.exception("comparability re-run failed for comparison %s", comparison_id)
        raise HTTPException(status_code=500, detail="The comparability check failed. Please try again.")
    return _matrix_out(db, matrix)


@router.post("/api/comparisons/{comparison_id}/verify", response_model=VerificationSummaryOut)
def verify_comparison_citations(comparison_id: str, db: Session = Depends(get_db)) -> VerificationSummaryOut:
    """Judge whether each cell's cited text backs its claim (Gemini as judge,
    plus code checks) and store the verdicts on the Claim/Evidence rows."""
    user = get_current_user(db)
    try:
        summary = verification_service.verify_comparison(db, comparison_id, user.id, settings=settings)
    except comparison_service.ComparisonNotFoundError:
        raise HTTPException(status_code=404, detail="Comparison not found.")
    except verification_service.VerifierUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception:
        logger.exception("citation verification failed for comparison %s", comparison_id)
        raise HTTPException(status_code=500, detail="Citation verification failed. Please try again.")
    return VerificationSummaryOut(**summary)


@router.get("/api/workspaces/{workspace_id}/comparisons", response_model=list[ComparisonSummaryOut])
def list_workspace_comparisons(
    workspace_id: str, db: Session = Depends(get_db)
) -> list[ComparisonSummaryOut]:
    user = get_current_user(db)
    matrices = comparison_service.list_comparisons_for_workspace(db, workspace_id, user.id)
    return [_summary_out(m) for m in matrices]


@router.get("/api/comparisons/{comparison_id}", response_model=ComparisonMatrixOut)
def get_comparison(comparison_id: str, db: Session = Depends(get_db)) -> ComparisonMatrixOut:
    user = get_current_user(db)
    try:
        matrix = comparison_service.get_comparison(db, comparison_id, user.id)
    except comparison_service.ComparisonNotFoundError:
        raise HTTPException(status_code=404, detail="Comparison not found.")
    return _matrix_out(db, matrix)


@router.delete("/api/comparisons/{comparison_id}", status_code=204, response_model=None)
def delete_comparison(comparison_id: str, db: Session = Depends(get_db)) -> None:
    user = get_current_user(db)
    try:
        comparison_service.delete_comparison(db, comparison_id, user.id)
    except comparison_service.ComparisonNotFoundError:
        raise HTTPException(status_code=404, detail="Comparison not found.")


# ---------------------------------------------------------------------------
# Related-work drafts (app/writing/review_writer.py)
# ---------------------------------------------------------------------------


def _draft_out(draft: LiteratureDraft) -> LiteratureDraftOut:
    return LiteratureDraftOut(
        id=draft.id,
        comparison_id=draft.comparison_id,
        content=draft.content,
        edited_text=draft.edited_text,
        created_at=draft.created_at,
        updated_at=draft.updated_at,
    )


def _get_owned_draft(db: Session, draft_id: str, owner_id: str) -> LiteratureDraft:
    draft = db.get(LiteratureDraft, draft_id)
    if draft is None or draft.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Draft not found.")
    return draft


@router.post("/api/comparisons/{comparison_id}/draft", response_model=LiteratureDraftOut)
def create_draft(comparison_id: str, db: Session = Depends(get_db)) -> LiteratureDraftOut:
    """Generate a cited related-work draft from the matrix and store it."""
    user = get_current_user(db)
    try:
        matrix = comparison_service.get_comparison(db, comparison_id, user.id)
    except comparison_service.ComparisonNotFoundError:
        raise HTTPException(status_code=404, detail="Comparison not found.")
    try:
        llm = get_llm_provider(settings)
        content = review_writer.generate_draft(db, settings, llm, matrix)
    except review_writer.DraftUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except review_writer.DraftError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("draft generation failed for comparison %s", comparison_id)
        raise HTTPException(status_code=422, detail=review_writer.NOT_GROUNDED_MESSAGE)
    draft = LiteratureDraft(comparison_id=matrix.id, owner_id=user.id, content=content)
    db.add(draft)
    db.commit()
    db.refresh(draft)
    return _draft_out(draft)


@router.get("/api/comparisons/{comparison_id}/draft", response_model=LiteratureDraftOut | None)
def get_latest_draft(comparison_id: str, db: Session = Depends(get_db)) -> LiteratureDraftOut | None:
    user = get_current_user(db)
    try:
        matrix = comparison_service.get_comparison(db, comparison_id, user.id)
    except comparison_service.ComparisonNotFoundError:
        raise HTTPException(status_code=404, detail="Comparison not found.")
    draft = (
        db.query(LiteratureDraft)
        .filter(LiteratureDraft.comparison_id == matrix.id, LiteratureDraft.owner_id == user.id)
        .order_by(LiteratureDraft.created_at.desc())
        .first()
    )
    return _draft_out(draft) if draft else None


@router.put("/api/drafts/{draft_id}", response_model=LiteratureDraftOut)
def save_draft(draft_id: str, payload: DraftEditIn, db: Session = Depends(get_db)) -> LiteratureDraftOut:
    """Save the user's edited text. The generated content is never modified."""
    user = get_current_user(db)
    draft = _get_owned_draft(db, draft_id, user.id)
    draft.edited_text = payload.edited_text
    db.commit()
    db.refresh(draft)
    return _draft_out(draft)


@router.post("/api/drafts/{draft_id}/verify", response_model=DraftVerificationOut)
def verify_draft(draft_id: str, db: Session = Depends(get_db)) -> DraftVerificationOut:
    """Judge each sentence against the evidence it cites (P4's verifier)."""
    user = get_current_user(db)
    draft = _get_owned_draft(db, draft_id, user.id)
    try:
        sentences = review_writer.verify_draft(db, get_llm_provider(settings), draft.content)
    except VerifierUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception:
        logger.exception("draft verification failed for draft %s", draft_id)
        raise HTTPException(status_code=500, detail="Draft verification failed. Please try again.")
    return DraftVerificationOut(sentences=sentences)
