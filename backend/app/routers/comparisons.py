"""Phase 6B — Comparison endpoints.

    POST   /api/workspaces/{workspace_id}/comparisons        create + run
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
from app.comparison.dimensions import DEFAULT_DIMENSIONS
from app.config import get_settings
from app.database import get_db
from app.deps import get_current_user
from app.models import ComparisonCell, ComparisonMatrix, Document
from app.rag.workspace_context import get_document_titles
from app.rag.workspace_retrieval import WorkspaceRetrievalError
from app.routers.documents import _to_out as document_to_out
from app.schemas import (
    ComparisonCellOut,
    ComparisonCreateIn,
    ComparisonDimensionOut,
    ComparisonMatrixOut,
    ComparisonSummaryOut,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["comparison"])
settings = get_settings()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _cell_out(cell: ComparisonCell, doc_titles: dict[str, str]) -> ComparisonCellOut:
    return ComparisonCellOut(
        id=cell.id,
        document_id=cell.document_id,
        document_title=doc_titles.get(cell.document_id),
        dimension_key=cell.dimension_key,
        dimension_label=cell.dimension_label,
        value_text=cell.value_text,
        insufficient_evidence=cell.insufficient_evidence,
        citations=cell.citations or [],
        llm_provider=cell.llm_provider,
        llm_model=cell.llm_model,
        error=cell.error,
        created_at=cell.created_at,
    )


def _matrix_out(db: Session, matrix: ComparisonMatrix) -> ComparisonMatrixOut:
    documents = list(
        db.query(Document).filter(Document.id.in_(matrix.document_ids)).all()
    )
    # Preserve the matrix's own column order (documents.query has no
    # guaranteed order matching document_ids).
    docs_by_id = {doc.id: doc for doc in documents}
    ordered_documents = [docs_by_id[doc_id] for doc_id in matrix.document_ids if doc_id in docs_by_id]
    doc_titles = get_document_titles(db, matrix.document_ids)

    return ComparisonMatrixOut(
        id=matrix.id,
        workspace_id=matrix.workspace_id,
        title=matrix.title,
        status=matrix.status,
        error=matrix.error,
        dimensions=[ComparisonDimensionOut(**dim) for dim in matrix.dimensions],
        documents=[document_to_out(doc) for doc in ordered_documents],
        cells=[_cell_out(cell, doc_titles) for cell in matrix.cells],
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
