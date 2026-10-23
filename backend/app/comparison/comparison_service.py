"""Phase 6B — Comparison service.

Orchestrates the full flow:

    Select 2-5 Papers -> Compare Papers
        -> Targeted Retrieval   (existing Retriever, per paper x dimension)
        -> Evidence             (existing Evidence layer)
        -> Structured Extraction(existing LLMProvider.generate(), one short
                                  grounded cell per paper x dimension)
        -> Comparison Matrix    (ComparisonMatrix + ComparisonCell)

This is the comparison counterpart to `RAGChatService`
(`app/rag/service.py`) and `WorkspaceChatService`
(`app/rag/workspace_chat.py`): same building blocks, reused directly
(`build_retriever`, `_parse_citations`, `_snippet`, `get_llm_provider`,
`evidence_from_retrieved_chunk` / `create_evidence`, `create_claim`),
composed into a batch of short "answers" instead of one streamed chat turn.
Nothing here runs a second retrieval implementation or a second citation
parser.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.comparison.dimensions import Dimension, resolve_dimensions
from app.config import Settings, get_settings
from app.evidence import claim_service, evidence_service
from app.models import (
    ComparisonCell,
    ComparisonMatrix,
    ComparisonStatus,
    Document,
    RagStatus,
    SupportStatus,
)
from app.rag.comparison_context import (
    COMPARISON_SYSTEM_PROMPT,
    CELL_NOT_FOUND_SENTENCE,
    build_comparison_cell_message,
)
from app.rag.factory import get_llm_provider
from app.rag.gemini_limits import short_error
from app.rag.interfaces import Citation, LLMMessage, LLMProvider, Retriever, RetrievedChunk
from app.rag.service import _parse_citations, build_retriever, llm_model_used
from app.rag.workspace_retrieval import WorkspaceRetrievalError, resolve_workspace_documents

logger = logging.getLogger(__name__)


class ComparisonError(RuntimeError):
    """A user-facing error: wrong paper count, papers not in this workspace,
    papers not indexed yet, etc. Deliberately a distinct type from
    `WorkspaceRetrievalError` (which this module also lets propagate for
    workspace-not-found / not-owned / no-documents) so the router can 404 vs
    400 correctly without string-matching error messages."""


class ComparisonNotFoundError(LookupError):
    pass


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_selection(
    settings: Settings, workspace_documents: list[Document], document_ids: list[str]
) -> list[Document]:
    """Resolve the requested document_ids against the workspace's own
    documents, enforcing the 2-5 paper rule and that every selected paper
    has finished RAG indexing (comparison retrieval needs embeddings exactly
    like chat does — see `RAGChatService.answer`'s own `rag_status` check)."""
    requested = list(dict.fromkeys(document_ids))  # dedupe, keep order

    if len(requested) < settings.comparison_min_papers:
        raise ComparisonError(
            f"Select at least {settings.comparison_min_papers} papers to compare."
        )
    if len(requested) > settings.comparison_max_papers:
        raise ComparisonError(
            f"Select at most {settings.comparison_max_papers} papers to compare."
        )

    by_id = {doc.id: doc for doc in workspace_documents}
    missing = [doc_id for doc_id in requested if doc_id not in by_id]
    if missing:
        raise ComparisonError(
            "One or more selected papers are not in this workspace: " + ", ".join(missing)
        )

    selected = [by_id[doc_id] for doc_id in requested]
    not_indexed = [doc for doc in selected if doc.rag_status != RagStatus.INDEXED or not doc.chunk_count]
    if not_indexed:
        names = ", ".join(
            f"{doc.title or doc.original_filename} ({doc.rag_status})" for doc in not_indexed
        )
        raise ComparisonError(
            "These papers have not finished indexing for chat yet, so they cannot be "
            f"compared: {names}. Try again shortly, or trigger a reindex."
        )
    return selected


# ---------------------------------------------------------------------------
# Cell extraction
# ---------------------------------------------------------------------------


@dataclass
class _PendingCell:
    """One (paper, dimension) cell between retrieval and persistence."""

    document: Document
    dimension: Dimension
    order_index: int
    user_content: str = ""
    used_chunks: list[RetrievedChunk] = field(default_factory=list)
    retrieval_error: str | None = None
    answer: str | None = None
    generation_error: str | None = None


def _prepare_cell(
    settings: Settings,
    retriever: Retriever,
    document: Document,
    dimension: Dimension,
    order_index: int,
) -> _PendingCell:
    """Targeted retrieval + prompt for one cell. Never raises: a retrieval
    failure is recorded on the pending cell and surfaces as that cell's
    `error`, so one bad cell cannot take down the rest of the matrix."""
    pending = _PendingCell(document=document, dimension=dimension, order_index=order_index)
    try:
        retrieved: list[RetrievedChunk] = retriever.retrieve(
            document.id, dimension.query, settings.comparison_top_k
        )
    except Exception as exc:
        logger.exception(
            "comparison retrieval failed for document %s, dimension %s", document.id, dimension.key
        )
        pending.retrieval_error = f"Retrieval failed: {short_error(exc)}"
        return pending
    pending.user_content, pending.used_chunks = build_comparison_cell_message(
        dimension.query, retrieved, settings.comparison_max_context_chars
    )
    return pending


def _generate_cell(llm: LLMProvider, pending: _PendingCell) -> None:
    """The LLM call for one cell (network only - safe to run in a thread).
    Never raises; the outcome lands on `pending`."""
    if pending.retrieval_error:
        return
    try:
        pending.answer = llm.generate(
            COMPARISON_SYSTEM_PROMPT, [LLMMessage(role="user", content=pending.user_content)]
        )
    except Exception as exc:
        logger.warning(
            "comparison generation failed for document %s, dimension %s: %s",
            pending.document.id, pending.dimension.key, short_error(exc),
        )
        pending.generation_error = f"Generation failed: {short_error(exc)}"


def _persist_cell(
    db: Session, matrix: ComparisonMatrix, llm: LLMProvider, pending: _PendingCell
) -> ComparisonCell:
    """Write the cell (and its evidence) for a prepared + generated cell."""
    document, dimension = pending.document, pending.dimension
    base = dict(
        comparison_id=matrix.id,
        document_id=document.id,
        dimension_key=dimension.key,
        dimension_label=dimension.label,
        order_index=pending.order_index,
    )
    error = pending.retrieval_error or pending.generation_error
    if error:
        cell = ComparisonCell(**base, value_text="", error=error)
        db.add(cell)
        db.flush()
        return cell

    answer = (pending.answer or "").strip() or CELL_NOT_FOUND_SENTENCE
    citations, insufficient = _parse_citations(answer, pending.used_chunks)

    cell = ComparisonCell(
        **base,
        value_text=answer,
        insufficient_evidence=insufficient,
        citations=[c.to_dict() for c in citations],
        llm_provider=llm.name,
        llm_model=llm_model_used(llm),
    )
    db.add(cell)
    db.flush()

    _ground_cell_in_evidence(db, matrix, document, dimension, cell, citations, pending.used_chunks, insufficient)
    cell.citations = [c.to_dict() for c in citations]
    return cell


def _ground_cell_in_evidence(
    db: Session,
    matrix: ComparisonMatrix,
    document: Document,
    dimension: Dimension,
    cell: ComparisonCell,
    citations: list[Citation],
    used_chunks: list[RetrievedChunk],
    insufficient: bool,
) -> None:
    """Same pattern as `RAGChatService._ground_citations_in_evidence`: wrap
    in a SAVEPOINT so an evidence-grounding failure never loses the cell's
    already-computed answer, just its evidence links."""
    if not citations or insufficient:
        return
    try:
        with db.begin_nested():
            claim = claim_service.create_claim(
                db,
                document_id=document.id,
                claim_text=cell.value_text,
                claim_type=dimension.claim_type,
                support_status=SupportStatus.UNCERTAIN,
                workspace_id=matrix.workspace_id,
                commit=False,
            )
            chunks_by_id = {chunk.chunk_id: chunk for chunk in used_chunks}
            for citation in citations:
                chunk = chunks_by_id.get(citation.chunk_id)
                if chunk is None:
                    continue
                kwargs = evidence_service.evidence_from_retrieved_chunk(chunk)
                evidence = evidence_service.create_evidence(db, claim_id=claim.id, commit=False, **kwargs)
                citation.evidence_id = evidence.id
            db.flush()
    except Exception:
        logger.exception(
            "evidence grounding failed for comparison cell (document %s, dimension %s) "
            "— cell answer still returned",
            document.id,
            dimension.key,
        )
        for citation in citations:
            citation.evidence_id = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_and_run_comparison(
    db: Session,
    workspace_id: str,
    owner_id: str,
    document_ids: list[str],
    dimensions_raw: list[str] | None = None,
    title: str | None = None,
    settings: Settings | None = None,
) -> ComparisonMatrix:
    """Validate the request, then run the whole
    Targeted-Retrieval -> Evidence -> Structured-Extraction pipeline
    synchronously, cell by cell, and persist the result.

    Raises `WorkspaceRetrievalError` (workspace not found/owned, or has no
    documents at all) or `ComparisonError` (bad paper count/selection, or
    selected papers not indexed) *before* creating any row, so a rejected
    request never leaves a partial matrix behind.
    """
    settings = settings or get_settings()

    workspace_documents = resolve_workspace_documents(db, workspace_id, owner_id)
    selected_docs = _validate_selection(settings, workspace_documents, document_ids)
    dimensions = resolve_dimensions(dimensions_raw, max_dimensions=settings.comparison_max_dimensions)

    matrix = ComparisonMatrix(
        workspace_id=workspace_id,
        owner_id=owner_id,
        title=title,
        document_ids=[doc.id for doc in selected_docs],
        dimensions=[dim.to_dict() for dim in dimensions],
        status=ComparisonStatus.RUNNING,
    )
    db.add(matrix)
    # Commit now: the matrix row is visible as RUNNING, and the write lock is
    # released while the (slow) LLM calls run.
    db.commit()

    try:
        retriever = build_retriever(db, settings)
        llm = get_llm_provider(settings)
    except Exception as exc:
        logger.exception("comparison %s: could not build retriever/LLM", matrix.id)
        matrix.status = ComparisonStatus.FAILED
        matrix.error = f"Retrieval/LLM setup failed: {exc}"
        db.commit()
        db.refresh(matrix)
        return matrix

    try:
        # Row-major (dimension outer, paper inner) so cells persist in the
        # exact order the matrix is rendered: one dimension's row across
        # every paper, then the next dimension.
        plan = [(dimension, document) for dimension in dimensions for document in selected_docs]

        # 1) Retrieval + prompts: database work, sequential (a Session is not
        #    thread-safe). Identical queries are served from the provider's
        #    embedding cache, so this is fast.
        pending = [
            _prepare_cell(settings, retriever, document, dimension, index)
            for index, (dimension, document) in enumerate(plan)
        ]

        # 2) The slow part - one LLM call per cell - runs in parallel.
        workers = max(1, min(settings.comparison_max_parallel, len(pending)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cmp-cell") as pool:
            list(pool.map(lambda cell: _generate_cell(llm, cell), pending))

        # 3) Persist in matrix order.
        for cell in pending:
            _persist_cell(db, matrix, llm, cell)
        matrix.status = ComparisonStatus.COMPLETED
    except Exception as exc:
        # _prepare_cell / _generate_cell catch per-cell failures; this is a
        # genuine defensive backstop for anything unexpected escaping.
        logger.exception("comparison %s failed unexpectedly", matrix.id)
        matrix.status = ComparisonStatus.FAILED
        matrix.error = str(exc)

    db.commit()
    db.refresh(matrix)
    return matrix


def get_comparison(db: Session, comparison_id: str, owner_id: str) -> ComparisonMatrix:
    matrix = db.scalar(
        select(ComparisonMatrix).where(
            ComparisonMatrix.id == comparison_id, ComparisonMatrix.owner_id == owner_id
        )
    )
    if matrix is None:
        raise ComparisonNotFoundError(f"comparison {comparison_id!r} not found")
    return matrix


def list_comparisons_for_workspace(
    db: Session, workspace_id: str, owner_id: str
) -> list[ComparisonMatrix]:
    return list(
        db.scalars(
            select(ComparisonMatrix)
            .where(
                ComparisonMatrix.workspace_id == workspace_id,
                ComparisonMatrix.owner_id == owner_id,
            )
            .order_by(ComparisonMatrix.created_at.desc())
        ).all()
    )


def delete_comparison(db: Session, comparison_id: str, owner_id: str) -> None:
    matrix = get_comparison(db, comparison_id, owner_id)
    db.delete(matrix)
    db.commit()
