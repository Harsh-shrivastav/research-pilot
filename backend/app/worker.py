"""Background document-processing worker.

Design notes
------------
The queue is the `documents` table itself rather than Redis/Celery, because
Phase 1's stack is FastAPI + SQLAlchemy + Postgres and adding a broker would be
a disproportionate amount of new infrastructure for this phase. What this gives
us that an in-memory `BackgroundTasks` would not:

  * **Restart-safe.** State lives in the database. If the container dies
    mid-parse, `migrate._requeue_interrupted` hands the document back on boot.
  * **Atomic claiming.** Workers claim via a conditional UPDATE, so two threads
    (or two backend replicas) never process the same document twice.
  * **Horizontally scalable.** `python -m app.worker` runs the same loop as a
    separate process/container with no code changes.

The API never blocks on parsing: upload writes the file, inserts a row with
status UPLOADED, and returns. A worker thread picks it up within `poll_interval`.
"""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
from datetime import datetime, timezone

from sqlalchemy import select, update

from app.config import get_settings
from app.database import SessionLocal, engine
from app.models import Document, ProcessingStatus, RagStatus
from app.persistence import persist
from app.processing.pipeline import ProcessingError, process_pdf
from app.rag.indexing import IndexingError, index_document, mark_indexing_failed

logger = logging.getLogger(__name__)
settings = get_settings()

_shutdown = threading.Event()
_threads: list[threading.Thread] = []


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Claiming
# ---------------------------------------------------------------------------


def claim_next_document(worker_name: str) -> str | None:
    """Atomically move one UPLOADED document to PROCESSING and return its id.

    The UPDATE carries `processing_status == UPLOADED` in its WHERE clause, so
    if another worker got there first our rowcount is 0 and we simply try the
    next candidate. This is the whole concurrency-control story — no locks, no
    broker, and correct under multiple processes.
    """
    with SessionLocal() as db:
        candidates = db.scalars(
            select(Document.id)
            .where(Document.processing_status == ProcessingStatus.UPLOADED)
            .order_by(Document.uploaded_at.asc())
            .limit(10)
        ).all()

        for document_id in candidates:
            result = db.execute(
                update(Document)
                .where(
                    Document.id == document_id,
                    Document.processing_status == ProcessingStatus.UPLOADED,
                )
                .values(
                    processing_status=ProcessingStatus.PROCESSING,
                    processing_started_at=_now(),
                    processing_completed_at=None,
                    processing_error=None,
                    processing_attempts=Document.processing_attempts + 1,
                )
            )
            if result.rowcount:
                db.commit()
                logger.info("%s claimed document %s", worker_name, document_id)
                return document_id
            db.rollback()
    return None


def claim_next_unindexed_document(worker_name: str) -> str | None:
    """Atomically move one READY-but-NOT_INDEXED document to INDEXING and
    return its id. Picks up documents whose indexing was interrupted (restart,
    database lock) so they resume on their own. FAILED documents are NOT
    retried automatically - that would loop on a permanent error; the user
    retriggers those via POST .../reindex."""
    with SessionLocal() as db:
        candidates = db.scalars(
            select(Document.id)
            .where(
                Document.processing_status == ProcessingStatus.READY,
                Document.rag_status == RagStatus.NOT_INDEXED,
            )
            .order_by(Document.uploaded_at.asc())
            .limit(5)
        ).all()
        for document_id in candidates:
            result = db.execute(
                update(Document)
                .where(
                    Document.id == document_id,
                    Document.rag_status == RagStatus.NOT_INDEXED,
                )
                .values(rag_status=RagStatus.INDEXING, rag_error=None)
            )
            if result.rowcount:
                db.commit()
                logger.info("%s resuming RAG indexing for document %s", worker_name, document_id)
                return document_id
            db.rollback()
    return None


# ---------------------------------------------------------------------------
# Processing one document
# ---------------------------------------------------------------------------


def process_document(document_id: str) -> bool:
    """Parse and persist one document. Returns True on success.

    Never raises: any failure is recorded on the row as FAILED with the reason,
    because a worker thread that dies takes the whole queue down with it.
    """
    started = time.monotonic()

    with SessionLocal() as db:
        document = db.get(Document, document_id)
        if document is None:
            logger.warning("document %s vanished before processing", document_id)
            return False

        storage_path = document.storage_path
        original_filename = document.original_filename

    try:
        if not os.path.exists(storage_path):
            raise ProcessingError(f"stored file is missing: {storage_path}")

        parsed = process_pdf(
            storage_path,
            source_filename=original_filename,
            preferred_backend=settings.parser_backend,
            enable_ocr=settings.enable_ocr,
            ocr_dpi=settings.ocr_dpi,
            max_ocr_pages=settings.max_ocr_pages,
        )
    except Exception as exc:
        duration = int((time.monotonic() - started) * 1000)
        logger.exception("processing failed for %s", document_id)
        _mark_failed(document_id, exc, duration)
        return False

    try:
        with SessionLocal() as db:
            document = db.get(Document, document_id)
            if document is None:
                logger.warning("document %s deleted during processing", document_id)
                return False

            counts = persist(db, document, parsed)

            document.processing_status = ProcessingStatus.READY
            document.processing_completed_at = _now()
            document.processing_duration_ms = int((time.monotonic() - started) * 1000)
            document.processing_error = None
            db.commit()

        logger.info(
            "document %s READY in %sms (%s)",
            document_id,
            int((time.monotonic() - started) * 1000),
            ", ".join(f"{k}={v}" for k, v in counts.items()),
        )
        # RAG indexing runs right after a successful parse, in the same
        # worker pass. It is intentionally decoupled from `processing_status`
        # (see Document.rag_status): if it fails, the document stays READY
        # and fully viewable, just not chat-searchable yet — the user (or a
        # future retry policy) can retrigger it via POST .../reindex without
        # re-parsing the PDF.
        index_document_rag(document_id)
        return True
    except Exception as exc:
        duration = int((time.monotonic() - started) * 1000)
        logger.exception("persisting failed for %s", document_id)
        _mark_failed(document_id, exc, duration)
        return False


def _mark_failed(document_id: str, exc: BaseException, duration_ms: int) -> None:
    message = f"{type(exc).__name__}: {exc}"
    try:
        with SessionLocal() as db:
            document = db.get(Document, document_id)
            if document is None:
                return
            document.processing_status = ProcessingStatus.FAILED
            document.processing_error = message[:2000]
            document.processing_completed_at = _now()
            document.processing_duration_ms = duration_ms
            db.commit()
    except Exception:
        logger.exception("could not record failure for %s", document_id)


# ---------------------------------------------------------------------------
# Phase 3: RAG indexing (chunk -> embed -> store)
# ---------------------------------------------------------------------------


def index_document_rag(document_id: str) -> bool:
    """Run chunking + embedding + storage for one document. Never raises —
    failures are recorded on `Document.rag_status`/`rag_error` and logged,
    same "a worker thread that dies takes the whole queue down with it"
    reasoning as `process_document`."""
    settings = get_settings()
    try:
        with SessionLocal() as db:
            document = db.get(Document, document_id)
            if document is None:
                return False
            document.rag_status = RagStatus.INDEXING
            document.rag_error = None
            db.commit()

        with SessionLocal() as db:
            document = db.get(Document, document_id)
            if document is None:
                return False
            counts = index_document(db, document, settings)
            db.commit()
        logger.info(
            "document %s RAG-indexed (%s)", document_id, ", ".join(f"{k}={v}" for k, v in counts.items())
        )
        return True
    except IndexingError as exc:
        logger.warning("RAG indexing failed for %s: %s", document_id, exc)
        _mark_rag_failed(document_id, str(exc))
        return False
    except Exception as exc:
        logger.exception("RAG indexing crashed for %s", document_id)
        _mark_rag_failed(document_id, f"{type(exc).__name__}: {exc}")
        return False


def _mark_rag_failed(document_id: str, message: str) -> None:
    try:
        with SessionLocal() as db:
            document = db.get(Document, document_id)
            if document is None:
                return
            mark_indexing_failed(db, document, message)
            db.commit()
    except Exception:
        logger.exception("could not record RAG indexing failure for %s", document_id)


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------


def worker_loop(worker_name: str, poll_interval: float) -> None:
    logger.info("%s started", worker_name)
    while not _shutdown.is_set():
        try:
            document_id = claim_next_document(worker_name)
        except Exception:
            logger.exception("%s failed to claim work", worker_name)
            _shutdown.wait(poll_interval)
            continue

        if document_id is None:
            try:
                pending_index = claim_next_unindexed_document(worker_name)
            except Exception:
                logger.exception("%s failed to claim an unindexed document", worker_name)
                pending_index = None
            if pending_index is not None:
                index_document_rag(pending_index)
                continue
            _shutdown.wait(poll_interval)
            continue

        process_document(document_id)

    logger.info("%s stopped", worker_name)


def start_workers(count: int | None = None, poll_interval: float | None = None) -> None:
    """Start in-process worker threads (called from FastAPI startup)."""
    if _threads:
        return
    count = count if count is not None else settings.worker_count
    poll_interval = (
        poll_interval if poll_interval is not None else settings.worker_poll_seconds
    )
    if count <= 0:
        logger.warning("worker_count=0: documents will stay UPLOADED until a "
                       "separate worker process is run")
        return

    _shutdown.clear()
    for i in range(count):
        thread = threading.Thread(
            target=worker_loop,
            args=(f"worker-{i + 1}", poll_interval),
            name=f"researchpilot-worker-{i + 1}",
            daemon=True,
        )
        thread.start()
        _threads.append(thread)
    logger.info("started %d document worker(s)", count)


def stop_workers(timeout: float = 5.0) -> None:
    _shutdown.set()
    for thread in _threads:
        thread.join(timeout=timeout)
    _threads.clear()


def main() -> None:
    """Entrypoint for running the worker as its own process.

        python -m app.worker
    """
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    from app.migrate import run_migrations

    run_migrations(engine)

    def handle_signal(signum, _frame):
        logger.info("received signal %s, shutting down", signum)
        _shutdown.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    count = max(1, settings.worker_count)
    for i in range(count - 1):
        thread = threading.Thread(
            target=worker_loop,
            args=(f"worker-{i + 2}", settings.worker_poll_seconds),
            daemon=True,
        )
        thread.start()
        _threads.append(thread)

    worker_loop("worker-1", settings.worker_poll_seconds)


if __name__ == "__main__":
    main()
