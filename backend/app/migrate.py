"""Lightweight additive schema migration.

Phase 1 shipped without Alembic, and there are existing databases in the wild
(there is a `researchpilot.db` in this very repo). `Base.metadata.create_all`
happily creates the four new Phase 2 tables, but it will **not** add the new
columns to the existing `documents` table — so without this module an upgraded
backend fails on the first query with "no such column: documents.title".

This does two things on startup:
  1. ALTER TABLE ... ADD COLUMN for any column the ORM knows about that the
     live table is missing.
  2. Normalise Phase 1's lowercase `processing_status` values ("pending") to
     the Phase 2 uppercase states ("UPLOADED").

Both are idempotent and safe to run on every boot. Anything more complex than
adding a column (renames, type changes, backfills) should move to Alembic —
this is intentionally the smallest thing that keeps existing installs working.
"""
from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateColumn

from app.database import Base
from app.models import LEGACY_STATUS_MAP, Document, ProcessingStatus, RagStatus

logger = logging.getLogger(__name__)


def run_migrations(engine: Engine) -> None:
    """Create missing tables, add missing columns, normalise legacy values."""
    _ensure_pgvector_extension(engine)
    Base.metadata.create_all(bind=engine)
    _add_missing_columns(engine)
    _normalise_statuses(engine)
    _requeue_interrupted(engine)
    _requeue_interrupted_rag(engine)


def _ensure_pgvector_extension(engine: Engine) -> None:
    """CREATE EXTENSION IF NOT EXISTS vector — required before create_all can
    build the `document_chunks.embedding` column as a native pgvector type.
    No-op (and safe) on SQLite / any non-Postgres dialect."""
    if engine.dialect.name != "postgresql":
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("migration: pgvector extension is present")
    except Exception as exc:
        logger.error(
            "migration: could not create the pgvector extension (%s). "
            "document_chunks.embedding will fail to create as a native vector "
            "column — is the 'vector' extension available on this Postgres?",
            exc,
        )


def _add_missing_columns(engine: Engine) -> None:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue  # create_all just made it, so it is already complete
        present = {col["name"] for col in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue
            ddl = _add_column_ddl(engine, table.name, column)
            if ddl is None:
                continue
            try:
                with engine.begin() as conn:
                    conn.execute(text(ddl))
                logger.info("migration: added %s.%s", table.name, column.name)
            except Exception as exc:
                logger.error(
                    "migration: could not add %s.%s (%s)", table.name, column.name, exc
                )


def _add_column_ddl(engine: Engine, table_name: str, column) -> str | None:
    """Render ADD COLUMN DDL that both SQLite and PostgreSQL accept."""
    try:
        spec = CreateColumn(column).compile(engine).string
    except Exception as exc:
        logger.error("migration: cannot compile column %s (%s)", column.name, exc)
        return None

    # SQLite cannot add a NOT NULL column without a default, and cannot add a
    # column carrying a UNIQUE or PRIMARY KEY constraint at all. New columns in
    # this codebase are all nullable or defaulted, so strip the qualifier
    # rather than failing the whole migration.
    if engine.dialect.name == "sqlite":
        spec = spec.replace(" NOT NULL", "").replace(" UNIQUE", "")

    return f"ALTER TABLE {table_name} ADD COLUMN {spec}"


def _normalise_statuses(engine: Engine) -> None:
    """Map Phase 1's lowercase statuses onto the Phase 2 state machine."""
    table = Document.__tablename__
    for legacy, modern in LEGACY_STATUS_MAP.items():
        if legacy == modern:
            continue
        try:
            with engine.begin() as conn:
                result = conn.execute(
                    text(
                        f"UPDATE {table} SET processing_status = :modern "
                        f"WHERE processing_status = :legacy"
                    ),
                    {"modern": modern, "legacy": legacy},
                )
                if result.rowcount:
                    logger.info(
                        "migration: %s document(s) %r -> %r",
                        result.rowcount,
                        legacy,
                        modern,
                    )
        except Exception as exc:
            logger.error("migration: status normalisation failed (%s)", exc)


def _requeue_interrupted(engine: Engine) -> None:
    """Any document stuck in PROCESSING is an interrupted run.

    A worker that dies mid-parse (container restart, OOM on a huge PDF) leaves
    its document claimed forever. On startup we hand those back to the queue
    so they get retried rather than hanging in the UI.
    """
    table = Document.__tablename__
    try:
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    f"UPDATE {table} SET processing_status = :uploaded "
                    f"WHERE processing_status = :processing"
                ),
                {
                    "uploaded": ProcessingStatus.UPLOADED,
                    "processing": ProcessingStatus.PROCESSING,
                },
            )
            if result.rowcount:
                logger.warning(
                    "migration: requeued %s interrupted document(s)", result.rowcount
                )
    except Exception as exc:
        logger.error("migration: requeue failed (%s)", exc)


def _requeue_interrupted_rag(engine: Engine) -> None:
    """Same idea as `_requeue_interrupted`, for the RAG indexing stage: a
    worker that dies mid-embed leaves a document stuck at INDEXING forever."""
    table = Document.__tablename__
    try:
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    f"UPDATE {table} SET rag_status = :not_indexed "
                    f"WHERE rag_status = :indexing"
                ),
                {"not_indexed": RagStatus.NOT_INDEXED, "indexing": RagStatus.INDEXING},
            )
            if result.rowcount:
                logger.warning(
                    "migration: requeued %s interrupted RAG-indexing document(s)",
                    result.rowcount,
                )
    except Exception as exc:
        logger.error("migration: RAG requeue failed (%s)", exc)
