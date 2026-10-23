"""Re-index every document embedded with a different provider than the one
currently configured (e.g. after switching EMBEDDING_PROVIDER to gemini).

    cd backend
    python -m scripts.reindex_all            # re-index stale documents
    python -m scripts.reindex_all --dry-run  # just list them

Documents are processed one at a time (Gemini rate limits; no worker
coordination needed) via `app.rag.indexing.index_document`. Only READY
documents are considered, and documents currently INDEXING are skipped.

A failed document is rolled back rather than marked FAILED, so it keeps its
previous chunks and stays searchable with the old provider until a later run
succeeds. Exits 1 if any document failed, 2 if the configured provider is
unavailable (it would fall back to local and re-index nothing useful).
"""
from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import select

from app.config import get_settings
from app.database import SessionLocal, engine
from app.migrate import run_migrations
from app.models import Document, ProcessingStatus, RagStatus
from app.rag.factory import get_embedding_provider
from app.rag.indexing import index_document, stale_embedding_filter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="list stale documents and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")
    settings = get_settings()
    run_migrations(engine)

    configured = (settings.embedding_provider or "local").strip().lower()
    provider = get_embedding_provider(settings)
    active = provider.name
    if active != configured:
        print(
            f"EMBEDDING_PROVIDER={configured} is unavailable (fell back to {active}); "
            "fix the configuration first. Nothing re-indexed.",
            file=sys.stderr,
        )
        return 2
    print(f"Active embedding provider: {active} (model: {getattr(provider, 'model', None)})")

    with SessionLocal() as db:
        stale = db.execute(
            select(Document.id, Document.original_filename, Document.embedding_provider)
            .where(
                Document.processing_status == ProcessingStatus.READY,
                Document.rag_status != RagStatus.INDEXING,
                stale_embedding_filter(active),
            )
            .order_by(Document.uploaded_at.asc())
        ).all()

    total = len(stale)
    if not total:
        print("No stale documents. Nothing to do.")
        return 0
    print(f"{total} stale document(s).")
    if args.dry_run:
        for doc_id, name, old in stale:
            print(f"  {doc_id}  {name}  (indexed with {old})")
        return 0

    failed = 0
    for i, (doc_id, name, old) in enumerate(stale, start=1):
        prefix = f"[{i}/{total}] {name} ({old} -> {active})"
        with SessionLocal() as db:
            document = db.get(Document, doc_id)
            if document is None or document.processing_status != ProcessingStatus.READY:
                print(f"{prefix}: skipped (no longer READY)")
                continue
            try:
                counts = index_document(db, document, settings)
                db.commit()
                print(f"{prefix}: ok, {counts['children']} chunks")
            except Exception as exc:
                db.rollback()
                failed += 1
                print(f"{prefix}: FAILED ({type(exc).__name__}: {exc})")

    print(f"Done: {total - failed} re-indexed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
