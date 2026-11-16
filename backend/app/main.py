"""FastAPI application entrypoint for ResearchPilot.

Phase 2 added the document-processing pipeline: schema migration on startup,
in-process background workers, and the extraction endpoints.

Phase 3 adds evidence-grounded paper chat: chunking + embedding + pgvector
storage running right after Phase 2 parsing, plus the streaming chat router.
"""
import logging
import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import worker
from app.config import get_settings
from app.database import engine, get_db
from app.migrate import run_migrations
from app.models import Document
from app.processing import ocr as ocr_module
from app.processing.pipeline import available_backends, resolve_order
from app.rag.factory import provider_status
from app.rag.indexing import stale_embedding_filter
from app.routers import chat, comparisons, dashboard, documents, evidence, gaps, workspaces

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="ResearchPilot API",
    description=(
        "AI-Powered Research Intelligence & Writing Workspace. "
        "Phase 3: evidence-grounded paper chat (RAG) on top of the Phase 2 "
        "document-intelligence pipeline."
    ),
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    os.makedirs(settings.upload_dir, exist_ok=True)

    # Creates new tables, adds missing columns to existing ones, normalises
    # Phase 1 status values and requeues anything left mid-processing.
    run_migrations(engine)

    backends = available_backends()
    if not backends:
        logger.error(
            "No PDF parser backend is installed. Uploads will be accepted but "
            "every document will FAIL. Install one of: docling, pymupdf, pdfplumber."
        )
    else:
        logger.info(
            "parser backends available: %s | order: %s | OCR: %s",
            ", ".join(backends),
            " -> ".join(resolve_order(settings.parser_backend)),
            "enabled" if settings.enable_ocr and ocr_module.ocr_available() else "unavailable",
        )

    rag = provider_status(settings)
    logger.info(
        "RAG chat: embeddings=%s (configured=%s) | llm=%s (configured=%s)",
        rag["embedding_provider"],
        rag["embedding_configured"],
        rag["llm_provider"],
        rag["llm_configured"],
    )

    worker.start_workers()


@app.on_event("shutdown")
def on_shutdown() -> None:
    worker.stop_workers()


# Mount after ensuring the directory exists (StaticFiles raises if dir is missing)
os.makedirs(settings.upload_dir, exist_ok=True)
app.mount("/files", StaticFiles(directory=settings.upload_dir), name="files")

app.include_router(documents.router)
app.include_router(workspaces.router)
app.include_router(dashboard.router)
app.include_router(chat.router)
app.include_router(evidence.router)
app.include_router(comparisons.router)
app.include_router(gaps.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/system/processing")
def processing_info() -> dict:
    """What the processing pipeline can actually do in this deployment.

    Surfaced in the UI so a missing optional dependency (no OCR engine, no
    Docling) is visible rather than silently degrading extraction quality.
    """
    return {
        "backends_available": available_backends(),
        "backend_order": resolve_order(settings.parser_backend),
        "configured_backend": settings.parser_backend,
        "ocr_enabled": settings.enable_ocr,
        "ocr_available": ocr_module.ocr_available(),
        "ocr_dpi": settings.ocr_dpi,
        "max_ocr_pages": settings.max_ocr_pages,
        "worker_count": settings.worker_count,
        "max_upload_mb": settings.max_upload_mb,
    }


@app.get("/api/system/rag")
def rag_info(db: Session = Depends(get_db)) -> dict:
    """Which embedding/LLM providers are active, and whether the hosted ones
    are actually configured (API key present) — surfaced in the UI so a
    misconfigured provider is visible rather than silently falling back.

    `stale_documents` counts documents embedded with a different provider
    than the active one (fix with `python -m scripts.reindex_all`). It is
    `null` if the count query fails — the rest of the status still returns."""
    status = provider_status(settings)
    try:
        status["stale_documents"] = db.scalar(
            select(func.count())
            .select_from(Document)
            .where(stale_embedding_filter(status["active_embedding_provider"]))
        )
    except Exception:
        logger.exception("could not count stale documents")
        status["stale_documents"] = None
    return status


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "researchpilot-api", "docs": "/docs", "version": "0.3.0"}

