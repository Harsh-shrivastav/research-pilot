"""Document upload / list / delete, plus the Phase 2 extraction endpoints.

Upload is deliberately non-blocking: it writes the file, records the row as
UPLOADED and returns. A background worker (see `app/worker.py`) claims it and
does the parsing, so a 300-page scanned PDF never holds an HTTP connection
open.
"""
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.deps import get_current_user
from app.models import (
    Document,
    DocumentElement,
    DocumentPage,
    DocumentReference,
    DocumentSection,
    ProcessingStatus,
    RagStatus,
    WorkspaceDocument,
)
from app.processing.schema import ELEMENT_TYPES
from app.schemas import (
    BBoxOut,
    DocumentDetailOut,
    DocumentOut,
    ElementOut,
    PageDetailOut,
    PageOut,
    ProcessingStatusOut,
    ReferenceOut,
    SectionDetailOut,
    SectionOut,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])
settings = get_settings()

ALLOWED_CONTENT_TYPES = {"application/pdf"}
PDF_MAGIC = b"%PDF-"
SECTION_PREVIEW_CHARS = 240


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _to_out(doc: Document) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        original_filename=doc.original_filename,
        file_size=doc.file_size,
        content_type=doc.content_type,
        processing_status=doc.processing_status,
        uploaded_at=doc.uploaded_at,
        file_url=f"/files/{doc.filename}",
        title=doc.title,
        authors=doc.authors,
        page_count=doc.page_count,
        word_count=doc.word_count,
        parser_backend=doc.parser_backend,
        ocr_used=bool(doc.ocr_used),
        section_count=doc.section_count or 0,
        element_count=doc.element_count or 0,
        table_count=doc.table_count or 0,
        figure_count=doc.figure_count or 0,
        reference_count=doc.reference_count or 0,
        processing_started_at=doc.processing_started_at,
        processing_completed_at=doc.processing_completed_at,
        processing_duration_ms=doc.processing_duration_ms,
        processing_error=doc.processing_error,
        rag_status=doc.rag_status or "NOT_INDEXED",
        rag_error=doc.rag_error,
        chunk_count=doc.chunk_count or 0,
        embedding_provider=doc.embedding_provider,
        embedding_model=doc.embedding_model,
    )


def _element_out(element: DocumentElement) -> ElementOut:
    bbox = None
    if element.bbox_x0 is not None and element.bbox_y0 is not None:
        bbox = BBoxOut(
            x0=element.bbox_x0,
            y0=element.bbox_y0,
            x1=element.bbox_x1 if element.bbox_x1 is not None else element.bbox_x0,
            y1=element.bbox_y1 if element.bbox_y1 is not None else element.bbox_y0,
        )
    return ElementOut(
        id=element.id,
        document_id=element.document_id,
        page_number=element.page_number,
        element_type=element.element_type,
        content=element.content,
        order_index=element.order_index,
        level=element.level,
        section_id=element.section_id,
        bbox=bbox,
        metadata=element.element_metadata,
    )


def _section_out(section: DocumentSection) -> SectionOut:
    text = section.text or ""
    preview = text[:SECTION_PREVIEW_CHARS]
    if len(text) > SECTION_PREVIEW_CHARS:
        preview = preview.rsplit(" ", 1)[0] + "\u2026"
    return SectionOut(
        id=section.id,
        document_id=section.document_id,
        parent_id=section.parent_id,
        title=section.title,
        level=section.level,
        order_index=section.order_index,
        start_page=section.start_page,
        end_page=section.end_page,
        char_count=section.char_count or 0,
        preview=preview,
    )


def _page_out(page: DocumentPage) -> PageOut:
    return PageOut(
        id=page.id,
        document_id=page.document_id,
        page_number=page.page_number,
        width=page.width,
        height=page.height,
        rotation=page.rotation or 0,
        char_count=page.char_count or 0,
        has_text_layer=bool(page.has_text_layer),
        ocr_applied=bool(page.ocr_applied),
        image_count=page.image_count or 0,
    )


def _get_owned_document(db: Session, document_id: str) -> Document:
    user = get_current_user(db)
    doc = db.scalar(
        select(Document).where(Document.id == document_id, Document.owner_id == user.id)
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


# ---------------------------------------------------------------------------
# Core document endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)) -> list[DocumentOut]:
    user = get_current_user(db)
    docs = db.scalars(
        select(Document).where(Document.owner_id == user.id).order_by(Document.uploaded_at.desc())
    ).all()
    return [_to_out(d) for d in docs]


@router.post("/upload", response_model=DocumentOut)
async def upload_document(file: UploadFile, db: Session = Depends(get_db)) -> DocumentOut:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_mb}MB upload limit.",
        )
    # Content-Type is client-supplied; check the real file signature so a
    # renamed non-PDF never reaches the parser.
    if not contents.startswith(PDF_MAGIC):
        raise HTTPException(
            status_code=400,
            detail="That file is not a valid PDF (missing PDF signature).",
        )

    user = get_current_user(db)
    os.makedirs(settings.upload_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or "document.pdf")[1] or ".pdf"
    stored_filename = f"{uuid.uuid4()}{ext}"
    storage_path = os.path.join(settings.upload_dir, stored_filename)

    with open(storage_path, "wb") as f:
        f.write(contents)

    doc = Document(
        filename=stored_filename,
        original_filename=file.filename or stored_filename,
        file_size=len(contents),
        content_type=file.content_type,
        storage_path=storage_path,
        # The worker takes it from here - the request does not wait for parsing.
        processing_status=ProcessingStatus.UPLOADED,
        owner_id=user.id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return _to_out(doc)


@router.get("/{document_id}", response_model=DocumentDetailOut)
def get_document(document_id: str, db: Session = Depends(get_db)) -> DocumentDetailOut:
    doc = _get_owned_document(db, document_id)

    sections = db.scalars(
        select(DocumentSection)
        .where(DocumentSection.document_id == doc.id)
        .order_by(DocumentSection.order_index)
    ).all()
    pages = db.scalars(
        select(DocumentPage)
        .where(DocumentPage.document_id == doc.id)
        .order_by(DocumentPage.page_number)
    ).all()

    metadata = doc.doc_metadata if isinstance(doc.doc_metadata, dict) else {}
    warnings = metadata.get("warnings")

    base = _to_out(doc)
    return DocumentDetailOut(
        **base.model_dump(),
        doc_metadata=metadata or None,
        sections=[_section_out(s) for s in sections],
        pages=[_page_out(p) for p in pages],
        warnings=list(warnings) if isinstance(warnings, list) else [],
    )


@router.get("/{document_id}/status", response_model=ProcessingStatusOut)
def get_document_status(document_id: str, db: Session = Depends(get_db)) -> ProcessingStatusOut:
    """Cheap endpoint the UI polls while processing is in flight."""
    doc = _get_owned_document(db, document_id)
    return ProcessingStatusOut(
        id=doc.id,
        processing_status=doc.processing_status,
        processing_error=doc.processing_error,
        processing_duration_ms=doc.processing_duration_ms,
        page_count=doc.page_count,
        section_count=doc.section_count or 0,
        element_count=doc.element_count or 0,
        table_count=doc.table_count or 0,
        figure_count=doc.figure_count or 0,
        reference_count=doc.reference_count or 0,
    )


@router.post("/{document_id}/reprocess", response_model=DocumentOut)
def reprocess_document(document_id: str, db: Session = Depends(get_db)) -> DocumentOut:
    """Hand a document back to the queue (after a failure, or a parser upgrade)."""
    doc = _get_owned_document(db, document_id)
    if doc.processing_status == ProcessingStatus.PROCESSING:
        raise HTTPException(
            status_code=409, detail="This document is already being processed."
        )
    if not os.path.exists(doc.storage_path):
        raise HTTPException(
            status_code=410, detail="The stored file for this document is missing."
        )

    doc.processing_status = ProcessingStatus.UPLOADED
    doc.processing_error = None
    doc.processing_started_at = None
    doc.processing_completed_at = None
    doc.processing_duration_ms = None
    # A re-parse can change page/section/element boundaries, so any existing
    # chunk index is now stale — it will be rebuilt once the worker's parse
    # succeeds again (see worker.process_document -> index_document_rag).
    doc.rag_status = RagStatus.NOT_INDEXED
    doc.rag_error = None
    doc.chunk_count = 0
    db.commit()
    db.refresh(doc)
    return _to_out(doc)


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, db: Session = Depends(get_db)) -> None:
    doc = _get_owned_document(db, document_id)

    db.query(WorkspaceDocument).filter(WorkspaceDocument.document_id == doc.id).delete()

    if os.path.exists(doc.storage_path):
        os.remove(doc.storage_path)

    # Pages/sections/elements/references go with it via cascade.
    db.delete(doc)
    db.commit()


# ---------------------------------------------------------------------------
# Extracted structure
# ---------------------------------------------------------------------------


@router.get("/{document_id}/sections", response_model=list[SectionOut])
def list_sections(document_id: str, db: Session = Depends(get_db)) -> list[SectionOut]:
    doc = _get_owned_document(db, document_id)
    sections = db.scalars(
        select(DocumentSection)
        .where(DocumentSection.document_id == doc.id)
        .order_by(DocumentSection.order_index)
    ).all()
    return [_section_out(s) for s in sections]


@router.get("/{document_id}/sections/{section_id}", response_model=SectionDetailOut)
def get_section(
    document_id: str, section_id: str, db: Session = Depends(get_db)
) -> SectionDetailOut:
    """Full extracted content of one section - what the UI shows when a user
    clicks a section in the outline."""
    doc = _get_owned_document(db, document_id)
    section = db.scalar(
        select(DocumentSection).where(
            DocumentSection.id == section_id, DocumentSection.document_id == doc.id
        )
    )
    if section is None:
        raise HTTPException(status_code=404, detail="Section not found.")

    elements = db.scalars(
        select(DocumentElement)
        .where(DocumentElement.section_id == section.id)
        .order_by(DocumentElement.order_index)
    ).all()

    base = _section_out(section)
    return SectionDetailOut(
        **base.model_dump(),
        text=section.text or "",
        elements=[_element_out(e) for e in elements],
    )


@router.get("/{document_id}/pages", response_model=list[PageOut])
def list_pages(document_id: str, db: Session = Depends(get_db)) -> list[PageOut]:
    doc = _get_owned_document(db, document_id)
    pages = db.scalars(
        select(DocumentPage)
        .where(DocumentPage.document_id == doc.id)
        .order_by(DocumentPage.page_number)
    ).all()
    return [_page_out(p) for p in pages]


@router.get("/{document_id}/pages/{page_number}", response_model=PageDetailOut)
def get_page(
    document_id: str, page_number: int, db: Session = Depends(get_db)
) -> PageDetailOut:
    doc = _get_owned_document(db, document_id)
    page = db.scalar(
        select(DocumentPage).where(
            DocumentPage.document_id == doc.id, DocumentPage.page_number == page_number
        )
    )
    if page is None:
        raise HTTPException(status_code=404, detail="Page not found.")

    elements = db.scalars(
        select(DocumentElement)
        .where(
            DocumentElement.document_id == doc.id,
            DocumentElement.page_number == page_number,
        )
        .order_by(DocumentElement.order_index)
    ).all()

    base = _page_out(page)
    return PageDetailOut(
        **base.model_dump(),
        text=page.text or "",
        elements=[_element_out(e) for e in elements],
    )


@router.get("/{document_id}/elements", response_model=list[ElementOut])
def list_elements(
    document_id: str,
    element_type: str | None = Query(
        default=None,
        description="Filter by element type, e.g. table, figure, caption, heading.",
    ),
    page_number: int | None = Query(default=None, ge=1),
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[ElementOut]:
    doc = _get_owned_document(db, document_id)

    if element_type is not None and element_type not in ELEMENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown element_type. Expected one of: {sorted(ELEMENT_TYPES)}",
        )

    query = select(DocumentElement).where(DocumentElement.document_id == doc.id)
    if element_type is not None:
        query = query.where(DocumentElement.element_type == element_type)
    if page_number is not None:
        query = query.where(DocumentElement.page_number == page_number)

    elements = db.scalars(
        query.order_by(DocumentElement.order_index).offset(offset).limit(limit)
    ).all()
    return [_element_out(e) for e in elements]


@router.get("/{document_id}/references", response_model=list[ReferenceOut])
def list_references(document_id: str, db: Session = Depends(get_db)) -> list[ReferenceOut]:
    doc = _get_owned_document(db, document_id)
    references = db.scalars(
        select(DocumentReference)
        .where(DocumentReference.document_id == doc.id)
        .order_by(DocumentReference.order_index)
    ).all()
    return [ReferenceOut.model_validate(r) for r in references]
