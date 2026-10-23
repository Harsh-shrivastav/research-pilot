"""The document processing pipeline.

    PDF -> backend parse -> OCR top-up -> section assembly -> reference
    extraction -> metadata inference -> ParsedDocument

Backend order is Docling (best structure) -> PyMuPDF (fast) -> pdfplumber
(pure Python). The first *available* backend is tried; if it raises, the next
one is tried, and only when all of them fail does the document fail.

Nothing in here invents content: every field is either read from the PDF or
left empty/None.
"""
from __future__ import annotations

import logging
import os
import re
import time

from . import ocr as ocr_module
from . import references as references_module
from . import structure as structure_module
from .backends.base import finalise_document
from .backends.docling_backend import DoclingBackend
from .backends.pdfplumber_backend import PdfPlumberBackend
from .backends.pymupdf_backend import PyMuPDFBackend
from .heuristics import clean_text
from .schema import HEADING, PARAGRAPH, ParsedDocument

logger = logging.getLogger(__name__)

BACKEND_CLASSES = {
    "docling": DoclingBackend,
    "pymupdf": PyMuPDFBackend,
    "pdfplumber": PdfPlumberBackend,
}

# Preference order when PARSER_BACKEND=auto.
DEFAULT_ORDER = ("docling", "pymupdf", "pdfplumber")

# A page below this many characters is a candidate for OCR.
OCR_TEXT_THRESHOLD = 24


class ProcessingError(RuntimeError):
    """Raised when no backend could parse the document at all."""


def available_backends() -> list[str]:
    return [name for name, cls in BACKEND_CLASSES.items() if cls.available()]


def resolve_order(preferred: str = "auto") -> list[str]:
    """Return the backend names to try, in order."""
    preferred = (preferred or "auto").strip().lower()
    if preferred in ("", "auto"):
        order = list(DEFAULT_ORDER)
    elif preferred in BACKEND_CLASSES:
        # Explicit choice first, then the rest as fallbacks.
        order = [preferred] + [n for n in DEFAULT_ORDER if n != preferred]
    else:
        logger.warning("unknown PARSER_BACKEND %r; falling back to auto", preferred)
        order = list(DEFAULT_ORDER)
    return [name for name in order if BACKEND_CLASSES[name].available()]


def process_pdf(
    path: str,
    *,
    source_filename: str | None = None,
    preferred_backend: str = "auto",
    enable_ocr: bool = True,
    ocr_dpi: int = ocr_module.DEFAULT_DPI,
    max_ocr_pages: int | None = 50,
) -> ParsedDocument:
    """Parse `path` into a fully structured document.

    `source_filename` is the user-facing original filename; it is used only to
    recognise (and reject) PDF metadata titles that are just the filename.
    """
    if not os.path.exists(path):
        raise ProcessingError(f"file not found: {path}")
    if os.path.getsize(path) == 0:
        raise ProcessingError("file is empty")

    order = resolve_order(preferred_backend)
    if not order:
        raise ProcessingError(
            "no PDF parser backend is installed (expected one of: "
            + ", ".join(BACKEND_CLASSES)
            + ")"
        )

    doc: ParsedDocument | None = None
    failures: list[str] = []

    for name in order:
        started = time.monotonic()
        try:
            doc = BACKEND_CLASSES[name]().parse(path)
            logger.info(
                "parsed %s with %s in %.2fs (%d pages)",
                os.path.basename(path),
                name,
                time.monotonic() - started,
                doc.page_count,
            )
            break
        except Exception as exc:
            logger.warning("backend %s failed on %s: %s", name, path, exc)
            failures.append(f"{name}: {exc}")

    if doc is None:
        raise ProcessingError(
            "all parser backends failed -> " + " | ".join(failures)
        )

    if failures:
        doc.warnings.extend(f"backend fallback -> {f}" for f in failures)

    if doc.page_count == 0:
        raise ProcessingError("document contains no pages")

    if enable_ocr:
        _apply_ocr(doc, path, dpi=ocr_dpi, max_pages=max_ocr_pages)

    doc.elements.sort(key=lambda e: (e.page_number, e.order_index))
    finalise_document(doc)

    doc.sections = structure_module.build_sections(doc.elements, doc.page_count)
    doc.references = references_module.extract_references(doc.elements)
    _infer_metadata(doc, source_filename or path)
    return doc


# ---------------------------------------------------------------------------
# OCR top-up
# ---------------------------------------------------------------------------


def _apply_ocr(
    doc: ParsedDocument, path: str, *, dpi: int, max_pages: int | None
) -> None:
    """OCR any page whose text layer is missing, and merge the result in."""
    candidates = [
        page
        for page in doc.pages
        if len(page.text.strip()) < OCR_TEXT_THRESHOLD and not page.ocr_applied
    ]
    if not candidates:
        return

    if not ocr_module.ocr_available():
        doc.warnings.append(
            f"{len(candidates)} page(s) have no text layer and OCR is unavailable "
            "(install tesseract-ocr to extract them)"
        )
        for page in candidates:
            page.has_text_layer = False
        return

    try:
        results = ocr_module.ocr_document_pages(
            path, candidates, dpi=dpi, max_pages=max_pages
        )
    except ocr_module.OcrUnavailable as exc:
        doc.warnings.append(f"OCR unavailable: {exc}")
        return
    except Exception as exc:
        logger.warning("OCR pass failed: %s", exc)
        doc.warnings.append(f"OCR failed: {exc}")
        return

    if max_pages is not None and len(candidates) > max_pages:
        doc.warnings.append(
            f"OCR limited to the first {max_pages} image-only pages "
            f"({len(candidates)} needed it)"
        )

    by_number = {p.page_number: p for p in doc.pages}
    added = 0
    for page_number, result in results.items():
        page = by_number.get(page_number)
        if page is None or not result.text.strip():
            continue
        page.text = clean_text(result.text)
        page.has_text_layer = False  # the text came from OCR, not the PDF
        page.ocr_applied = True
        doc.elements.extend(result.elements)
        added += len(result.elements)
        doc.ocr_used = True

    if added:
        logger.info("OCR recovered %d elements across %d pages", added, len(results))

    unresolved = [p for p in candidates if not p.ocr_applied]
    if unresolved:
        doc.warnings.append(
            f"OCR produced no text for page(s): "
            f"{', '.join(str(p.page_number) for p in unresolved)}"
        )


# ---------------------------------------------------------------------------
# Document-level metadata
# ---------------------------------------------------------------------------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_AUTHOR_LINE = re.compile(
    r"^(?:[A-Z][A-Za-z'’\-]+(?:\s+[A-Z]\.)?\s+[A-Z][A-Za-z'’\-]+|[A-Z]\.\s*[A-Z][A-Za-z'’\-]+)"
    r"(?:\s*(?:,|and|&)\s*[A-Z][A-Za-z'’.\-]+(?:\s+[A-Z][A-Za-z'’\-]+)?)*$"
)

_GENERIC_TITLES = {"untitled", "microsoft word", "document", "pdf document", ""}

# Values PDF producers write when nobody actually set an author.
_GENERIC_AUTHORS = {
    "anonymous",
    "unknown",
    "user",
    "administrator",
    "admin",
    "author",
    "none",
    "n/a",
    "default",
    "owner",
    "windows user",
    "microsoft office user",
    "pc",
}


def _infer_metadata(doc: ParsedDocument, source_filename: str | None = None) -> None:
    """Fill title/authors from the document body when PDF metadata is absent.

    Falls back to None rather than guessing wildly — an empty title is honest,
    a wrong one is not.
    """
    if not _is_useful_title(doc.title, source_filename):
        doc.title = _title_from_body(doc)

    if not _is_useful_author(doc.authors):
        doc.authors = _authors_from_body(doc)

    doc.metadata = dict(doc.metadata or {})
    doc.metadata.update(
        {
            "detected_backend": doc.backend,
            "ocr_used": doc.ocr_used,
            "page_count": doc.page_count,
            "word_count": doc.word_count,
        }
    )
    if doc.warnings:
        doc.metadata["warnings"] = doc.warnings[:10]


def _is_useful_title(title: str | None, source_filename: str | None = None) -> bool:
    if not title:
        return False
    normalised = title.strip().lower()
    if normalised in _GENERIC_TITLES or len(normalised) < 4:
        return False
    # PDF producers often stuff the source filename into the Title field.
    if normalised.endswith((".pdf", ".docx", ".doc", ".tex")):
        return False
    if source_filename:
        stem = os.path.splitext(os.path.basename(source_filename))[0].strip().lower()
        if stem and normalised == stem:
            return False
    return True


def _is_useful_author(author: str | None) -> bool:
    if not author:
        return False
    normalised = " ".join(author.strip().lower().split())
    if normalised in _GENERIC_AUTHORS or len(normalised) < 3:
        return False
    return True


def _title_from_body(doc: ParsedDocument) -> str | None:
    """The first substantial heading (or first line) on page 1."""
    page_one = [e for e in doc.elements if e.page_number == 1]
    for element in page_one:
        if element.element_type == HEADING and 4 <= len(element.content) <= 220:
            return element.content
    for element in page_one:
        if element.element_type == PARAGRAPH and 8 <= len(element.content) <= 220:
            return element.content.split(". ")[0]
    return None


def _authors_from_body(doc: ParsedDocument) -> str | None:
    """Author lines sit between the title and the abstract on page 1."""
    page_one = [e for e in doc.elements if e.page_number == 1][:12]
    seen_title = False
    for element in page_one:
        text = element.content.strip()
        if not text:
            continue
        if element.element_type == HEADING:
            if seen_title:
                break  # reached "Abstract"/"Introduction"
            seen_title = True
            continue
        if _EMAIL.search(text):
            continue
        if len(text) <= 200 and _AUTHOR_LINE.match(text):
            return text
    return None
