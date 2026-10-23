"""PyMuPDF (fitz) parser backend.

Considerably faster than pdfplumber and gives richer span metadata (real bold
flags, per-span sizes) plus native table detection on PyMuPDF >= 1.23. Used as
the default when Docling is not installed.
"""
from __future__ import annotations

import logging

from ..heuristics import clean_text, find_repeating_lines, is_running_line
from ..schema import FIGURE, TABLE, BBox, ParsedDocument, ParsedElement, ParsedPage
from .base import (
    BackendUnavailable,
    Word,
    estimate_body_size,
    finalise_document,
    group_words_into_lines,
    lines_to_elements,
    rows_to_markdown,
    split_columns,
)

logger = logging.getLogger(__name__)

MIN_TEXT_LAYER_CHARS = 24
_BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit for bold


class PyMuPDFBackend:
    name = "pymupdf"

    @staticmethod
    def available() -> bool:
        try:
            import fitz  # noqa: F401
        except Exception:
            return False
        return True

    def parse(self, path: str) -> ParsedDocument:
        try:
            import fitz
        except Exception as exc:  # pragma: no cover - guarded by available()
            raise BackendUnavailable("PyMuPDF (pymupdf) is not installed") from exc

        doc = ParsedDocument(backend=self.name)

        with fitz.open(path) as pdf:
            doc.metadata = {
                k: str(v) for k, v in (pdf.metadata or {}).items() if v
            }
            doc.title = _clean_meta(doc.metadata.get("title"))
            doc.authors = _clean_meta(doc.metadata.get("author"))

            page_words: list[list[Word]] = []
            page_line_texts: list[list[str]] = []

            for page in pdf:
                words = _extract_words(page)
                page_words.append(words)
                page_line_texts.append(
                    [l.text for l in group_words_into_lines(words)]
                )

            body_size = estimate_body_size([w for ws in page_words for w in ws])
            repeating = find_repeating_lines(page_line_texts)

            for index, page in enumerate(pdf):
                page_number = index + 1
                words = page_words[index]
                rect = page.rect
                width, height = float(rect.width), float(rect.height)

                tables, table_bboxes = _extract_tables(page, page_number)
                figures = _extract_figures(page, page_number, width * height)

                body_words = [w for w in words if not _inside_any(w, table_bboxes)]
                lines = split_columns(group_words_into_lines(body_words), width)

                elements = lines_to_elements(
                    lines,
                    page_number,
                    body_size,
                    start_order=0,
                    skip_line=lambda text: is_running_line(text, repeating),
                )
                elements.extend(tables)
                elements.extend(figures)

                page_text = clean_text(
                    "\n".join(l.text for l in group_words_into_lines(words))
                )
                doc.pages.append(
                    ParsedPage(
                        page_number=page_number,
                        width=width,
                        height=height,
                        rotation=int(page.rotation or 0),
                        text=page_text,
                        has_text_layer=len(page_text.strip()) >= MIN_TEXT_LAYER_CHARS,
                        image_count=len(page.get_images(full=True) or []),
                    )
                )
                doc.elements.extend(elements)

        return finalise_document(doc)


def _extract_words(page) -> list[Word]:
    words: list[Word] = []
    try:
        data = page.get_text("dict")
    except Exception as exc:
        logger.warning("text extraction failed: %s", exc)
        return words

    for block in data.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = float(span.get("size") or 0)
                bold = bool(int(span.get("flags") or 0) & _BOLD_FLAG) or (
                    "bold" in str(span.get("font", "")).lower()
                )
                x0, y0, x1, y1 = span.get("bbox", (0, 0, 0, 0))
                text = span.get("text", "")
                if not text.strip():
                    continue
                # Split the span into words, apportioning width evenly. Span
                # level geometry is accurate; word level is interpolated.
                tokens = text.split()
                if not tokens:
                    continue
                span_width = float(x1) - float(x0)
                total = sum(len(t) for t in tokens) or 1
                cursor = float(x0)
                for token in tokens:
                    token_width = span_width * (len(token) / total)
                    words.append(
                        Word(
                            text=token,
                            x0=cursor,
                            top=float(y0),
                            x1=cursor + token_width,
                            bottom=float(y1),
                            size=size,
                            bold=bold,
                        )
                    )
                    cursor += token_width
    return words


def _extract_tables(page, page_number: int) -> tuple[list[ParsedElement], list[tuple]]:
    elements: list[ParsedElement] = []
    bboxes: list[tuple] = []

    finder = getattr(page, "find_tables", None)
    if finder is None:  # PyMuPDF < 1.23
        return elements, bboxes

    try:
        found = finder()
    except Exception as exc:
        logger.warning("table detection failed on page %s: %s", page_number, exc)
        return elements, bboxes

    for i, table in enumerate(getattr(found, "tables", []) or []):
        try:
            rows = table.extract()
        except Exception as exc:
            logger.warning("table extraction failed on page %s: %s", page_number, exc)
            continue
        markdown = rows_to_markdown(rows)
        if not markdown:
            continue
        x0, y0, x1, y1 = table.bbox
        bboxes.append((x0, y0, x1, y1))
        elements.append(
            ParsedElement(
                page_number=page_number,
                element_type=TABLE,
                content=markdown,
                order_index=10_000 + i,
                bbox=BBox(float(x0), float(y0), float(x1), float(y1)),
                meta={
                    "format": "markdown",
                    "rows": len(rows),
                    "columns": max((len(r) for r in rows), default=0),
                },
            )
        )
    return elements, bboxes


def _extract_figures(page, page_number: int, page_area: float) -> list[ParsedElement]:
    elements: list[ParsedElement] = []
    try:
        images = page.get_images(full=True) or []
    except Exception:
        return elements

    for i, img in enumerate(images):
        xref = img[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        rect = rects[0] if rects else None
        if rect is not None:
            area = abs(rect.width * rect.height)
            if page_area and area < page_area * 0.01:
                continue
            bbox = BBox(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
            size_meta = {"width": round(rect.width, 1), "height": round(rect.height, 1)}
        else:
            bbox = None
            size_meta = {}
        elements.append(
            ParsedElement(
                page_number=page_number,
                element_type=FIGURE,
                content="",
                order_index=20_000 + i,
                bbox=bbox,
                meta={
                    "label": f"Figure on page {page_number}",
                    "xref": xref,
                    "source": "embedded_image",
                    **size_meta,
                },
            )
        )
    return elements


def _inside_any(word: Word, bboxes: list[tuple]) -> bool:
    cx = (word.x0 + word.x1) / 2
    cy = (word.top + word.bottom) / 2
    for x0, y0, x1, y1 in bboxes:
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return True
    return False


def _clean_meta(value) -> str | None:
    if not value:
        return None
    text = clean_text(str(value))
    return text or None
