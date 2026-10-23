"""pdfplumber parser backend.

The pure-Python fallback. Slower than PyMuPDF and without Docling's layout
model, but it depends only on pdfminer.six, so it works anywhere the rest of
the stack does. It still produces real word-level geometry, tables and image
regions — nothing here is synthesised.
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

# A page with fewer than this many characters of embedded text is treated as
# image-only and handed to OCR by the pipeline.
MIN_TEXT_LAYER_CHARS = 24


class PdfPlumberBackend:
    name = "pdfplumber"

    @staticmethod
    def available() -> bool:
        try:
            import pdfplumber  # noqa: F401
        except Exception:
            return False
        return True

    def parse(self, path: str) -> ParsedDocument:
        try:
            import pdfplumber
        except Exception as exc:  # pragma: no cover - guarded by available()
            raise BackendUnavailable("pdfplumber is not installed") from exc

        doc = ParsedDocument(backend=self.name)

        with pdfplumber.open(path) as pdf:
            doc.metadata = _safe_metadata(pdf)
            doc.title = _clean_meta(doc.metadata.get("Title"))
            doc.authors = _clean_meta(doc.metadata.get("Author"))

            page_words: list[list[Word]] = []
            page_line_texts: list[list[str]] = []
            pages_raw = []

            for page in pdf.pages:
                words = _extract_words(page)
                page_words.append(words)
                lines = group_words_into_lines(words)
                page_line_texts.append([l.text for l in lines])
                pages_raw.append(page)

            all_words = [w for words in page_words for w in words]
            body_size = estimate_body_size(all_words)
            repeating = find_repeating_lines(page_line_texts)

            for index, page in enumerate(pages_raw):
                page_number = index + 1
                words = page_words[index]
                width = float(page.width or 0)
                height = float(page.height or 0)

                tables, table_bboxes = _extract_tables(page, page_number)
                figures = _extract_figures(page, page_number)

                # Words that fall inside a detected table are represented by
                # the table element, not duplicated as paragraph text.
                body_words = [
                    w for w in words if not _inside_any(w, table_bboxes)
                ]
                lines = group_words_into_lines(body_words)
                lines = split_columns(lines, width)

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
                        rotation=int(getattr(page, "rotation", 0) or 0),
                        text=page_text,
                        has_text_layer=len(page_text.strip()) >= MIN_TEXT_LAYER_CHARS,
                        image_count=len(getattr(page, "images", []) or []),
                    )
                )
                doc.elements.extend(elements)

        return finalise_document(doc)


def _extract_words(page) -> list[Word]:
    try:
        raw = page.extract_words(
            extra_attrs=["size", "fontname"],
            keep_blank_chars=False,
            use_text_flow=False,
        )
    except Exception as exc:
        logger.warning("word extraction failed on a page: %s", exc)
        return []

    words: list[Word] = []
    for w in raw:
        fontname = (w.get("fontname") or "").lower()
        words.append(
            Word(
                text=w.get("text", ""),
                x0=float(w.get("x0", 0)),
                top=float(w.get("top", 0)),
                x1=float(w.get("x1", 0)),
                bottom=float(w.get("bottom", 0)),
                size=float(w.get("size") or 0),
                bold="bold" in fontname or "black" in fontname or ",b" in fontname,
            )
        )
    return words


def _extract_tables(page, page_number: int) -> tuple[list[ParsedElement], list[tuple]]:
    elements: list[ParsedElement] = []
    bboxes: list[tuple] = []
    try:
        found = page.find_tables()
    except Exception as exc:
        logger.warning("table detection failed on page %s: %s", page_number, exc)
        return elements, bboxes

    for i, table in enumerate(found):
        try:
            rows = table.extract()
        except Exception as exc:
            logger.warning("table extraction failed on page %s: %s", page_number, exc)
            continue
        markdown = rows_to_markdown(rows)
        if not markdown:
            continue
        x0, top, x1, bottom = table.bbox
        bboxes.append((x0, top, x1, bottom))
        elements.append(
            ParsedElement(
                page_number=page_number,
                element_type=TABLE,
                content=markdown,
                order_index=10_000 + i,
                bbox=BBox(float(x0), float(top), float(x1), float(bottom)),
                meta={
                    "format": "markdown",
                    "rows": len(rows),
                    "columns": max((len(r) for r in rows), default=0),
                },
            )
        )
    return elements, bboxes


def _extract_figures(page, page_number: int) -> list[ParsedElement]:
    elements: list[ParsedElement] = []
    images = getattr(page, "images", []) or []
    page_area = float(page.width or 1) * float(page.height or 1)

    for i, img in enumerate(images):
        x0 = float(img.get("x0", 0))
        top = float(img.get("top", 0))
        x1 = float(img.get("x1", 0))
        bottom = float(img.get("bottom", 0))
        area = abs((x1 - x0) * (bottom - top))
        # Ignore hairline rules, bullets and logos masquerading as images.
        if area < page_area * 0.01:
            continue
        elements.append(
            ParsedElement(
                page_number=page_number,
                element_type=FIGURE,
                content="",
                order_index=20_000 + i,
                bbox=BBox(x0, top, x1, bottom),
                meta={
                    "label": f"Figure on page {page_number}",
                    "width": round(x1 - x0, 1),
                    "height": round(bottom - top, 1),
                    "source": "embedded_image",
                },
            )
        )
    return elements


def _inside_any(word: Word, bboxes: list[tuple]) -> bool:
    cx = (word.x0 + word.x1) / 2
    cy = (word.top + word.bottom) / 2
    for x0, top, x1, bottom in bboxes:
        if x0 <= cx <= x1 and top <= cy <= bottom:
            return True
    return False


def _safe_metadata(pdf) -> dict:
    try:
        meta = dict(pdf.metadata or {})
    except Exception:
        return {}
    return {k: str(v) for k, v in meta.items() if v is not None}


def _clean_meta(value) -> str | None:
    if not value:
        return None
    text = clean_text(str(value))
    return text or None
