"""Docling parser backend — the preferred parser when it is installed.

Docling runs a learned layout model over the rendered page, so it classifies
headings, tables, figures, captions and formulas far more reliably than font
heuristics can, and it handles scanned input through its own OCR stage.

Docling is an optional dependency (it pulls in a heavy ML stack). When it is
absent, `available()` returns False and `pipeline.py` falls back to PyMuPDF and
then pdfplumber. Because Docling's object model has changed across major
versions, everything here is read defensively via getattr and any structural
surprise raises, which the pipeline treats as "try the next backend" rather
than "fail the document".
"""
from __future__ import annotations

import logging

from ..heuristics import clean_text, parse_caption
from ..schema import (
    CAPTION,
    FIGURE,
    FORMULA,
    HEADING,
    LIST_ITEM,
    PARAGRAPH,
    TABLE,
    BBox,
    ParsedDocument,
    ParsedElement,
    ParsedPage,
)
from .base import BackendUnavailable, finalise_document

logger = logging.getLogger(__name__)

# Docling label -> our element type. Labels are compared lowercased as strings
# so we do not depend on importing Docling's enum.
_LABEL_MAP = {
    "title": HEADING,
    "section_header": HEADING,
    "subtitle-level-1": HEADING,
    "text": PARAGRAPH,
    "paragraph": PARAGRAPH,
    "list_item": LIST_ITEM,
    "caption": CAPTION,
    "table": TABLE,
    "picture": FIGURE,
    "figure": FIGURE,
    "chart": FIGURE,
    "formula": FORMULA,
    "equation": FORMULA,
    "code": PARAGRAPH,
    "footnote": PARAGRAPH,
}

# Layout regions we deliberately drop — they are page furniture, not content.
_SKIP_LABELS = {"page_header", "page_footer", "document_index"}


class DoclingBackend:
    name = "docling"

    @staticmethod
    def available() -> bool:
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
        except Exception:
            return False
        return True

    def __init__(self, do_ocr: bool = True) -> None:
        self.do_ocr = do_ocr
        self._converter = None

    def _get_converter(self):
        if self._converter is not None:
            return self._converter
        try:
            from docling.document_converter import DocumentConverter
        except Exception as exc:  # pragma: no cover - guarded by available()
            raise BackendUnavailable("docling is not installed") from exc

        # Try the richer pipeline-options constructor first; fall back to the
        # bare constructor on versions where those symbols moved.
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption

            options = PdfPipelineOptions()
            options.do_ocr = self.do_ocr
            options.do_table_structure = True
            try:
                options.table_structure_options.do_cell_matching = True
            except Exception:
                pass
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        except Exception as exc:
            logger.info("Docling pipeline options unavailable (%s); using defaults", exc)
            self._converter = DocumentConverter()
        return self._converter

    def parse(self, path: str) -> ParsedDocument:
        converter = self._get_converter()
        result = converter.convert(path)
        dl_doc = getattr(result, "document", None)
        if dl_doc is None:
            raise BackendUnavailable("docling returned no document")

        doc = ParsedDocument(backend=self.name)
        page_sizes = _page_sizes(dl_doc)
        doc.pages = _build_pages(dl_doc, page_sizes)

        order = 0
        pending_caption_for: dict[int, str] = {}

        for item, _level in _iterate(dl_doc):
            label = _label_of(item)
            if label in _SKIP_LABELS:
                continue
            element_type = _LABEL_MAP.get(label)
            if element_type is None:
                continue

            page_number, bbox = _provenance(item, page_sizes)
            if page_number is None:
                page_number = 1

            if element_type == TABLE:
                content = _table_markdown(item, dl_doc)
                meta = {"format": "markdown", "source": "docling"}
                rows, cols = _table_shape(item)
                if rows:
                    meta["rows"] = rows
                if cols:
                    meta["columns"] = cols
                caption = _caption_text(item, dl_doc)
                if caption:
                    meta["caption"] = caption
                doc.elements.append(
                    ParsedElement(
                        page_number=page_number,
                        element_type=TABLE,
                        content=content,
                        order_index=order,
                        bbox=bbox,
                        meta=meta,
                    )
                )
                order += 1
                continue

            if element_type == FIGURE:
                caption = _caption_text(item, dl_doc)
                meta = {
                    "label": caption or f"Figure on page {page_number}",
                    "source": "docling",
                }
                if caption:
                    meta["caption"] = caption
                doc.elements.append(
                    ParsedElement(
                        page_number=page_number,
                        element_type=FIGURE,
                        content="",
                        order_index=order,
                        bbox=bbox,
                        meta=meta,
                    )
                )
                order += 1
                continue

            text = clean_text(getattr(item, "text", "") or "")
            if not text:
                continue

            meta: dict = {"source": "docling"}
            level = None
            if element_type == HEADING:
                level = int(getattr(item, "level", 0) or 0) or (1 if label == "title" else 2)
                level = max(1, min(level, 6))
            elif element_type == CAPTION:
                parsed = parse_caption(text)
                if parsed:
                    kind, number, body = parsed
                    meta.update(
                        {"refers_to": kind, "label": f"{kind.title()} {number}", "body": body}
                    )

            doc.elements.append(
                ParsedElement(
                    page_number=page_number,
                    element_type=element_type,
                    content=text,
                    order_index=order,
                    bbox=bbox,
                    level=level,
                    meta=meta,
                )
            )
            order += 1

        _fill_page_text(doc)
        doc.title = _document_title(dl_doc, doc)
        doc.metadata = {"parser": "docling", **_origin_metadata(dl_doc)}
        doc.ocr_used = any(p.ocr_applied for p in doc.pages)
        return finalise_document(doc)


# ---------------------------------------------------------------------------
# Defensive accessors over Docling's object model
# ---------------------------------------------------------------------------


def _iterate(dl_doc):
    iterator = getattr(dl_doc, "iterate_items", None)
    if callable(iterator):
        for entry in iterator():
            if isinstance(entry, tuple):
                yield entry[0], (entry[1] if len(entry) > 1 else 0)
            else:
                yield entry, 0
        return
    # Very old versions: walk the flat collections in document order.
    for name in ("texts", "tables", "pictures"):
        for item in getattr(dl_doc, name, []) or []:
            yield item, 0


def _label_of(item) -> str:
    label = getattr(item, "label", None)
    if label is None:
        return ""
    value = getattr(label, "value", label)
    return str(value).lower().replace("-", "_")


def _page_sizes(dl_doc) -> dict[int, tuple[float, float]]:
    sizes: dict[int, tuple[float, float]] = {}
    pages = getattr(dl_doc, "pages", None) or {}
    items = pages.items() if hasattr(pages, "items") else enumerate(pages, start=1)
    for key, page in items:
        size = getattr(page, "size", None)
        width = float(getattr(size, "width", 0) or 0)
        height = float(getattr(size, "height", 0) or 0)
        page_no = int(getattr(page, "page_no", key) or key)
        sizes[page_no] = (width, height)
    return sizes


def _build_pages(dl_doc, sizes: dict[int, tuple[float, float]]) -> list[ParsedPage]:
    if not sizes:
        count = 0
        num_pages = getattr(dl_doc, "num_pages", None)
        if callable(num_pages):
            try:
                count = int(num_pages())
            except Exception:
                count = 0
        sizes = {i: (0.0, 0.0) for i in range(1, max(count, 1) + 1)}

    pages = []
    for page_no in sorted(sizes):
        width, height = sizes[page_no]
        pages.append(
            ParsedPage(
                page_number=page_no,
                width=width,
                height=height,
                text="",
                has_text_layer=True,
            )
        )
    return pages


def _provenance(item, sizes: dict[int, tuple[float, float]]) -> tuple[int | None, BBox | None]:
    provs = getattr(item, "prov", None) or []
    if not provs:
        return None, None
    prov = provs[0]
    page_no = getattr(prov, "page_no", None)
    page_no = int(page_no) if page_no is not None else None

    raw = getattr(prov, "bbox", None)
    if raw is None:
        return page_no, None

    _, page_height = sizes.get(page_no or 0, (0.0, 0.0))

    # Docling boxes may use a bottom-left origin; normalise to top-left.
    converter = getattr(raw, "to_top_left_origin", None)
    box = raw
    if callable(converter) and page_height:
        try:
            box = converter(page_height)
        except Exception:
            box = raw

    try:
        left = float(getattr(box, "l"))
        right = float(getattr(box, "r"))
        top = float(getattr(box, "t"))
        bottom = float(getattr(box, "b"))
    except Exception:
        return page_no, None

    if top > bottom:  # still bottom-left origin
        if page_height:
            top, bottom = page_height - top, page_height - bottom
        else:
            top, bottom = bottom, top
    return page_no, BBox(left, top, right, bottom)


def _table_markdown(item, dl_doc) -> str:
    for method, args in (
        ("export_to_markdown", (dl_doc,)),
        ("export_to_markdown", ()),
    ):
        fn = getattr(item, method, None)
        if callable(fn):
            try:
                out = fn(*args)
                if out:
                    return str(out).strip()
            except Exception:
                continue

    fn = getattr(item, "export_to_dataframe", None)
    if callable(fn):
        try:
            df = fn()
            return df.to_markdown(index=False)
        except Exception:
            pass
    return ""


def _table_shape(item) -> tuple[int | None, int | None]:
    data = getattr(item, "data", None)
    rows = getattr(data, "num_rows", None)
    cols = getattr(data, "num_cols", None)
    try:
        return (int(rows) if rows else None, int(cols) if cols else None)
    except Exception:
        return None, None


def _caption_text(item, dl_doc) -> str | None:
    fn = getattr(item, "caption_text", None)
    if callable(fn):
        try:
            text = fn(dl_doc)
            if text:
                return clean_text(str(text))
        except Exception:
            pass
    return None


def _fill_page_text(doc: ParsedDocument) -> None:
    """Docling yields elements, not page text; rebuild page text from them."""
    buckets: dict[int, list[str]] = {}
    for element in doc.elements:
        if element.content.strip():
            buckets.setdefault(element.page_number, []).append(element.content.strip())

    known = {p.page_number for p in doc.pages}
    for page_number in sorted(buckets):
        if page_number not in known:
            doc.pages.append(
                ParsedPage(page_number=page_number, width=0.0, height=0.0)
            )
    doc.pages.sort(key=lambda p: p.page_number)

    for page in doc.pages:
        page.text = "\n".join(buckets.get(page.page_number, []))
        page.has_text_layer = bool(page.text.strip())
        page.image_count = sum(
            1
            for e in doc.elements
            if e.element_type == FIGURE and e.page_number == page.page_number
        )


def _document_title(dl_doc, doc: ParsedDocument) -> str | None:
    name = getattr(dl_doc, "name", None)
    for element in doc.elements:
        if element.element_type == HEADING and element.level == 1:
            return element.content
    return clean_text(str(name)) if name else None


def _origin_metadata(dl_doc) -> dict:
    origin = getattr(dl_doc, "origin", None)
    if origin is None:
        return {}
    out = {}
    for field in ("filename", "mimetype", "binary_hash"):
        value = getattr(origin, field, None)
        if value is not None:
            out[field] = str(value)
    return out
