"""OCR top-up for pages that carry no embedded text layer.

Scanned PDFs parse "successfully" with every backend — they just yield zero
text. Rather than reporting an empty document, the pipeline rasterises those
specific pages and runs Tesseract over them, then feeds the recognised words
back through the same line/heading/caption heuristics used for born-digital
pages. OCR is only applied to pages that need it, so mixed documents pay only
for their scanned pages.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .heuristics import find_repeating_lines, is_running_line
from .schema import ParsedElement, ParsedPage
from .backends.base import (
    Word,
    estimate_body_size,
    group_words_into_lines,
    lines_to_elements,
    split_columns,
)

logger = logging.getLogger(__name__)

DEFAULT_DPI = 200
# Tesseract confidence below this is treated as noise.
MIN_CONFIDENCE = 40.0


class OcrUnavailable(RuntimeError):
    pass


@dataclass
class OcrPageResult:
    page_number: int
    text: str
    elements: list[ParsedElement]
    word_count: int


def ocr_available() -> bool:
    """True only when both the Python binding and the tesseract binary exist."""
    try:
        import pytesseract
    except Exception:
        return False
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return _rasteriser() is not None


def _rasteriser() -> str | None:
    """Pick whichever rasterisation route is installed."""
    try:
        import fitz  # PyMuPDF  # noqa: F401

        return "pymupdf"
    except Exception:
        pass
    try:
        import pdf2image  # noqa: F401

        # pdf2image needs the poppler binaries, not just the wrapper.
        from pdf2image.exceptions import PDFInfoNotInstalledError  # noqa: F401
        import shutil

        if shutil.which("pdftoppm"):
            return "pdf2image"
    except Exception:
        pass
    try:
        import pypdfium2  # noqa: F401

        return "pypdfium2"
    except Exception:
        pass
    return None


def render_page(path: str, page_number: int, dpi: int = DEFAULT_DPI):
    """Render a 1-based page to a PIL image using whatever is available."""
    route = _rasteriser()
    if route is None:
        raise OcrUnavailable("no PDF rasteriser installed")

    if route == "pymupdf":
        import fitz
        from PIL import Image

        with fitz.open(path) as pdf:
            page = pdf[page_number - 1]
            zoom = dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            mode = "RGBA" if pix.alpha else "RGB"
            return Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("RGB")

    if route == "pdf2image":
        from pdf2image import convert_from_path

        images = convert_from_path(
            path, dpi=dpi, first_page=page_number, last_page=page_number
        )
        if not images:
            raise OcrUnavailable(f"could not rasterise page {page_number}")
        return images[0].convert("RGB")

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        page = pdf[page_number - 1]
        bitmap = page.render(scale=dpi / 72.0)
        return bitmap.to_pil().convert("RGB")
    finally:
        pdf.close()


def ocr_page(
    path: str,
    page: ParsedPage,
    *,
    dpi: int = DEFAULT_DPI,
    repeating: set[str] | None = None,
) -> OcrPageResult:
    """Run OCR on one page and return recognised text plus parsed elements."""
    import pytesseract
    from PIL import Image  # noqa: F401  (ensures Pillow is present)

    image = render_page(path, page.page_number, dpi=dpi)
    data = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT, config="--psm 3"
    )

    # Tesseract works in image pixels; convert back to PDF points so bounding
    # boxes are in the same coordinate space as born-digital elements.
    scale = 72.0 / dpi
    words: list[Word] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < MIN_CONFIDENCE:
            continue
        left = float(data["left"][i]) * scale
        top = float(data["top"][i]) * scale
        width = float(data["width"][i]) * scale
        height = float(data["height"][i]) * scale
        words.append(
            Word(
                text=text,
                x0=left,
                top=top,
                x1=left + width,
                bottom=top + height,
                # Tesseract reports no font size; glyph height is a usable proxy
                # and keeps the relative-size heading heuristic working.
                size=round(height, 1),
                bold=False,
            )
        )

    if not words:
        return OcrPageResult(page.page_number, "", [], 0)

    lines = group_words_into_lines(words, tolerance=3.5)
    page_text = "\n".join(l.text for l in lines)

    body_size = estimate_body_size(words)
    ordered = split_columns(lines, page.width or 0)
    elements = lines_to_elements(
        ordered,
        page.page_number,
        body_size,
        start_order=0,
        skip_line=(
            (lambda text: is_running_line(text, repeating)) if repeating else (lambda text: False)
        ),
        size_reliable=False,
    )
    for element in elements:
        element.meta.setdefault("source", "ocr")

    return OcrPageResult(page.page_number, page_text, elements, len(words))


def ocr_document_pages(
    path: str,
    pages: list[ParsedPage],
    *,
    dpi: int = DEFAULT_DPI,
    max_pages: int | None = None,
) -> dict[int, OcrPageResult]:
    """OCR every page in `pages` (already filtered to text-less pages)."""
    if not pages:
        return {}
    if not ocr_available():
        raise OcrUnavailable("pytesseract/tesseract or a rasteriser is unavailable")

    targets = pages if max_pages is None else pages[:max_pages]
    results: dict[int, OcrPageResult] = {}

    # Two passes so running headers/footers can be detected across OCR'd pages.
    raw: dict[int, OcrPageResult] = {}
    for page in targets:
        try:
            raw[page.page_number] = ocr_page(path, page, dpi=dpi)
        except Exception as exc:
            logger.warning("OCR failed on page %s: %s", page.page_number, exc)

    if len(raw) >= 4:
        repeating = find_repeating_lines(
            [r.text.split("\n") for r in raw.values()]
        )
        for page in targets:
            if page.page_number not in raw:
                continue
            try:
                results[page.page_number] = ocr_page(
                    path, page, dpi=dpi, repeating=repeating
                )
            except Exception as exc:
                logger.warning("OCR re-pass failed on page %s: %s", page.page_number, exc)
                results[page.page_number] = raw[page.page_number]
    else:
        results = raw

    return results
