"""Parser backend protocol plus shared line/block assembly.

A backend's only job is to turn a PDF into ordered `ParsedElement`s and
`ParsedPage`s. Section building, reference extraction and OCR top-up all happen
above it in `pipeline.py`, so every backend benefits from them equally.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Protocol

from ..heuristics import (
    clean_text,
    heading_level,
    is_formula,
    is_list_item,
    is_page_number,
    parse_caption,
)
from ..schema import (
    CAPTION,
    FORMULA,
    HEADING,
    LIST_ITEM,
    PARAGRAPH,
    BBox,
    ParsedDocument,
    ParsedElement,
)


class ParserBackend(Protocol):
    name: str

    @staticmethod
    def available() -> bool:
        """True when the backend's third-party dependency is importable."""

    def parse(self, path: str) -> ParsedDocument:
        """Parse `path` into a ParsedDocument (sections/references left empty)."""


class BackendUnavailable(RuntimeError):
    """Raised when a backend is asked to parse but its dependency is missing."""


# ---------------------------------------------------------------------------
# Shared geometry helpers for word-level backends
# ---------------------------------------------------------------------------


@dataclass
class Word:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    size: float
    bold: bool = False

    @property
    def height(self) -> float:
        return max(self.bottom - self.top, 0.0)


@dataclass
class Line:
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()

    @property
    def x0(self) -> float:
        return min(w.x0 for w in self.words)

    @property
    def x1(self) -> float:
        return max(w.x1 for w in self.words)

    @property
    def top(self) -> float:
        return min(w.top for w in self.words)

    @property
    def bottom(self) -> float:
        return max(w.bottom for w in self.words)

    @property
    def size(self) -> float:
        sizes = [w.size for w in self.words if w.size]
        return statistics.median(sizes) if sizes else 0.0

    @property
    def bold(self) -> bool:
        if not self.words:
            return False
        return sum(1 for w in self.words if w.bold) >= len(self.words) * 0.6

    def bbox(self) -> BBox:
        return BBox(self.x0, self.top, self.x1, self.bottom)


def group_words_into_lines(words: list[Word], tolerance: float = 2.5) -> list[Line]:
    """Cluster words that share a baseline into lines, left-to-right."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(w.top, 1), w.x0))
    lines: list[Line] = []
    current = Line(words=[ordered[0]])
    for word in ordered[1:]:
        ref = current.words[-1]
        same_row = abs(word.top - ref.top) <= max(tolerance, ref.height * 0.5)
        if same_row:
            current.words.append(word)
        else:
            lines.append(current)
            current = Line(words=[word])
    lines.append(current)
    for line in lines:
        line.words.sort(key=lambda w: w.x0)
    return sorted(lines, key=lambda l: (l.top, l.x0))


def split_columns(lines: list[Line], page_width: float) -> list[Line]:
    """Reorder lines for a two-column layout, if one is detected.

    Academic PDFs are very often two-column; reading them in raw `top` order
    interleaves the columns and destroys paragraph continuity.
    """
    if len(lines) < 12 or page_width <= 0:
        return lines

    mid = page_width / 2
    left = [l for l in lines if l.x1 <= mid + page_width * 0.04]
    right = [l for l in lines if l.x0 >= mid - page_width * 0.04]
    spanning = [l for l in lines if l not in left and l not in right]

    # Require both columns to be substantial and spanning lines to be rare.
    if len(left) < 5 or len(right) < 5:
        return lines
    if len(spanning) > len(lines) * 0.25:
        return lines

    top_span = [l for l in spanning if l.top < min((l.top for l in left + right), default=0) + 1]
    rest_span = [l for l in spanning if l not in top_span]
    return (
        sorted(top_span, key=lambda l: l.top)
        + sorted(left, key=lambda l: l.top)
        + sorted(right, key=lambda l: l.top)
        + sorted(rest_span, key=lambda l: l.top)
    )


def estimate_body_size(all_words: list[Word]) -> float:
    """Most common font size, which is the body text size in practice."""
    sizes = [round(w.size, 1) for w in all_words if w.size and w.size > 0]
    if not sizes:
        return 0.0
    try:
        return statistics.mode(sizes)
    except statistics.StatisticsError:
        return statistics.median(sizes)


def lines_to_elements(
    lines: list[Line],
    page_number: int,
    body_size: float,
    *,
    start_order: int,
    skip_line: callable = lambda text: False,
    size_reliable: bool = True,
) -> list[ParsedElement]:
    """Assemble ordered lines into headings / paragraphs / captions / lists.

    Consecutive body lines are merged into a paragraph; a heading, caption,
    list item or formula always ends the current paragraph.

    `size_reliable=False` tells the heading detector to ignore font size. OCR
    has no real font metrics — the only size proxy is glyph bounding-box
    height, which varies with ascenders and descenders and would otherwise
    promote random body lines to headings. Textual cues (numeric prefixes,
    canonical section names, all-caps titles) still apply.
    """
    elements: list[ParsedElement] = []
    order = start_order
    buffer: list[Line] = []

    def flush() -> None:
        nonlocal order, buffer
        if not buffer:
            return
        text = clean_text(" ".join(l.text for l in buffer))
        if text:
            elements.append(
                ParsedElement(
                    page_number=page_number,
                    element_type=PARAGRAPH,
                    content=text,
                    order_index=order,
                    bbox=_union_bbox(buffer),
                )
            )
            order += 1
        buffer = []

    for line in lines:
        raw = line.text.strip()
        if not raw or is_page_number(raw) or skip_line(raw):
            flush()
            continue

        caption = parse_caption(raw)
        if caption is not None:
            flush()
            kind, label, body = caption
            elements.append(
                ParsedElement(
                    page_number=page_number,
                    element_type=CAPTION,
                    content=clean_text(raw),
                    order_index=order,
                    bbox=line.bbox(),
                    meta={"refers_to": kind, "label": f"{kind.title()} {label}", "body": body},
                )
            )
            order += 1
            continue

        level = heading_level(
            raw,
            size=line.size if size_reliable else None,
            body_size=body_size if size_reliable else None,
            bold=line.bold if size_reliable else False,
        )
        if level is not None:
            flush()
            elements.append(
                ParsedElement(
                    page_number=page_number,
                    element_type=HEADING,
                    content=clean_text(raw),
                    order_index=order,
                    bbox=line.bbox(),
                    level=level,
                )
            )
            order += 1
            continue

        if is_list_item(raw):
            flush()
            elements.append(
                ParsedElement(
                    page_number=page_number,
                    element_type=LIST_ITEM,
                    content=clean_text(raw),
                    order_index=order,
                    bbox=line.bbox(),
                )
            )
            order += 1
            continue

        if is_formula(raw):
            flush()
            elements.append(
                ParsedElement(
                    page_number=page_number,
                    element_type=FORMULA,
                    content=clean_text(raw),
                    order_index=order,
                    bbox=line.bbox(),
                )
            )
            order += 1
            continue

        buffer.append(line)

    flush()
    return elements


def _union_bbox(lines: list[Line]) -> BBox | None:
    if not lines:
        return None
    return BBox(
        min(l.x0 for l in lines),
        min(l.top for l in lines),
        max(l.x1 for l in lines),
        max(l.bottom for l in lines),
    )


def rows_to_markdown(rows: list[list[str | None]]) -> str:
    """Render an extracted table as GitHub-flavoured Markdown.

    Markdown keeps the table readable as plain text (for search and for the
    section view) while still being trivially renderable in the UI.
    """
    cleaned = [
        [(" ".join((cell or "").split())) for cell in row]
        for row in rows
        if row is not None and any((cell or "").strip() for cell in row)
    ]
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]

    header, *body = cleaned
    if not any(header):
        header = [f"col{i + 1}" for i in range(width)]
    out = ["| " + " | ".join(c.replace("|", "\\|") for c in header) + " |"]
    out.append("| " + " | ".join("---" for _ in header) + " |")
    for row in body:
        out.append("| " + " | ".join(c.replace("|", "\\|") for c in row) + " |")
    return "\n".join(out)


def finalise_document(doc: ParsedDocument) -> ParsedDocument:
    """Renumber element order globally so it is unique document-wide."""
    for i, element in enumerate(doc.elements):
        element.order_index = i
    return doc
