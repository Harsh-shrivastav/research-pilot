"""Parser-neutral structured representation of a PDF.

This module deliberately has **no** FastAPI / SQLAlchemy imports so the whole
parsing layer can be unit-tested (and run) standalone, without a database or a
web server. The persistence layer maps these dataclasses onto ORM rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# --------------------------------------------------------------------------
# Element types. Kept as plain string constants so they serialise directly into
# the `document_elements.element_type` column and into JSON for the frontend.
# --------------------------------------------------------------------------
HEADING = "heading"
PARAGRAPH = "paragraph"
TABLE = "table"
FIGURE = "figure"
CAPTION = "caption"
LIST_ITEM = "list_item"
FORMULA = "formula"
REFERENCE = "reference"

ELEMENT_TYPES = {
    HEADING,
    PARAGRAPH,
    TABLE,
    FIGURE,
    CAPTION,
    LIST_ITEM,
    FORMULA,
    REFERENCE,
}


@dataclass
class BBox:
    """Bounding box in PDF points, origin at the *top-left* of the page.

    Every backend normalises into this convention so the frontend can overlay
    boxes without knowing which parser produced them.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class ParsedElement:
    """A single atomic piece of content extracted from a page."""

    page_number: int
    element_type: str
    content: str
    order_index: int = 0
    bbox: Optional[BBox] = None
    level: Optional[int] = None  # headings only: 1 = top level
    section_index: Optional[int] = None  # filled in by structure.build_sections
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.element_type not in ELEMENT_TYPES:
            raise ValueError(f"unknown element_type: {self.element_type!r}")


@dataclass
class ParsedPage:
    page_number: int
    width: float
    height: float
    rotation: int = 0
    text: str = ""
    has_text_layer: bool = True
    ocr_applied: bool = False
    image_count: int = 0

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class ParsedSection:
    """A logical section, derived from the heading hierarchy."""

    index: int
    title: str
    level: int
    order_index: int
    start_page: int
    end_page: int
    parent_index: Optional[int] = None
    text: str = ""

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class ParsedReference:
    order_index: int
    raw_text: str
    authors: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    page_number: Optional[int] = None


@dataclass
class ParsedDocument:
    """Everything the pipeline knows about one PDF."""

    backend: str
    pages: list[ParsedPage] = field(default_factory=list)
    elements: list[ParsedElement] = field(default_factory=list)
    sections: list[ParsedSection] = field(default_factory=list)
    references: list[ParsedReference] = field(default_factory=list)
    title: Optional[str] = None
    authors: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    ocr_used: bool = False
    warnings: list[str] = field(default_factory=list)

    # -- convenience aggregates used by the API layer ----------------------
    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def word_count(self) -> int:
        return sum(len(p.text.split()) for p in self.pages)

    def count_of(self, element_type: str) -> int:
        return sum(1 for e in self.elements if e.element_type == element_type)

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "pages": self.page_count,
            "elements": len(self.elements),
            "sections": len(self.sections),
            "references": len(self.references),
            "tables": self.count_of(TABLE),
            "figures": self.count_of(FIGURE),
            "captions": self.count_of(CAPTION),
            "headings": self.count_of(HEADING),
            "paragraphs": self.count_of(PARAGRAPH),
            "ocr_used": self.ocr_used,
            "word_count": self.word_count,
        }
