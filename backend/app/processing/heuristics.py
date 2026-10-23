"""Shared, backend-agnostic layout heuristics.

The Docling backend gets most of this for free from its layout model. The
PyMuPDF / pdfplumber backends need explicit heuristics to tell a heading from a
paragraph, spot a caption, and recognise the reference section. Keeping them in
one module means all backends agree on what a "heading" is.
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------

_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
}


def clean_text(text: str) -> str:
    """Normalise unicode, expand ligatures, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    # de-hyphenate words broken across lines: "docu-\nment" -> "document"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\n", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------

# "1", "1.2", "3.4.1", "IV", "A." style prefixes
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
_ROMAN = re.compile(r"^([IVXL]{1,6})\.?\s+(\S.+)$")
_LETTERED = re.compile(r"^([A-Z])\.\s+(\S.+)$")

# Canonical IMRaD-ish section names that are headings even without styling cues
_CANONICAL_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "related works",
    "prior work",
    "motivation",
    "methods",
    "method",
    "methodology",
    "materials and methods",
    "approach",
    "model",
    "architecture",
    "experiments",
    "experimental setup",
    "experimental results",
    "evaluation",
    "results",
    "results and discussion",
    "analysis",
    "ablation study",
    "ablation studies",
    "discussion",
    "limitations",
    "threats to validity",
    "future work",
    "conclusion",
    "conclusions",
    "conclusion and future work",
    "acknowledgements",
    "acknowledgments",
    "references",
    "bibliography",
    "appendix",
    "supplementary material",
    "appendices",
    "data availability",
    "funding",
    "author contributions",
    "declaration of competing interest",
    "ethics statement",
}

_REFERENCE_TITLES = {
    "references",
    "reference",
    "bibliography",
    "works cited",
    "literature cited",
    "references cited",
}

_MAX_HEADING_WORDS = 18
_MAX_HEADING_CHARS = 160


def normalise_heading(title: str) -> str:
    """Lowercase, strip numbering and trailing punctuation, for comparisons."""
    t = title.strip()
    for pattern in (_NUMBERED, _ROMAN, _LETTERED):
        m = pattern.match(t)
        if m:
            t = m.group(2)
            break
    t = t.strip().strip(".:;,").strip()
    return " ".join(t.lower().split())


def is_reference_heading(title: str) -> bool:
    return normalise_heading(title) in _REFERENCE_TITLES


def numbering_depth(text: str) -> int | None:
    """Return heading depth implied by a numeric prefix, else None."""
    m = _NUMBERED.match(text.strip())
    if not m:
        return None
    return min(m.group(1).count(".") + 1, 6)


def heading_level(
    text: str,
    *,
    size: float | None = None,
    body_size: float | None = None,
    bold: bool = False,
) -> int | None:
    """Decide whether `text` is a heading; return its level (1-based) or None.

    `size`/`body_size` are font sizes in points; when a backend cannot supply
    them the decision falls back to purely textual cues.
    """
    t = (text or "").strip()
    if not t:
        return None
    words = t.split()
    if len(words) > _MAX_HEADING_WORDS or len(t) > _MAX_HEADING_CHARS:
        return None
    # Paragraph-ending punctuation is a strong negative signal.
    if t.endswith((".", ";", ",")) and numbering_depth(t) is None:
        # "3.1 Setup." is still a heading; "We ran the model." is not.
        if not _looks_like_short_titlecase(t):
            return None
    if t.endswith(("?", "!")):
        return None

    depth = numbering_depth(t)
    canonical = normalise_heading(t) in _CANONICAL_HEADINGS
    bigger = (
        size is not None
        and body_size is not None
        and body_size > 0
        and size >= body_size * 1.12
    )
    same_size_bold = (
        bold
        and size is not None
        and body_size is not None
        and size >= body_size * 0.98
    )

    if canonical and (bigger or same_size_bold or depth is not None or len(words) <= 6):
        return depth or 1
    if depth is not None and (bigger or same_size_bold or _starts_titlecase(t)):
        return depth
    if bigger and len(words) <= _MAX_HEADING_WORDS:
        # Size alone: rank by how much bigger it is.
        assert size is not None and body_size is not None
        ratio = size / body_size
        if ratio >= 1.45:
            return 1
        if ratio >= 1.25:
            return 2
        return 3
    if same_size_bold and len(words) <= 10 and _starts_titlecase(t):
        return 3
    if _is_allcaps_title(t):
        return 1
    return None


def _starts_titlecase(text: str) -> bool:
    stripped = text.lstrip("0123456789.IVXL ").strip()
    return bool(stripped) and stripped[0].isupper()


def _looks_like_short_titlecase(text: str) -> bool:
    words = text.rstrip(".").split()
    if not (1 <= len(words) <= 6):
        return False
    caps = sum(1 for w in words if w[:1].isupper())
    return caps >= max(1, len(words) - 1)


def _is_allcaps_title(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return False
    if not all(c.isupper() for c in letters):
        return False
    return len(text.split()) <= 8


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

_CAPTION = re.compile(
    r"^\s*(figure|fig\.?|table|tbl\.?|chart|algorithm|listing|scheme|exhibit)\s*"
    r"([0-9]+(?:\.[0-9]+)*|[IVXL]+)\s*([.:\)\u2013\u2014-])?\s*(.*)$",
    re.IGNORECASE,
)

_CAPTION_KIND = {
    "figure": "figure",
    "fig": "figure",
    "fig.": "figure",
    "chart": "figure",
    "scheme": "figure",
    "exhibit": "figure",
    "table": "table",
    "tbl": "table",
    "tbl.": "table",
    "algorithm": "figure",
    "listing": "figure",
}


def parse_caption(text: str) -> tuple[str, str, str] | None:
    """Return (kind, label, caption_body) when `text` opens like a caption.

    kind is "figure" or "table" — i.e. what the caption refers to.
    """
    t = (text or "").strip()
    if not t:
        return None
    m = _CAPTION.match(t)
    if not m:
        return None
    word = m.group(1).lower().rstrip(".")
    kind = _CAPTION_KIND.get(word) or _CAPTION_KIND.get(word + ".")
    if kind is None:
        return None
    separator = m.group(3)
    body = m.group(4).strip()

    if not separator:
        # No "Figure 1:" style separator. "Figure 1 shows that ..." is a prose
        # cross-reference, not a caption — a real caption's body starts with a
        # capital or the line is just the bare label.
        if body and not body[0].isupper():
            return None
        # "Table 3" alone mid-sentence is also a cross-reference.
        if not body and len(t) > 24:
            return None
    return kind, m.group(2), body


# ---------------------------------------------------------------------------
# Lists & formulas
# ---------------------------------------------------------------------------

_BULLET = re.compile(r"^\s*([\u2022\u2023\u25cf\u25aa\u2043\u2219*\-\u2013])\s+\S")
_ENUM = re.compile(r"^\s*(\(?\d{1,2}[\.\)]|\(?[a-z][\.\)])\s+\S")


def is_list_item(text: str) -> bool:
    t = text or ""
    return bool(_BULLET.match(t)) or bool(_ENUM.match(t))


_MATH_CHARS = set("=∑∫∏≤≥≈≠∈∀∃√±∇∂αβγδθλμσφψω")


def is_formula(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 220:
        return False
    if " " not in t and len(t) < 4:
        return False
    math_hits = sum(1 for c in t if c in _MATH_CHARS)
    letters = sum(1 for c in t if c.isalpha())
    if math_hits == 0:
        return False
    # Formula lines are math-dense relative to their prose content.
    return math_hits >= 1 and letters <= max(12, len(t) * 0.45)


# ---------------------------------------------------------------------------
# Running headers / footers / page numbers
# ---------------------------------------------------------------------------

_PAGE_NUM_ONLY = re.compile(r"^\s*[-–—\[\(]?\s*(?:page\s*)?\d{1,4}\s*(?:/\s*\d{1,4})?\s*[-–—\]\)]?\s*$", re.I)


def is_page_number(text: str) -> bool:
    return bool(_PAGE_NUM_ONLY.match(text or ""))


def find_repeating_lines(page_texts: list[list[str]], threshold: float = 0.6) -> set[str]:
    """Lines appearing near-identically on most pages are headers/footers."""
    if len(page_texts) < 4:
        return set()
    counts: dict[str, int] = {}
    for lines in page_texts:
        for line in set(_norm_running(l) for l in lines if l.strip()):
            counts[line] = counts.get(line, 0) + 1
    cutoff = max(3, int(len(page_texts) * threshold))
    return {line for line, n in counts.items() if n >= cutoff and len(line) > 3}


def _norm_running(line: str) -> str:
    # Strip digits so "Smith et al.  page 4" matches across pages.
    return re.sub(r"\d+", "#", " ".join((line or "").lower().split()))


def is_running_line(line: str, repeating: set[str]) -> bool:
    return _norm_running(line) in repeating
