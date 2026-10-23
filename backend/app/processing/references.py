"""Reference/bibliography extraction.

Finds the references section in the element stream, splits it into individual
entries, and does a light structured parse (authors / title / year / venue /
DOI / URL). This is deliberately heuristic — it reports what it can actually
find in the PDF and leaves fields as None when they are not determinable,
rather than inventing plausible-looking metadata.
"""
from __future__ import annotations

import re

from .heuristics import clean_text, is_reference_heading
from .schema import HEADING, LIST_ITEM, PARAGRAPH, ParsedElement, ParsedReference

# "[1]" / "[12]" markers
_BRACKET_MARKER = re.compile(r"^\s*\[(\d{1,3})\]\s*")
# "1." / "12)" markers
_NUMERIC_MARKER = re.compile(r"^\s*(\d{1,3})\s*[\.\)]\s+")

_DOI = re.compile(r"\b(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.I)
_URL = re.compile(r"https?://[^\s,;]+")
_YEAR = re.compile(r"(?:\(|\b)((?:19|20)\d{2})(?:\)|\b)")

# An entry usually ends in a year, page range, DOI or URL; and a new entry
# usually starts with an author-ish token.
_AUTHOR_START = re.compile(
    r"^(?:[A-Z][A-Za-z'’\-]+,\s*(?:[A-Z]\.\s*)+|[A-Z]\.\s*[A-Z][A-Za-z'’\-]+|[A-Z][A-Za-z'’\-]+\s+[A-Z]\.)"
)

_MIN_ENTRY_CHARS = 18
_MAX_ENTRY_CHARS = 1200


def extract_references(elements: list[ParsedElement]) -> list[ParsedReference]:
    """Return parsed references, or an empty list when none are detectable."""
    lines = _collect_reference_lines(elements)
    if not lines:
        return []

    raw_entries = _split_entries(lines)
    references: list[ParsedReference] = []
    for i, (text, page_number) in enumerate(raw_entries):
        text = clean_text(text)
        if len(text) < _MIN_ENTRY_CHARS:
            continue
        references.append(_parse_entry(text[:_MAX_ENTRY_CHARS], i, page_number))
    return references


def _collect_reference_lines(
    elements: list[ParsedElement],
) -> list[tuple[str, int]]:
    """Grab the (text, page) lines that sit under a references heading."""
    collecting = False
    start_level = 1
    out: list[tuple[str, int]] = []

    for element in elements:
        if element.element_type == HEADING:
            if is_reference_heading(element.content):
                collecting = True
                start_level = element.level or 1
                out = []  # a later references heading supersedes an earlier one
                continue
            if collecting and (element.level or 1) <= start_level:
                # Appendices etc. after the bibliography end the collection.
                break
            if collecting:
                continue
        if collecting and element.element_type in (PARAGRAPH, LIST_ITEM):
            content = element.content.strip()
            if content:
                for piece in _explode_markers(content):
                    out.append((piece, element.page_number))

    return out


# Multiple entries frequently land in a single paragraph element, because the
# line-merging step upstream cannot know it is inside a bibliography. Split
# them back apart on entry markers before doing anything else.
_INLINE_BRACKET = re.compile(r"(?=\[\d{1,3}\]\s)")
_INLINE_NUMERIC = re.compile(r"(?<=[.\)])\s+(?=\d{1,3}\.\s+[A-Z])")


def _explode_markers(text: str) -> list[str]:
    bracket_pieces = [p.strip() for p in _INLINE_BRACKET.split(text) if p.strip()]
    if len(bracket_pieces) > 1:
        return bracket_pieces

    numeric_pieces = [p.strip() for p in _INLINE_NUMERIC.split(text) if p.strip()]
    if len(numeric_pieces) > 1:
        return numeric_pieces

    return [text]


def _split_entries(lines: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Group physical lines into bibliography entries.

    Strategy, in order of reliability:
      1. If most lines carry "[n]" markers, split on those.
      2. Else if most lines carry "n." markers, split on those.
      3. Else start a new entry whenever a line begins like an author name and
         the previous entry already looks complete (ends with year/DOI/period).
    """
    if not lines:
        return []

    bracket_hits = sum(1 for text, _ in lines if _BRACKET_MARKER.match(text))
    numeric_hits = sum(1 for text, _ in lines if _NUMERIC_MARKER.match(text))

    marker: re.Pattern[str] | None = None
    if bracket_hits >= max(2, len(lines) * 0.3):
        marker = _BRACKET_MARKER
    elif numeric_hits >= max(2, len(lines) * 0.3):
        marker = _NUMERIC_MARKER

    entries: list[tuple[list[str], int]] = []

    for text, page in lines:
        starts_new = False
        if marker is not None:
            starts_new = bool(marker.match(text))
        else:
            starts_new = bool(_AUTHOR_START.match(text)) and (
                not entries or _looks_complete(" ".join(entries[-1][0]))
            )

        if starts_new or not entries:
            if marker is not None:
                text = marker.sub("", text, count=1)
            entries.append(([text], page))
        else:
            entries[-1][0].append(text)

    return [(" ".join(parts).strip(), page) for parts, page in entries]


def _looks_complete(entry: str) -> bool:
    tail = entry.rstrip()[-40:]
    if _YEAR.search(tail) or _DOI.search(tail) or _URL.search(tail):
        return True
    return tail.endswith(".") and len(entry) > 40


def _parse_entry(text: str, order_index: int, page_number: int) -> ParsedReference:
    doi_match = _DOI.search(text)
    url_match = _URL.search(text)
    year_match = _YEAR.search(text)

    authors, title, venue = _split_author_title_venue(text)

    return ParsedReference(
        order_index=order_index,
        raw_text=text,
        authors=authors,
        title=title,
        year=int(year_match.group(1)) if year_match else None,
        venue=venue,
        doi=doi_match.group(1) if doi_match else None,
        url=url_match.group(0).rstrip(".") if url_match else None,
        page_number=page_number,
    )


# A run of "Surname, A. B.," style names, optionally joined by "and"/"&".
# Anchored at the start so it only ever consumes a leading author list.
_AUTHOR_LIST = re.compile(
    r"^((?:(?:and\s+|&\s+)?[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+)?,\s*"
    r"(?:[A-Z]\.\s*)+,?\s*)+)"
)


def _split_author_title_venue(text: str) -> tuple[str | None, str | None, str | None]:
    """Split an entry into Authors / Title / Venue.

    Returns None for any part that cannot be determined confidently — an
    unparsed field is more useful than a wrong one, and `raw_text` always
    keeps the verbatim entry.
    """
    authors: str | None = None
    remainder = text

    match = _AUTHOR_LIST.match(text)
    if match:
        authors = match.group(1).strip().rstrip(",").strip()
        remainder = text[match.end() :].strip()

    parts = _sentence_split(remainder)
    if not parts:
        return _tidy(authors), None, None

    if authors is None:
        # No recognisable author list: fall back to "Authors. Title. Venue."
        if len(parts) < 2:
            return None, None, None
        authors, title = parts[0], parts[1]
        venue = ". ".join(parts[2:]) if len(parts) > 2 else None
    else:
        title = parts[0]
        venue = ". ".join(parts[1:]) if len(parts) > 1 else None

    return _tidy(authors), _tidy(title), _tidy(venue, limit=400)


def _sentence_split(text: str) -> list[str]:
    """Split on ". " while keeping initials like "J. Smith" intact."""
    if not text.strip():
        return []
    protected = re.sub(r"\b([A-Z])\.(\s)", r"\1<DOT>\2", text)
    return [
        p.strip().replace("<DOT>", ".")
        for p in protected.split(". ")
        if p.strip()
    ]


def _tidy(value: str | None, limit: int = 320) -> str | None:
    if not value:
        return None
    cleaned = value.strip().rstrip(",;").strip()
    # Strip a trailing period unless it belongs to an author initial ("Lopez, P.")
    if cleaned.endswith(".") and not re.search(r"\b[A-Z]\.$", cleaned):
        cleaned = cleaned[:-1].strip()
    if not cleaned:
        return None
    return cleaned[:limit]
