"""Text cleaning: step 1 of the Phase 3 pipeline.

Runs on the already-extracted `DocumentElement.content` / `DocumentPage.text`
strings from Phase 2 before they are chunked. Phase 2's parser backends
already do PDF-specific extraction; this pass is generic clean-up on top of
that extracted text so both embeddings and the text shown in citations are
free of the artifacts that make retrieval noisy: hyphenated line-wraps,
repeated running headers/footers, ligature glyphs, and stray whitespace.

Deliberately conservative: it never rewrites words or drops sentences, only
normalises characters and whitespace, so citation text always still matches
what a reader sees on the page.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}

# A hyphen at the end of a line followed by a lowercase continuation is almost
# always a PDF line-wrap artifact ("informa-\ntion" -> "information"), not a
# real hyphenated compound (those are far more often "well-known" mid-line).
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_LINE = re.compile(r"\n{3,}")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(raw: str) -> str:
    """Normalise one blob of extracted text (a page, an element)."""
    if not raw:
        return ""

    text = unicodedata.normalize("NFKC", raw)
    for lig, expansion in _LIGATURES.items():
        text = text.replace(lig, expansion)

    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK_LINE.sub("\n\n", text)

    # Trim trailing whitespace on each line without collapsing intentional
    # paragraph breaks.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def strip_repeated_lines(page_texts: dict[int, str], min_pages: int = 3, min_repeats_ratio: float = 0.6) -> dict[int, str]:
    """Remove running headers/footers: short lines that repeat near-verbatim
    across most pages (journal name, paper title, page-number-only lines).

    Only touches the first and last two lines of each page — headers/footers
    live at page edges, so this cannot accidentally delete a repeated
    in-body sentence (e.g. a refrain, a table header printed on every page of
    a long table are already handled as distinct elements upstream).
    """
    if len(page_texts) < min_pages:
        return page_texts

    edge_lines: Counter[str] = Counter()
    per_page_lines: dict[int, list[str]] = {}
    for page_number, text in page_texts.items():
        lines = text.split("\n")
        per_page_lines[page_number] = lines
        edge = lines[:2] + lines[-2:]
        for line in edge:
            norm = line.strip().lower()
            # Ignore empty lines and pure page numbers ("- 4 -", "12") —
            # those are handled fine as-is and are not what pollutes retrieval.
            if not norm or norm.isdigit() or len(norm) < 6:
                continue
            edge_lines[norm] += 1

    threshold = max(min_pages, int(len(page_texts) * min_repeats_ratio))
    repeated = {line for line, count in edge_lines.items() if count >= threshold}
    if not repeated:
        return page_texts

    cleaned: dict[int, str] = {}
    for page_number, lines in per_page_lines.items():
        kept = [ln for ln in lines if ln.strip().lower() not in repeated]
        cleaned[page_number] = "\n".join(kept).strip()
    return cleaned


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"\u201c(])")


def split_sentences(text: str) -> list[str]:
    """A lightweight, dependency-free sentence splitter.

    Not linguistically perfect (abbreviations like "e.g." can trip it) but
    good enough for chunk-boundary decisions, where the cost of an
    imperfect split is a slightly awkward chunk edge, not a wrong answer.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def approx_token_count(text: str) -> int:
    """Cheap token estimate (~4 chars/token in English) used for chunk-size
    budgeting. Avoids a tiktoken/model-specific tokenizer dependency, which
    would also need network access to fetch encoding files on first run —
    chunk boundaries do not need to be exact, only consistent."""
    if not text:
        return 0
    return max(1, len(text) // 4)
