"""Structured comparison extraction (Phase P2): ONE LLM call per paper fills
every requested dimension, and every answer carries a quote the backend
verifies against the paper's stored text before it is shown as verified.

    load_paper_material(db, doc)      chunks + elements -> plain data (no ORM)
    extract_paper_dimensions(...)     context -> generate_json -> {key: {found, value, quote, page}}
    check_quote(material, quote, pg)  pure-Python verification, no LLM

Why this shape
--------------
* Long context instead of per-cell retrieval: a research paper fits in a
  long-context model's window, so the model reads the whole paper once
  (page-labelled) and answers all dimensions together. When a paper is too
  long for `comparison_full_paper_max_chars`, only the parent chunks that
  retrieval ranks highest for each dimension are sent instead.
* Evidence-first: the model must return an EXACT quote and a page. The quote
  is never trusted: `check_quote` finds it in the stored element text
  (whitespace, ligatures and quote/dash variants normalised the same way the
  chunker normalises text) and derives the page from the stored data, so a
  wrong page number from the model is corrected and an invented quote fails.
* Two verification tiers. (1) Exact: the normalised quote is a substring of
  the page text. (2) Column-tolerant: two-column PDFs are often extracted
  with the columns interleaved line by line, and a model that reads that text
  quotes it in the correct reading order, which is not a literal substring of
  the stored text. Tier 2 accepts a quote whose words all appear in the page
  text, in order, within a bounded window (what interleaving looks like),
  tolerating words split by hyphenation at a line end. Paraphrases and
  invented sentences do not pass either tier. Tier 2 can be switched off
  (COMPARISON_TOLERANT_QUOTES=false) to require exact substrings.
* DB-safe parallelism: `load_paper_material` returns plain dataclasses, so
  the LLM work for different papers can run in threads (each with its own
  short-lived Session, the way HybridRetriever does it) while all writes stay
  on the caller's session.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.comparison.dimensions import Dimension
from app.config import Settings
from app.models import ChunkLevel, Document, DocumentChunk, DocumentElement
from app.rag.interfaces import LLMMessage, LLMProvider, RetrievedChunk
from app.rag.text_clean import clean_text

logger = logging.getLogger(__name__)

MAX_QUOTE_CHARS = 300
MAX_VALUE_CHARS = 1200
# A quote shorter than this proves nothing (it matches almost anywhere).
MIN_QUOTE_CHARS = 12

# Tier-2 (column-tolerant) matching limits.
TOLERANT_MAX_GAP_TOKENS = 40  # next quote word must appear within this many page words
TOLERANT_MAX_MISS_RATIO = 0.05  # at most 5% of the quote's words may be unmatched
TOLERANT_MIN_WORDS = 6  # shorter quotes are not specific enough to match loosely
TOLERANT_MIN_CHUNK_COVERAGE = 0.8

NOT_FOUND_ITEM = {"found": False, "value": "", "quote": "", "page": None}


class ExtractionError(RuntimeError):
    """The structured path cannot produce a usable result for this paper;
    the caller falls back to the per-cell path for it."""


# ---------------------------------------------------------------------------
# Plain-data view of one paper (no ORM objects, safe to share across threads)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkInfo:
    id: str
    parent_id: str | None
    level: str
    page: int  # start page
    end_page: int
    order: int
    text: str
    section_id: str | None
    section_title: str | None
    element_types: tuple[str, ...]
    bbox: dict | None


@dataclass(frozen=True)
class ElementInfo:
    page: int
    text: str  # already normalised for matching
    bbox: dict | None


_PUNCT_MAP = str.maketrans(
    {
        "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
        "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
        "―": "-", "−": "-",
        "­": None, "​": None, "‌": None, "‍": None, "﻿": None,
    }
)


def normalize_for_match(text: str) -> str:
    """The comparison form used for quote checking: the chunker's own
    `clean_text` (NFKC, ligatures, hyphenated line-wraps, whitespace) plus
    curly->straight quotes, dash variants and invisible characters, with all
    whitespace collapsed. Case is NOT folded: a quote must be exact."""
    cleaned = clean_text(text or "").translate(_PUNCT_MAP)
    return " ".join(cleaned.split())


def _bbox(x0, y0, x1, y1) -> dict | None:
    if x0 is None or y0 is None:
        return None
    return {"x0": x0, "y0": y0, "x1": x1 if x1 is not None else x0, "y1": y1 if y1 is not None else y0}


class PaperMaterial:
    """Everything quote verification and context building need about one
    paper, loaded once."""

    def __init__(self, parents: list[ChunkInfo], children: list[ChunkInfo], elements: list[ElementInfo]):
        self.parents = parents
        self.children = children
        self.elements = elements
        self.elements_by_page: dict[int, list[ElementInfo]] = {}
        for element in elements:
            self.elements_by_page.setdefault(element.page, []).append(element)
        self.pages: dict[int, str] = {
            page: " ".join(e.text for e in els if e.text) for page, els in self.elements_by_page.items()
        }
        self._norm: dict[str, str] = {}
        self._page_tokens: dict[int, list[str]] = {}
        self._chunk_keys: dict[str, frozenset[str]] = {}

    def page_tokens(self, page: int) -> list[str]:
        cached = self._page_tokens.get(page)
        if cached is None:
            cached = self._page_tokens[page] = _word_keys(self.pages.get(page, ""))
        return cached

    def chunk_keys(self, chunk: ChunkInfo) -> frozenset[str]:
        cached = self._chunk_keys.get(chunk.id)
        if cached is None:
            cached = self._chunk_keys[chunk.id] = frozenset(_word_keys(self.norm(chunk)))
        return cached

    def norm(self, chunk: ChunkInfo) -> str:
        cached = self._norm.get(chunk.id)
        if cached is None:
            cached = self._norm[chunk.id] = normalize_for_match(chunk.text)
        return cached


def load_paper_material(db: Session, document_id: str) -> PaperMaterial:
    # Column selects (not entities): skips the embedding vector on every child chunk.
    chunk_rows = db.execute(
        select(
            DocumentChunk.id, DocumentChunk.parent_chunk_id, DocumentChunk.chunk_level,
            DocumentChunk.page_number, DocumentChunk.order_index, DocumentChunk.text,
            DocumentChunk.section_id, DocumentChunk.chunk_metadata,
        )
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.order_index)
    ).all()
    parents: list[ChunkInfo] = []
    children: list[ChunkInfo] = []
    for cid, parent_id, level, page, order, text, section_id, meta in chunk_rows:
        meta = meta or {}
        info = ChunkInfo(
            id=cid, parent_id=parent_id, level=level, page=page,
            end_page=int(meta.get("end_page") or page), order=order, text=text or "",
            section_id=section_id, section_title=meta.get("section_title"),
            element_types=tuple(meta.get("element_types") or ()), bbox=meta.get("bbox"),
        )
        (parents if level == ChunkLevel.PARENT else children).append(info)

    element_rows = db.execute(
        select(
            DocumentElement.page_number, DocumentElement.content, DocumentElement.bbox_x0,
            DocumentElement.bbox_y0, DocumentElement.bbox_x1, DocumentElement.bbox_y1,
        )
        .where(DocumentElement.document_id == document_id)
        .order_by(DocumentElement.order_index)
    ).all()
    elements = [
        ElementInfo(page=page, text=normalize_for_match(content or ""), bbox=_bbox(x0, y0, x1, y1))
        for page, content, x0, y0, x1, y1 in element_rows
    ]
    return PaperMaterial(parents, children, elements)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def _page_label(chunk: ChunkInfo) -> str:
    if chunk.end_page > chunk.page:
        return f"[Pages {chunk.page}-{chunk.end_page}]"
    return f"[Page {chunk.page}]"


def build_paper_context(parents: list[ChunkInfo]) -> str:
    return "\n\n".join(f"{_page_label(p)}\n{p.text}" for p in parents)


def select_context(
    db: Session,
    settings: Settings,
    document: Document,
    material: PaperMaterial,
    dimensions: list[Dimension],
    retriever=None,
) -> tuple[list[ChunkInfo], str]:
    """The parent chunks to send, in page order, and which strategy chose
    them: "full" (whole paper, if it fits the budget) or "retrieved" (the
    top retrieved parents per dimension, round-robin, within the budget)."""
    budget = settings.comparison_full_paper_max_chars
    if sum(len(p.text) for p in material.parents) <= budget:
        return material.parents, "full"

    if retriever is None:
        from app.rag.service import build_retriever  # local: avoids an import cycle at module load

        retriever = build_retriever(db, settings)
    by_id = {p.id: p for p in material.parents}
    ranked: list[list[str]] = []
    for dimension in dimensions:
        try:
            hits = retriever.retrieve(document.id, dimension.query, settings.comparison_top_k)
        except Exception as exc:
            logger.warning("structured comparison: retrieval failed for %s (%s)", dimension.key, type(exc).__name__)
            hits = []
        ranked.append([h.parent_chunk_id for h in hits if h.parent_chunk_id in by_id])

    chosen: list[ChunkInfo] = []
    seen: set[str] = set()
    used = 0
    for rank in range(max((len(r) for r in ranked), default=0)):
        for per_dimension in ranked:
            if rank >= len(per_dimension) or per_dimension[rank] in seen:
                continue
            parent = by_id[per_dimension[rank]]
            if chosen and used + len(parent.text) > budget:
                continue
            seen.add(parent.id)
            chosen.append(parent)
            used += len(parent.text)
    if not chosen:
        raise ExtractionError("paper is too long to send whole and retrieval found no passages")
    return sorted(chosen, key=lambda p: p.order), "retrieved"


# ---------------------------------------------------------------------------
# Prompt, schema, response handling
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
You extract facts about ONE research paper into a fixed set of comparison fields, using ONLY \
the paper text provided. The text is divided into blocks; each block starts with a marker like \
[Page 4] or [Pages 4-5] saying which page(s) it comes from.

Rules you must follow:
1. Use ONLY the paper text. No outside knowledge, no assumptions, no guessing — even if you \
recognise the paper.
2. Return one entry for every field listed, using its key exactly.
3. found = true only if the paper explicitly states the answer. Otherwise return found = false \
with value = "", quote = "" and page = 0. Prefer found = false over a guess.
4. value: your answer, at most 2 sentences, plain text, no citation markers.
5. quote: an EXACT substring copied character-for-character from the paper text that supports \
the value. At most 300 characters. Do not paraphrase, reorder, merge separate passages, add \
ellipses, or include the [Page N] markers.
6. page: the page number (an integer) where that quote appears, from the [Page N] marker of the \
block you copied it from.
"""


def build_schema(dimensions: list[Dimension]) -> dict:
    """JSON Schema for the structured response: one object per field."""
    return {
        "type": "object",
        "properties": {
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "enum": [d.key for d in dimensions]},
                        "found": {"type": "boolean"},
                        "value": {"type": "string"},
                        "quote": {"type": "string"},
                        "page": {"type": "integer"},
                    },
                    "required": ["key", "found", "value", "quote", "page"],
                },
            }
        },
        "required": ["fields"],
    }


def build_user_message(document: Document, dimensions: list[Dimension], context: str) -> str:
    fields = "\n".join(f"- key: {d.key} | {d.label}: {d.query}" for d in dimensions)
    title = document.title or document.original_filename or "Untitled paper"
    return (
        f"Paper: {title}\n\n"
        f"Fields to fill:\n{fields}\n\n"
        f"Paper text:\n\n{context}\n\n"
        "Return the JSON object with one entry per field."
    )


def normalise_response(data: dict, dimensions: list[Dimension]) -> dict[str, dict]:
    """Coerce the model's JSON into {key: {found, value, quote, page}} for
    every requested dimension; anything missing or malformed is "not found"."""
    fields = data.get("fields") if isinstance(data, dict) else None
    if not isinstance(fields, list):
        raise ExtractionError("structured response has no 'fields' list")

    valid = {d.key for d in dimensions}
    out: dict[str, dict] = {}
    for item in fields:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if key not in valid or key in out:
            continue
        value = str(item.get("value") or "").strip()
        if item.get("found") is not True or not value:
            out[key] = dict(NOT_FOUND_ITEM)
            continue
        quote = " ".join(str(item.get("quote") or "").split())[:MAX_QUOTE_CHARS]
        try:
            page = int(item.get("page"))
        except (TypeError, ValueError):
            page = 0
        out[key] = {"found": True, "value": value[:MAX_VALUE_CHARS], "quote": quote, "page": page if page > 0 else None}
    for dimension in dimensions:
        out.setdefault(dimension.key, dict(NOT_FOUND_ITEM))
    return out


def extract_paper_dimensions(
    db: Session,
    settings: Settings,
    llm: LLMProvider,
    document: Document,
    dimensions: list[Dimension],
    *,
    material: PaperMaterial | None = None,
    retriever=None,
) -> dict[str, dict]:
    """One structured LLM call for one paper. Returns
    {dimension_key: {found, value, quote, page}}. Raises on any failure
    (provider without JSON support, invalid JSON twice, no text): the caller
    falls back to the per-cell path for this paper."""
    material = material or load_paper_material(db, document.id)
    if not material.parents:
        raise ExtractionError("document has no parent chunks to read")
    parents, scope = select_context(db, settings, document, material, dimensions, retriever)
    context = build_paper_context(parents)
    logger.info(
        "structured comparison: document %s, %d dimension(s), context=%s (%d parents, %d chars)",
        document.id, len(dimensions), scope, len(parents), len(context),
    )
    data = llm.generate_json(
        EXTRACTION_SYSTEM_PROMPT,
        [LLMMessage(role="user", content=build_user_message(document, dimensions, context))],
        build_schema(dimensions),
    )
    return normalise_response(data, dimensions)


# ---------------------------------------------------------------------------
# Quote verification (pure Python — no LLM)
# ---------------------------------------------------------------------------


_EDGE_PUNCT = ".,;:()[]{}\"'"


def _word_keys(text: str) -> list[str]:
    """Words with edge punctuation stripped (case is kept: a quote stays
    exact); pure-punctuation tokens are dropped. The unit tier-2 matching
    works on."""
    keys = []
    for token in text.split():
        key = token.strip(_EDGE_PUNCT)
        if key:
            keys.append(key)
    return keys


def _word_matches(quote_word: str, page_word: str) -> bool:
    if quote_word == page_word:
        return True
    # A page word cut by a line-end hyphen ("exception-") matches the whole
    # word the quote has ("exceptionally"); the rest of it sits elsewhere in
    # the interleaved text and is simply not needed.
    return len(page_word) >= 5 and page_word.endswith("-") and quote_word.startswith(page_word[:-1])


def _ordered_match(quote_words: list[str], page_words: list[str]) -> bool:
    """Do the quote's words occur in `page_words` in order, each within
    TOLERANT_MAX_GAP_TOKENS of the previous one, with (almost) none missing
    and the last word matched? See the module docstring, tier 2."""
    n = len(quote_words)
    if n < TOLERANT_MIN_WORDS:
        return False
    allowed_misses = int(n * TOLERANT_MAX_MISS_RATIO)
    for start, word in enumerate(page_words):
        if not _word_matches(quote_words[0], word):
            continue
        position = start
        misses = 0
        last_matched = True
        for quote_word in quote_words[1:]:
            window = page_words[position + 1 : position + 1 + TOLERANT_MAX_GAP_TOKENS]
            hit = next((i for i, w in enumerate(window) if _word_matches(quote_word, w)), None)
            if hit is None:
                misses += 1
                last_matched = False
                if misses > allowed_misses:
                    break
                continue
            position += 1 + hit
            last_matched = True
        else:
            if last_matched:
                return True
    return False


@dataclass
class QuoteCheck:
    verified: bool
    reason: str = ""
    page: int | None = None  # derived from stored data, never the model's claim
    chunk: ChunkInfo | None = None
    bbox: dict | None = None
    page_corrected: bool = False
    match: str = ""  # "exact" | "column-tolerant"


def _page_order(pages: list[int], cited: int | None) -> list[int]:
    if cited is None:
        return sorted(pages)
    return sorted(pages, key=lambda p: (p != cited, abs(p - cited), p))


def _find_chunk(material: PaperMaterial, qn: str, page: int) -> ChunkInfo | None:
    """The chunk that contains the quote: a child first (what retrieval
    cites), then a parent, then any chunk covering that page."""
    for chunk in material.children:
        if qn in material.norm(chunk):
            return chunk
    for chunk in material.parents:
        if qn in material.norm(chunk):
            return chunk
    for chunk in material.parents:
        if chunk.page <= page <= chunk.end_page:
            return chunk
    return None


def _find_chunk_tolerant(material: PaperMaterial, quote_words: list[str], page: int) -> ChunkInfo | None:
    """For a column-tolerant match there is no literal substring to locate:
    pick the chunk that holds the largest share of the quote's words."""
    wanted = set(quote_words)
    best: tuple[float, ChunkInfo] | None = None
    for chunk in material.children + material.parents:
        coverage = len(wanted & material.chunk_keys(chunk)) / len(wanted)
        if coverage >= TOLERANT_MIN_CHUNK_COVERAGE and (best is None or coverage > best[0]):
            best = (coverage, chunk)
    if best is not None:
        return best[1]
    return next((c for c in material.parents if c.page <= page <= c.end_page), None)


def check_quote(
    material: PaperMaterial, quote: str, cited_page: int | None, tolerant: bool = True
) -> QuoteCheck:
    """Is `quote` a verbatim passage of the paper? The cited page is tried
    first, then neighbouring pages (a chunk spanning pages is labelled with
    only its first page, so the model's page can be off by one); the page
    reported is the one the stored text actually has it on."""
    qn = normalize_for_match(quote)
    if len(qn) < MIN_QUOTE_CHARS:
        return QuoteCheck(False, reason="quote missing or too short to verify")

    found_page: int | None = None
    bbox: dict | None = None
    for page in _page_order(list(material.pages), cited_page):
        if qn in material.pages[page]:
            found_page = page
            bbox = next((e.bbox for e in material.elements_by_page[page] if qn in e.text), None)
            break
    if found_page is None:  # a quote that straddles a page break
        for page in sorted(material.pages):
            if page + 1 in material.pages and qn in f"{material.pages[page]} {material.pages[page + 1]}":
                found_page = page
                break
    match = "exact"
    quote_words: list[str] = []
    if found_page is None and tolerant:
        quote_words = _word_keys(qn)
        for page in _page_order(list(material.pages), cited_page):
            if _ordered_match(quote_words, material.page_tokens(page)):
                found_page, match = page, "column-tolerant"
                break
    if found_page is None:
        return QuoteCheck(False, reason="quote not found in the paper's text")

    if match == "exact":
        chunk = _find_chunk(material, qn, found_page)
    else:
        chunk = _find_chunk_tolerant(material, quote_words, found_page)
    if chunk is None:
        return QuoteCheck(False, reason="no stored chunk covers the quote")
    return QuoteCheck(
        True, page=found_page, chunk=chunk, bbox=bbox or chunk.bbox,
        page_corrected=cited_page is not None and found_page != cited_page, match=match,
    )


def quote_as_retrieved_chunk(document_id: str, quote: str, check: QuoteCheck) -> RetrievedChunk:
    """A verified quote in the shape the existing evidence flow consumes
    (`evidence_from_retrieved_chunk`): the snippet is the quote itself and the
    chunk is the stored chunk that contains it. Score 1.0 = exact match."""
    chunk = check.chunk
    assert chunk is not None and check.page is not None
    return RetrievedChunk(
        chunk_id=chunk.id,
        parent_chunk_id=chunk.parent_id,
        document_id=document_id,
        page_number=check.page,
        section_id=chunk.section_id,
        section_title=chunk.section_title,
        child_text=" ".join(quote.split()),
        context_text=chunk.text,
        score=1.0,
        element_types=list(chunk.element_types),
        bbox=check.bbox,
    )
