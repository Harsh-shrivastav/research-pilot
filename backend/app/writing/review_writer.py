"""Cited related-work draft from a finished comparison matrix.

The model only writes sentence text and says which evidence rows (E1..En) each
sentence rests on. Everything a reader sees as a citation is built here, from
the evidence table, never from model output: unknown IDs drop the sentence,
citation-like text inside a sentence is stripped, and the comparability caveat
is appended by code.

The Gemini provider already takes a `gemini_slot()` for each request (the slot
is a non-reentrant bounded semaphore), so `generate_json` is not wrapped in a
second one here — nesting could deadlock under load.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import Settings
from app.evidence import paper_service
from app.evidence.verification_service import load_support_info
from app.models import ComparisonMatrix, SupportStatus
from app.rag.gemini_limits import short_error
from app.rag.interfaces import LLMMessage, LLMProvider
from app.rag.workspace_context import get_document_titles

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2
MIN_PAPERS = 2
MIN_SENTENCE_WORDS = 4
MAX_PARAGRAPHS = 3

REQUIRES_GEMINI_MESSAGE = "Drafting requires Gemini (LLM_PROVIDER=gemini)."
NOT_ENOUGH_EVIDENCE_MESSAGE = (
    "Not enough verified evidence to draft: at least two papers need cells with grounded, "
    "supported evidence."
)
NOT_GROUNDED_MESSAGE = "Could not generate a grounded draft. Please try again."


class DraftUnavailableError(RuntimeError):
    """The configured provider cannot write drafts (only Gemini can)."""


class DraftError(RuntimeError):
    """No grounded draft could be produced; the message is safe to show."""


@dataclass
class EvidenceRow:
    id: str  # "E1"
    paper_label: str
    document_id: str
    page_number: int
    dimension_label: str
    value_text: str
    evidence_id: str
    support_status: str

    @property
    def citation_label(self) -> str:
        return f"[{self.paper_label}, p.{self.page_number}]"


WRITER_SYSTEM_PROMPT = """You write the related-work paragraph(s) of a literature review from a table of evidence rows. Each row is a fact extracted from a paper, with an ID (E1, E2, ...).

Rules:
- Academic, third person, neutral tone. At most 3 short paragraphs.
- Every sentence must cite at least one provided evidence ID in its "evidence_ids". Never invent an ID.
- Say nothing beyond what the rows state. No outside knowledge, no new numbers, no speculation.
- Do NOT put citations, author names, years, brackets or evidence IDs inside the sentence text; citations are added by the system. Refer to papers generically ("one study", "another approach") or by what they do.
- Use hedged wording ("appears to", "is reported to") for rows marked partially_supported.
- Compare or contrast papers only where the rows support it."""

_NOT_COMPARABLE_RULE = (
    "\n- The papers are NOT directly comparable ({overall}). Do not rank papers or say one is better, "
    "faster or more accurate than another based on numbers; describe what each paper does or reports separately."
)

# Citation-like text a model may add despite instructions.
_STRIP_PATTERNS = [
    re.compile(r"\(\s*(?:[A-Z][^()]*?)\b(?:19|20)\d{2}[a-z]?[^()]*\)"),  # (Smith et al., 2020)
    re.compile(r"\[[^\]]*\]"),  # [3], [Smith 2020, p.7]
    re.compile(r"\(\s*E\d+(?:\s*[,;]\s*E\d+)*\s*\)"),  # (E1, E2)
    re.compile(r"\bE\d+\b"),  # bare E-IDs
]


def clean_sentence(text: str) -> str:
    for pattern in _STRIP_PATTERNS:
        text = pattern.sub("", text or "")
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return " ".join(text.split())


def _surname(author: str) -> str:
    words = [w for w in re.split(r"\s+", author.strip()) if w]
    return words[-1].strip(".,;") if words else ""


def paper_label(db: Session, document_id: str, titles: dict[str, str], position: int) -> str:
    """"Smith et al. 2020" / "Smith 2020" when authors and year are known,
    else the document title, else "Paper N"."""
    paper = paper_service.get_paper_for_document(db, document_id)
    if paper is not None and paper.authors and paper.publication_year:
        names = [n for n in re.split(r"\s*(?:;|,|&|\band\b)\s*", paper.authors) if n.strip()]
        surname = _surname(names[0]) if names else ""
        if surname:
            return f"{surname} et al. {paper.publication_year}" if len(names) > 1 else f"{surname} {paper.publication_year}"
    return titles.get(document_id) or f"Paper {position}"


def build_evidence_table(db: Session, matrix: ComparisonMatrix) -> list[EvidenceRow]:
    titles = get_document_titles(db, list(matrix.document_ids))
    positions = {doc_id: i for i, doc_id in enumerate(matrix.document_ids, start=1)}
    labels: dict[str, str] = {}

    candidates = []
    for cell in matrix.cells:
        if not (cell.value_text or "").strip() or cell.insufficient_evidence or cell.error:
            continue
        if cell.quote_verified is False:
            continue
        citation = next((c for c in cell.citations or [] if c.get("evidence_id")), None)
        if citation is None:
            continue
        candidates.append((cell, citation))

    support = load_support_info(db, [c["evidence_id"] for _, c in candidates])
    rows: list[EvidenceRow] = []
    for cell, citation in candidates:
        info = support.get(citation["evidence_id"]) or {}
        status = info.get("status") if info.get("checked") else SupportStatus.UNCERTAIN
        if status == SupportStatus.UNSUPPORTED:
            continue
        if cell.document_id not in labels:
            labels[cell.document_id] = paper_label(db, cell.document_id, titles, positions.get(cell.document_id, 0))
        rows.append(
            EvidenceRow(
                id=f"E{len(rows) + 1}",
                paper_label=labels[cell.document_id],
                document_id=cell.document_id,
                page_number=int(citation.get("page_number") or 0),
                dimension_label=cell.dimension_label,
                value_text=" ".join(cell.value_text.split()),
                evidence_id=citation["evidence_id"],
                support_status=status or SupportStatus.UNCERTAIN,
            )
        )
    return rows


def _draft_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "paragraphs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sentences": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["text", "evidence_ids"],
                            },
                        }
                    },
                    "required": ["sentences"],
                },
            }
        },
        "required": ["paragraphs"],
    }


def _comparability_caveat(matrix: ComparisonMatrix) -> tuple[str | None, str | None]:
    """(overall, caveat) when the stored verdict says the papers are not
    directly comparable, else (None, None)."""
    stored = matrix.comparability or {}
    overall = stored.get("overall")
    if not overall or overall == "comparable":
        return None, None
    caveat = " ".join(str(stored.get("caveat") or "").split())
    return overall, caveat or None


def _user_message(rows: list[EvidenceRow]) -> str:
    lines = [
        f"{r.id} | {r.paper_label} | {r.dimension_label} | status={r.support_status} | {r.value_text}"
        for r in rows
    ]
    return "Evidence table (ID | paper | dimension | support status | fact):\n" + "\n".join(lines)


def validate_draft(raw: dict, rows: list[EvidenceRow]) -> tuple[list[dict], int]:
    """Model output -> paragraphs of validated sentences, plus the number of
    sentences dropped. A sentence is "ok" only with cleaned text and at least
    one evidence ID, all of them in the table; its labels come from the table."""
    by_id = {r.id: r for r in rows}
    paragraphs: list[dict] = []
    dropped = 0
    for raw_paragraph in (raw or {}).get("paragraphs") or []:
        sentences: list[dict] = []
        for raw_sentence in (raw_paragraph or {}).get("sentences") or []:
            if not isinstance(raw_sentence, dict):
                dropped += 1
                continue
            ids = [str(i).strip() for i in raw_sentence.get("evidence_ids") or []]
            text = clean_sentence(str(raw_sentence.get("text") or ""))
            valid = bool(ids) and all(i in by_id for i in ids) and len(text.split()) >= MIN_SENTENCE_WORDS
            if not valid:
                dropped += 1
                sentences.append({"text": text, "evidence_ids": [], "citation_labels": [], "status": "dropped"})
                continue
            ids = list(dict.fromkeys(ids))
            sentences.append(
                {
                    "text": text,
                    "evidence_ids": [by_id[i].evidence_id for i in ids],
                    "citation_labels": [by_id[i].citation_label for i in ids],  # parallel to evidence_ids
                    "status": "ok",
                }
            )
        if any(s["status"] == "ok" for s in sentences):
            paragraphs.append({"sentences": sentences})
    return paragraphs[:MAX_PARAGRAPHS], dropped


def generate_draft(db: Session, settings: Settings, llm: LLMProvider, matrix: ComparisonMatrix) -> dict:
    """Draft content for `matrix` (not persisted here).

    Raises DraftUnavailableError (provider cannot do JSON output) or
    DraftError (nothing grounded to write / nothing survived validation)."""
    if getattr(llm, "name", "") != "gemini":
        raise DraftUnavailableError(REQUIRES_GEMINI_MESSAGE)

    rows = build_evidence_table(db, matrix)
    if len({r.document_id for r in rows}) < MIN_PAPERS:
        raise DraftError(NOT_ENOUGH_EVIDENCE_MESSAGE)

    overall, caveat = _comparability_caveat(matrix)
    system = WRITER_SYSTEM_PROMPT
    if overall:
        system += _NOT_COMPARABLE_RULE.format(overall=overall.replace("_", " "))

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = llm.generate_json(system, [LLMMessage(role="user", content=_user_message(rows))], _draft_schema())
        except NotImplementedError as exc:
            raise DraftUnavailableError(REQUIRES_GEMINI_MESSAGE) from exc
        except Exception as exc:
            logger.warning("draft generation failed (attempt %d/%d): %s", attempt, MAX_ATTEMPTS, short_error(exc))
            continue
        paragraphs, dropped = validate_draft(raw, rows)
        if paragraphs:
            return {"paragraphs": paragraphs, "caveat": caveat, "dropped_count": dropped}
        logger.warning("draft attempt %d/%d: no sentence survived validation", attempt, MAX_ATTEMPTS)
    raise DraftError(NOT_GROUNDED_MESSAGE)


def verify_draft(db: Session, llm: LLMProvider, content: dict) -> list[dict]:
    """Run P4's verifier on every kept sentence against its cited evidence.
    Nothing is persisted. Returns [{paragraph, sentence, status, confidence,
    reason}]; sentences whose judging failed are omitted (stay unchecked).
    Raises VerifierUnavailableError for a provider that cannot judge."""
    from app.evidence.verification_service import _load_evidence, _passages
    from app.evidence.verifier import verify_pairs

    pairs, index = [], {}
    for p_i, paragraph in enumerate(content.get("paragraphs") or []):
        for s_i, sentence in enumerate(paragraph.get("sentences") or []):
            if sentence.get("status") != "ok" or not sentence.get("evidence_ids"):
                continue
            pair_id = f"{p_i}:{s_i}"
            evidence = _load_evidence(db, sentence["evidence_ids"])
            pairs.append({"id": pair_id, "claim_text": sentence["text"], "passages": _passages(db, evidence)})
            index[pair_id] = (p_i, s_i)
    results = verify_pairs(llm, pairs)
    return [
        {
            "paragraph": index[pid][0],
            "sentence": index[pid][1],
            "status": r["status"],
            "confidence": r["confidence"],
            "reason": r["reason"],
        }
        for pid, r in results.items()
    ]
