"""Citation verification — loads claim/evidence pairs, runs the verifier
(`app/evidence/verifier.py`) and persists the verdicts.

Only `Claim.support_status`/`confidence`, `Evidence.support_status` and
`Evidence.evidence_metadata["verification"]` are ever written. Answers,
citations and cells are never touched, and any failure leaves the affected
rows at their previous status (`uncertain` for never-verified ones).

Not exported from `app.evidence` (it imports the comparison package, which
itself imports `app.evidence`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.comparison.paper_extractor import PaperMaterial, check_quote, load_paper_material
from app.config import Settings, get_settings
from app.evidence.verifier import VerifierUnavailableError, verify_pairs
from app.models import (
    ChatConversation,
    ChatMessage,
    ChatRole,
    Claim,
    ComparisonCell,
    ComparisonMatrix,
    DocumentChunk,
    Evidence,
    SupportStatus,
)
from app.rag.factory import get_llm_provider
from app.rag.service import llm_model_used

logger = logging.getLogger(__name__)

__all__ = [
    "VerifierUnavailableError",
    "verify_comparison",
    "verify_conversation",
    "load_support_info",
]


@dataclass
class _Group:
    """One thing to verify: a comparison cell or a chat answer."""

    id: str
    claim_text: str
    evidence: list[Evidence]
    quote_ok: bool | None = None
    passages: list[str] = field(default_factory=list)


def _evidence_ids(citations: list | None) -> list[str]:
    return [c["evidence_id"] for c in citations or [] if isinstance(c, dict) and c.get("evidence_id")]


def _load_evidence(db: Session, ids: list[str]) -> list[Evidence]:
    if not ids:
        return []
    rows = {e.id: e for e in db.scalars(select(Evidence).where(Evidence.id.in_(ids))).all()}
    return [rows[i] for i in dict.fromkeys(ids) if i in rows]


def _passages(db: Session, evidence: list[Evidence]) -> list[str]:
    """The snippet shown to the user plus, when stored, the full chunk it came
    from (the snippet is capped at ~500 characters)."""
    passages: list[str] = []
    for item in evidence:
        passages.append(item.snippet)
        if item.chunk_id:
            chunk_text = db.scalar(select(DocumentChunk.text).where(DocumentChunk.id == item.chunk_id))
            if chunk_text and chunk_text.strip() != item.snippet.strip():
                passages.append(chunk_text)
    return passages


def _quote_ok(settings: Settings, materials: dict[str, PaperMaterial], db: Session, evidence: list[Evidence]) -> bool | None:
    """True if any cited quote is found on its cited page, False if none is,
    None if it could not be checked."""
    if not evidence:
        return None
    try:
        for item in evidence:
            material = materials.get(item.document_id)
            if material is None:
                material = materials[item.document_id] = load_paper_material(db, item.document_id)
            quote = (item.snippet or "").rstrip("…").strip()
            if check_quote(material, quote, item.page_number, tolerant=settings.comparison_tolerant_quotes).verified:
                return True
        return False
    except Exception:
        logger.exception("quote check failed during citation verification")
        return None


def _persist(db: Session, groups: list[_Group], results: dict[str, dict], model: str | None) -> None:
    claims: dict[str, Claim] = {}
    for group in groups:
        verdict = results.get(group.id)
        if verdict is None:
            continue
        info = {
            "reason": verdict["reason"],
            "verified_by": verdict["verified_by"],
            "model": model if verdict["verified_by"] == "gemini" else None,
        }
        for item in group.evidence:
            item.support_status = verdict["status"]
            metadata = dict(item.evidence_metadata or {})
            metadata["verification"] = info
            item.evidence_metadata = metadata
            if item.claim_id and item.claim_id not in claims:
                claim = db.get(Claim, item.claim_id)
                if claim is not None:
                    claims[claim.id] = claim
                    claim.support_status = verdict["status"]
                    claim.confidence = verdict["confidence"]
    db.commit()


def _run(db: Session, settings: Settings, groups: list[_Group], llm=None) -> dict[str, int]:
    """Verify `groups`, persist, and return the status counts over all of
    them (verified now or earlier)."""
    llm = llm or get_llm_provider(settings)
    pairs = [
        {"id": g.id, "claim_text": g.claim_text, "passages": g.passages, "quote_ok": g.quote_ok}
        for g in groups
        if g.evidence
    ]
    try:
        results = verify_pairs(llm, pairs)
        _persist(db, groups, results, llm_model_used(llm))
    except VerifierUnavailableError:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        logger.exception("citation verification failed; statuses left unchanged")
    return _summary(db, groups)


def _group_status(group: _Group) -> str:
    return group.evidence[0].support_status if group.evidence else SupportStatus.UNCERTAIN


def _summary(db: Session, groups: list[_Group]) -> dict[str, int]:
    counts = {"total": 0, **{status: 0 for status in SupportStatus.ALL}}
    for group in groups:
        if not group.evidence:
            continue
        for item in group.evidence:
            db.refresh(item)
        counts["total"] += 1
        status = _group_status(group)
        counts[status if status in SupportStatus.ALL else SupportStatus.UNCERTAIN] += 1
    return counts


def verify_comparison(db: Session, comparison_id: str, owner_id: str, settings: Settings | None = None, llm=None) -> dict[str, int]:
    """Verify every cell of a comparison that carries evidence. Raises
    `ComparisonNotFoundError` for a missing/unowned comparison and
    `VerifierUnavailableError` when the provider cannot judge."""
    from app.comparison.comparison_service import get_comparison

    settings = settings or get_settings()
    matrix: ComparisonMatrix = get_comparison(db, comparison_id, owner_id)
    materials: dict[str, PaperMaterial] = {}
    groups: list[_Group] = []
    for cell in matrix.cells:
        evidence = _load_evidence(db, _evidence_ids(cell.citations))
        if not evidence or not cell.value_text:
            continue
        group = _Group(id=cell.id, claim_text=cell.value_text, evidence=evidence)
        if cell.quote_verified is True:  # only structured cells carry a model-given quote
            group.quote_ok = _quote_ok(settings, materials, db, evidence)
        group.passages = _passages(db, evidence)
        groups.append(group)
    return _run(db, settings, groups, llm)


def verify_conversation(db: Session, conversation_id: str, owner_id: str, settings: Settings | None = None, llm=None) -> dict[str, int] | None:
    """Verify every cited assistant answer of a conversation. None when the
    conversation does not exist or is not owned by `owner_id`."""
    settings = settings or get_settings()
    conversation = db.scalar(
        select(ChatConversation).where(ChatConversation.id == conversation_id, ChatConversation.owner_id == owner_id)
    )
    if conversation is None:
        return None
    groups: list[_Group] = []
    for message in conversation.messages:
        if message.role != ChatRole.ASSISTANT or message.error or not (message.content or "").strip():
            continue
        evidence = _load_evidence(db, _evidence_ids(message.citations))
        if not evidence:
            continue
        groups.append(
            _Group(id=message.id, claim_text=message.content, evidence=evidence, passages=_passages(db, evidence))
        )
    return _run(db, settings, groups, llm)


def load_support_info(db: Session, evidence_ids: list[str]) -> dict[str, dict]:
    """{evidence_id: {status, reason, verified_by, checked}} for the read
    paths (cells, chat citations, evidence dialog)."""
    ids = [i for i in dict.fromkeys(evidence_ids) if i]
    info: dict[str, dict] = {}
    for item in _load_evidence(db, ids):
        verification = (item.evidence_metadata or {}).get("verification") or {}
        info[item.id] = {
            "status": item.support_status,
            "reason": verification.get("reason"),
            "verified_by": verification.get("verified_by"),
            "checked": bool(verification),
        }
    return info
