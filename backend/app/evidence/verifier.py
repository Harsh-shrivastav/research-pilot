"""Citation verifier — does the cited text actually back the claim?

Two layers, in this order:

1. Code (no LLM). A pair whose cited quote failed `check_quote` is
   `unsupported` outright (the LLM is never called for it). Every number in the
   claim must appear in the passages, otherwise the status is capped at
   `partially_supported`. A code downgrade is final: the LLM verdict can lower
   a status further but never raise it past the cap.
2. Gemini as judge (`LLMProvider.generate_json`), batched, for the rest.

This module is pure (no database). `verification_service.py` loads the pairs
and persists the results. A failed batch yields no result for its ids, so the
caller leaves them as they were (`uncertain`).

The Gemini provider already takes a `gemini_slot()` around every request it
makes, so the call here is not wrapped in a second one: the slot is a
non-reentrant bounded semaphore and nesting it could deadlock under load.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.models import SupportStatus
from app.rag.interfaces import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

BATCH_SIZE = 8
MAX_CLAIM_CHARS = 2000
MAX_PASSAGE_CHARS = 3000
MAX_REASON_WORDS = 25

REQUIRES_GEMINI_MESSAGE = "Citation verification requires the Gemini provider (LLM_PROVIDER=gemini)."

_CITATION_MARKER_RE = re.compile(r"\[\d+(?:\s*[,–-]\s*\d+)*\]")
_NUMBER_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?|(?<![\w])\.\d+")

VERIFIER_SYSTEM_PROMPT = """You are a strict citation checker for research papers.

For each item you get a CLAIM and the PASSAGES cited as its evidence. Judge ONLY whether the passages entail the claim. Use no outside knowledge and do not give credit for things that are merely plausible or true in general.

Rules:
- Every number, unit and dataset name in the claim must appear in the passages, otherwise the status cannot be "supported".
- "supported": the passages state everything the claim says.
- "partially_supported": only part of the claim is backed by the passages.
- "unsupported": the passages do not back the claim, or contradict it.
- "uncertain": the claim is too vague to judge, or the passages are too fragmentary.

For each item return its id, the status, a confidence between 0 and 1, and a reason of at most 25 words."""

_STATUS_RANK = {
    SupportStatus.UNSUPPORTED: 0,
    SupportStatus.UNCERTAIN: 1,
    SupportStatus.PARTIALLY_SUPPORTED: 2,
    SupportStatus.SUPPORTED: 3,
}


class VerifierUnavailableError(RuntimeError):
    """The configured provider cannot act as the judge (only Gemini can)."""


def _verifier_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "status": {"type": "string", "enum": list(SupportStatus.ALL)},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "status", "confidence", "reason"],
                },
            }
        },
        "required": ["results"],
    }


def _canonical_number(raw: str) -> str | None:
    try:
        return f"{float(raw.replace(',', '')):g}"
    except ValueError:
        return None


def extract_numbers(text: str) -> set[str]:
    """Numbers in `text`, canonicalised so "1,000", "1000.0" and "1000" agree.
    Citation markers like [3] are not numbers."""
    text = _CITATION_MARKER_RE.sub(" ", text or "")
    found = set()
    for match in _NUMBER_RE.findall(text):
        canonical = _canonical_number(match.rstrip(".,"))
        if canonical is not None:
            found.add(canonical)
    return found


def missing_numbers(claim_text: str, passages: list[str]) -> list[str]:
    """Numbers the claim states that no passage contains (sorted, as written
    canonically)."""
    wanted = extract_numbers(claim_text)
    if not wanted:
        return []
    available: set[str] = set()
    for passage in passages:
        available |= extract_numbers(passage)
    return sorted(wanted - available, key=lambda n: (len(n), n))


def _trim_reason(reason: str) -> str:
    words = (reason or "").split()
    return " ".join(words[:MAX_REASON_WORDS])


def _result(status: str, confidence: float, reason: str, verified_by: str) -> dict:
    return {
        "status": status,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": _trim_reason(reason),
        "verified_by": verified_by,
    }


def _prompt_for(batch: list[dict]) -> str:
    parts = []
    for pair in batch:
        passages = "\n".join(
            f"  [{i}] {p[:MAX_PASSAGE_CHARS]}" for i, p in enumerate(pair["passages"], start=1)
        )
        parts.append(
            f"ITEM id={pair['id']}\nCLAIM: {_CITATION_MARKER_RE.sub('', pair['claim_text'])[:MAX_CLAIM_CHARS].strip()}\n"
            f"PASSAGES:\n{passages}"
        )
    return "\n\n".join(parts)


def _judge_batch(llm: LLMProvider, batch: list[dict]) -> dict[str, dict]:
    data = llm.generate_json(
        VERIFIER_SYSTEM_PROMPT,
        [LLMMessage(role="user", content=_prompt_for(batch))],
        _verifier_schema(),
    )
    judged: dict[str, dict] = {}
    for item in (data or {}).get("results") or []:
        if not isinstance(item, dict):
            continue
        pair_id, status = str(item.get("id", "")), item.get("status")
        if status not in SupportStatus.ALL:
            continue
        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.5
        judged[pair_id] = _result(status, confidence, str(item.get("reason") or ""), "gemini")
    return judged


def verify_pairs(llm: LLMProvider, pairs: list[dict[str, Any]]) -> dict[str, dict]:
    """Verify claim/passage pairs.

    pair: {id, claim_text, passages: [str], quote_ok: bool | None (optional)}.
    `quote_ok=False` means the cited quote was not found in the paper.

    Returns {id: {status, confidence, reason, verified_by: "code"|"gemini"}}.
    Ids whose judging failed are absent (the caller keeps them `uncertain`).
    Raises VerifierUnavailableError when an LLM verdict is needed but the
    provider cannot produce JSON.
    """
    results: dict[str, dict] = {}
    to_judge: list[dict] = []
    caps: dict[str, str] = {}  # id -> reason for a partially_supported ceiling

    for pair in pairs:
        pair_id = str(pair["id"])
        passages = [p for p in pair.get("passages") or [] if (p or "").strip()]
        if pair.get("quote_ok") is False:
            results[pair_id] = _result(
                SupportStatus.UNSUPPORTED, 1.0, "The cited quote was not found in the paper's text.", "code"
            )
            continue
        if not passages:
            results[pair_id] = _result(SupportStatus.UNCERTAIN, 0.0, "No cited passage to check against.", "code")
            continue
        missing = missing_numbers(pair["claim_text"], passages)
        if missing:
            caps[pair_id] = "Not in the cited text: " + ", ".join(missing[:5]) + "."
        to_judge.append({"id": pair_id, "claim_text": pair["claim_text"], "passages": passages})

    if to_judge and getattr(llm, "name", "") != "gemini":
        raise VerifierUnavailableError(REQUIRES_GEMINI_MESSAGE)

    for start in range(0, len(to_judge), BATCH_SIZE):
        batch = to_judge[start : start + BATCH_SIZE]
        try:
            judged = _judge_batch(llm, batch)
        except NotImplementedError as exc:
            raise VerifierUnavailableError(REQUIRES_GEMINI_MESSAGE) from exc
        except Exception as exc:
            logger.warning("citation verification batch failed (%d pair(s)): %s: %s", len(batch), type(exc).__name__, exc)
            judged = {}
        for pair in batch:
            pair_id = pair["id"]
            verdict = judged.get(pair_id)
            if pair_id in caps:
                if verdict is None:
                    # LLM unavailable for this item, but the code check still stands.
                    results[pair_id] = _result(SupportStatus.PARTIALLY_SUPPORTED, 0.3, caps[pair_id], "code")
                    continue
                if _STATUS_RANK[verdict["status"]] > _STATUS_RANK[SupportStatus.PARTIALLY_SUPPORTED]:
                    verdict = _result(
                        SupportStatus.PARTIALLY_SUPPORTED,
                        min(verdict["confidence"], 0.6),
                        f"{caps[pair_id]} {verdict['reason']}",
                        "code",
                    )
                else:
                    verdict["reason"] = _trim_reason(f"{caps[pair_id]} {verdict['reason']}")
            if verdict is not None:
                results[pair_id] = verdict
    return results
