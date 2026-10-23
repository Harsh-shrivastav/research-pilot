"""Workspace query rewriting: one focused retrieval sub-query per paper.

A workspace question like "compare their methods" retrieves poorly when it is
sent verbatim to every paper: there is no paper-specific term to match. The
LLM rewrites it into one self-contained search query per paper, e.g.
"What method does Dense Passage Retrieval propose for training the retriever?".

This only changes what is *searched for*. The original question is still
what the answer prompt asks, and every answer is still grounded in the
retrieved passages, so a bad rewrite can only cost recall, never invent
evidence.

Degrades, never raises: on any LLM error, non-JSON reply, or missing/invalid
entry, that paper (or every paper) falls back to the original question.
"""
from __future__ import annotations

import logging

from app.rag.interfaces import LLMMessage, LLMProvider
from app.rag.json_utils import parse_json_object

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 300

REWRITE_SYSTEM_PROMPT = """\
You rewrite a researcher's question into search queries, one per paper, for a \
retrieval system that searches each paper separately.

Rules:
1. Output ONLY a JSON object, no prose and no code fences, of the form:
   {"queries": [{"document_id": "<id>", "query": "<search query>"}]}
2. Include exactly one entry for every paper listed, using its document_id verbatim.
3. Each query must be a single, self-contained question about THAT paper alone, \
using the paper's title to resolve references like "they", "their method", "both".
4. Keep the researcher's intent. Do not add facts, assumptions or answers. Under 30 words each.
"""


def _build_prompt(question: str, papers: list[dict]) -> str:
    listing = "\n".join(f'- document_id: {p["document_id"]} | title: "{p.get("title") or "Untitled"}"' for p in papers)
    return f"Papers:\n{listing}\n\nResearcher's question: {question}"


def _parse(raw: str, papers: list[dict]) -> dict[str, str]:
    """Extract {document_id: query}; raises ValueError if nothing usable."""
    data = parse_json_object(raw)
    entries = data.get("queries")
    if not isinstance(entries, list):
        raise ValueError("rewrite output has no 'queries' list")

    known = {p["document_id"] for p in papers}
    out: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        doc_id, query = entry.get("document_id"), entry.get("query")
        if doc_id in known and isinstance(query, str) and query.strip():
            out[doc_id] = " ".join(query.split())[:MAX_QUERY_CHARS]
    if not out:
        raise ValueError("rewrite output had no valid entries")
    return out


def rewrite_for_papers(llm: LLMProvider, question: str, papers: list[dict]) -> dict[str, str]:
    """Return {document_id: sub_query} for every paper in `papers`
    ([{document_id, title}]). Papers the model skipped, or every paper if the
    call or parse fails, get the original question."""
    fallback = {p["document_id"]: question for p in papers}
    if not papers or not question.strip():
        return fallback
    try:
        raw = llm.generate(REWRITE_SYSTEM_PROMPT, [LLMMessage(role="user", content=_build_prompt(question, papers))])
        rewritten = _parse(raw, papers)
    except Exception as exc:
        logger.warning("query rewrite failed (%s: %s); using the original question", type(exc).__name__, str(exc)[:200])
        return fallback
    missing = [p["document_id"] for p in papers if p["document_id"] not in rewritten]
    if missing:
        logger.warning("query rewrite skipped %d paper(s); using the original question for them", len(missing))
    return {**fallback, **rewritten}


def rewrite_enabled(setting: bool | None, llm: LLMProvider) -> bool:
    """`query_rewrite_enabled` None means auto: on only for Gemini."""
    return llm.name == "gemini" if setting is None else bool(setting)
