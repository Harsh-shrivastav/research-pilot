"""Research topic -> focused search queries (Gemini via generate_json).

The model only writes queries and related terms. It is told never to name
papers or authors, and nothing it returns is shown as a result: papers come
only from the search API (app/discovery/openalex.py).
"""
from __future__ import annotations

import logging

from app.rag.gemini_limits import short_error
from app.rag.interfaces import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

MIN_QUERIES = 4
MAX_QUERIES = 6
MAX_RELATED = 8
MAX_QUERY_CHARS = 120

REQUIRES_GEMINI_MESSAGE = "Query expansion requires Gemini (LLM_PROVIDER=gemini)."

SYSTEM_PROMPT = """You help a researcher search academic databases.

Given a research topic, write 4 to 6 focused search queries (each at most 8 words) that together cover it: the core topic, key methods, synonyms or alternative terminology, and neighbouring problems. Also list up to 8 related terms.

Rules:
- Write search queries only. Do NOT name or invent papers, authors, titles, years or DOIs.
- Queries must be plain keywords, no boolean operators or quotes.
- Each query must be meaningfully different from the others."""


class QueryExpansionUnavailableError(RuntimeError):
    """The configured provider cannot expand queries (only Gemini can)."""


class QueryExpansionError(RuntimeError):
    """The model did not return usable queries; the message is safe to show."""


class QueryExpansionFailedError(QueryExpansionError):
    """The model call itself failed (outage, quota)."""


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "queries": {"type": "array", "items": {"type": "string"}},
            "related_terms": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["queries", "related_terms"],
    }


def _clean_list(raw, limit: int, max_chars: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in raw or []:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split()).strip(" \"'")
        if not text or len(text) > max_chars or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(text)
        if len(out) == limit:
            break
    return out


def expand_topic(llm: LLMProvider, topic: str) -> dict:
    """{queries: [up to 6 strings], related_terms: [...]}."""
    topic = " ".join((topic or "").split())
    if not topic:
        raise QueryExpansionError("Enter a research topic.")
    if getattr(llm, "name", "") != "gemini":
        raise QueryExpansionUnavailableError(REQUIRES_GEMINI_MESSAGE)
    try:
        data = llm.generate_json(
            SYSTEM_PROMPT, [LLMMessage(role="user", content=f"Research topic: {topic}")], _schema()
        )
    except NotImplementedError as exc:
        raise QueryExpansionUnavailableError(REQUIRES_GEMINI_MESSAGE) from exc
    except Exception as exc:
        logger.warning("query expansion failed: %s", short_error(exc))
        raise QueryExpansionFailedError("Could not expand the topic right now. Please try again.") from exc
    queries = _clean_list((data or {}).get("queries"), MAX_QUERIES, MAX_QUERY_CHARS)
    if not queries:
        raise QueryExpansionError("The model returned no usable queries. Please try again.")
    return {
        "queries": queries,
        "related_terms": _clean_list((data or {}).get("related_terms"), MAX_RELATED, MAX_QUERY_CHARS),
    }
