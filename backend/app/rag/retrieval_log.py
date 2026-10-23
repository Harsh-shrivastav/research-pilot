"""Retrieval logging — item 6 of the Phase 4 spec, and deliberately a
separate concern from the Retrieval Debug panel (`interfaces.RetrievalDebug`,
gated behind `DEBUG_MODE`).

The distinction matters:

* The **debug panel** is user/admin-facing: it is only ever computed,
  stored, or served when an operator has explicitly turned `DEBUG_MODE` on,
  because it contains full candidate text snippets and is meant to be looked
  at interactively (see `hybrid_retriever.py`, `routers/chat.py`).
* **Retrieval logging** is an always-on ops trail: one structured INFO line
  per retrieval call, regardless of `DEBUG_MODE`, on its own logger name
  (`app.rag.retrieval`) so it can be filtered, shipped, or sampled
  independently of the rest of the application's logs. It intentionally logs
  *counts and scores*, not chunk text — safe to leave on in production
  without leaking document content into log aggregation.

This is what lets you answer "what did retrieval actually do for this
question, in this deployment, right now" from `docker compose logs` alone,
without needing DEBUG_MODE on and without re-running anything.
"""
from __future__ import annotations

import logging

from app.rag.interfaces import RetrievalDebug

retrieval_logger = logging.getLogger("app.rag.retrieval")


def log_retrieval(document_id: str, debug: RetrievalDebug | None, result_count: int) -> None:
    if debug is None:
        retrieval_logger.info(
            "document=%s mode=empty query -> 0 results (empty/whitespace-only question)",
            document_id,
        )
        return

    candidates = debug.candidates
    vector_hits = sum(1 for c in candidates if c.vector_score is not None)
    keyword_hits = sum(1 for c in candidates if c.keyword_score is not None)
    selected = [c for c in candidates if c.selected]

    def _display_score(c) -> str:
        score = c.rerank_score if c.rerank_score is not None else c.fusion_score
        return f"{c.chunk_id[:8]}:{score:.4f}" if score is not None else f"{c.chunk_id[:8]}:n/a"

    top_scores = ", ".join(_display_score(c) for c in selected[:5])
    timings = ", ".join(f"{k}={v:.1f}ms" for k, v in debug.timings_ms.items())

    # Logging should never be able to take down a retrieval call — guard the
    # actual emit so a future field-shape change degrades to a plain message
    # instead of raising out of the retrieval path.
    try:
        retrieval_logger.info(
            "document=%s mode=%s query=%r candidates=%d (vector=%d keyword=%d) "
            "selected=%d/%d config=%s top_scores=[%s] timings=[%s]",
            document_id,
            debug.mode,
            debug.query.get("normalized", ""),
            len(candidates),
            vector_hits,
            keyword_hits,
            len(selected),
            result_count,
            debug.config,
            top_scores,
            timings,
        )
    except Exception:
        retrieval_logger.exception("retrieval logging failed for document=%s", document_id)
