"""Phase 6B — Comparison cell context construction.

One comparison cell answers exactly one targeted question ("What dataset does
this paper use?") for exactly one paper, from that paper's own targeted
retrieval — so this reuses the *single-document* source-block renderer in
``app/rag/context.py`` unchanged (``build_source_blocks``), rather than the
multi-paper one in ``app/rag/workspace_context.py`` (which annotates every
block with a paper label — unnecessary here since a cell is always scoped to
one paper already).

This module only adds what a comparison cell needs on top of that: a system
prompt tuned for short, table-cell-sized answers, and the user-turn wrapper
that asks the specific dimension question. Citation parsing is NOT
duplicated here — ``app/comparison/comparison_service.py`` reuses
``app.rag.service._parse_citations`` directly, exactly like
``app/rag/workspace_chat.py`` reuses ``_snippet`` and ``build_retriever``
from the same module.
"""
from __future__ import annotations

from app.rag.context import NOT_FOUND_SENTENCE, build_source_blocks
from app.rag.interfaces import RetrievedChunk

COMPARISON_SYSTEM_PROMPT = """\
You are the AI Research Assistant inside ResearchPilot, filling in ONE cell of a \
multi-paper comparison table. You are answering a single, specific question about a \
single paper, using ONLY the numbered source passages provided — those passages were \
retrieved from that paper.

Rules you must follow:
1. Answer ONLY the question asked, about ONLY this paper. Do not compare it to other \
papers — you are not shown any other paper's content, and the comparison itself \
happens elsewhere once every cell is filled in.
2. Ground your answer in the numbered sources. Cite the sources you used with bracketed \
markers like [1] or [2][3] placed right after the claim they support.
3. Never invent, assume, or fill in information that is not present in the provided \
sources.
4. Be extremely concise — this is a table cell, not a chat answer. One to three short \
sentences at most. No preamble ("Based on the paper..."), no restating the question, \
just the fact(s).
5. If the sources do not contain enough information to answer the question at all, \
reply with exactly this sentence and nothing else: "The paper does not provide \
sufficient information to answer this question based on the content retrieved from it."
"""

# Re-exported so callers (comparison_service.py) can detect the
# insufficient-evidence case with the exact same check
# `app.rag.service._parse_citations` already uses for chat answers — one
# sentence, one place it's defined (app/rag/context.py), reused everywhere.
CELL_NOT_FOUND_SENTENCE = NOT_FOUND_SENTENCE


def build_comparison_cell_message(
    dimension_query: str,
    chunks: list[RetrievedChunk],
    max_chars: int,
) -> tuple[str, list[RetrievedChunk]]:
    """Build the user-turn content for one comparison cell: this paper's
    targeted-retrieval sources, followed by the dimension's question."""
    sources_text, used = build_source_blocks(chunks, max_chars=max_chars)
    if not sources_text:
        sources_text = "(No relevant passages were retrieved from this paper for this question.)"

    content = (
        f"Sources retrieved from this paper:\n\n{sources_text}\n\n"
        f"Question: {dimension_query}\n\n"
        "Answer using only the sources above, with [N] citation markers. "
        "Keep the answer to 1-3 concise sentences suitable for a comparison table cell."
    )
    return content, used
