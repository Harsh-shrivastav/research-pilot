"""Phase 6A — Multi-paper context construction.

Extends the single-document context builder (``app/rag/context.py``) to
produce LLM context that clearly identifies which paper each source came
from.  This is critical for grounded, multi-paper answers — without paper
identity in the prompt, the model may silently mix evidence from different
papers.

The single-document context module is NOT modified; this module imports and
delegates to it for helper functions, and adds workspace-specific formatting.

Context format per workspace source block:

    [Paper 1: "Title of Paper A"]
    [Source 1] (Page 4, Section: Methodology)
    <text>

    [Paper 2: "Title of Paper B"]
    [Source 2] (Page 6, Section: Dataset)
    <text>
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Document
from app.rag.interfaces import LLMMessage, RetrievedChunk

WORKSPACE_SYSTEM_PROMPT = """\
You are the AI Research Assistant inside ResearchPilot, embedded in a multi-paper \
research workspace. The workspace contains several uploaded research papers. Answer \
the researcher's questions using ONLY the numbered source passages provided with each \
question — those passages were retrieved from the papers in this workspace.

Rules you must follow:
1. Ground every factual claim in the numbered sources. Cite the sources you used with \
bracketed markers like [1] or [2][3] placed right after the claim they support.
2. Always identify WHICH PAPER a piece of evidence comes from. Use the paper title or \
a short label (e.g. "Paper A", "Paper B") consistently throughout your answer.
3. Never invent, assume, or fill in information that is not present in the provided \
sources — not citation details, not numbers, not methodology steps, not authors, \
not DOIs. If the sources only partly answer the question, answer the part they support \
and say plainly what is not covered.
4. If the sources do not contain enough information to answer the question at all, reply \
with exactly this sentence and nothing else: "The workspace papers do not provide \
sufficient information to answer this question based on the content retrieved."
5. Never use outside knowledge about these papers, their authors or the field — only the \
numbered sources. Answer in 3-6 sentences (or a short per-paper list for comparisons) unless \
the researcher explicitly asks for more. No preamble and no closing summary.
6. If two papers disagree or present different findings on the same topic, present BOTH \
perspectives and clearly identify which paper each comes from. Do not silently pick one.
7. Do not refer to "the sources" or "the context" as such in your answer; refer to the \
papers by their titles or labels naturally.
"""

WORKSPACE_NOT_FOUND_SENTENCE = (
    "The workspace papers do not provide sufficient information to answer "
    "this question based on the content retrieved."
)

MAX_CONTEXT_CHARS_DEFAULT = 16000  # slightly larger budget for multi-paper


def build_workspace_source_blocks(
    chunks: list[RetrievedChunk],
    doc_titles: dict[str, str],
    max_chars: int = MAX_CONTEXT_CHARS_DEFAULT,
) -> tuple[str, list[RetrievedChunk]]:
    """Render retrieved chunks as numbered [Source N] blocks, each annotated
    with its paper title, trimmed to a character budget.

    Returns the rendered text and the (possibly trimmed) chunk list, kept in
    the same order/index so callers can map [N] markers back to a specific
    RetrievedChunk for citations.
    """
    blocks: list[str] = []
    used: list[RetrievedChunk] = []
    total = 0

    for i, chunk in enumerate(chunks, start=1):
        paper_title = doc_titles.get(chunk.document_id, "Unknown Paper")
        paper_label = f'[Paper: "{paper_title}"]'

        header = f"[Source {i}] (Page {chunk.page_number}" + (
            f", Section: {chunk.section_title})" if chunk.section_title else ")"
        )
        text = chunk.context_text.strip()
        block = f"{paper_label}\n{header}\n{text}"
        if total + len(block) > max_chars and used:
            break
        blocks.append(block)
        used.append(chunk)
        total += len(block)

    return "\n\n".join(blocks), used


def build_workspace_user_message(
    question: str,
    chunks: list[RetrievedChunk],
    doc_titles: dict[str, str],
    max_chars: int = MAX_CONTEXT_CHARS_DEFAULT,
) -> tuple[str, list[RetrievedChunk]]:
    """Build the final user-turn content for workspace chat: numbered sources
    with paper identity, followed by the question."""
    sources_text, used = build_workspace_source_blocks(
        chunks, doc_titles, max_chars=max_chars
    )
    if not sources_text:
        sources_text = (
            "(No relevant passages were retrieved from the workspace papers "
            "for this question.)"
        )

    content = (
        f"Sources retrieved from the workspace papers:\n\n{sources_text}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the sources above, with [N] citation markers. "
        "Identify which paper each piece of evidence comes from."
    )
    return content, used


def get_document_titles(db: Session, document_ids: list[str]) -> dict[str, str]:
    """Look up document titles for a set of document IDs.
    Falls back to original_filename if title is not available."""
    from sqlalchemy import select
    from app.models import Document

    titles: dict[str, str] = {}
    if not document_ids:
        return titles

    documents = db.scalars(
        select(Document).where(Document.id.in_(document_ids))
    ).all()
    for doc in documents:
        titles[doc.id] = doc.title or doc.original_filename or "Unknown Paper"
    return titles
