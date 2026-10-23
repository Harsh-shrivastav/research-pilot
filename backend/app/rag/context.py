"""Context construction: step 7 of the USER QUERY FLOW.

Turns the retriever's ranked `RetrievedChunk` list into the exact prompt text
handed to the LLM, and defines the system prompt that keeps the model
grounded in the paper. Kept as plain string templates (not a prompt-template
library) since there is exactly one prompt shape in this app.

The `[Source N] (Page P, Section: S)` block format is a *contract* with
`app/rag/llm/local_provider.py`, which parses it back out — real hosted
providers don't need to parse it, they just read it as instructed context.
"""
from __future__ import annotations

from app.rag.interfaces import LLMMessage, RetrievedChunk

SYSTEM_PROMPT = """You are the AI Research Assistant inside ResearchPilot, embedded next to a \
PDF viewer showing one uploaded research paper. Answer the researcher's questions using ONLY \
the numbered source passages provided with each question — those passages were retrieved from \
the paper itself.

Rules you must follow:
1. Use ONLY the numbered sources. Never use outside knowledge — not what you know about this \
paper, its authors, the field, or related work — even if you are confident it is correct.
2. Every sentence that states a fact must end with the bracketed marker of the source(s) it comes \
from, like [1] or [2][3]. Only cite source numbers that appear above. A sentence you cannot cite \
does not belong in the answer.
3. Never invent, assume, or fill in information that is not present in the sources — not numbers, \
names, citation details, or methodology steps. If the sources only partly answer the question, \
answer the part they support and say in one sentence what is not covered.
4. If the sources do not contain enough information to answer the question at all, reply with \
exactly this sentence and nothing else: "The paper does not provide sufficient information to \
answer this question based on the content retrieved from it."
5. Answer in 3-6 sentences unless the researcher explicitly asks for more detail, a list, or a \
different format. No preamble, no restating the question, no closing summary.
6. Do not refer to "the sources" or "the context" as such in your answer; refer to "the paper" \
naturally, the way a colleague who had just read it would.
"""

NOT_FOUND_SENTENCE = (
    "The paper does not provide sufficient information to answer this question "
    "based on the content retrieved from it."
)

MAX_CONTEXT_CHARS_DEFAULT = 14000


def build_source_blocks(chunks: list[RetrievedChunk], max_chars: int = MAX_CONTEXT_CHARS_DEFAULT) -> tuple[str, list[RetrievedChunk]]:
    """Render retrieved chunks as numbered `[Source N]` blocks, trimmed to a
    character budget so one very large parent chunk cannot blow out the
    prompt. Returns the rendered text and the (possibly trimmed) chunk list,
    kept in the same order/index so callers can map `[N]` markers back to a
    specific `RetrievedChunk` for citations.
    """
    blocks: list[str] = []
    used: list[RetrievedChunk] = []
    total = 0

    for i, chunk in enumerate(chunks, start=1):
        header = f"[Source {i}] (Page {chunk.page_number}" + (
            f", Section: {chunk.section_title})" if chunk.section_title else ")"
        )
        text = chunk.context_text.strip()
        block = f"{header}\n{text}"
        if total + len(block) > max_chars and used:
            break
        blocks.append(block)
        used.append(chunk)
        total += len(block)

    return "\n\n".join(blocks), used


def build_user_message(question: str, chunks: list[RetrievedChunk], max_chars: int = MAX_CONTEXT_CHARS_DEFAULT) -> tuple[str, list[RetrievedChunk]]:
    """Build the final user-turn content: the numbered sources followed by
    the question, in the exact shape the system prompt promises."""
    sources_text, used = build_source_blocks(chunks, max_chars=max_chars)
    if not sources_text:
        sources_text = "(No relevant passages were retrieved from the paper for this question.)"

    content = (
        f"Sources retrieved from the paper:\n\n{sources_text}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the sources above, with [N] citation markers."
    )
    return content, used


def build_history_messages(
    history: list[tuple[str, str]], max_turns: int
) -> list[LLMMessage]:
    """Convert prior (role, content) turns into LLMMessages, most recent
    `max_turns` only — enough for natural follow-ups ("what about the
    baseline they compared against?") without growing the prompt unbounded."""
    trimmed = history[-max_turns:] if max_turns > 0 else []
    return [LLMMessage(role=role, content=content) for role, content in trimmed]
