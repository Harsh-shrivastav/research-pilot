"""Deterministic, offline "LLM" provider.

No network access and no API key, so `LLM_PROVIDER=local` (the default)
makes chat fully functional without any hosted account. It does not
paraphrase or reason the way a real LLM does — instead it is strictly
*extractive*: it can only ever return sentences that are already present in
the retrieved source blocks passed to it, which is by construction the
strongest possible guarantee against inventing information. Swap
`LLM_PROVIDER` to `anthropic` or `openai` for real generative answers.

It parses the exact prompt format `app/rag/context.py` builds (delimited
`[Source N] (Page P, Section: S)` blocks), so it is coupled to that module by
convention, not by a shared structured argument — kept this way so the
`LLMProvider.generate(system, messages)` contract stays identical to what a
real hosted API takes.
"""
from __future__ import annotations

import re
from collections.abc import AsyncIterator

from app.rag.interfaces import LLMMessage, LLMProvider
from app.rag.text_clean import split_sentences

_SOURCE_BLOCK = re.compile(
    r"\[Source (\d+)\]\s*\(Page (\d+)(?:, Section: ([^)]*))?\)\s*\n(.*?)(?=\n\[Source \d+\]|\n\nQuestion:|\Z)",
    re.DOTALL,
)
_QUESTION_LINE = re.compile(r"^Question:\s*(.+)$", re.MULTILINE)

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "this", "that", "these", "those",
    "it", "its", "as", "by", "with", "from", "what", "which", "who", "how",
    "does", "do", "did", "was", "paper", "please", "can", "you", "tell",
    "me", "about",
}

NOT_FOUND_MESSAGE = (
    "The paper does not provide sufficient information to answer this "
    "question based on the content retrieved from it."
)


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


class LocalExtractiveLLMProvider(LLMProvider):
    name = "local"
    model = "extractive-v1"

    def __init__(self, min_overlap: float = 0.08, max_sentences: int = 5):
        self.min_overlap = min_overlap
        self.max_sentences = max_sentences

    def _synthesize(self, prompt: str) -> str:
        question_match = _QUESTION_LINE.search(prompt)
        question = question_match.group(1).strip() if question_match else ""
        q_keywords = _keywords(question)

        sources = []
        for match in _SOURCE_BLOCK.finditer(prompt):
            idx, page, section, text = match.groups()
            sources.append(
                {"idx": int(idx), "page": page, "section": (section or "").strip(), "text": text.strip()}
            )

        if not sources or not q_keywords:
            return NOT_FOUND_MESSAGE

        # Score every sentence in every source by keyword overlap with the
        # question, then take the best few — this is the entire "model".
        scored: list[tuple[float, dict, str]] = []
        for source in sources:
            for sentence in split_sentences(source["text"]):
                s_keywords = _keywords(sentence)
                if not s_keywords:
                    continue
                overlap = len(q_keywords & s_keywords) / max(1, len(q_keywords))
                if overlap > 0:
                    scored.append((overlap, source, sentence))

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[: self.max_sentences]

        if not top or top[0][0] < self.min_overlap:
            return NOT_FOUND_MESSAGE

        # Preserve the sources' natural order (by [Source N]) rather than
        # pure score order, so the answer reads coherently.
        top.sort(key=lambda t: t[1]["idx"])

        seen_idx: list[int] = []
        lines: list[str] = []
        for _score, source, sentence in top:
            marker = f"[{source['idx']}]"
            lines.append(f"{sentence.rstrip('.')} {marker}.")
            if source["idx"] not in seen_idx:
                seen_idx.append(source["idx"])

        body = " ".join(lines)
        return (
            f"Based on the paper, {body}\n\n"
            "(This is an extractive summary assembled from the retrieved passages "
            "above — configure a hosted LLM_PROVIDER for a fully generative answer.)"
        )

    def generate(self, system: str, messages: list[LLMMessage]) -> str:
        prompt = messages[-1].content if messages else ""
        return self._synthesize(prompt)

    async def stream(self, system: str, messages: list[LLMMessage]) -> AsyncIterator[str]:
        text = self.generate(system, messages)
        # Yield word-by-word so the frontend's streaming UI has something
        # real to animate even with the offline provider.
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
