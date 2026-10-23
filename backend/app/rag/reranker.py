"""Reranking: step 4 of the Phase 4 USER QUERY FLOW.

    Candidate pool -> [neural/cross-encoder reranking] -> Top-K evidence

`Reranker` is the provider-neutral interface (same seam as `EmbeddingProvider`
/ `LLMProvider` in `interfaces.py`): the hybrid retriever only ever talks to
this ABC, so which scorer actually runs is a config choice
(`RERANK_PROVIDER` in `.env`), never a code change.

`LocalCrossEncoderReranker` is the "local" provider for this seam, matching
the project-wide pattern set by `embeddings/local_provider.py` and
`llm/local_provider.py`: it needs no model weights, no GPU and no network
access, so hybrid retrieval + reranking works end to end with nothing
installed, and a real cross-encoder (e.g. a `sentence-transformers`
`CrossEncoder`, or a hosted reranking API like Cohere Rerank / Voyage
Rerank) can be dropped in later behind this same interface without touching
`hybrid_retriever.py`.

What it actually computes, given it is not a trained model: a query/chunk
*relevance* score built from lexical evidence a real cross-encoder would also
pick up on — exact term coverage, phrase adjacency, and term position —
rather than the bag-of-words IDF weighting BM25 already contributed at the
fusion stage. It is not a substitute for a trained cross-encoder's semantic
judgement, so treat it as a precision-oriented re-sort of the candidate pool
that vector + keyword search already assembled, not as a source of new
recall.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.rag.query_processing import tokenize


@dataclass
class RerankResult:
    chunk_id: str
    score: float  # higher = more relevant; roughly [0, 1] for the local scorer
    matched_terms: list[str]


class Reranker(ABC):
    """Re-scores a candidate pool against the query, independent of whatever
    upstream retrieval score(s) each candidate arrived with."""

    name: str = "base"

    @abstractmethod
    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int) -> list[RerankResult]:
        """`candidates` is a list of (chunk_id, chunk_text). Returns results
        sorted best-first, at most `top_k` of them."""


class LocalCrossEncoderReranker(Reranker):
    """Dependency-free lexical reranker — see module docstring for the
    rationale and what it does and does not substitute for."""

    name = "local"

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int) -> list[RerankResult]:
        query_tokens = tokenize(query)
        if not query_tokens or not candidates:
            return [RerankResult(chunk_id=cid, score=0.0, matched_terms=[]) for cid, _ in candidates[:top_k]]

        query_terms = list(dict.fromkeys(query_tokens))
        query_bigrams = set(zip(query_tokens, query_tokens[1:]))

        results: list[RerankResult] = []
        for chunk_id, text in candidates:
            chunk_tokens = tokenize(text)
            if not chunk_tokens:
                results.append(RerankResult(chunk_id=chunk_id, score=0.0, matched_terms=[]))
                continue

            chunk_token_set = set(chunk_tokens)
            matched = [t for t in query_terms if t in chunk_token_set]

            # 1. Term coverage: fraction of distinct query terms present at
            #    all in the chunk. This is what a cross-encoder attending
            #    over both sequences would most directly pick up on for a
            #    short, specific question.
            coverage = len(matched) / len(query_terms)

            # 2. Phrase adjacency: query bigrams that appear adjacent in the
            #    chunk in the same order score extra — rewards a chunk that
            #    contains the query's exact phrasing ("cross encoder
            #    reranking") over one that merely contains the same words
            #    scattered far apart.
            chunk_bigrams = set(zip(chunk_tokens, chunk_tokens[1:]))
            bigram_hits = len(query_bigrams & chunk_bigrams)
            bigram_bonus = bigram_hits / max(1, len(query_bigrams)) if query_bigrams else 0.0

            # 3. Density: matched terms as a fraction of the chunk's own
            #    length, so a short, tightly-on-topic chunk outranks a long
            #    chunk that happens to contain the same terms once each
            #    among a lot of unrelated text.
            term_hits_in_chunk = sum(1 for t in chunk_tokens if t in matched)
            density = term_hits_in_chunk / math.sqrt(len(chunk_tokens))

            # 4. Early-position bonus: research-paper answers to "what is
            #    the main contribution" style questions are disproportionately
            #    likely to be in a chunk's opening sentence (topic sentences),
            #    so give a small boost when a match occurs early.
            first_match_idx = next((i for i, t in enumerate(chunk_tokens) if t in matched), None)
            position_bonus = 0.0
            if first_match_idx is not None:
                position_bonus = 1.0 / (1.0 + first_match_idx / 20.0)

            score = (
                0.55 * coverage
                + 0.20 * bigram_bonus
                + 0.15 * min(1.0, density)
                + 0.10 * position_bonus
            )
            results.append(RerankResult(chunk_id=chunk_id, score=round(score, 6), matched_terms=matched))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
