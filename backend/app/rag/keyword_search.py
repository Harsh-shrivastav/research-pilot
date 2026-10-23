"""Keyword search: the BM25 half of Phase 4's hybrid retrieval.

    Query
      |
      +-- Semantic vector search  (app/rag/retriever.py)
      +-- Keyword/BM25 search     (this module)
              |
        Result fusion (app/rag/fusion.py)

Why BM25, and why implemented by hand instead of a library:

* Vector search (see `retriever.py`) is good at *meaning* — it finds chunks
  that are semantically related to the question even when the wording
  differs. It is comparatively weak at exact-term precision: an uncommon
  identifier, a model name, a specific number, or jargon that is rare in the
  embedding model's training distribution can end up looking similar to
  lots of unrelated text after being squeezed into a fixed-size vector.
* BM25 is a term-frequency / inverse-document-frequency ranking function
  that is the opposite: excellent for "does this chunk contain the literal
  term(s) the user asked about", weak at paraphrase and synonymy. That
  complementary failure mode is exactly why hybrid retrieval fuses the two
  (see `fusion.py`) instead of picking one.
* This codebase's stated design principle (see `embeddings/local_provider.py`,
  `llm/local_provider.py`) is that the app must be fully functional with zero
  external services and zero extra heavyweight dependencies. `rank_bm25` is a
  tiny library, but scope here is the same as the existing brute-force vector
  path in `retriever.py`: chunks of *one* uploaded paper (low hundreds), not a
  corpus-wide search engine. At that scale a ~60-line pure-Python BM25 is not
  a performance compromise, and it keeps the RAG stack dependency-free end to
  end, consistent with the rest of `app/rag/`.

Scope/complexity note mirrors `retriever.py`'s bruteforce path: this reindexes
in memory on every call. Fine for "chunks of one paper"; would need a
persistent inverted index for a corpus-wide search engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ChunkLevel, DocumentChunk
from app.rag.query_processing import tokenize

# Standard Okapi BM25 constants. k1 controls term-frequency saturation (how
# much repeating a term within a chunk keeps helping — 1.2-2.0 is the usual
# range); b controls length normalisation (0 = ignore chunk length, 1 = fully
# normalise). These are the values from the original Robertson/Sparck Jones
# formulation and virtually every BM25 implementation (Lucene, Elasticsearch,
# rank_bm25) defaults to them, so they are a well-justified starting point
# rather than an arbitrary guess.
BM25_K1 = 1.5
BM25_B = 0.75


@dataclass
class KeywordHit:
    chunk: DocumentChunk
    score: float  # raw BM25 score, unbounded >= 0; 0 = no query term matched
    matched_terms: list[str]


class BM25Index:
    """An in-memory BM25 index over one document's child chunks."""

    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        self._doc_tokens: list[list[str]] = [tokenize(c.text) for c in chunks]
        self._doc_len = [len(toks) for toks in self._doc_tokens]
        self._avg_doc_len = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0

        # document frequency: how many chunks contain each term at least once
        df: dict[str, int] = {}
        self._term_freqs: list[dict[str, int]] = []
        for toks in self._doc_tokens:
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self._term_freqs.append(tf)
            for t in tf:
                df[t] = df.get(t, 0) + 1
        self._df = df
        self._n = len(chunks)

    def _idf(self, term: str) -> float:
        n = self._n
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        # BM25's IDF variant (with the +1 inside the log) — always
        # non-negative, unlike the classic Robertson-Sparck-Jones IDF, which
        # can go negative for terms that appear in most documents and would
        # then *penalise* a chunk for containing the query term.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query_terms: list[str], top_k: int) -> list[KeywordHit]:
        if not query_terms or self._n == 0:
            return []

        # Score each unique query term once, not once per occurrence in the
        # query — repeating a word in the question shouldn't double-count it.
        unique_terms = list(dict.fromkeys(query_terms))
        idfs = {t: self._idf(t) for t in unique_terms}

        scored: list[KeywordHit] = []
        for i, chunk in enumerate(self.chunks):
            tf = self._term_freqs[i]
            dl = self._doc_len[i]
            score = 0.0
            matched: list[str] = []
            for term in unique_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                matched.append(term)
                idf = idfs[term]
                denom = f + BM25_K1 * (1 - BM25_B + BM25_B * dl / (self._avg_doc_len or 1.0))
                score += idf * (f * (BM25_K1 + 1)) / (denom or 1.0)
            if score > 0:
                scored.append(KeywordHit(chunk=chunk, score=score, matched_terms=matched))

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]


def build_index(db: Session, document_id: str) -> BM25Index:
    chunks = db.scalars(
        select(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.chunk_level == ChunkLevel.CHILD,
        )
    ).all()
    return BM25Index(list(chunks))
