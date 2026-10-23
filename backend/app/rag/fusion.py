"""Result fusion: step 3 of the Phase 4 USER QUERY FLOW.

    Parallel retrieval (vector + keyword) -> [result fusion] -> candidate pool

Fusion strategy: Reciprocal Rank Fusion (RRF), not score normalisation.

Why RRF over e.g. min-max normalising and summing the two raw scores:

* Vector similarity (cosine, roughly [-1, 1] in practice usually [0, 1] for
  this app's embeddings) and BM25 score (unbounded >= 0, depends on corpus
  size and term rarity) live on completely different, incommensurable
  scales. Averaging or weighting raw scores from two different scoring
  functions requires calibrating that relationship (e.g. via held-out
  relevance judgments) — this app has no labelled relevance data to fit that
  calibration against, and getting it wrong silently lets whichever score
  happens to have larger magnitude dominate.
* RRF sidesteps the calibration problem entirely by fusing on *rank*, not
  score: `score(d) = sum over lists containing d of 1 / (k + rank(d))`. A
  chunk ranked #1 by BM25 and a chunk ranked #1 by vector search contribute
  the same amount to their respective fused scores regardless of what the
  underlying similarity/BM25 numbers were. This is the standard choice for
  combining lexical + semantic search in production hybrid-search systems
  (Elasticsearch, Azure AI Search, Weaviate all ship RRF as their default
  hybrid fusion) precisely because it needs no tuning to work reasonably
  well out of the box.
* The constant `k` (default 60, from the original Cormack et al. 2009 RRF
  paper, and the same default Elasticsearch/OpenSearch ship) dampens the
  impact of very high ranks so the fused order isn't overly dominated by
  whichever single list a chunk happened to top.

A chunk that only one of the two searches found still enters the pool (with
only that list's contribution) — that is the point of hybrid retrieval:
letting a chunk that vector search missed but is a strong keyword match (or
vice versa) surface into the candidate pool for the reranker to judge on its
actual content rather than being cut before it gets a chance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_RRF_K = 60


@dataclass
class FusedCandidate:
    chunk_id: str
    fusion_score: float
    vector_rank: int | None = None  # 1-based; None = not found by this path
    vector_score: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    matched_keyword_terms: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    vector_ranked: list[tuple[str, float]],
    keyword_ranked: list[tuple[str, float]],
    keyword_terms: dict[str, list[str]] | None = None,
    k: int = DEFAULT_RRF_K,
) -> list[FusedCandidate]:
    """Fuse two ranked lists of (chunk_id, raw_score) into one ranked pool.

    `keyword_terms` optionally maps chunk_id -> the query terms that chunk
    matched, purely so the debug panel can show *why* BM25 ranked it there.
    """
    keyword_terms = keyword_terms or {}
    candidates: dict[str, FusedCandidate] = {}

    for rank, (chunk_id, score) in enumerate(vector_ranked, start=1):
        c = candidates.setdefault(chunk_id, FusedCandidate(chunk_id=chunk_id, fusion_score=0.0))
        c.vector_rank = rank
        c.vector_score = score
        c.fusion_score += 1.0 / (k + rank)

    for rank, (chunk_id, score) in enumerate(keyword_ranked, start=1):
        c = candidates.setdefault(chunk_id, FusedCandidate(chunk_id=chunk_id, fusion_score=0.0))
        c.keyword_rank = rank
        c.keyword_score = score
        c.matched_keyword_terms = keyword_terms.get(chunk_id, [])
        c.fusion_score += 1.0 / (k + rank)

    fused = sorted(candidates.values(), key=lambda c: c.fusion_score, reverse=True)
    return fused
