"""HybridRetriever — the full Phase 4 USER QUERY FLOW:

    Query
      |
      v
    Query preprocessing            (app/rag/query_processing.py)
      |
      v
    Parallel retrieval
      +-- Semantic vector search   (app/rag/retriever.py)
      +-- Keyword/BM25 search      (app/rag/keyword_search.py)
      |
      v
    Result fusion (RRF)            (app/rag/fusion.py)
      |
      v
    Candidate pool
      |
      v
    Neural/cross-encoder reranking (app/rag/reranker.py)
      |
      v
    Top-K evidence
      |
      v
    Context construction           (app/rag/context.py, unchanged)

`RAGChatService` (Phase 3, unchanged in shape) talks to this exactly like it
talked to `VectorRetriever` — same `Retriever` interface, same
`list[RetrievedChunk]` return — so upgrading retrieval is a `RETRIEVAL_MODE`
config change, not a change to the chat flow, context construction, citation
parsing, or the frontend chat panel. The only new surface is `last_debug`,
which the router reads off *after* calling `retrieve()` to build the
Retrieval Debug panel payload (see `app/routers/chat.py`) — normal request
handling never has to look at it.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.orm import Session, sessionmaker

from app.models import DocumentChunk
from app.rag.fusion import DEFAULT_RRF_K, FusedCandidate, reciprocal_rank_fusion
from app.rag.interfaces import (
    DebugCandidate,
    EmbeddingProvider,
    RetrievalDebug,
    RetrievedChunk,
    Retriever,
)
from app.rag.keyword_search import build_index
from app.rag.query_processing import preprocess_query
from app.rag.reranker import Reranker
from app.rag.retrieval_log import log_retrieval
from app.rag.retriever import VectorRetriever, expand_to_parents

logger = logging.getLogger(__name__)


def _snippet(text: str, max_chars: int = 220) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "\u2026"


class HybridRetriever(Retriever):
    def __init__(
        self,
        db: Session,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker,
        candidate_pool_size: int = 20,
        rrf_k: int = DEFAULT_RRF_K,
        enable_reranking: bool = True,
    ):
        self.db = db
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.candidate_pool_size = max(candidate_pool_size, 1)
        self.rrf_k = rrf_k
        self.enable_reranking = enable_reranking
        self.last_debug: RetrievalDebug | None = None

    def retrieve(self, document_id: str, query: str, top_k: int) -> list[RetrievedChunk]:
        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        # -- 1. query preprocessing ------------------------------------------
        preprocessed = preprocess_query(query)
        if not preprocessed.normalized:
            self.last_debug = None
            log_retrieval(document_id, None, 0)
            return []

        pool_size = max(self.candidate_pool_size, top_k)

        # -- 2. parallel retrieval --------------------------------------------
        # Vector search (DB round-trip + brute-force cosine on SQLite) and
        # BM25 (in-memory scan) are both blocking, independent, single-input
        # operations, so a 2-worker thread pool runs them concurrently rather
        # than one after another — a real wall-clock win on SQLite's
        # brute-force path where vector search alone can take tens of ms.
        #
        # Each branch gets its own short-lived Session bound to the same
        # engine as `self.db`: a single SQLAlchemy Session is not safe to use
        # from two threads at once, so sharing `self.db` across the two
        # futures would be a real (if easy-to-miss) concurrency bug, not just
        # a style nit.
        session_factory = sessionmaker(bind=self.db.get_bind())
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            vector_future = pool.submit(
                self._vector_search, session_factory, document_id, preprocessed.normalized, pool_size
            )
            keyword_future = pool.submit(
                self._keyword_search, session_factory, document_id, preprocessed.keywords, pool_size
            )
            keyword_hits = keyword_future.result()
            try:
                vector_hits = vector_future.result()
            except Exception as exc:
                # Degrade, don't crash: if the embedding service is down or
                # out of quota, answer from keyword (BM25) search alone
                # rather than failing the whole question.
                logger.warning(
                    "vector search unavailable for document %s (%s: %s); using keyword search only",
                    document_id, type(exc).__name__, str(exc)[:160],
                )
                vector_hits = []
        timings["retrieval_parallel_ms"] = (time.perf_counter() - t0) * 1000

        chunk_by_id: dict[str, DocumentChunk] = {}
        vector_ranked: list[tuple[str, float]] = []
        for chunk, score in vector_hits:
            chunk_by_id[chunk.id] = chunk
            vector_ranked.append((chunk.id, score))

        keyword_ranked: list[tuple[str, float]] = []
        keyword_terms: dict[str, list[str]] = {}
        for hit in keyword_hits:
            chunk_by_id[hit.chunk.id] = hit.chunk
            keyword_ranked.append((hit.chunk.id, hit.score))
            keyword_terms[hit.chunk.id] = hit.matched_terms

        # -- 3. result fusion (RRF) -------------------------------------------
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(vector_ranked, keyword_ranked, keyword_terms, k=self.rrf_k)
        timings["fusion_ms"] = (time.perf_counter() - t0) * 1000

        # -- 4. candidate pool --------------------------------------------------
        pool_candidates = fused[:pool_size]
        if not pool_candidates:
            self.last_debug = self._build_debug(preprocessed, [], top_k, timings)
            log_retrieval(document_id, self.last_debug, 0)
            return []

        # -- 5. neural/cross-encoder reranking ------------------------------
        t0 = time.perf_counter()
        rerank_scores: dict[str, float] = {}
        if self.enable_reranking:
            texts = [(c.chunk_id, chunk_by_id[c.chunk_id].text) for c in pool_candidates]
            reranked = self.reranker.rerank(preprocessed.normalized, texts, top_k=len(texts))
            rerank_scores = {r.chunk_id: r.score for r in reranked}
            ordered = sorted(pool_candidates, key=lambda c: rerank_scores.get(c.chunk_id, 0.0), reverse=True)
        else:
            ordered = pool_candidates
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

        # -- 6. Top-K evidence ------------------------------------------------
        selected = ordered[:top_k]
        selected_ids = {c.chunk_id for c in selected}

        final_score_of = rerank_scores if self.enable_reranking else {c.chunk_id: c.fusion_score for c in ordered}
        hits = [(chunk_by_id[c.chunk_id], final_score_of.get(c.chunk_id, c.fusion_score)) for c in selected]

        # -- 7. context construction (parent expansion) ------------------------
        results = expand_to_parents(self.db, hits)
        for result, candidate in zip(results, selected):
            result.vector_score = candidate.vector_score
            result.keyword_score = candidate.keyword_score
            result.fusion_score = candidate.fusion_score
            result.rerank_score = rerank_scores.get(candidate.chunk_id)

        timings["total_ms"] = (time.perf_counter() - t_start) * 1000
        self.last_debug = self._build_debug(
            preprocessed, ordered, top_k, timings, chunk_by_id, rerank_scores, selected_ids
        )
        log_retrieval(document_id, self.last_debug, len(results))
        return results

    # -- parallel retrieval paths -----------------------------------------

    def _vector_search(
        self, session_factory: sessionmaker, document_id: str, query: str, limit: int
    ) -> list[tuple[DocumentChunk, float]]:
        session = session_factory()
        try:
            query_embedding = self.embedding_provider.embed_query(query)
            vector_retriever = VectorRetriever(session, self.embedding_provider)
            hits = vector_retriever.retrieve_candidates(document_id, query_embedding, limit)
            # Expunge so the chunk objects outlive this thread's session (the
            # caller reads their attributes after we close it below).
            for chunk, _ in hits:
                session.expunge(chunk)
            return hits
        finally:
            session.close()

    def _keyword_search(self, session_factory: sessionmaker, document_id: str, keywords: list[str], limit: int):
        session = session_factory()
        try:
            index = build_index(session, document_id)
            hits = index.search(keywords, limit)
            for hit in hits:
                session.expunge(hit.chunk)
            return hits
        finally:
            session.close()

    # -- debug payload ------------------------------------------------------

    def _build_debug(
        self,
        preprocessed,
        ordered: list[FusedCandidate],
        top_k: int,
        timings: dict[str, float],
        chunk_by_id: dict[str, DocumentChunk] | None = None,
        rerank_scores: dict[str, float] | None = None,
        selected_ids: set[str] | None = None,
    ) -> RetrievalDebug:
        chunk_by_id = chunk_by_id or {}
        rerank_scores = rerank_scores or {}
        selected_ids = selected_ids or set()

        debug_candidates = []
        for c in ordered:
            chunk = chunk_by_id.get(c.chunk_id)
            if chunk is None:
                continue
            debug_candidates.append(
                DebugCandidate(
                    chunk_id=c.chunk_id,
                    page_number=chunk.page_number,
                    section_title=(chunk.chunk_metadata or {}).get("section_title"),
                    snippet=_snippet(chunk.text),
                    vector_rank=c.vector_rank,
                    vector_score=c.vector_score,
                    keyword_rank=c.keyword_rank,
                    keyword_score=c.keyword_score,
                    matched_keyword_terms=c.matched_keyword_terms,
                    fusion_score=c.fusion_score,
                    rerank_score=rerank_scores.get(c.chunk_id),
                    selected=c.chunk_id in selected_ids,
                )
            )

        return RetrievalDebug(
            query=preprocessed.to_dict(),
            mode="hybrid",
            candidates=debug_candidates,
            config={
                "top_k": top_k,
                "candidate_pool_size": self.candidate_pool_size,
                "rrf_k": self.rrf_k,
                "reranking_enabled": self.enable_reranking,
                "reranker": self.reranker.name if self.enable_reranking else None,
            },
            timings_ms=timings,
        )
