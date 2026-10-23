"""Vector retrieval: the semantic half of Phase 4's hybrid retrieval (and the
whole retrieval path in Phase 3 / `RETRIEVAL_MODE=vector_only`).

See `app/rag/hybrid_retriever.py` for how this is combined with
`app/rag/keyword_search.py` (BM25) via `app/rag/fusion.py` (RRF) and
`app/rag/reranker.py` when `RETRIEVAL_MODE=hybrid` (the default).

Two search paths, chosen automatically by DB dialect (see
`app/rag/vector_type.py`):

* **PostgreSQL + pgvector** — native ANN search using pgvector's
  `cosine_distance` operator, pushed down to SQL (`ORDER BY ... LIMIT k`).
* **SQLite (local dev)** — pgvector has nothing to push down to, so this
  loads the document's child-chunk embeddings and does an in-process
  brute-force cosine similarity scan. That is only acceptable because the
  scope is "chunks of one uploaded paper" (typically low hundreds), not a
  corpus-wide index; it would not scale past that, which is exactly the
  scenario pgvector is for in production (set DATABASE_URL to PostgreSQL).

Either path only ever searches **child** chunks (the precise, embedded
level); each hit is then expanded to its **parent** chunk's fuller text
before being returned, per the small-to-big retrieval design in
`app/rag/chunking.py`.
"""
from __future__ import annotations

import logging
import math

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ChunkLevel, DocumentChunk, DocumentSection
from app.rag.interfaces import DebugCandidate, EmbeddingProvider, RetrievalDebug, RetrievedChunk, Retriever
from app.rag.query_processing import preprocess_query
from app.rag.retrieval_log import log_retrieval
from app.rag.vector_type import is_pgvector_dialect

logger = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def expand_to_parents(
    db: Session, hits: list[tuple[DocumentChunk, float]]
) -> list[RetrievedChunk]:
    """Module-level version of the small-to-big expansion, shared by
    `VectorRetriever` (Phase 3, vector-only) and `HybridRetriever` (Phase 4,
    hybrid + rerank) so both retrieval paths hand the LLM the same fuller
    parent-chunk context regardless of how the child chunk was found."""
    if not hits:
        return []

    parent_ids = {c.parent_chunk_id for c, _ in hits if c.parent_chunk_id}
    parents: dict[str, DocumentChunk] = {}
    if parent_ids:
        for parent in db.scalars(select(DocumentChunk).where(DocumentChunk.id.in_(parent_ids))):
            parents[parent.id] = parent

    section_ids = {c.section_id for c, _ in hits if c.section_id}
    section_ids |= {p.section_id for p in parents.values() if p.section_id}
    sections: dict[str, DocumentSection] = {}
    if section_ids:
        for section in db.scalars(select(DocumentSection).where(DocumentSection.id.in_(section_ids))):
            sections[section.id] = section

    results: list[RetrievedChunk] = []
    for child, score in hits:
        parent = parents.get(child.parent_chunk_id) if child.parent_chunk_id else None
        context_text = parent.text if parent is not None else child.text
        section = sections.get(child.section_id)
        metadata = child.chunk_metadata or {}

        results.append(
            RetrievedChunk(
                chunk_id=child.id,
                parent_chunk_id=child.parent_chunk_id,
                document_id=child.document_id,
                page_number=child.page_number,
                section_id=child.section_id,
                section_title=(section.title if section else metadata.get("section_title")),
                child_text=child.text,
                context_text=context_text,
                score=score,
                element_types=list(metadata.get("element_types") or []),
                bbox=metadata.get("bbox"),
            )
        )
    return results


class VectorRetriever(Retriever):
    def __init__(self, db: Session, embedding_provider: EmbeddingProvider):
        self.db = db
        self.embedding_provider = embedding_provider
        self.last_debug: RetrievalDebug | None = None

    def retrieve(self, document_id: str, query: str, top_k: int) -> list[RetrievedChunk]:
        # -- 1. query processing -------------------------------------------
        preprocessed = preprocess_query(query)
        if not preprocessed.normalized:
            self.last_debug = None
            log_retrieval(document_id, None, 0)
            return []

        # -- 2. embedding -----------------------------------------------------
        query_embedding = self.embedding_provider.embed_query(preprocessed.normalized)

        # -- 3. vector retrieval -----------------------------------------------
        hits = self.retrieve_candidates(document_id, query_embedding, top_k)

        # -- 4. expand to parent context ---------------------------------------
        results = self._expand_to_parents(hits)

        # RETRIEVAL_MODE=vector_only still populates the Retrieval Debug panel
        # (when DEBUG_MODE is on) — just with only the vector column filled
        # in, no keyword/fusion/rerank scores, so switching modes doesn't
        # silently make the debug panel disappear.
        self.last_debug = RetrievalDebug(
            query=preprocessed.to_dict(),
            mode="vector_only",
            candidates=[
                DebugCandidate(
                    chunk_id=chunk.id,
                    page_number=chunk.page_number,
                    section_title=(chunk.chunk_metadata or {}).get("section_title"),
                    snippet=" ".join(chunk.text.split())[:220],
                    vector_rank=rank,
                    vector_score=score,
                    selected=True,
                )
                for rank, (chunk, score) in enumerate(hits, start=1)
            ],
            config={"top_k": top_k},
        )
        log_retrieval(document_id, self.last_debug, len(results))
        return results

    def retrieve_candidates(
        self, document_id: str, query_embedding: list[float], limit: int
    ) -> list[tuple[DocumentChunk, float]]:
        """Raw (child chunk, cosine similarity) hits, *not* expanded to parent
        context — what `HybridRetriever` needs for its candidate pool, before
        fusion/reranking decide which chunks are even worth expanding."""
        dialect_name = self.db.bind.dialect.name if self.db.bind is not None else "sqlite"
        if is_pgvector_dialect(dialect_name):
            return self._retrieve_pgvector(document_id, query_embedding, limit)
        return self._retrieve_bruteforce(document_id, query_embedding, limit)

    # -- search paths -----------------------------------------------------

    def _retrieve_pgvector(
        self, document_id: str, query_embedding: list[float], top_k: int
    ) -> list[tuple[DocumentChunk, float]]:
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        rows = self.db.execute(
            select(DocumentChunk, distance.label("distance"))
            .where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.chunk_level == ChunkLevel.CHILD,
                DocumentChunk.embedding.is_not(None),
            )
            .order_by(distance.asc())
            .limit(top_k)
        ).all()
        # pgvector's cosine_distance is 1 - cosine_similarity.
        return [(chunk, 1.0 - float(dist)) for chunk, dist in rows]

    def _retrieve_bruteforce(
        self, document_id: str, query_embedding: list[float], top_k: int
    ) -> list[tuple[DocumentChunk, float]]:
        chunks = self.db.scalars(
            select(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.chunk_level == ChunkLevel.CHILD,
                DocumentChunk.embedding.is_not(None),
            )
        ).all()
        scored = [(c, _cosine(query_embedding, c.embedding)) for c in chunks if c.embedding]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    # -- parent expansion ---------------------------------------------------
    # Two different child chunks can share the same parent (both matched the
    # query) — `expand_to_parents` keeps both as separate citations (their
    # snippets differ); that's where you'd dedupe if the goal were "one card
    # per parent" instead of "one card per match".

    def _expand_to_parents(self, hits: list[tuple[DocumentChunk, float]]) -> list[RetrievedChunk]:
        return expand_to_parents(self.db, hits)
