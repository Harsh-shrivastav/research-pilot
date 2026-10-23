"""Provider-neutral interfaces for the RAG pipeline.

Nothing outside `app/rag/factory.py` should import a concrete provider class.
Everything else (the chunker, the retriever, the chat service, the router)
talks only to these three interfaces, so switching embedding or LLM vendors
is a config change (`EMBEDDING_PROVIDER` / `LLM_PROVIDER` in `.env`), never a
code change. This is the same seam `deps.get_current_user` leaves for real
auth later, or `processing/backends/*` leaves for parser backends.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """One retrieval hit, already expanded to its parent chunk's text.

    `score` is the *final* relevance score used to order/select this chunk —
    for vector-only retrieval that's cosine similarity; for hybrid retrieval
    (Phase 4) that's the reranker score, with the earlier-stage scores kept
    on `vector_score` / `keyword_score` / `fusion_score` for the retrieval
    debug panel (see `app/rag/hybrid_retriever.py`). All four are optional
    except `score` because vector-only mode never populates the other three.
    """

    chunk_id: str
    parent_chunk_id: Optional[str]
    document_id: str
    page_number: int
    section_id: Optional[str]
    section_title: Optional[str]
    child_text: str  # the precise text that actually matched the query
    context_text: str  # the (usually larger) parent text handed to the LLM
    score: float  # final relevance score, higher is better
    element_types: list[str] = field(default_factory=list)
    bbox: Optional[dict[str, float]] = None  # best-effort, for citation highlighting

    # -- Phase 4: hybrid retrieval score breakdown (debug/eval only) --------
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    fusion_score: Optional[float] = None
    rerank_score: Optional[float] = None


@dataclass
class Citation:
    """What the API returns to the frontend for one source card."""

    chunk_id: str
    document_id: str
    page_number: int
    section_title: Optional[str]
    snippet: str  # the "relevant evidence" text
    score: float
    marker: Optional[int] = None  # the [N] index used inline in the answer, if any
    # Evidence & Research Object Layer (see app/evidence/): the persisted
    # Evidence row this citation was grounded in, when evidence creation
    # succeeded. None for older messages answered before this layer existed,
    # or if evidence persistence failed for this citation — the citation
    # itself (and the answer) is never blocked on it, see app/rag/service.py.
    evidence_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "snippet": self.snippet,
            "score": round(self.score, 4),
            "marker": self.marker,
            "evidence_id": self.evidence_id,
        }


@dataclass
class LLMMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class DebugCandidate:
    """One candidate chunk's full score breakdown through the Phase 4
    pipeline, for the Retrieval Debug panel — never sent to normal users."""

    chunk_id: str
    page_number: int
    section_title: Optional[str]
    snippet: str
    vector_rank: Optional[int] = None
    vector_score: Optional[float] = None
    keyword_rank: Optional[int] = None
    keyword_score: Optional[float] = None
    matched_keyword_terms: list[str] = field(default_factory=list)
    fusion_score: Optional[float] = None
    rerank_score: Optional[float] = None
    selected: bool = False  # made it into the final Top-K handed to the LLM

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "snippet": self.snippet,
            "vector_rank": self.vector_rank,
            "vector_score": _round(self.vector_score),
            "keyword_rank": self.keyword_rank,
            "keyword_score": _round(self.keyword_score),
            "matched_keyword_terms": self.matched_keyword_terms,
            "fusion_score": _round(self.fusion_score),
            "rerank_score": _round(self.rerank_score),
            "selected": self.selected,
        }


def _round(value: Optional[float], ndigits: int = 5) -> Optional[float]:
    return round(value, ndigits) if value is not None else None


@dataclass
class RetrievalDebug:
    """Everything the Retrieval Debug panel shows for one question: the
    preprocessed query, every candidate that reached the fusion pool with its
    full score breakdown, which ones were finally selected, and the
    retrieval config that produced this run (so a screenshot is
    self-explanatory without cross-referencing `.env`)."""

    query: dict[str, Any]
    mode: str  # "hybrid" | "vector_only"
    candidates: list[DebugCandidate] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "candidates": [c.to_dict() for c in self.candidates],
            "config": self.config,
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
        }


@dataclass
class GenerationResult:
    text: str
    provider: str
    model: str
    insufficient_evidence: bool = False


# ---------------------------------------------------------------------------
# EmbeddingProvider
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Turns text into fixed-dimensionality vectors for storage/search."""

    name: str = "base"
    dimensions: int = 0

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts (indexing time)."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single user question (query time). Some hosted APIs use a
        different mode/prefix for queries vs. documents, hence the split."""


# ---------------------------------------------------------------------------
# LLMProvider
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Generates the final answer from a system prompt + conversation."""

    name: str = "base"
    model: str = ""

    @abstractmethod
    def generate(self, system: str, messages: list[LLMMessage]) -> str:
        """Non-streaming generation. Used for tests and non-streaming callers."""

    @abstractmethod
    async def stream(self, system: str, messages: list[LLMMessage]) -> AsyncIterator[str]:
        """Yield the answer incrementally, token-ish chunk by chunk."""


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever(ABC):
    """Finds the chunks of one document most relevant to a query.

    `VectorRetriever` (Phase 3) is query -> embedding -> vector search ->
    parent expansion. `HybridRetriever` (Phase 4) additionally fuses in BM25
    keyword search and reranks the fused pool before parent expansion — see
    `app/rag/hybrid_retriever.py`. Both implement this same interface, so
    `RAGChatService` and the `/ask` router never need to know which one is
    configured (`RETRIEVAL_MODE` in `.env`).
    """

    #: Populated by the most recent `retrieve()` call, or left `None` if the
    #: implementation doesn't produce debug detail (kept off the return type
    #: so nothing outside debug tooling has to unpack a tuple).
    last_debug: Optional["RetrievalDebug"] = None

    @abstractmethod
    def retrieve(
        self, document_id: str, query: str, top_k: int
    ) -> list[RetrievedChunk]:
        """Query processing -> retrieval -> (fusion -> reranking) -> parent expansion."""
