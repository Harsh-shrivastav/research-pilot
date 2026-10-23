"""Environment-based application settings. No secrets are hardcoded here —
everything falls back to safe local-dev defaults but is meant to be overridden
via environment variables / the .env file in production."""
from functools import lru_cache
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./researchpilot.db"
    upload_dir: str = "./uploads"
    max_upload_mb: int = 50
    cors_origins: str = "http://localhost:3000"

    # -- Phase 2: document processing -------------------------------------
    # "auto" tries docling -> pymupdf -> pdfplumber and uses the first one
    # installed. Set explicitly to pin a backend (it still falls back if that
    # backend errors on a given file).
    parser_backend: str = "auto"
    enable_ocr: bool = True
    ocr_dpi: int = 200
    # Cap OCR work so one huge scanned book cannot occupy a worker for hours.
    max_ocr_pages: int = 50
    # In-process worker threads. Set to 0 to disable and run `python -m
    # app.worker` as a separate process/container instead.
    worker_count: int = 2
    worker_poll_seconds: float = 2.0

    # -- Phase 3: RAG chat ---------------------------------------------------
    # Provider selection is entirely config-driven — nothing in app/rag/ or
    # the chat router imports a concrete provider directly. "local" needs no
    # API key and no network access, so the app is fully functional (chunking,
    # embedding, retrieval, grounded generation, citations) out of the box;
    # point these at "openai" / "anthropic" / "voyage" for hosted quality.
    embedding_provider: str = "local"  # local | openai | voyage | gemini
    embedding_model: str = ""  # empty = provider default
    llm_provider: str = "local"  # local | anthropic | openai | gemini
    llm_model: str = ""  # empty = provider default
    # Gemini only: models to switch to when LLM_MODEL has no quota left or is
    # overloaded, comma-separated. Unset = the provider's built-in list;
    # set to an empty value to disable fallback.
    llm_fallback_models: str | None = None
    # Gemini only: how much a thinking model reasons before answering.
    # "minimal" is several times faster and enough for answering from
    # retrieved passages; empty = the model's own default (slow, thorough).
    gemini_thinking_level: str = "minimal"

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    voyage_api_key: str | None = None
    gemini_api_key: str | None = None  # Google AI Studio key (GEMINI_API_KEY)
    # Process-wide cap on in-flight Gemini calls, shared by the embedding and
    # LLM providers (see app/rag/gemini_limits.py). Keeps worker threads +
    # chat requests from tripping free-tier rate limits all at once.
    gemini_max_concurrency: int = 4
    # Process-wide pacing of embedding traffic (texts sent per minute). The
    # free tier allows 100/min and counts each text in a batch; 0 = no pacing
    # (paid tiers).
    gemini_embed_texts_per_minute: int = 90

    # Chunking. Sizes are approximate token counts (chars // 4).
    chunk_child_tokens: int = 320
    chunk_child_overlap_tokens: int = 60
    chunk_parent_tokens: int = 1400

    # Retrieval.
    retrieval_top_k: int = 6
    retrieval_max_context_chars: int = 14000
    # Optional pre-LLM gate: if a document's best final retrieval score is
    # below this, answer "not found" without calling the LLM. 0.0 = off. Pick
    # a value from data: `python -m scripts.print_score_distribution`.
    retrieval_min_score: float = 0.0

    # Workspace chat: have the LLM turn one question into a focused sub-query
    # per paper before retrieval. None = auto (on only for LLM_PROVIDER=gemini;
    # the local extractive provider cannot rewrite).
    query_rewrite_enabled: bool | None = None
    query_rewrite_timeout_seconds: float = 8.0
    # Every workspace paper contributes at least this many chunks (or all it
    # has) to the context before any paper gets more.
    workspace_min_chunks_per_paper: int = 2

    # -- Phase 4: hybrid retrieval --------------------------------------------
    # "hybrid" runs vector + BM25 keyword search in parallel, fuses with
    # Reciprocal Rank Fusion, then reranks (see app/rag/hybrid_retriever.py).
    # "vector_only" keeps the Phase 3 behaviour (VectorRetriever alone) —
    # useful as the baseline side of the internal comparison in
    # backend/eval_retrieval.py, or to fall back to if hybrid ever needs to
    # be ruled out while debugging.
    retrieval_mode: str = "hybrid"  # hybrid | vector_only
    # How many candidates each of vector search and keyword search contribute
    # to the fused pool before reranking trims down to retrieval_top_k. Kept
    # well above retrieval_top_k so the reranker has real material to work
    # with instead of just re-sorting a list already the size it needs to be.
    retrieval_candidate_pool_size: int = 20
    # RRF's damping constant — see app/rag/fusion.py's docstring for why 60
    # (the value from the original RRF paper, and Elasticsearch/OpenSearch's
    # default) is a reasonable, non-arbitrary choice.
    retrieval_rrf_k: int = 60
    retrieval_enable_reranking: bool = True
    rerank_provider: str = "local"  # local (see app/rag/reranker.py)

    # Surfaces the Retrieval Debug panel (query, candidates, and every score
    # the hybrid pipeline computed) via the chat SSE stream and API. OFF by
    # default: this is meant for development/admin use, not end users — see
    # app/routers/chat.py and app/schemas.py for where this gates responses.
    debug_mode: bool = False

    chat_history_turns: int = 6  # prior user/assistant turns kept as context
    llm_request_timeout_seconds: float = 60.0

    # -- Phase 6B: multi-paper comparison ------------------------------------
    # Each comparison cell (one paper x one dimension) runs its own targeted,
    # single-document retrieval — deliberately smaller than chat's since a
    # cell answer is 1-3 sentences, not a full grounded chat turn.
    comparison_min_papers: int = 2
    comparison_max_papers: int = 5
    comparison_top_k: int = 4
    comparison_max_context_chars: int = 6000
    comparison_max_dimensions: int = 15
    # Comparison cells are independent LLM calls; run this many at once
    # (Gemini calls are additionally capped by GEMINI_MAX_CONCURRENCY).
    comparison_max_parallel: int = 4
    # How a comparison fills its cells. "structured": ONE LLM call per paper
    # fills every dimension at once, constrained to a JSON schema, and each
    # answer must carry an exact quote that the backend verifies against the
    # stored page text. "per_cell": one retrieval + one LLM call per
    # (paper x dimension). Unset = auto: structured for Gemini, per_cell for
    # everything else. A paper whose structured call fails falls back to
    # per_cell for that paper only.
    comparison_mode: str | None = None
    # Structured mode sends the paper's full text when it fits in this many
    # characters (~4 chars per token; 250k = ~60k tokens, comfortably inside a
    # long-context model and a free-tier per-minute token budget even with
    # several papers at once). Longer papers send only the passages retrieval
    # ranks highest for each dimension.
    comparison_full_paper_max_chars: int = 250_000
    # When on, a structured answer whose quote cannot be found in the paper's
    # stored text is NOT shown: the cell reads "not found" instead. When off,
    # the answer is kept but flagged unverified (quote_verified=false).
    comparison_strict_quotes: bool = True
    # Quote verification, tier 2: also accept a quote whose words all appear
    # in the page text in order within a bounded window. Two-column PDFs are
    # often extracted with the columns interleaved line by line, and a model
    # reading that text quotes it in the correct reading order, which is not a
    # literal substring. Off = exact substrings only.
    comparison_tolerant_quotes: bool = True
    # After a matrix is filled, judge whether the papers are directly
    # comparable (app/comparison/comparability.py). The rules always run; they
    # work with any LLM provider. Off = never run it.
    comparability_enabled: bool = True
    # Let the LLM decide the aspects the rules cannot (ambiguous cell text).
    # Unset = auto: only for Gemini (needs generate_json). false = rules only.
    comparability_use_llm: bool | None = None

    @field_validator("query_rewrite_enabled", "comparison_mode", "comparability_use_llm", mode="before")
    @classmethod
    def _blank_means_auto(cls, value):
        # `QUERY_REWRITE_ENABLED=` / `COMPARISON_MODE=` /
        # `COMPARABILITY_USE_LLM=` (empty) in .env mean auto, not a parse error.
        return None if isinstance(value, str) and not value.strip() else value

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
