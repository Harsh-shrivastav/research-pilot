"""Resolve the configured EmbeddingProvider / LLMProvider.

This is the *only* module in the codebase that imports concrete provider
classes — everything else (chunking, the retriever, the chat service, the
router) depends only on the `EmbeddingProvider` / `LLMProvider` interfaces.
Swapping vendors is therefore a `.env` change (`EMBEDDING_PROVIDER`,
`LLM_PROVIDER`), never a code change, mirroring how `PARSER_BACKEND` selects
a Phase 2 parser backend in `app/processing/pipeline.py`.

If a hosted provider is selected but misconfigured (missing API key, bad
value), we log an error and fall back to the local provider rather than
taking the whole chat feature down — the same "degrade, don't crash"
philosophy `main.py` already applies when no PDF parser backend is installed.
"""
from __future__ import annotations

import logging

from app.config import Settings
from app.models import EMBEDDING_DIM
from app.rag.interfaces import EmbeddingProvider, LLMProvider
from app.rag.reranker import Reranker

logger = logging.getLogger(__name__)

EMBEDDING_PROVIDERS = ("local", "openai", "voyage", "gemini")
LLM_PROVIDERS = ("local", "anthropic", "openai", "gemini")
# Only "local" is implemented today (see app/rag/reranker.py's module
# docstring for why, and for the interface a real cross-encoder / hosted
# reranker would plug into). Kept as a tuple, matching the embedding/LLM
# provider lists, so `provider_status()` and the settings docstring stay one
# shape as more rerankers are added.
RERANK_PROVIDERS = ("local",)


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    from app.rag.embeddings.local_provider import LocalHashEmbeddingProvider

    choice = (settings.embedding_provider or "local").strip().lower()

    if choice == "openai":
        try:
            from app.rag.embeddings.openai_provider import OpenAIEmbeddingProvider

            return OpenAIEmbeddingProvider(
                api_key=settings.openai_api_key or "",
                dimensions=EMBEDDING_DIM,
                model=settings.embedding_model or None,
            )
        except Exception as exc:
            logger.error("EMBEDDING_PROVIDER=openai unavailable (%s); falling back to local", exc)
            return LocalHashEmbeddingProvider(dimensions=EMBEDDING_DIM)

    if choice == "voyage":
        try:
            from app.rag.embeddings.voyage_provider import VoyageEmbeddingProvider

            return VoyageEmbeddingProvider(
                api_key=settings.voyage_api_key or "",
                dimensions=EMBEDDING_DIM,
                model=settings.embedding_model or None,
            )
        except Exception as exc:
            logger.error("EMBEDDING_PROVIDER=voyage unavailable (%s); falling back to local", exc)
            return LocalHashEmbeddingProvider(dimensions=EMBEDDING_DIM)

    if choice == "gemini":
        try:
            from app.rag.embeddings.gemini_provider import GeminiEmbeddingProvider

            return GeminiEmbeddingProvider(
                api_key=settings.gemini_api_key or "",
                dimensions=EMBEDDING_DIM,
                model=settings.embedding_model or None,
            )
        except Exception as exc:
            logger.error("EMBEDDING_PROVIDER=gemini unavailable (%s); falling back to local", exc)
            return LocalHashEmbeddingProvider(dimensions=EMBEDDING_DIM)

    if choice != "local":
        logger.warning("unknown EMBEDDING_PROVIDER %r; using local", choice)

    return LocalHashEmbeddingProvider(dimensions=EMBEDDING_DIM)


def get_llm_provider(settings: Settings) -> LLMProvider:
    from app.rag.llm.local_provider import LocalExtractiveLLMProvider

    choice = (settings.llm_provider or "local").strip().lower()

    if choice == "anthropic":
        try:
            from app.rag.llm.anthropic_provider import AnthropicLLMProvider

            return AnthropicLLMProvider(
                api_key=settings.anthropic_api_key or "",
                model=settings.llm_model or None,
                timeout=settings.llm_request_timeout_seconds,
            )
        except Exception as exc:
            logger.error("LLM_PROVIDER=anthropic unavailable (%s); falling back to local", exc)
            return LocalExtractiveLLMProvider()

    if choice == "openai":
        try:
            from app.rag.llm.openai_provider import OpenAILLMProvider

            return OpenAILLMProvider(
                api_key=settings.openai_api_key or "",
                model=settings.llm_model or None,
                timeout=settings.llm_request_timeout_seconds,
            )
        except Exception as exc:
            logger.error("LLM_PROVIDER=openai unavailable (%s); falling back to local", exc)
            return LocalExtractiveLLMProvider()

    if choice == "gemini":
        try:
            from app.rag.llm.gemini_provider import DEFAULT_FALLBACK_MODELS, GeminiLLMProvider

            raw_fallbacks = settings.llm_fallback_models
            fallbacks = (
                DEFAULT_FALLBACK_MODELS
                if raw_fallbacks is None
                else tuple(m.strip() for m in raw_fallbacks.split(",") if m.strip())
            )
            return GeminiLLMProvider(
                api_key=settings.gemini_api_key or "",
                model=settings.llm_model or None,
                timeout=settings.llm_request_timeout_seconds,
                fallback_models=fallbacks,
                thinking_level=settings.gemini_thinking_level,
            )
        except Exception as exc:
            logger.error("LLM_PROVIDER=gemini unavailable (%s); falling back to local", exc)
            return LocalExtractiveLLMProvider()

    if choice != "local":
        logger.warning("unknown LLM_PROVIDER %r; using local", choice)

    return LocalExtractiveLLMProvider()


def get_reranker(settings: Settings) -> Reranker:
    from app.rag.reranker import LocalCrossEncoderReranker

    choice = (settings.rerank_provider or "local").strip().lower()
    if choice != "local":
        logger.warning("unknown RERANK_PROVIDER %r; using local", choice)
    return LocalCrossEncoderReranker()


def provider_status(settings: Settings) -> dict:
    """Surfaced at /api/system/rag so the UI can show which providers are
    actually active (and whether a hosted one is misconfigured)."""
    embedding_choice = (settings.embedding_provider or "local").strip().lower()
    llm_choice = (settings.llm_provider or "local").strip().lower()

    embedding_configured = embedding_choice == "local" or bool(
        {"openai": settings.openai_api_key, "voyage": settings.voyage_api_key, "gemini": settings.gemini_api_key}.get(embedding_choice)
    )
    llm_configured = llm_choice == "local" or bool(
        {"anthropic": settings.anthropic_api_key, "openai": settings.openai_api_key, "gemini": settings.gemini_api_key}.get(llm_choice)
    )

    # Resolve the providers actually in use (after any fallback) so the UI
    # shows real model names. Constructing a provider makes no network call.
    active_embedding = get_embedding_provider(settings)
    active_llm = get_llm_provider(settings)

    return {
        "embedding_provider": embedding_choice,
        "embedding_configured": embedding_configured,
        "embedding_dimensions": EMBEDDING_DIM,
        # The provider/model that indexing would really use right now — differs
        # from embedding_provider when a hosted provider fell back to local.
        "active_embedding_provider": active_embedding.name,
        "embedding_model": getattr(active_embedding, "model", None),
        "llm_provider": llm_choice,
        "llm_configured": llm_configured,
        "active_llm_provider": active_llm.name,
        "llm_model": getattr(active_llm, "model", None),
        "available_embedding_providers": list(EMBEDDING_PROVIDERS),
        "available_llm_providers": list(LLM_PROVIDERS),
        # -- Phase 4: hybrid retrieval -----------------------------------
        "retrieval_mode": settings.retrieval_mode,
        "reranking_enabled": settings.retrieval_enable_reranking,
        "rerank_provider": (settings.rerank_provider or "local").strip().lower(),
        "available_rerank_providers": list(RERANK_PROVIDERS),
        # Whether the Retrieval Debug panel is available at all in this
        # deployment — the UI only offers to show it when this is true, and
        # the API omits debug payloads from responses regardless of what the
        # UI asks for when it is false. See DEBUG_MODE in app/config.py.
        "debug_mode": settings.debug_mode,
    }
