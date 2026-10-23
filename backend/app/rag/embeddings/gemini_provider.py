"""Gemini embeddings via the official `google-genai` SDK.

Requires `GEMINI_API_KEY`. Selected via `EMBEDDING_PROVIDER=gemini`.

Notes
-----
* Uses `gemini-embedding-001` (stable) by default. Override with
  `EMBEDDING_MODEL=...` in `.env` if you want a different Gemini embedding model.
* Documents are embedded with task_type RETRIEVAL_DOCUMENT and questions with
  RETRIEVAL_QUERY. This is why `EmbeddingProvider` has two methods.
* The app stores a fixed 768-wide vector (`EMBEDDING_DIM`). We ask Gemini for
  768 dimensions directly (`output_dimensionality`) and then L2-normalise,
  because truncated embeddings are not unit-length and the retriever uses
  cosine similarity.
* Switching providers means every document must be re-indexed
  (POST /api/documents/{id}/reindex), since old vectors came from a different
  model and are not comparable.
"""
from __future__ import annotations

import logging
import math
import time

from app.rag.embeddings.util import resize_embedding
from app.rag.gemini_limits import (
    MAX_RETRIES,
    backoff_seconds,
    gemini_slot,
    is_retryable,
    short_error,
    throttle_embed_texts,
)
from app.rag.interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-embedding-001"
BATCH_SIZE = 50  # well under the per-request batch limit
QUERY_CACHE_SIZE = 256


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm > 0 else vector


class GeminiEmbeddingProvider(EmbeddingProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        dimensions: int = 768,
        model: str | None = None,
        timeout: float = 30.0,
        client=None,  # injectable for tests
    ):
        if not api_key and client is None:
            raise ValueError("GEMINI_API_KEY is required for EMBEDDING_PROVIDER=gemini")
        self.model = model or DEFAULT_MODEL
        self.dimensions = dimensions
        self.timeout = timeout
        self._query_cache: dict[str, list[float]] = {}
        if client is not None:
            self._client = client
        else:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(timeout * 1000)),
            )

    def _embed(self, texts: list[str], task_type: str, paced: bool = True) -> list[list[float]]:
        from google.genai import types

        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.dimensions,
        )
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                if paced:  # bulk indexing only; see embed_query
                    throttle_embed_texts(len(texts))  # before taking a slot, so waiting holds none
                with gemini_slot():
                    result = self._client.models.embed_content(
                        model=self.model, contents=texts, config=config
                    )
                vectors = [list(e.values) for e in result.embeddings]
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        f"Gemini returned {len(vectors)} embeddings for {len(texts)} inputs"
                    )
                return [_normalise(resize_embedding(v, self.dimensions)) for v in vectors]
            except Exception as exc:
                # Only 429 / 5xx / timeouts are worth retrying; a bad key or
                # bad request fails fast instead of burning ~15s of backoff.
                if not is_retryable(exc):
                    raise
                last_exc = exc
                if attempt == MAX_RETRIES - 1:
                    break
                wait = backoff_seconds(attempt, exc)
                logger.warning(
                    "Gemini embed attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, short_error(exc), wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"Gemini embeddings failed after {MAX_RETRIES} attempts: {short_error(last_exc)}"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            out.extend(self._embed(texts[i : i + BATCH_SIZE], "RETRIEVAL_DOCUMENT"))
        return out

    def embed_query(self, text: str) -> list[float]:
        # A question is one text. It must not queue behind a paper that is
        # being indexed (that would freeze chat for a minute), and the pacer's
        # headroom under the real limit leaves room for it. The same query is
        # embedded again for every paper in a comparison / gap run, so cache.
        cached = self._query_cache.get(text)
        if cached is not None:
            return list(cached)
        vector = self._embed([text], "RETRIEVAL_QUERY", paced=False)[0]
        if len(self._query_cache) >= QUERY_CACHE_SIZE:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[text] = vector
        return list(vector)
