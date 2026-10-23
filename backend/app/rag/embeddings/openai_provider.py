"""OpenAI embeddings — real API calls to https://api.openai.com/v1/embeddings.

Requires `OPENAI_API_KEY`. Selected via `EMBEDDING_PROVIDER=openai`.
"""
from __future__ import annotations

import logging

import httpx

from app.rag.embeddings.util import resize_embedding
from app.rag.interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"
API_URL = "https://api.openai.com/v1/embeddings"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, api_key: str, dimensions: int = 768, model: str | None = None, timeout: float = 30.0):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai")
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.dimensions = dimensions
        self.timeout = timeout

    def _call(self, inputs: list[str]) -> list[list[float]]:
        # text-embedding-3-* models accept a `dimensions` param that returns a
        # native, shortened embedding (better than post-hoc truncation) —
        # request our storage width directly when it's supported.
        payload: dict = {"model": self.model, "input": inputs}
        if self.model.startswith("text-embedding-3"):
            payload["dimensions"] = self.dimensions

        resp = httpx.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenAI embeddings request failed ({resp.status_code}): {resp.text[:500]}")
        data = resp.json()
        vectors = [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]
        return [resize_embedding(v, self.dimensions) for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # OpenAI batches happily; keep batches modest so one call cannot time
        # out on a very large document.
        out: list[list[float]] = []
        batch_size = 96
        for i in range(0, len(texts), batch_size):
            out.extend(self._call(texts[i : i + batch_size]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._call([text])[0]
