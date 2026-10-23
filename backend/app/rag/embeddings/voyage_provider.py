"""Voyage AI embeddings — real API calls to https://api.voyageai.com/v1/embeddings.

Voyage is Anthropic's recommended embedding partner (Claude itself does not
expose an embeddings endpoint). Requires `VOYAGE_API_KEY`. Selected via
`EMBEDDING_PROVIDER=voyage`.
"""
from __future__ import annotations

import logging

import httpx

from app.rag.embeddings.util import resize_embedding
from app.rag.interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "voyage-3.5"
API_URL = "https://api.voyageai.com/v1/embeddings"


class VoyageEmbeddingProvider(EmbeddingProvider):
    name = "voyage"

    def __init__(self, api_key: str, dimensions: int = 768, model: str | None = None, timeout: float = 30.0):
        if not api_key:
            raise ValueError("VOYAGE_API_KEY is required for EMBEDDING_PROVIDER=voyage")
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.dimensions = dimensions
        self.timeout = timeout

    def _call(self, inputs: list[str], input_type: str) -> list[list[float]]:
        resp = httpx.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": inputs, "input_type": input_type},
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Voyage embeddings request failed ({resp.status_code}): {resp.text[:500]}")
        data = resp.json()
        vectors = [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]
        return [resize_embedding(v, self.dimensions) for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        batch_size = 128
        for i in range(0, len(texts), batch_size):
            out.extend(self._call(texts[i : i + batch_size], input_type="document"))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._call([text], input_type="query")[0]
