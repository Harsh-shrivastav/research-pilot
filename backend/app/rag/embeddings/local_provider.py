"""Deterministic, offline embedding provider.

No network access and no API key, so `EMBEDDING_PROVIDER=local` (the
default) makes the whole chat feature work out of the box — including in
environments with no outbound internet access at all.

This is a classic "hashing trick" bag-of-words embedder: each token (and
character bigram, to give partial credit for morphological overlap like
"train"/"training") is hashed into one of `dimensions` buckets, weighted by a
log-dampened term frequency, and the resulting vector is L2-normalised so
cosine similarity behaves sensibly. It is a lexical-overlap embedding, not a
learned semantic one — good enough to retrieve the right paragraph for
keyword-bearing research questions ("What dataset was used?", "What
methodology..."), materially worse than a real sentence embedding for
paraphrase-heavy queries. Swap `EMBEDDING_PROVIDER` to `openai` or `voyage`
for production-grade semantic retrieval.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from app.rag.interfaces import EmbeddingProvider

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "as", "by", "with", "from", "we", "our", "which",
    "such", "into", "than", "then", "but", "not", "can", "also", "their",
}


def _tokens(text: str) -> list[str]:
    words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def _hash_index(token: str, dimensions: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


class LocalHashEmbeddingProvider(EmbeddingProvider):
    name = "local"

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions
        self.model = "hashing-bow-v1"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _tokens(text)
        if not tokens:
            return vector
        counts = Counter(tokens)
        for token, count in counts.items():
            idx = _hash_index(token, self.dimensions)
            # Sign hashing (second hash bit) reduces collision bias, and
            # log-dampened TF keeps one very repeated word from dominating.
            sign = 1.0 if (_hash_index(token + "#sign", 2) == 0) else -1.0
            vector[idx] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
