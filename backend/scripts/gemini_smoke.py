"""Live smoke test for the Gemini providers — the one place that calls the
real API on purpose (unit tests never do).

    cd backend
    python -m scripts.gemini_smoke          # key from backend/.env
    GEMINI_API_KEY=... python -m scripts.gemini_smoke

Embeds three sentences (two related, one unrelated), prints the vector
dimensions and pairwise cosine similarities, makes one generate() call, and
prints the model names used. Exits non-zero on any failure, including the
factory silently falling back to the local provider. The API key is never
printed: it is scrubbed from every message this script writes.
"""
from __future__ import annotations

import math
import os
import sys

from app.config import Settings, get_settings
from app.rag.factory import get_embedding_provider, get_llm_provider
from app.rag.interfaces import LLMMessage

SENTENCES = [
    "The transformer model was trained on a large corpus of English text.",
    "A neural language model was pre-trained on billions of English words.",
    "The bakery on Main Street sells sourdough bread every Tuesday.",
]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> int:
    # The environment wins; otherwise use the key the app itself would read
    # from backend/.env.
    key = os.environ.get("GEMINI_API_KEY", "").strip() or (get_settings().gemini_api_key or "").strip()
    if not key:
        print("FAIL: GEMINI_API_KEY is not set (environment or backend/.env).", file=sys.stderr)
        return 2

    def safe(text: object) -> str:
        return str(text).replace(key, "***")

    # Model overrides (EMBEDDING_MODEL / LLM_MODEL) still come from env/.env.
    settings = Settings(embedding_provider="gemini", llm_provider="gemini", gemini_api_key=key)

    try:
        embedder = get_embedding_provider(settings)
        if embedder.name != "gemini":
            print("FAIL: Gemini embedding provider unavailable (fell back to local; "
                  "is google-genai installed?)", file=sys.stderr)
            return 1
        print(f"Embedding model: {embedder.model}")
        vectors = embedder.embed_documents(SENTENCES)
        print(f"Embedded {len(vectors)} sentences, dimensions: {[len(v) for v in vectors]}")
        related = _cosine(vectors[0], vectors[1])
        unrelated_a = _cosine(vectors[0], vectors[2])
        unrelated_b = _cosine(vectors[1], vectors[2])
        print(f"cosine(related 1, related 2)  = {related:.4f}")
        print(f"cosine(related 1, unrelated)  = {unrelated_a:.4f}")
        print(f"cosine(related 2, unrelated)  = {unrelated_b:.4f}")
        if not related > max(unrelated_a, unrelated_b):
            print("WARN: related pair is not the most similar — check the embedding model.")

        llm = get_llm_provider(settings)
        if llm.name != "gemini":
            print("FAIL: Gemini LLM provider unavailable (fell back to local).", file=sys.stderr)
            return 1
        print(f"LLM model: {llm.model}")
        reply = llm.generate(
            "You are a terse assistant.",
            [LLMMessage("user", "Reply with exactly the word: pong")],
        )
        if not reply.strip():
            print("FAIL: generate() returned empty text.", file=sys.stderr)
            return 1
        print(f"generate() -> {safe(reply.strip()[:200])!r}")
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {safe(exc)}", file=sys.stderr)
        return 1

    print("OK: Gemini embeddings + generation working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
