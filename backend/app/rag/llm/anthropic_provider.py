"""Anthropic LLM provider — real API calls to https://api.anthropic.com/v1/messages.

Requires `ANTHROPIC_API_KEY`. Selected via `LLM_PROVIDER=anthropic`. Uses
plain `httpx` rather than the `anthropic` SDK to avoid adding another
dependency for a single endpoint; the request/response shape is stable and
documented at https://docs.claude.com/en/api/messages.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from app.rag.interfaces import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicLLMProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str | None = None, timeout: float = 60.0, max_tokens: int = 1200):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic")
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.max_tokens = max_tokens

    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    def _payload(self, system: str, messages: list[LLMMessage], stream: bool) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
        }

    def generate(self, system: str, messages: list[LLMMessage]) -> str:
        resp = httpx.post(
            API_URL,
            headers=self._headers(),
            json=self._payload(system, messages, stream=False),
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Anthropic request failed ({resp.status_code}): {resp.text[:500]}")
        data = resp.json()
        return "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")

    async def stream(self, system: str, messages: list[LLMMessage]) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                API_URL,
                headers=self._headers(),
                json=self._payload(system, messages, stream=True),
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise RuntimeError(f"Anthropic request failed ({resp.status_code}): {body[:500]}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")
