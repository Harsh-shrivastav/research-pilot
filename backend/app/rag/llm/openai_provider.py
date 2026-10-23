"""OpenAI LLM provider — real API calls to
https://api.openai.com/v1/chat/completions.

Requires `OPENAI_API_KEY`. Selected via `LLM_PROVIDER=openai`.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from app.rag.interfaces import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAILLMProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str | None = None, timeout: float = 60.0, max_tokens: int = 1200):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.max_tokens = max_tokens

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _payload(self, system: str, messages: list[LLMMessage], stream: bool) -> dict:
        chat_messages = [{"role": "system", "content": system}]
        chat_messages.extend({"role": m.role, "content": m.content} for m in messages)
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": chat_messages,
            "stream": stream,
        }

    def generate(self, system: str, messages: list[LLMMessage]) -> str:
        resp = httpx.post(
            API_URL, headers=self._headers(), json=self._payload(system, messages, stream=False), timeout=self.timeout
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenAI request failed ({resp.status_code}): {resp.text[:500]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""

    async def stream(self, system: str, messages: list[LLMMessage]) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", API_URL, headers=self._headers(), json=self._payload(system, messages, stream=True)
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise RuntimeError(f"OpenAI request failed ({resp.status_code}): {body[:500]}")
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
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    text = delta.get("content")
                    if text:
                        yield text
