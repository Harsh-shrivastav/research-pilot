"""Gemini LLM provider via the official `google-genai` SDK.

Requires `GEMINI_API_KEY`. Selected via `LLM_PROVIDER=gemini`.

Uses the stable `generateContent` API (Google documents it as fully supported;
the newer Interactions API is optional). Default model is `gemini-3.8-flash`;
override with `LLM_MODEL=...` (for example a Flash-Lite model for cheap calls).

Both calls retry 429 / 5xx / timeouts with exponential backoff + jitter and
share the process-wide Gemini concurrency limit (app/rag/gemini_limits.py).
`stream()` only retries if nothing has been yielded yet — once text has
reached the client, a retry would duplicate it, so the error is raised.

Model fallback: free-tier keys have small per-model daily caps and the newest
model is often overloaded. When the primary model's quota is gone, it was
retired, or it stays overloaded through its retries, the call moves on to the
next model in `fallback_models` (LLM_FALLBACK_MODELS). A model that failed is
remembered for a while, so later calls skip it without another round trip.
`last_model_used` reports which model actually answered.

`generate_json()` is the structured-output variant: the same call with
`response_mime_type="application/json"` and a JSON Schema (`response_json_schema`
in google-genai >= 1.x), so the model is constrained to the requested shape.
It reuses the retry / fallback / thinking handling above, strips a stray code
fence defensively, parses the object, and retries ONCE on invalid JSON.

Speed: "thinking" models reason silently before the first word, which costs
seconds per answer and counts against the output-token budget (answers can
come back truncated). RAG answers only restate retrieved passages, so
`thinking_level` defaults to "minimal" (GEMINI_THINKING_LEVEL; empty = the
model's own default). A model that rejects the setting is remembered and
called without it.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from app.rag.gemini_limits import (
    FALLBACK_ATTEMPTS,
    FAST_FAILOVER_SECONDS,
    MAX_RETRIES,
    ModelUnavailable,
    async_gemini_slot,
    backoff_seconds,
    gemini_slot,
    is_retryable,
    model_unavailable,
    note_model_failure,
    retry_after_seconds,
    short_error,
    should_try_next_model,
)
from app.rag.interfaces import LLMMessage, LLMProvider
from app.rag.json_utils import parse_json_object

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"
# Tried in order when DEFAULT_MODEL / LLM_MODEL can't serve (see module doc).
DEFAULT_FALLBACK_MODELS = ("gemini-3.5-flash", "gemini-3.1-flash-lite")
DEFAULT_THINKING_LEVEL = "minimal"
# A structured answer can be long (one object per comparison field); give it
# room so the JSON is never cut off mid-object.
JSON_MIN_OUTPUT_TOKENS = 8192

# Models that answered 400 to a thinking config; called without it from then on.
_thinking_rejected: set[str] = set()


def _rejects_thinking_config(exc: BaseException) -> bool:
    return getattr(exc, "code", None) == 400 and "thinking" in str(exc).lower()


def _to_contents(messages: list[LLMMessage]):
    """Our roles are user/assistant; Gemini wants user/model."""
    from google.genai import types

    return [
        types.Content(
            role="model" if m.role == "assistant" else "user",
            parts=[types.Part.from_text(text=m.content)],
        )
        for m in messages
    ]


class GeminiLLMProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 4096,  # generous: thinking models may spend tokens reasoning
        temperature: float = 0.2,  # low = more faithful to the retrieved passages
        client=None,  # injectable for tests
        fallback_models: tuple[str, ...] | list[str] = (),
        thinking_level: str | None = None,
    ):
        if not api_key and client is None:
            raise ValueError("GEMINI_API_KEY is required for LLM_PROVIDER=gemini")
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fallback_models = tuple(m for m in fallback_models if m and m != self.model)
        self.last_model_used = self.model
        # None -> the module default; "" -> leave the model's own default.
        self.thinking_level = DEFAULT_THINKING_LEVEL if thinking_level is None else thinking_level
        if client is not None:
            self._client = client
        else:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(timeout * 1000)),
            )

    def _config(self, system: str, model: str, json_schema: dict | None = None):
        from google.genai import types

        extra = {}
        if self.thinking_level and model not in _thinking_rejected:
            extra["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        max_tokens = self.max_tokens
        if json_schema is not None:
            extra["response_mime_type"] = "application/json"
            extra["response_json_schema"] = json_schema
            max_tokens = max(max_tokens, JSON_MIN_OUTPUT_TOKENS)
        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=self.temperature,
            **extra,
        )

    @staticmethod
    def _note_thinking_rejected(model: str, exc: BaseException) -> bool:
        """True (once per model) if `exc` is the model refusing our thinking
        config: the caller then retries immediately without it."""
        if model in _thinking_rejected or not _rejects_thinking_config(exc):
            return False
        _thinking_rejected.add(model)
        logger.warning("Gemini model %s rejected the thinking config; calling it without", model)
        return True

    def _model_chain(self) -> list[str]:
        """Primary then fallbacks, skipping models known to be down right now.
        If all are marked down, try them all anyway (each fails fast)."""
        models = [self.model, *self.fallback_models]
        return [m for m in models if not model_unavailable(m)] or models

    # -- generate ---------------------------------------------------------

    def _generate_with(
        self,
        model: str,
        system: str,
        messages: list[LLMMessage],
        attempts: int,
        can_fail_over: bool = False,
        json_schema: dict | None = None,
    ) -> str:
        failures = 0
        while True:
            try:
                with gemini_slot():
                    response = self._client.models.generate_content(
                        model=model,
                        contents=_to_contents(messages),
                        config=self._config(system, model, json_schema),
                    )
                return response.text or ""
            except Exception as exc:
                if self._note_thinking_rejected(model, exc):
                    continue
                if not is_retryable(exc):
                    raise
                hint = retry_after_seconds(exc)
                if can_fail_over and hint is not None and hint > FAST_FAILOVER_SECONDS:
                    raise ModelUnavailable(
                        f"Gemini rate limit on {model}, retry in {hint:.0f}s", retry_after=hint
                    ) from exc
                failures += 1
                if failures >= attempts:
                    raise ModelUnavailable(
                        f"Gemini generate failed after {attempts} attempts: {short_error(exc)}"
                    ) from exc
                wait = backoff_seconds(failures - 1, exc)
                logger.warning(
                    "Gemini generate attempt %d/%d on %s failed (%s); retrying in %.1fs",
                    failures, attempts, model, short_error(exc), wait,
                )
                time.sleep(wait)

    def generate(self, system: str, messages: list[LLMMessage]) -> str:
        return self._generate_text(system, messages)

    def generate_json(self, system: str, messages: list[LLMMessage], schema: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(2):  # one retry on invalid JSON
            sent = messages
            if attempt:
                sent = [*messages[:-1], LLMMessage(
                    role=messages[-1].role,
                    content=messages[-1].content + "\n\nReturn ONLY one valid JSON object that matches the schema.",
                )]
            raw = self._generate_text(system, sent, json_schema=schema)
            try:
                return parse_json_object(raw)
            except ValueError as exc:
                last_error = exc
                logger.warning("Gemini JSON output was not valid (attempt %d/2): %s", attempt + 1, short_error(exc))
        raise ValueError(f"Gemini returned invalid JSON twice: {last_error}")

    def _generate_text(self, system: str, messages: list[LLMMessage], json_schema: dict | None = None) -> str:
        chain = self._model_chain()
        for index, model in enumerate(chain):
            is_last = index == len(chain) - 1
            try:
                text = self._generate_with(
                    model, system, messages, MAX_RETRIES if is_last else FALLBACK_ATTEMPTS,
                    can_fail_over=not is_last, json_schema=json_schema,
                )
                self.last_model_used = model
                return text
            except Exception as exc:
                if is_last or not should_try_next_model(exc):
                    raise
                note_model_failure(model, exc)
                logger.warning(
                    "Gemini model %s unavailable (%s); falling back to %s",
                    model, short_error(exc), chain[index + 1],
                )
        raise AssertionError("unreachable")

    # -- stream -----------------------------------------------------------

    async def _stream_with(
        self, model: str, system: str, messages: list[LLMMessage], attempts: int, can_fail_over: bool = False
    ) -> AsyncIterator[str]:
        failures = 0
        while True:
            yielded = False
            try:
                async with async_gemini_slot():
                    stream = await self._client.aio.models.generate_content_stream(
                        model=model,
                        contents=_to_contents(messages),
                        config=self._config(system, model),
                    )
                    async for chunk in stream:
                        text = getattr(chunk, "text", None)
                        if text:
                            yielded = True
                            yield text
                return
            except Exception as exc:
                if yielded:
                    raise
                if self._note_thinking_rejected(model, exc):
                    continue
                if not is_retryable(exc):
                    raise
                hint = retry_after_seconds(exc)
                if can_fail_over and hint is not None and hint > FAST_FAILOVER_SECONDS:
                    raise ModelUnavailable(
                        f"Gemini rate limit on {model}, retry in {hint:.0f}s", retry_after=hint
                    ) from exc
                failures += 1
                if failures >= attempts:
                    raise ModelUnavailable(
                        f"Gemini stream failed after {attempts} attempts: {short_error(exc)}"
                    ) from exc
                wait = backoff_seconds(failures - 1, exc)
                logger.warning(
                    "Gemini stream attempt %d/%d on %s failed before first chunk (%s); retrying in %.1fs",
                    failures, attempts, model, short_error(exc), wait,
                )
                await asyncio.sleep(wait)

    async def stream(self, system: str, messages: list[LLMMessage]) -> AsyncIterator[str]:
        chain = self._model_chain()
        for index, model in enumerate(chain):
            is_last = index == len(chain) - 1
            yielded = False
            try:
                async for text in self._stream_with(
                    model, system, messages, MAX_RETRIES if is_last else FALLBACK_ATTEMPTS,
                    can_fail_over=not is_last,
                ):
                    yielded = True
                    yield text
                self.last_model_used = model
                return
            except Exception as exc:
                # Once text reached the client, a different model would
                # produce a different answer: raise instead of switching.
                if yielded or is_last or not should_try_next_model(exc):
                    raise
                note_model_failure(model, exc)
                logger.warning(
                    "Gemini model %s unavailable (%s); falling back to %s",
                    model, short_error(exc), chain[index + 1],
                )
