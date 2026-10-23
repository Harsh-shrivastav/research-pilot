"""Shared retry + concurrency plumbing for the Gemini providers.

Not a provider itself (nothing here talks to Gemini), so importing it from
`app/rag/embeddings/gemini_provider.py` and `app/rag/llm/gemini_provider.py`
does not break the "concrete providers only via app/rag/factory.py" rule.

* One process-wide `threading.BoundedSemaphore` caps in-flight Gemini calls
  across BOTH providers (worker threads embedding documents and chat requests
  generating answers share one budget), sized by `GEMINI_MAX_CONCURRENCY`.
* Async callers (`stream()`) acquire the same semaphore without blocking the
  event loop: a non-blocking acquire polled with `asyncio.sleep`. That keeps
  one shared budget and is cancellation-safe — a cancelled waiter never ends
  up holding a slot it can't release (which `asyncio.to_thread(sem.acquire)`
  could leak).
* A sliding-window pacer (`throttle_embed_texts`) keeps embedding traffic
  under `GEMINI_EMBED_TEXTS_PER_MINUTE`. The free tier counts every text in
  a batch against its per-minute request quota, so a re-index of a few
  papers would otherwise exhaust it within seconds.
* On a 429, Gemini says how long to wait ("Please retry in 59s"); the
  backoff honours that instead of retrying inside the same exhausted window.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import random
import re
import threading
import time
from contextlib import asynccontextmanager, contextmanager

logger = logging.getLogger(__name__)

MAX_RETRIES = 4
# Per-minute limits ask for <= ~60s. A longer "retry in" means a daily cap
# (e.g. 13636s on the free tier's 20 requests/day): not worth retrying.
MAX_RETRY_DELAY_SECONDS = 60.0
_ASYNC_POLL_SECONDS = 0.05
_WINDOW_SECONDS = 60.0
_RETRY_HINT_RE = re.compile(
    r"retry in (\d+(?:\.\d+)?)s|retryDelay['\"]?:\s*['\"](\d+(?:\.\d+)?)s", re.IGNORECASE
)

_lock = threading.Lock()
_semaphore: threading.BoundedSemaphore | None = None

_rate_lock = threading.Lock()
_rate_limit: int | None = None  # texts per minute; 0 = unlimited; None = read setting
_rate_window: collections.deque[tuple[float, int]] = collections.deque()


def _get_semaphore() -> threading.BoundedSemaphore:
    global _semaphore
    if _semaphore is None:
        with _lock:
            if _semaphore is None:
                from app.config import get_settings

                _semaphore = threading.BoundedSemaphore(max(1, get_settings().gemini_max_concurrency))
    return _semaphore


def reset_limiter(max_concurrency: int | None = None) -> None:
    """Re-size the limiter (tests only — never call while calls are in flight).
    `None` means re-read the setting on next use."""
    global _semaphore
    with _lock:
        _semaphore = threading.BoundedSemaphore(max(1, max_concurrency)) if max_concurrency else None


@contextmanager
def gemini_slot():
    sem = _get_semaphore()
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


@asynccontextmanager
async def async_gemini_slot():
    sem = _get_semaphore()
    while not sem.acquire(blocking=False):
        await asyncio.sleep(_ASYNC_POLL_SECONDS)
    try:
        yield
    finally:
        sem.release()


def reset_rate_limit(texts_per_minute: int | None = None) -> None:
    """Re-size the embedding pacer (tests only). `None` means re-read the
    setting on next use; 0 disables pacing."""
    global _rate_limit
    with _rate_lock:
        _rate_limit = texts_per_minute
        _rate_window.clear()


def throttle_embed_texts(count: int) -> None:
    """Block until sending `count` more texts keeps the last minute's total
    under the limit. Shared by every thread in the process."""
    global _rate_limit
    with _rate_lock:
        if _rate_limit is None:
            from app.config import get_settings

            _rate_limit = max(0, get_settings().gemini_embed_texts_per_minute)
        limit = _rate_limit
        if limit <= 0:
            return
        count = min(count, limit)  # a batch larger than the limit still gets through
        while True:
            now = time.monotonic()
            while _rate_window and now - _rate_window[0][0] >= _WINDOW_SECONDS:
                _rate_window.popleft()
            used = sum(n for _, n in _rate_window)
            if used + count <= limit:
                _rate_window.append((now, count))
                return
            wait = _WINDOW_SECONDS - (now - _rate_window[0][0]) + 0.05
            logger.info("Gemini embedding pacer: %d/%d texts this minute; waiting %.0fs", used, limit, wait)
            time.sleep(wait)


def retry_after_seconds(exc: BaseException | None) -> float | None:
    """The server's "retry in Ns" hint from a 429, if it gave one."""
    if exc is None:
        return None
    match = _RETRY_HINT_RE.search(str(exc))
    if not match:
        return None
    return float(match.group(1) or match.group(2))


def backoff_seconds(attempt: int, exc: BaseException | None = None) -> float:
    """Exponential backoff (1s, 2s, 4s, ...) plus up to 1s of jitter so
    parallel workers that hit a 429 together don't retry in lockstep. If the
    server said how long to wait, wait at least that (capped)."""
    wait = 2**attempt + random.uniform(0, 1)
    hint = retry_after_seconds(exc)
    if hint is not None:
        wait = max(wait, min(hint + random.uniform(0, 1), MAX_RETRY_DELAY_SECONDS))
    return wait


def short_error(exc: BaseException, limit: int = 160) -> str:
    """One-line error for logs: SDK errors embed the whole JSON body."""
    text = " ".join(str(exc).split()) or type(exc).__name__  # timeouts often have no message
    return text if len(text) <= limit else text[: limit - 3] + "..."


def is_retryable(exc: BaseException) -> bool:
    """429 / 5xx / timeouts. `google.genai.errors.APIError` carries the HTTP
    status as `.code`; other HTTP clients use `.status_code`."""
    if is_quota_exhausted(exc):
        return False  # a daily cap won't clear in a retry window: fail fast / switch model
    for attr in ("code", "status_code"):
        code = getattr(exc, attr, None)
        if isinstance(code, int):
            return code == 429 or 500 <= code < 600
    if isinstance(exc, TimeoutError):  # includes asyncio.TimeoutError on 3.11+
        return True
    return "timeout" in type(exc).__name__.lower()  # httpx.ReadTimeout etc.


# ---------------------------------------------------------------------------
# Model fallback: remember which models can't serve right now
# ---------------------------------------------------------------------------

FALLBACK_ATTEMPTS = 2  # tries on a model that has a fallback behind it
# A 429 asking us to wait longer than this isn't worth sleeping through when
# another model is available: switch immediately.
FAST_FAILOVER_SECONDS = 8.0
_MAX_UNAVAILABLE_SECONDS = 6 * 3600.0
_UNAVAILABLE_AFTER_OUTAGE_SECONDS = 30.0

_unavailable_lock = threading.Lock()
_unavailable_until: dict[str, float] = {}


class ModelUnavailable(RuntimeError):
    """A model can't serve right now: it stayed overloaded through its retries,
    or it asked us to wait longer than a fallback model is worth waiting for.
    `retry_after` (seconds) is the wait the server asked for, if any."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def reset_model_registry() -> None:
    """Forget which models are unavailable (tests only)."""
    with _unavailable_lock:
        _unavailable_until.clear()


def mark_model_unavailable(model: str, seconds: float) -> None:
    with _unavailable_lock:
        _unavailable_until[model] = time.monotonic() + min(max(seconds, 1.0), _MAX_UNAVAILABLE_SECONDS)


def model_unavailable(model: str) -> bool:
    with _unavailable_lock:
        until = _unavailable_until.get(model)
        if until is None:
            return False
        if time.monotonic() >= until:
            del _unavailable_until[model]
            return False
        return True


def is_quota_exhausted(exc: BaseException) -> bool:
    """A 429 that will not clear within a retry window (a per-DAY cap)."""
    if getattr(exc, "code", None) != 429 and getattr(exc, "status_code", None) != 429:
        return False
    hint = retry_after_seconds(exc)
    return "PerDay" in str(exc) or (hint is not None and hint > MAX_RETRY_DELAY_SECONDS)


def should_try_next_model(exc: BaseException) -> bool:
    """Worth switching to a fallback model: quota gone, model retired (404),
    or the model stayed overloaded through every retry."""
    if isinstance(exc, ModelUnavailable) or is_quota_exhausted(exc):
        return True
    return getattr(exc, "code", None) == 404


def note_model_failure(model: str, exc: BaseException) -> None:
    """Record why `model` failed so later calls skip it instead of re-learning."""
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        mark_model_unavailable(model, retry_after)
    elif is_quota_exhausted(exc):
        mark_model_unavailable(model, retry_after_seconds(exc) or 3600.0)
    elif getattr(exc, "code", None) == 404:
        mark_model_unavailable(model, _MAX_UNAVAILABLE_SECONDS)
    else:
        mark_model_unavailable(model, _UNAVAILABLE_AFTER_OUTAGE_SECONDS)
