"""Parse a JSON object out of LLM output.

Even with a JSON response mode, models occasionally wrap the object in a
```json fence or add a stray sentence around it. This strips both, then
parses; anything else raises `ValueError` (`json.JSONDecodeError` is one), so
callers can treat "not usable JSON" as one failure type.
"""
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(raw: str) -> dict:
    text = _FENCE_RE.sub("", (raw or "").strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model output is JSON but not an object")
    return data
