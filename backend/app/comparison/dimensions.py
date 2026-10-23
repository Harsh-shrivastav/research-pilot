"""Phase 6B — Comparison dimension registry.

A "dimension" is one row of the comparison matrix: a short label plus the
targeted-retrieval question used to fill that row for each paper. The
defaults deliberately mirror ``app/rag/service.py:SUGGESTED_QUESTIONS`` (the
single-document chat suggestions) — the same underlying questions a
researcher already asks one paper at a time in chat are exactly the useful
axes to compare several papers on.

Researchers are not limited to the defaults: any free-text dimension (e.g.
"evaluation metrics") is accepted and turned into a reasonable retrieval
query (`resolve_dimensions`). Nothing here talks to the database or runs
retrieval — this module is pure data + string formatting so it's trivially
unit-testable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import ClaimType


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    query: str
    claim_type: str = ClaimType.UNKNOWN

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "query": self.query}


# Ordered: this is also the default row order shown in the matrix.
# Phase 7: all 11 standard research-paper comparison aspects. The system is
# still extensible — researchers can add arbitrary custom dimensions on top
# of these via the comparison UI or API.
DEFAULT_DIMENSIONS: list[Dimension] = [
    Dimension(
        key="research_problem",
        label="Research Problem",
        query="What is the research problem or gap this paper addresses?",
        claim_type=ClaimType.REPORTED_FACT,
    ),
    Dimension(
        key="research_objective",
        label="Research Objective",
        query="What is the stated objective or goal of this paper?",
        claim_type=ClaimType.REPORTED_FACT,
    ),
    Dimension(
        key="methodology",
        label="Methodology",
        query="What methodology or approach does this paper use?",
        claim_type=ClaimType.REPORTED_METHOD,
    ),
    Dimension(
        key="model_algorithm",
        label="Model / Algorithm",
        query="What model, algorithm, or system does this paper propose or use?",
        claim_type=ClaimType.REPORTED_METHOD,
    ),
    Dimension(
        key="dataset",
        label="Dataset",
        query="What dataset(s) does this paper use for its experiments or evaluation?",
        claim_type=ClaimType.REPORTED_FACT,
    ),
    Dimension(
        key="dataset_size",
        label="Dataset Size",
        query="What is the size of the dataset(s) used in this paper (e.g. number of samples, documents, images)?",
        claim_type=ClaimType.REPORTED_FACT,
    ),
    Dimension(
        key="evaluation_metrics",
        label="Evaluation Metrics",
        query="What evaluation metrics does this paper use to measure performance?",
        claim_type=ClaimType.REPORTED_FACT,
    ),
    Dimension(
        key="main_results",
        label="Main Results",
        query="What are the main quantitative or qualitative results reported in this paper?",
        claim_type=ClaimType.REPORTED_RESULT,
    ),
    Dimension(
        key="baselines",
        label="Baselines",
        query="What baseline methods or systems does this paper compare against?",
        claim_type=ClaimType.REPORTED_FACT,
    ),
    Dimension(
        key="limitations",
        label="Limitations",
        query="What limitations does this paper report or discuss?",
        claim_type=ClaimType.REPORTED_FACT,
    ),
    Dimension(
        key="future_work",
        label="Future Work",
        query="What future work or directions does this paper suggest?",
        claim_type=ClaimType.REPORTED_FACT,
    ),
]

DEFAULT_DIMENSION_KEYS = [d.key for d in DEFAULT_DIMENSIONS]
_BY_KEY = {d.key: d for d in DEFAULT_DIMENSIONS}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    slug = _SLUG_RE.sub("_", text.strip().lower()).strip("_")
    return slug or "dimension"


def _custom_dimension(raw: str) -> Dimension:
    """Turn a free-text comparison criterion into a Dimension. Never
    fabricates a canned question for something a researcher typed
    themselves — the raw text becomes both the label and, wrapped in a
    neutral question frame, the retrieval query."""
    label = raw.strip()
    query = raw.strip()
    if not query.endswith("?"):
        query = f"What does this paper report regarding {query.rstrip('.')}?"
    return Dimension(key=_slugify(raw), label=label, query=query, claim_type=ClaimType.UNKNOWN)


def resolve_dimensions(raw: list[str] | None, max_dimensions: int = 8) -> list[Dimension]:
    """Turn the (optional) list of dimension strings from a comparison
    request into concrete `Dimension`s. `None` or an empty list -> the
    default five. A string matching a known default key/label (case
    insensitive) reuses that canonical dimension (including its
    claim_type); anything else becomes a custom dimension via
    `_custom_dimension`. Duplicate keys are dropped, keeping the first
    occurrence's position, and the result is capped at `max_dimensions`.
    """
    if not raw:
        return list(DEFAULT_DIMENSIONS)

    label_lookup = {d.label.lower(): d for d in DEFAULT_DIMENSIONS}

    resolved: list[Dimension] = []
    seen_keys: set[str] = set()
    for item in raw:
        text = (item or "").strip()
        if not text:
            continue
        dim = _BY_KEY.get(text.lower()) or label_lookup.get(text.lower()) or _custom_dimension(text)
        if dim.key in seen_keys:
            continue
        seen_keys.add(dim.key)
        resolved.append(dim)
        if len(resolved) >= max_dimensions:
            break

    return resolved or list(DEFAULT_DIMENSIONS)
