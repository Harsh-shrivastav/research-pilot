"""Query preprocessing: step 1 of the Phase 4 USER QUERY FLOW.

    Query -> [query preprocessing] -> parallel retrieval -> ...

Kept deliberately small and dependency-free (same philosophy as
`text_clean.py`'s sentence splitter): this is normalisation and tokenisation
for lexical search, not an NLP pipeline. The embedding side of retrieval
still embeds the raw (lightly-stripped) question — only the BM25 path and
the debug panel need tokens.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# A short, generic English stopword list. Deliberately conservative: it only
# drops words with essentially no discriminative value for a research paper
# ("the", "of", "and"), and keeps anything that could plausibly be a
# methodology or technical term. This is a retrieval-quality knob, not a
# linguistic resource, so a hand-picked list beats pulling in nltk/spacy.
STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have if in into is it its of on
    or such that the their there these this to was were will with what
    when where which who why how do does did can could should would
    not no nor but so than then them they i you he she we
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_.]*[a-z0-9]|[a-z0-9]")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-ish tokens. Keeps internal hyphens/dots/
    underscores so things like "gpt-4", "bert-base", "3.5" and "in-context"
    survive as single tokens instead of being shredded — exactly the kind of
    uncommon technical term BM25 needs to match exactly."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def keywords(text: str) -> list[str]:
    """Tokens with stopwords removed — used to build the BM25 query and for
    the "which query terms actually appear in this chunk" debug signal."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


@dataclass
class PreprocessedQuery:
    """What step 1 of the pipeline produces, and what the debug panel shows
    under "Query" — the raw text plus what retrieval actually keyed off of."""

    raw: str
    normalized: str
    tokens: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "normalized": self.normalized,
            "tokens": self.tokens,
            "keywords": self.keywords,
        }


def preprocess_query(question: str) -> PreprocessedQuery:
    normalized = " ".join((question or "").strip().split())
    return PreprocessedQuery(
        raw=question,
        normalized=normalized,
        tokens=tokenize(normalized),
        keywords=keywords(normalized),
    )
