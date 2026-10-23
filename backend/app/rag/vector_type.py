"""A SQLAlchemy column type that stores an embedding as a native pgvector
column on PostgreSQL and transparently falls back to a JSON-encoded TEXT
column on SQLite.

This is what lets the exact same `DocumentChunk` model run against the
project's default local-dev SQLite database *and* the docker-compose
PostgreSQL + pgvector service with no code branching anywhere else — the
retriever is the only other place that needs to know which storage is in use
(to pick native ANN search vs. an in-process brute-force scan), see
`app/rag/retriever.py`.

pgvector is an optional import: it is only required when the process actually
connects to PostgreSQL, so a SQLite-only dev install does not need the
`vector` extension or the `pgvector` Python package installed to boot.
"""
from __future__ import annotations

import json

from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator

try:
    from pgvector.sqlalchemy import Vector as _PGVector
except ImportError:  # pragma: no cover - pgvector always installed per requirements.txt
    _PGVector = None


class EmbeddingVector(TypeDecorator):
    """Usage: `Mapped[list[float] | None] = mapped_column(EmbeddingVector(dim))`."""

    impl = Text
    cache_ok = True

    def __init__(self, dimensions: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dimensions = dimensions

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql" and _PGVector is not None:
            return dialect.type_descriptor(_PGVector(self.dimensions))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql" and _PGVector is not None:
            return list(value)
        return json.dumps(list(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql" and _PGVector is not None:
            # psycopg + pgvector already hand back a list[float] / ndarray
            return list(value)
        return json.loads(value)

    @property
    def python_type(self):  # pragma: no cover
        return list


def is_pgvector_dialect(dialect_name: str) -> bool:
    return dialect_name == "postgresql" and _PGVector is not None
