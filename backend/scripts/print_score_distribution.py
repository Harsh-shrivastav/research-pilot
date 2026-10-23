"""Print the final retrieval scores chat would see, to choose RETRIEVAL_MIN_SCORE
from data instead of guessing.

    cd backend
    python -m scripts.print_score_distribution <document_id> questions.txt [--top-k 6]

`questions.txt` holds one question per line (blank lines and lines starting
with # are ignored). Mix questions the paper answers with ones it doesn't
("What is the capital of France?"): a useful threshold sits between the two
groups' best scores. Uses the same retriever chat uses (`build_retriever`,
so RETRIEVAL_MODE / EMBEDDING_PROVIDER from .env apply). Scores are only
comparable within one retrieval mode and embedding provider.
"""
from __future__ import annotations

import argparse
import statistics
import sys

from app.config import get_settings
from app.database import SessionLocal, engine
from app.migrate import run_migrations
from app.models import Document, RagStatus
from app.rag.service import build_retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("document_id")
    parser.add_argument("questions_file")
    parser.add_argument("--top-k", type=int, default=None, help="default: RETRIEVAL_TOP_K")
    args = parser.parse_args(argv)

    settings = get_settings()
    top_k = args.top_k or settings.retrieval_top_k
    with open(args.questions_file, encoding="utf-8") as fh:
        questions = [q.strip() for q in fh if q.strip() and not q.lstrip().startswith("#")]
    if not questions:
        print("No questions in file.", file=sys.stderr)
        return 2

    run_migrations(engine)
    with SessionLocal() as db:
        document = db.get(Document, args.document_id)
        if document is None or document.rag_status != RagStatus.INDEXED:
            print(f"Document {args.document_id} not found or not indexed.", file=sys.stderr)
            return 2
        retriever = build_retriever(db, settings)
        print(f"Document: {document.title or document.original_filename}")
        print(f"Mode: {settings.retrieval_mode} | embeddings: {document.embedding_provider} | top_k: {top_k} | "
              f"current RETRIEVAL_MIN_SCORE: {settings.retrieval_min_score}\n")

        bests: list[float] = []
        for question in questions:
            try:
                chunks = retriever.retrieve(document.id, question, top_k)
            except Exception as exc:
                print(f"? {question}\n    FAILED: {type(exc).__name__}: {exc}")
                continue
            scores = [c.score for c in chunks]
            best = max(scores) if scores else 0.0
            bests.append(best)
            print(f"best={best:.4f}  {question}")
            print("    " + "  ".join(f"{s:.4f}" for s in scores) if scores else "    (no chunks)")

    if bests:
        print(f"\nBest-score summary over {len(bests)} question(s): "
              f"min={min(bests):.4f} median={statistics.median(bests):.4f} max={max(bests):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
