#!/usr/bin/env python3
"""Phase 4 retrieval evaluation: baseline vector search vs. hybrid + rerank.

Standalone, like `verify_pipeline.py`: no pytest, real database, real
providers, real (already-indexed) document. Run it against any document this
app has already chunked and embedded:

    python eval_retrieval.py                 # picks the most recently indexed doc
    python eval_retrieval.py <document_id>    # evaluate a specific document

What this does NOT do, on purpose: invent precision/recall/nDCG numbers.
There is no labelled relevance data for an arbitrary uploaded PDF (no human
has marked which chunks are "correct" for a given question), so this script
never prints a made-up accuracy figure. What it prints instead is real,
directly measured comparisons between the two retrieval paths on the exact
same document and query:

  * literal-term hit rank — the rank (1 = first) of the first returned chunk
    that actually contains one of the query's keyword terms, verbatim. This
    is not an accuracy claim, but it IS a real, computable signal for "did a
    lexically-relevant chunk surface near the top", which is precisely the
    failure mode hybrid retrieval is meant to improve over vector-only search
    on exact/rare terminology.
  * top-K overlap between the two retrieval paths, and whether the chunks
    that differ are ones with strong keyword matches.
  * wall-clock timings for each stage (already computed by
    `HybridRetriever`, not re-derived here).

Five query categories are generated FROM the document's own indexed text
(never hand-authored fixtures the document might not contain terms for): the
document is the ground truth for "does this term/number/phrase actually
appear in it", which is a claim this script CAN make honestly.
"""
from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import select  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import ChunkLevel, Document, DocumentChunk, RagStatus  # noqa: E402
from app.rag.factory import get_embedding_provider, get_reranker  # noqa: E402
from app.rag.hybrid_retriever import HybridRetriever  # noqa: E402
from app.rag.keyword_search import build_index  # noqa: E402
from app.rag.query_processing import keywords as extract_keywords  # noqa: E402
from app.rag.query_processing import tokenize  # noqa: E402
from app.rag.retriever import VectorRetriever  # noqa: E402

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m",
)

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  {GREEN}PASS{RESET} {label}")
    else:
        failed += 1
        print(f"  {RED}FAIL{RESET} {label}" + (f" {DIM}({detail}){RESET}" if detail else ""))


# ---------------------------------------------------------------------------
# Query generation — derived from the document's own indexed chunks.
# ---------------------------------------------------------------------------

# A small, hand-picked synonym map used only to build the "paraphrased
# terminology" query from whatever exact term the document happens to
# contain. This is a query-construction convenience, not a claimed accuracy
# resource, so a short list is fine — it only needs to occasionally hit.
SYNONYMS = {
    "method": "approach", "methods": "approaches", "approach": "technique",
    "result": "finding", "results": "findings", "show": "demonstrate",
    "shows": "demonstrates", "use": "employ", "uses": "employs",
    "propose": "introduce", "proposed": "introduced", "evaluate": "assess",
    "evaluation": "assessment", "significant": "notable", "dataset": "data set",
    "model": "system", "improve": "enhance", "improves": "enhances",
    "baseline": "reference model", "performance": "effectiveness",
    "accuracy": "correctness rate", "training": "learning",
}

METHODOLOGY_TERMS = [
    "method", "methodology", "approach", "algorithm", "dataset", "baseline",
    "experiment", "experiments", "evaluation", "training", "model",
    "framework", "architecture", "procedure",
]

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


def build_bm25_and_chunks(db, document_id: str):
    index = build_index(db, document_id)
    return index


def generate_queries(db, document_id: str, index) -> dict[str, str | None]:
    chunks: list[DocumentChunk] = db.scalars(
        select(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.chunk_level == ChunkLevel.CHILD,
        )
    ).all()

    queries: dict[str, str | None] = {
        "exact_terminology": None,
        "paraphrased_terminology": None,
        "numbers": None,
        "methodology_terms": None,
        "uncommon_technical_terms": None,
    }

    # -- exact terminology: the chunk with the highest average term rarity
    # (lowest average document frequency) contains this document's most
    # distinctive language — use a real 3-word phrase straight out of it.
    if chunks:
        def avg_rarity(chunk: DocumentChunk) -> float:
            toks = extract_keywords(chunk.text)
            if not toks:
                return 0.0
            dfs = [index._df.get(t, index._n) for t in toks]  # noqa: SLF001 (eval script, same package)
            return sum(1.0 / (df or 1) for df in dfs) / len(toks)

        distinctive = max(chunks, key=avg_rarity)
        words = _WORD_RE.findall(distinctive.text)
        if len(words) >= 3:
            # A real 3-gram of running words from the chunk, verbatim.
            for i in range(len(words) - 2):
                phrase = " ".join(words[i : i + 3])
                if len(phrase) >= 12:
                    queries["exact_terminology"] = phrase
                    break

    # -- paraphrased terminology: substitute synonyms into the exact-term
    # query where possible; otherwise wrap it as a natural-language question.
    exact = queries["exact_terminology"]
    if exact:
        paraphrased_words = []
        changed = False
        for w in exact.split():
            sub = SYNONYMS.get(w.lower())
            if sub:
                paraphrased_words.append(sub)
                changed = True
            else:
                paraphrased_words.append(w)
        queries["paraphrased_terminology"] = (
            " ".join(paraphrased_words) if changed else f"What does the paper say about {exact}?"
        )

    # -- numbers: a chunk containing a numeric literal; query = the sentence
    # around it, so the number is embedded in real surrounding context.
    for chunk in chunks:
        m = _NUMBER_RE.search(chunk.text)
        if not m:
            continue
        start = chunk.text.rfind(".", 0, m.start()) + 1
        end = chunk.text.find(".", m.end())
        end = end + 1 if end != -1 else min(len(chunk.text), m.end() + 80)
        sentence = chunk.text[start:end].strip()
        if len(sentence) >= 15:
            queries["numbers"] = sentence[:200]
            break

    # -- methodology terms: whichever fixed methodology term this document
    # actually contains (first hit by document frequency in the BM25 index).
    for term in METHODOLOGY_TERMS:
        if index._df.get(term, 0) > 0:  # noqa: SLF001
            queries["methodology_terms"] = f"What {term} did the paper use?"
            break

    # -- uncommon technical terms: the rarest (lowest document-frequency,
    # ties broken by length) alphabetic token with df >= 1 in the corpus.
    candidates = [
        (df, -len(term), term)
        for term, df in index._df.items()  # noqa: SLF001
        if term.isalpha() and len(term) >= 6
    ]
    if candidates:
        candidates.sort()
        queries["uncommon_technical_terms"] = candidates[0][2]

    return queries


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def literal_hit_rank(query: str, chunks: list) -> int | None:
    """1-based rank of the first chunk whose text literally contains one of
    the query's keyword terms. None if no returned chunk contains any."""
    terms = set(extract_keywords(query))
    if not terms:
        terms = set(tokenize(query))
    for i, chunk in enumerate(chunks, start=1):
        chunk_terms = set(tokenize(chunk.child_text))
        if terms & chunk_terms:
            return i
    return None


def run_category(db, document_id: str, category: str, query: str, top_k: int, reranker) -> None:
    print(f"\n{BOLD}{category}{RESET}  {DIM}query: {query!r}{RESET}")

    embedding_provider = get_embedding_provider(get_settings())

    baseline = VectorRetriever(db, embedding_provider)
    t0 = time.perf_counter()
    baseline_results = baseline.retrieve(document_id, query, top_k)
    baseline_ms = (time.perf_counter() - t0) * 1000

    settings = get_settings()
    hybrid = HybridRetriever(
        db,
        embedding_provider,
        reranker,
        candidate_pool_size=max(settings.retrieval_candidate_pool_size, top_k),
        rrf_k=settings.retrieval_rrf_k,
        enable_reranking=True,
    )
    t0 = time.perf_counter()
    hybrid_results = hybrid.retrieve(document_id, query, top_k)
    hybrid_ms = (time.perf_counter() - t0) * 1000

    baseline_ids = [c.chunk_id for c in baseline_results]
    hybrid_ids = [c.chunk_id for c in hybrid_results]
    overlap = len(set(baseline_ids) & set(hybrid_ids))

    baseline_rank = literal_hit_rank(query, baseline_results)
    hybrid_rank = literal_hit_rank(query, hybrid_results)

    print(f"  {DIM}baseline (vector-only):{RESET} {baseline_ms:.1f} ms, {len(baseline_results)} chunks,"
          f" literal-term hit at rank {baseline_rank or 'none in top-k'}")
    print(f"  {DIM}hybrid + rerank:       {RESET} {hybrid_ms:.1f} ms, {len(hybrid_results)} chunks,"
          f" literal-term hit at rank {hybrid_rank or 'none in top-k'}")
    print(f"  {DIM}top-{top_k} overlap: {overlap}/{top_k} chunks in common{RESET}")

    if baseline_rank is None and hybrid_rank is not None:
        print(f"  {GREEN}hybrid surfaced a literal term match that vector-only missed entirely{RESET}")
    elif hybrid_rank is not None and baseline_rank is not None and hybrid_rank < baseline_rank:
        print(f"  {GREEN}hybrid ranked the literal term match earlier ({hybrid_rank} vs {baseline_rank}){RESET}")
    elif hybrid_rank is not None and baseline_rank is not None and hybrid_rank > baseline_rank:
        print(f"  {YELLOW}vector-only ranked the literal term match earlier this time ({baseline_rank} vs {hybrid_rank}){RESET}")

    # -- structural invariants (real assertions, not accuracy claims) -------
    check(f"[{category}] hybrid returned <= top_k results", len(hybrid_results) <= top_k)
    check(f"[{category}] every hybrid result has a fusion_score", all(c.fusion_score is not None for c in hybrid_results))
    check(f"[{category}] every hybrid result has a rerank_score", all(c.rerank_score is not None for c in hybrid_results))

    debug = hybrid.last_debug
    check(f"[{category}] hybrid retriever produced debug info", debug is not None)
    if debug is not None:
        selected_in_debug = {c.chunk_id for c in debug.candidates if c.selected}
        check(
            f"[{category}] debug 'selected' set matches returned chunk ids",
            selected_in_debug == set(hybrid_ids),
            f"{selected_in_debug} vs {set(hybrid_ids)}",
        )
        fusion_scores = [c.fusion_score for c in debug.candidates if c.fusion_score is not None]
        check(
            f"[{category}] candidate pool is sorted by descending rerank/fusion score",
            all(a.rerank_score is None or b.rerank_score is None or a.rerank_score >= b.rerank_score
                for a, b in zip(debug.candidates, debug.candidates[1:])),
        )


def main() -> int:
    settings = get_settings()
    db = SessionLocal()
    try:
        document_id = sys.argv[1] if len(sys.argv) > 1 else None
        if document_id:
            document = db.get(Document, document_id)
        else:
            document = db.scalar(
                select(Document)
                .where(Document.rag_status == RagStatus.INDEXED, Document.chunk_count > 0)
                .order_by(Document.uploaded_at.desc())
            )

        if document is None:
            print(f"{RED}No indexed document found.{RESET}")
            print("Upload and let a document finish indexing first (via the running app), "
                  "or pass a specific --document-id.")
            return 1

        print(f"document: {document.title or document.original_filename} ({document.id})")
        print(f"chunks: {document.chunk_count}  embedding_provider: {document.embedding_provider}")
        print(f"retrieval config: candidate_pool_size={settings.retrieval_candidate_pool_size} "
              f"rrf_k={settings.retrieval_rrf_k} reranker={settings.rerank_provider}")

        index = build_bm25_and_chunks(db, document.id)
        queries = generate_queries(db, document.id, index)
        reranker = get_reranker(settings)

        print(f"\n{'=' * 72}\ngenerated test queries (derived from this document's own text)\n{'=' * 72}")
        for category, query in queries.items():
            print(f"  {category:28s} {query!r}" if query else f"  {category:28s} {YELLOW}(could not derive one){RESET}")

        print(f"\n{'=' * 72}\nbaseline (vector-only) vs. hybrid + RRF + rerank\n{'=' * 72}")
        for category, query in queries.items():
            if not query:
                print(f"\n{BOLD}{category}{RESET}  {YELLOW}skipped — no suitable query found in this document{RESET}")
                continue
            run_category(db, document.id, category, query, settings.retrieval_top_k, reranker)

        total = passed + failed
        colour = GREEN if failed == 0 else RED
        print(f"\n{'=' * 72}")
        print(f"{colour}{passed}/{total} structural checks passed{RESET}")
        print(
            f"{DIM}Note: the metrics above (literal-term hit rank, top-k overlap, timings) are "
            f"measured directly from this run against this document. No accuracy/precision/recall "
            f"numbers are reported because no labelled relevance judgments exist for it.{RESET}"
        )
        print(f"{'=' * 72}")
        return 1 if failed else 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
