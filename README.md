# ResearchPilot

**An evidence-first AI research workspace.** Upload academic PDFs, chat with
them, compare several papers side by side, and surface potential research
gaps — with every AI output traced back to a source passage in the paper.

> If the papers don't support an answer, ResearchPilot says so instead of
> guessing. Every claim carries its citation: paper, page, section and the
> exact retrieved passage.

Built for the Google Gemini hackathon. Gemini does the real work — embeddings
for retrieval and the LLM for grounded answers, comparisons and gap
synthesis — while a fully offline local provider keeps the app working.

---

## Features

| Feature | What it does |
|---|---|
| **Document intelligence** | Background parsing of PDFs into pages, sections, paragraphs, tables, figures, captions and references, with OCR for scanned pages. Browsable in the *Explore* tab. |
| **Paper chat** | Ask questions about one paper. Answers stream token by token with inline `[N]` citations; clicking one jumps the PDF viewer to that page. |
| **Workspace chat** | Group up to 5 papers in a workspace and ask questions across all of them. Citations name which paper each passage came from. |
| **Comparison matrix** | Compare 2–5 papers across dimensions (methodology, dataset, results, limitations, or your own free-text ones). Each cell is a short grounded answer with its evidence. |
| **Research gaps** | Cross-paper synthesis of *potential* research gaps (repeated limitations, future work, contradictions). Gaps without retrieved evidence are rejected; confidence is computed from evidence breadth, never taken from the LLM. |
| **Evidence & claims** | Every answer, comparison cell and gap is stored as Claim → Evidence rows pointing at real chunks/pages, so any output can be audited. |

## Architecture

```mermaid
graph TB
    subgraph Frontend ["Frontend · Next.js 14"]
        Lib["Document library"]
        WS["Workspace: PDF viewer + Chat / Explore / Compare / Gaps"]
    end

    subgraph Backend ["Backend · FastAPI"]
        API["REST API + SSE streaming"]
        Worker["DB-backed worker queue"]
        Parse["PDF parsing + OCR"]
        Index["Chunking + embeddings"]
        Retr["Hybrid retrieval: vector + BM25 → RRF → rerank"]
        Svc["Chat · Workspace chat · Comparison · Gap services"]
        Ev["Evidence & claim layer"]
        Factory["Provider factory: gemini · local · openai · anthropic · voyage"]
    end

    DB[("SQLite (dev) / PostgreSQL + pgvector (prod)")]

    Frontend <-->|HTTP + SSE| API
    API --> Svc --> Retr --> DB
    Svc --> Ev --> DB
    Worker --> Parse --> Index --> DB
    Index -.-> Factory
    Svc -.-> Factory
```

**Ingestion** (automatic after upload): `UPLOADED → PROCESSING → READY` (parse),
then RAG indexing `INDEXING → INDEXED`. Elements are packed into ~1400-token
*parent* chunks (never crossing a section) and split into ~320-token
sentence-aligned *child* chunks. Only children are embedded and searched; a
match is expanded to its parent for fuller context.

**Answering a question:** query preprocessing → vector search and BM25 keyword
search in parallel → Reciprocal Rank Fusion → reranking → top-K passages as
`[Source N] (Page P, Section S)` blocks → LLM (streamed) → citations parsed and
persisted as evidence. `RETRIEVAL_MODE=vector_only` skips BM25/fusion/rerank.

**Design rules the code follows**

- **Providers only via [`app/rag/factory.py`](backend/app/rag/factory.py).** Nothing else
  imports a concrete provider; switching vendor is an `.env` change.
- **Degrade, don't crash.** A hosted provider with a missing key or package
  falls back to the local one and logs why; a failing optional step records
  an error and lets the request succeed.
- **One retrieval and one citation implementation**, reused by every feature
  (`build_retriever`, `_parse_citations`, `evidence_service`, `claim_service`).
- **Additive migrations only.** [`app/migrate.py`](backend/app/migrate.py) adds missing
  columns/tables on startup. No Alembic, nothing destructive.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14 (App Router), TypeScript, Tailwind, Radix/shadcn-style components |
| Backend | FastAPI, Pydantic v2, SQLAlchemy 2.0 |
| Database | SQLite by default; PostgreSQL + pgvector optional (just change `DATABASE_URL`) |
| Parsing | PyMuPDF → pdfplumber (optional Docling), Tesseract OCR |
| Embeddings | **Gemini** `gemini-embedding-001` · local hashing embedder · OpenAI · Voyage |
| LLM | **Gemini** `gemini-3.8-flash` · local extractive · Anthropic · OpenAI |
| Retrieval | Hybrid vector + BM25, Reciprocal Rank Fusion, local reranker |

---

## Quick start

Requirements: Python 3.12+, Node 18+. No database server is needed (SQLite).
Tesseract is optional (only for scanned PDFs).

**1. Backend**

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate     macOS/Linux: source venv/bin/activate
pip install -r requirements-local.txt
cp .env.example .env
```

In `backend/.env`, turn on Gemini (or leave the defaults to run fully offline
with the local providers):

```env
EMBEDDING_PROVIDER=gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-google-ai-studio-key
```

```bash
uvicorn app.main:app --reload --port 8000
```

**2. Frontend**

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open http://localhost:3000, upload a PDF, wait for processing and indexing to
finish (a few seconds per paper), then open it and start asking.

| URL | |
|---|---|
| http://localhost:3000 | App |
| http://localhost:8000/docs | Swagger / OpenAPI |
| http://localhost:8000/api/system/rag | Active providers, models, stale document count |
| http://localhost:8000/api/system/processing | Parser backends / OCR availability |

**Optional extras**

- **PostgreSQL + pgvector** (native ANN vector search instead of SQLite's
  brute-force scan): `pip install -r requirements.txt`, then set
  `DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/researchpilot`.
  The `vector` extension and all tables are created on startup.
- **Docling** (better layout recovery, ~2–3 GB of ML dependencies):
  `pip install -r requirements-docling.txt`. It is picked up automatically.
- **Separate worker process:** set `WORKER_COUNT=0` and run `python -m app.worker`.

---

## Using Gemini

Three variables turn it on:

```env
EMBEDDING_PROVIDER=gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-google-ai-studio-key
```

| Optional variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Override the embedding model |
| `LLM_MODEL` | `gemini-3.8-flash` | Override the generation model (e.g. a Flash-Lite model) |
| `GEMINI_MAX_CONCURRENCY` | `4` | Max in-flight Gemini calls per process, shared by embeddings and chat |
| `GEMINI_EMBED_TEXTS_PER_MINUTE` | `90` | Paces embedding traffic under the free tier's 100/min (every text in a batch counts). `0` disables pacing on a paid tier. |

Behaviour:

- Embeddings use task type `RETRIEVAL_DOCUMENT` for chunks and
  `RETRIEVAL_QUERY` for questions, requested at 768 dimensions and
  L2-normalised.
- 429 / 5xx / timeouts are retried up to 4 times with exponential backoff and
  jitter; on a 429 the wait honours Gemini's own "retry in Ns" hint (capped
  at 90s). Other errors (bad key, bad request) fail immediately. Streaming
  only retries before the first token, so text is never duplicated.
- On the free tier, re-indexing is paced to ~90 chunks per minute, so a
  100-chunk paper takes a bit over a minute. That's expected, not a hang.
- If the key or `google-genai` package is missing, the app logs an error and
  falls back to the local provider. `GET /api/system/rag` shows which
  provider is *actually* active.

**Check your key works** — the only script that calls the real API. It never
prints the key and exits non-zero on any failure:

```bash
cd backend
python -m scripts.gemini_smoke    # uses GEMINI_API_KEY from the environment or backend/.env
```

It embeds three sentences (two related, one not), prints the dimensions and
cosine similarities, makes one generation call, and prints the model names.

**Re-index after switching provider.** Vectors from different embedding
models aren't comparable, so documents indexed before the switch are reported
as `stale_documents` by `GET /api/system/rag`. Re-index them one at a time:

```bash
cd backend
python -m scripts.reindex_all --dry-run   # list stale documents
python -m scripts.reindex_all             # re-index them
```

Only `READY` documents are touched. A document that fails keeps its old
chunks; re-run to retry. A single document can also be re-indexed with
`POST /api/documents/{id}/reindex`.

## Configuration reference

All settings live in [`backend/app/config.py`](backend/app/config.py) and are read
from environment variables / `backend/.env`.
[`backend/.env.example`](backend/.env.example) documents each one. The most useful:

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./researchpilot.db` | Postgres: `postgresql+psycopg://user:pass@host:5432/db` |
| `EMBEDDING_PROVIDER` | `local` | `local` · `gemini` · `openai` · `voyage` |
| `LLM_PROVIDER` | `local` | `local` · `gemini` · `openai` · `anthropic` |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `VOYAGE_API_KEY` | — | Only the one for your chosen provider |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid` or `vector_only` |
| `RETRIEVAL_TOP_K` | `6` | Passages handed to the LLM per chat answer |
| `RETRIEVAL_MIN_SCORE` | `0.0` (off) | Optional pre-LLM gate: a paper whose best retrieval score is below this gets "not found" without an LLM call. Pick it with `python -m scripts.print_score_distribution <document_id> questions.txt` |
| `QUERY_REWRITE_ENABLED` | auto | Workspace chat: rewrite the question into one sub-query per paper (auto = on for Gemini only) |
| `WORKSPACE_MIN_CHUNKS_PER_PAPER` | `2` | Every workspace paper contributes at least this many passages before any paper gets more |
| `PARSER_BACKEND` | `auto` | `auto` · `docling` · `pymupdf` · `pdfplumber` |
| `ENABLE_OCR` | `true` | Needs the `tesseract` binary installed |
| `WORKER_COUNT` | `2` | In-process worker threads; `0` to use a separate worker |
| `DEBUG_MODE` | `false` | Shows the Retrieval Debug panel (every candidate's scores). Dev only. |
| `NEXT_PUBLIC_API_URL` | — | Frontend → backend URL (`frontend/.env.local`) |

## API overview

Full interactive docs at `/docs`. Main groups:

| Area | Endpoints |
|---|---|
| Documents | `POST /api/documents/upload`, `GET /api/documents`, `GET /api/documents/{id}` (+ `/status`, `/sections`, `/pages`, `/elements`, `/references`), `POST .../reprocess`, `DELETE` |
| Indexing | `GET /api/documents/{id}/rag-status`, `POST /api/documents/{id}/reindex` |
| Paper chat | `POST /api/documents/{id}/chat/ask` (SSE), conversations, suggested questions |
| Workspaces | `POST/GET /api/workspaces`, add/remove documents, `POST /api/workspaces/{id}/chat/ask` (SSE) |
| Comparison | `GET /api/comparisons/default-dimensions`, `POST/GET /api/workspaces/{id}/comparisons`, `GET/DELETE /api/comparisons/{id}` |
| Gaps | `POST /api/workspaces/{id}/gaps/analyze`, `GET /api/workspaces/{id}/gaps`, `GET /api/gaps/{id}` (+ `/evidence`, `/claims`) |
| Evidence | `GET /api/evidence/{id}`, `GET /api/claims/{id}` (+ `/evidence`), per-document evidence/claims/paper metadata |
| System | `GET /health`, `GET /api/system/rag`, `GET /api/system/processing`, `GET /api/dashboard` |

## Project structure

```
.
├── backend/
│   ├── .env.example            Every setting, documented (copy to .env)
│   ├── requirements-local.txt  SQLite install (default)
│   ├── requirements.txt        + PostgreSQL driver
│   ├── app/
│   │   ├── main.py             FastAPI app, startup (migrations, workers), /api/system/*
│   │   ├── config.py           Settings (env-driven)
│   │   ├── models.py           SQLAlchemy models
│   │   ├── migrate.py          Additive startup migrations
│   │   ├── worker.py           DB-backed queue: parse → index
│   │   ├── processing/         PDF parsing backends, OCR, structure recovery
│   │   ├── rag/                Chunking, indexing, retrieval, chat services
│   │   │   ├── factory.py      The only place concrete providers are chosen
│   │   │   ├── gemini_limits.py  Shared Gemini retry + concurrency limiter
│   │   │   ├── embeddings/     gemini · local · openai · voyage
│   │   │   └── llm/            gemini · local · openai · anthropic
│   │   ├── evidence/           Evidence, claim and paper-metadata services
│   │   ├── comparison/         Comparison matrix service + dimensions
│   │   ├── gaps/               Research gap detection
│   │   └── routers/            HTTP endpoints
│   ├── scripts/                gemini_smoke.py, reindex_all.py
│   ├── tests/                  unittest suite (in-memory SQLite, mocked providers)
│   ├── eval_retrieval.py       Hybrid vs vector-only comparison on an indexed doc
│   └── verify_pipeline.py      Parser check against real PDFs
└── frontend/
    ├── .env.example            NEXT_PUBLIC_API_URL (copy to .env.local)
    ├── app/                    Pages: /, /dashboard, /documents, /workspace/[id], /settings
    ├── components/workspace/   PDF, chat, explore, comparison, gaps, evidence panels
    └── lib/                    API client + shared types
```

## Testing

```bash
cd backend
python -m unittest discover -s tests -p "test_*.py"
```

The suite uses an in-memory SQLite database and `MagicMock` providers. It
**never calls the real Gemini API** (use `scripts.gemini_smoke` for that).

Other checks against real data:

```bash
python tests/make_test_pdfs.py && python verify_pipeline.py   # parser on generated PDFs (needs reportlab)
python eval_retrieval.py [document_id]                        # hybrid vs vector-only on an indexed paper
```

