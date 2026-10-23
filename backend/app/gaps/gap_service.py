"""Phase 8 — Research Gap Detection Service.

Evidence-first gap detection pipeline:

    Workspace
     ↓
    Selected Papers (must be indexed)
     ↓
    Per-Paper Evidence Extraction (reuses existing retrieval)
     ↓
    Cross-Paper Pattern Analysis (LLM-assisted synthesis over structured evidence)
     ↓
    Candidate Validation (Pydantic schema + evidence checks; unsupported gaps rejected)
     ↓
    Persisted ResearchGap → Claim → Evidence

Reuses existing infrastructure — no second retrieval system:
- `build_retriever` (hybrid retrieval), `get_llm_provider`
- `evidence_service.create_evidence` / `evidence_from_retrieved_chunk`
- `claim_service.create_claim`
- `resolve_workspace_documents` (ownership + workspace scoping)

Confidence is computed from the breadth of verified evidence, never taken
from the LLM. The system never claims novelty: gaps are "potential research
gaps identified from the analyzed papers".
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.evidence import claim_service, evidence_service
from app.models import (
    Claim,
    ClaimType,
    Document,
    Evidence,
    GapClaim,
    GapConfidence,
    GapEvidence,
    GapStatus,
    GapType,
    RagStatus,
    ResearchGap,
    SupportStatus,
)
from app.rag.factory import get_llm_provider
from app.rag.interfaces import LLMMessage, RetrievedChunk
from app.rag.service import build_retriever
from app.rag.workspace_retrieval import resolve_workspace_documents

logger = logging.getLogger(__name__)


class GapDetectionError(RuntimeError):
    """User-facing error for gap detection failures."""


class GapNotFoundError(LookupError):
    pass


# ---------------------------------------------------------------------------
# Analysis dimensions for gap detection (targeted retrieval queries)
# ---------------------------------------------------------------------------

_GAP_EXTRACTION_QUERIES = [
    ("limitations", "What limitations does this paper report or discuss?"),
    ("future_work", "What future work or directions does this paper suggest?"),
    ("methodology", "What methodology or approach does this paper use?"),
    ("dataset", "What dataset(s) does this paper use for experiments or evaluation?"),
    ("evaluation_metrics", "What evaluation metrics does this paper use to measure performance?"),
    ("challenges", "What challenges or difficulties does this paper identify?"),
    ("baselines", "What baseline methods or systems does this paper compare against?"),
    ("main_results", "What are the main quantitative or qualitative results reported?"),
]


# ---------------------------------------------------------------------------
# LLM prompt for cross-paper gap synthesis
# ---------------------------------------------------------------------------

_ASPECT_KEYS = tuple(k for k, _ in _GAP_EXTRACTION_QUERIES)

_GAP_SYNTHESIS_SYSTEM = """\
You are a research analysis engine inside ResearchPilot. You are given structured \
evidence snippets extracted from several research papers in a workspace. Identify \
POTENTIAL research gaps by analyzing patterns ACROSS this evidence.

Rules:
1. Every gap MUST cite the specific evidence it rests on using paper_index and aspect \
   (matching the [Paper N, aspect] labels you were given).
2. Do NOT invent papers, datasets, metrics, results, limitations or future work. If the \
   evidence does not support a gap, do not report it.
3. Use cautious language. You cannot establish novelty; say "potential research gap".
4. For contradictory findings, cite evidence from BOTH conflicting papers and do not \
   decide who is right.
5. Confidence is computed by the system; do not output a confidence score.
6. Return ONLY a valid JSON array. No markdown, no commentary.

Patterns to look for: repeated limitations across papers; methodological gaps or missing \
comparisons; dataset/domain/population limitations; evaluation gaps; contradictory \
findings; explicit future-work themes shared by several papers; reproducibility concerns.

Each array element:
{
  "title": "Short gap title",
  "description": "2-3 sentences describing the gap, grounded in the evidence",
  "gap_type": one of ["methodological","dataset","evaluation","domain","population",
"comparative","theoretical","reproducibility","contradictory","future_work"],
  "confidence_reasoning": "Optional: which patterns in the evidence support this",
  "supporting_evidence": [{"paper_index": 0, "aspect": "limitations"}, ...],
  "research_direction": "Optional: one sentence suggesting a direction to address the gap"
}

If there are no supportable gaps return: []
"""


# ---------------------------------------------------------------------------
# Structured output validation (never trust raw LLM JSON)
# ---------------------------------------------------------------------------


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    paper_index: int = Field(ge=0)
    aspect: str

    @field_validator("aspect")
    @classmethod
    def _known_aspect(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in _ASPECT_KEYS:
            raise ValueError(f"unknown aspect {v!r}")
        return v


class GapCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=3, max_length=500)
    description: str = Field(min_length=10)
    gap_type: str
    confidence_reasoning: str | None = None
    supporting_evidence: list[EvidenceRef] = Field(min_length=1)
    research_direction: str | None = None

    @field_validator("title", "description")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @field_validator("gap_type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in GapType.ALL:
            raise ValueError(f"unknown gap_type {v!r}")
        return v

    @field_validator("research_direction", "confidence_reasoning")
    @classmethod
    def _blank_to_none(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None


def _load_json_array(raw_text: str) -> list:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.split("\n") if not line.strip().startswith("```")
        ).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            raise GapDetectionError("The language model returned malformed output (not a JSON array).")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raise GapDetectionError("The language model returned malformed output (invalid JSON).")
    if not isinstance(data, list):
        raise GapDetectionError("The language model returned malformed output (expected a JSON array).")
    return data


def _parse_gap_candidates(raw_text: str) -> list[GapCandidate]:
    """Validate LLM output. Unparseable output raises GapDetectionError;
    individually malformed candidates are dropped (and logged)."""
    valid: list[GapCandidate] = []
    for item in _load_json_array(raw_text):
        try:
            valid.append(GapCandidate.model_validate(item))
        except ValidationError as exc:
            logger.warning("gap detection: rejected malformed candidate: %s", exc.errors()[:2])
    return valid


# ---------------------------------------------------------------------------
# Transparent confidence
# ---------------------------------------------------------------------------


def compute_confidence(
    supporting_paper_count: int,
    total_papers: int,
    aspects: list[str] | None = None,
) -> tuple[str, str]:
    """Confidence from evidence breadth only (never from the LLM).

    HIGH   — verified evidence from >= 3 independent papers, or from every
             analyzed paper when the workspace has 2 papers.
    MEDIUM — verified evidence from 2 papers; needs further validation.
    LOW    — evidence from a single paper (or none): indirect/speculative.
    """
    n, total = max(supporting_paper_count, 0), max(total_papers, 0)
    aspect_note = ""
    if aspects:
        aspect_note = " Evidence drawn from: " + ", ".join(sorted(set(aspects))) + "."
    if n >= 3 or (n >= 2 and n == total):
        level = GapConfidence.HIGH
        reason = f"Supported by evidence from {n} of {total} analyzed papers (multiple independent sources)."
    elif n == 2:
        level = GapConfidence.MEDIUM
        reason = (
            f"Supported by evidence from 2 of {total} analyzed papers; "
            "additional validation is needed."
        )
    elif n == 1:
        level = GapConfidence.LOW
        reason = (
            f"Supported by evidence from only 1 of {total} analyzed papers; "
            "treat as speculative until corroborated."
        )
    else:
        level = GapConfidence.LOW
        reason = "No verified supporting evidence."
    return level, reason + aspect_note + " Novelty requires verification against the broader literature."


# ---------------------------------------------------------------------------
# Evidence extraction + synthesis prompt
# ---------------------------------------------------------------------------


def _extract_paper_evidence(
    settings: Settings, retriever, document: Document
) -> dict[str, list[RetrievedChunk]]:
    """Targeted retrieval per gap-relevant aspect. {aspect: [chunks]}."""
    result: dict[str, list[RetrievedChunk]] = {}
    for aspect, query in _GAP_EXTRACTION_QUERIES:
        try:
            chunks = retriever.retrieve(document.id, query, settings.comparison_top_k)
            # Defensive: only keep chunks that really belong to this paper.
            result[aspect] = [
                c for c in chunks
                if c.document_id == document.id and (c.child_text or "").strip()
            ]
        except Exception:
            logger.warning(
                "gap detection: retrieval failed for doc %s aspect %s", document.id, aspect
            )
            result[aspect] = []
    return result


def _build_synthesis_prompt(
    papers: list[Document],
    all_evidence: dict[str, dict[str, list[RetrievedChunk]]],
) -> str:
    per_aspect = 3 if len(papers) <= 5 else 2  # keep prompt size bounded
    sections = []
    for i, paper in enumerate(papers):
        title = paper.title or paper.original_filename
        block = f"=== Paper {i}: {title} ===\n"
        for aspect in _ASPECT_KEYS:
            chunks = all_evidence.get(paper.id, {}).get(aspect, [])
            if not chunks:
                block += f"\n[Paper {i}, {aspect}]: Not found in paper\n"
                continue
            snippets = [
                " ".join((c.child_text or "").split())[:350]
                for c in chunks[:per_aspect]
            ]
            block += f"\n[Paper {i}, {aspect}]:\n" + "\n".join(f"  - {t}" for t in snippets) + "\n"
        sections.append(block)
    return (
        "Structured evidence from the papers in a workspace follows. Identify potential "
        "research gaps across papers: repeated limitations, methodological differences, "
        "dataset limitations, evaluation gaps, contradictions and shared future-work themes.\n\n"
        + "\n".join(sections)
        + "\n\nReturn a JSON array only."
    )


# ---------------------------------------------------------------------------
# Validation + persistence
# ---------------------------------------------------------------------------


@dataclass
class _ResolvedRef:
    paper: Document
    aspect: str
    chunk: RetrievedChunk


def _resolve_refs(
    candidate: GapCandidate,
    papers: list[Document],
    all_evidence: dict[str, dict[str, list[RetrievedChunk]]],
) -> list[_ResolvedRef]:
    """Map the candidate's evidence references onto real retrieved chunks.
    References to nonexistent papers/aspects/empty retrievals are discarded."""
    workspace_doc_ids = {p.id for p in papers}
    seen: set[tuple[int, str]] = set()
    resolved: list[_ResolvedRef] = []
    for ref in candidate.supporting_evidence:
        key = (ref.paper_index, ref.aspect)
        if key in seen or ref.paper_index >= len(papers):
            continue
        seen.add(key)
        paper = papers[ref.paper_index]
        chunks = all_evidence.get(paper.id, {}).get(ref.aspect, [])
        # Evidence must belong to a document in this workspace (and this paper)
        chunks = [c for c in chunks if c.document_id == paper.id and c.document_id in workspace_doc_ids]
        if chunks:
            resolved.append(_ResolvedRef(paper=paper, aspect=ref.aspect, chunk=chunks[0]))
    return resolved


def _persist_gap(
    db: Session,
    workspace_id: str,
    owner_id: str,
    candidate: GapCandidate,
    papers: list[Document],
    all_evidence: dict[str, dict[str, list[RetrievedChunk]]],
) -> ResearchGap | None:
    """Validate then persist one gap. Returns None (persisting nothing) if
    the candidate lacks verified evidence."""
    refs = _resolve_refs(candidate, papers, all_evidence)
    if not refs:
        return None  # unsupported gap — never persisted

    paper_ids = list(dict.fromkeys(r.paper.id for r in refs))
    # A contradiction needs both sides shown.
    if candidate.gap_type == GapType.CONTRADICTORY and len(paper_ids) < 2:
        logger.info("gap detection: rejected contradictory gap with a single source: %s", candidate.title)
        return None

    confidence, reasoning = compute_confidence(
        len(paper_ids), len(papers), [r.aspect for r in refs]
    )

    gap = ResearchGap(
        workspace_id=workspace_id,
        owner_id=owner_id,
        title=candidate.title,
        description=candidate.description,
        gap_type=candidate.gap_type,
        confidence=confidence,
        confidence_reasoning=reasoning,
        status=GapStatus.IDENTIFIED,
        research_direction=candidate.research_direction,
        source_paper_ids=paper_ids,
        supporting_paper_count=len(paper_ids),
    )
    db.add(gap)
    db.flush()

    for r in refs:
        kwargs = evidence_service.evidence_from_retrieved_chunk(r.chunk)
        snippet = " ".join((kwargs["snippet"] or "").split())
        # Claim = the paper's own statement (snippet), attached to its Evidence.
        claim = claim_service.create_claim(
            db,
            document_id=r.paper.id,
            claim_text=snippet[:1000],
            claim_type=ClaimType.REPORTED_FACT,
            support_status=SupportStatus.UNCERTAIN,
            workspace_id=workspace_id,
            commit=False,
        )
        ev = evidence_service.create_evidence(
            db, claim_id=claim.id, commit=False, **kwargs
        )
        db.add(GapClaim(gap_id=gap.id, claim_id=claim.id))
        db.add(GapEvidence(gap_id=gap.id, evidence_id=ev.id, role=r.aspect))
    db.flush()
    return gap


def _delete_gaps_no_commit(db: Session, gaps: list[ResearchGap]) -> None:
    """Remove gaps plus the Claims (and, via cascade, Evidence) that were
    created solely to support them."""
    claim_ids = [
        gc.claim_id for g in gaps for gc in g.gap_claim_items
    ]
    for g in gaps:
        db.delete(g)
    db.flush()
    for cid in claim_ids:
        claim = db.get(Claim, cid)
        if claim is not None:
            db.delete(claim)
    db.flush()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class GapAnalysisOutcome:
    gaps: list[ResearchGap] = field(default_factory=list)
    papers_analyzed: int = 0
    candidates_rejected: int = 0


def analyze_gaps(
    db: Session,
    workspace_id: str,
    owner_id: str,
    settings: Settings | None = None,
) -> GapAnalysisOutcome:
    """Run the pipeline. Previous gaps for the workspace are replaced only
    after the new analysis succeeds."""
    settings = settings or get_settings()

    # Ownership + workspace scoping (raises WorkspaceRetrievalError)
    workspace_documents = resolve_workspace_documents(db, workspace_id, owner_id)
    indexed = [
        d for d in workspace_documents
        if d.rag_status == RagStatus.INDEXED and d.chunk_count
    ]
    if len(indexed) < 2:
        raise GapDetectionError(
            "At least 2 indexed papers are required for gap detection. "
            f"This workspace has {len(indexed)} indexed paper(s)."
        )

    retriever = build_retriever(db, settings)
    llm = get_llm_provider(settings)
    if llm.name == "local":
        raise GapDetectionError(
            "Gap detection requires a generative LLM to synthesize cross-paper patterns. "
            "The offline local provider is strictly extractive. Please configure "
            "LLM_PROVIDER=gemini, openai or anthropic in your .env file."
        )

    all_evidence = {d.id: _extract_paper_evidence(settings, retriever, d) for d in indexed}

    prompt = _build_synthesis_prompt(indexed, all_evidence)
    try:
        raw = llm.generate(_GAP_SYNTHESIS_SYSTEM, [LLMMessage(role="user", content=prompt)])
    except Exception as exc:
        logger.exception("gap detection: LLM synthesis failed")
        raise GapDetectionError(f"Gap synthesis failed: {exc}")

    candidates = _parse_gap_candidates(raw or "[]")

    try:
        existing = list(
            db.scalars(
                select(ResearchGap).where(
                    ResearchGap.workspace_id == workspace_id,
                    ResearchGap.owner_id == owner_id,
                )
            ).all()
        )
        _delete_gaps_no_commit(db, existing)

        gaps: list[ResearchGap] = []
        rejected = 0
        for cand in candidates:
            gap = _persist_gap(db, workspace_id, owner_id, cand, indexed, all_evidence)
            if gap is None:
                rejected += 1
            else:
                gaps.append(gap)
        db.commit()
    except Exception:
        db.rollback()
        raise

    for g in gaps:
        db.refresh(g)
    return GapAnalysisOutcome(
        gaps=gaps, papers_analyzed=len(indexed), candidates_rejected=rejected
    )


def list_gaps(db: Session, workspace_id: str, owner_id: str) -> list[ResearchGap]:
    resolve_workspace_documents(db, workspace_id, owner_id)  # ownership check
    return list(
        db.scalars(
            select(ResearchGap)
            .where(
                ResearchGap.workspace_id == workspace_id,
                ResearchGap.owner_id == owner_id,
            )
            .order_by(ResearchGap.created_at.desc())
        ).all()
    )


def get_gap(db: Session, gap_id: str, owner_id: str) -> ResearchGap:
    gap = db.scalar(
        select(ResearchGap).where(
            ResearchGap.id == gap_id, ResearchGap.owner_id == owner_id
        )
    )
    if gap is None:
        raise GapNotFoundError(f"gap {gap_id!r} not found")
    return gap


def get_gap_evidence(db: Session, gap_id: str, owner_id: str) -> list[GapEvidence]:
    gap = get_gap(db, gap_id, owner_id)
    return list(gap.gap_evidence_items)


def get_gap_claims(db: Session, gap_id: str, owner_id: str) -> list[GapClaim]:
    gap = get_gap(db, gap_id, owner_id)
    return list(gap.gap_claim_items)
