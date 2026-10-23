"""Evidence & Research Object Layer.

The foundational core described in the Phase 5 brief: a normalized Evidence
model, a paper-level metadata model, and a Claim -> Evidence ledger, sitting
on top of (never replacing) the existing Phase 2-4 pipeline:

    Document -> Document Structure -> Chunk -> Retrieval Result
             -> Evidence -> Claim -> Answer / Citation

This package is the only place outside `app/models.py` that should construct
or query `Evidence` / `Claim` / `ResearchPaper` rows, so later phases
(citation verification, multi-paper comparison, literature discovery,
research gap detection, ideation, writing assistance, similarity checking)
can depend on `app.evidence` without depending on the retrieval
implementation or the chat service directly.
"""
from app.evidence.evidence_service import (
    EvidenceNotFoundError,
    attach_evidence_to_claim,
    create_evidence,
    evidence_from_retrieved_chunk,
    get_evidence,
    list_evidence_for_document,
    persist_evidence_from_retrieved_chunks,
)
from app.evidence.claim_service import (
    ClaimNotFoundError,
    create_claim,
    get_claim,
    list_claims_for_document,
)
from app.evidence.paper_service import (
    build_arxiv_url,
    build_doi_url,
    get_or_create_paper,
    get_paper_for_document,
)

__all__ = [
    "EvidenceNotFoundError",
    "attach_evidence_to_claim",
    "create_evidence",
    "evidence_from_retrieved_chunk",
    "get_evidence",
    "list_evidence_for_document",
    "persist_evidence_from_retrieved_chunks",
    "ClaimNotFoundError",
    "create_claim",
    "get_claim",
    "list_claims_for_document",
    "build_arxiv_url",
    "build_doi_url",
    "get_or_create_paper",
    "get_paper_for_document",
]
