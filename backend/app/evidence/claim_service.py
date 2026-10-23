"""Claim service — Task 4 of the Evidence & Research Object Layer.

Deliberately thin: this task establishes the Claim data model and its
relationship to Evidence, not claim *verification*. Verifying whether a
claim's attached evidence actually supports it is future work built on top
of this foundation.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Claim, ClaimType, Document, SupportStatus


class ClaimNotFoundError(LookupError):
    pass


class ClaimValidationError(ValueError):
    pass


def create_claim(
    db: Session,
    *,
    document_id: str,
    claim_text: str,
    claim_type: str = ClaimType.UNKNOWN,
    support_status: str = SupportStatus.UNCERTAIN,
    confidence: float | None = None,
    workspace_id: str | None = None,
    conversation_id: str | None = None,
    message_id: str | None = None,
    commit: bool = True,
) -> Claim:
    document = db.get(Document, document_id)
    if document is None:
        raise ClaimValidationError(f"document {document_id!r} does not exist")
    if claim_type not in ClaimType.ALL:
        raise ClaimValidationError(f"unknown claim_type {claim_type!r}")
    if support_status not in SupportStatus.ALL:
        raise ClaimValidationError(f"unknown support_status {support_status!r}")
    if not (claim_text or "").strip():
        raise ClaimValidationError("claim_text cannot be empty")

    claim = Claim(
        document_id=document_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        message_id=message_id,
        claim_text=claim_text.strip(),
        claim_type=claim_type,
        support_status=support_status,
        confidence=confidence,
    )
    db.add(claim)
    if commit:
        db.commit()
        db.refresh(claim)
    else:
        db.flush()
    return claim


def get_claim(db: Session, claim_id: str) -> Claim:
    claim = db.get(Claim, claim_id)
    if claim is None:
        raise ClaimNotFoundError(f"claim {claim_id!r} not found")
    return claim


def list_claims_for_document(db: Session, document_id: str, limit: int = 100, offset: int = 0) -> list[Claim]:
    return list(
        db.scalars(
            select(Claim)
            .where(Claim.document_id == document_id)
            .order_by(Claim.created_at.desc())
            .offset(offset)
            .limit(limit)
        ).all()
    )
