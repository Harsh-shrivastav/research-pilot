"""Paper metadata service — Task 3 + Task 9 of the Evidence & Research Object
Layer.

Seeds a `ResearchPaper` row from what Phase 2 already extracted
(`Document.title` / `Document.authors`) and never invents anything beyond
that: no DOI, no URL, no year is fabricated. `build_doi_url` / `build_arxiv_url`
only ever construct a link from a value that is actually present.
"""
from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, MetadataSource, ResearchPaper

DOI_URL_PREFIX = "https://doi.org/"
ARXIV_URL_PREFIX = "https://arxiv.org/abs/"

# Loose but conservative DOI shape check (10.<registrant>/<suffix>) — used
# only to avoid constructing an obviously-broken link, never to "fix" or
# reformat a DOI the parser produced.
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def build_doi_url(doi: str | None) -> str | None:
    """Only construct a DOI URL when a valid-looking DOI exists — per spec,
    never fabricate one."""
    if not doi:
        return None
    doi = doi.strip()
    if doi.lower().startswith(("http://", "https://")):
        return doi  # already a URL (e.g. someone stored the full link)
    if not _DOI_RE.match(doi):
        return None
    return f"{DOI_URL_PREFIX}{doi}"


def build_arxiv_url(arxiv_id: str | None) -> str | None:
    if not arxiv_id:
        return None
    arxiv_id = arxiv_id.strip()
    if arxiv_id.lower().startswith(("http://", "https://")):
        return arxiv_id
    if not _ARXIV_ID_RE.match(arxiv_id):
        return None
    return f"{ARXIV_URL_PREFIX}{arxiv_id}"


def get_paper_for_document(db: Session, document_id: str) -> ResearchPaper | None:
    return db.scalar(select(ResearchPaper).where(ResearchPaper.document_id == document_id))


def get_or_create_paper(db: Session, document: Document, commit: bool = True) -> ResearchPaper:
    """Return the document's ResearchPaper, creating it from Phase 2's
    already-extracted title/authors if it doesn't exist yet. Safe to call
    repeatedly (idempotent): an existing row is returned as-is, never
    overwritten, so any later manual/external-lookup edits are preserved.
    """
    paper = get_paper_for_document(db, document.id)
    if paper is not None:
        return paper

    paper = ResearchPaper(
        document_id=document.id,
        title=document.title,
        authors=document.authors,
        metadata_source=MetadataSource.DOCUMENT_PARSER,
        # Phase 2 extraction is best-effort text-layout heuristics, not a
        # verified bibliographic record — mid confidence by default.
        metadata_confidence=0.5 if (document.title or document.authors) else None,
    )
    db.add(paper)
    if commit:
        db.commit()
        db.refresh(paper)
    else:
        db.flush()
    return paper
