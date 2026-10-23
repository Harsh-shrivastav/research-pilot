"""Discover: find papers by topic.

    POST   /api/discover/queries       topic -> 4-6 search queries (Gemini)
    POST   /api/discover/search        queries -> de-duplicated OpenAlex results
    GET    /api/discover/saved         the user's saved papers
    POST   /api/discover/saved         save a result to the library
    DELETE /api/discover/saved/{id}    remove a saved paper

Gemini only writes queries; every paper shown comes from OpenAlex.
`search` also works with LLM_PROVIDER=local (queries typed by hand).
"""
import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.deps import get_current_user
from app.discovery import openalex, query_expander
from app.discovery.openalex import normalize_doi
from app.models import SavedPaper
from app.rag.factory import get_llm_provider
from app.schemas import (
    DiscoverPaperOut,
    DiscoverQueriesIn,
    DiscoverQueriesOut,
    DiscoverSearchIn,
    DiscoverSearchOut,
    SavedPaperOut,
    SavePaperIn,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/discover", tags=["discover"])
settings = get_settings()

MAX_SEARCH_QUERIES = 6
MAX_QUERY_CHARS = 200


def _saved_out(paper: SavedPaper) -> SavedPaperOut:
    return SavedPaperOut(
        id=paper.id,
        title=paper.title,
        authors=paper.authors or [],
        year=paper.year,
        venue=paper.venue,
        doi=paper.doi,
        openalex_id=paper.openalex_id,
        abstract=paper.abstract,
        pdf_url=paper.pdf_url,
        landing_url=paper.landing_url,
        source=paper.source,
        created_at=paper.created_at,
    )


def _find_saved(db: Session, owner_id: str, doi: str | None, openalex_id: str | None) -> SavedPaper | None:
    if doi:
        found = db.scalar(select(SavedPaper).where(SavedPaper.owner_id == owner_id, SavedPaper.doi == doi))
        if found:
            return found
    if openalex_id:
        return db.scalar(
            select(SavedPaper).where(SavedPaper.owner_id == owner_id, SavedPaper.openalex_id == openalex_id)
        )
    return None


# The HTTP client is created per request; tests replace this to inject a mock transport.
def _http_client() -> httpx.Client | None:
    return None


@router.post("/queries", response_model=DiscoverQueriesOut)
def expand_queries(payload: DiscoverQueriesIn, db: Session = Depends(get_db)) -> DiscoverQueriesOut:
    get_current_user(db)
    try:
        result = query_expander.expand_topic(get_llm_provider(settings), payload.topic)
    except query_expander.QueryExpansionUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except query_expander.QueryExpansionFailedError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except query_expander.QueryExpansionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("query expansion failed")
        raise HTTPException(status_code=502, detail="Could not expand the topic right now. Please try again.")
    return DiscoverQueriesOut(**result)


@router.post("/search", response_model=DiscoverSearchOut)
def search_papers(payload: DiscoverSearchIn, db: Session = Depends(get_db)) -> DiscoverSearchOut:
    user = get_current_user(db)
    queries = list(
        dict.fromkeys(" ".join(q.split())[:MAX_QUERY_CHARS] for q in payload.queries if q and q.strip())
    )[:MAX_SEARCH_QUERIES]
    if not queries:
        raise HTTPException(status_code=400, detail="Enter at least one search query.")
    try:
        records, warnings = openalex.search_many(
            queries,
            per_page=settings.discover_results_per_query,
            mailto=settings.openalex_mailto,
            client=_http_client(),
        )
    except openalex.RateLimitedError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except openalex.SearchError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    results = []
    for record in records:
        saved = _find_saved(db, user.id, record["doi"], record["openalex_id"])
        results.append(DiscoverPaperOut(**record, saved_id=saved.id if saved else None))
    return DiscoverSearchOut(results=results, warnings=warnings)


@router.get("/saved", response_model=list[SavedPaperOut])
def list_saved(db: Session = Depends(get_db)) -> list[SavedPaperOut]:
    user = get_current_user(db)
    rows = db.scalars(
        select(SavedPaper).where(SavedPaper.owner_id == user.id).order_by(SavedPaper.created_at.desc())
    ).all()
    return [_saved_out(p) for p in rows]


@router.post("/saved", response_model=SavedPaperOut)
def save_paper(payload: SavePaperIn, db: Session = Depends(get_db)) -> SavedPaperOut:
    """Save a search result to the library. Saving the same paper twice returns
    the existing record."""
    user = get_current_user(db)
    title = " ".join(payload.title.split())
    if not title:
        raise HTTPException(status_code=400, detail="A paper needs a title.")
    doi = normalize_doi(payload.doi)
    existing = _find_saved(db, user.id, doi, payload.openalex_id)
    if existing:
        return _saved_out(existing)
    paper = SavedPaper(
        owner_id=user.id,
        title=title[:1000],
        authors=payload.authors[:50],
        year=payload.year,
        venue=payload.venue,
        doi=doi,
        openalex_id=payload.openalex_id,
        abstract=payload.abstract,
        pdf_url=payload.pdf_url,
        landing_url=payload.landing_url,
    )
    db.add(paper)
    db.commit()
    db.refresh(paper)
    return _saved_out(paper)


@router.delete("/saved/{paper_id}", status_code=204, response_model=None)
def delete_saved(paper_id: str, db: Session = Depends(get_db)) -> None:
    user = get_current_user(db)
    paper = db.scalar(select(SavedPaper).where(SavedPaper.id == paper_id, SavedPaper.owner_id == user.id))
    if paper is None:
        raise HTTPException(status_code=404, detail="Saved paper not found.")
    db.delete(paper)
    db.commit()
