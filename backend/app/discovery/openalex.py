"""OpenAlex search (https://docs.openalex.org) - the only paper source.

Google Scholar has no API and forbids scraping, so OpenAlex (free, open,
covers most scholarly works) is used. Results are normalised to one shape and
de-duplicated by DOI. Rate limits and outages surface as `SearchError`s the
caller turns into per-query warnings.
"""
from __future__ import annotations

import logging
import re
import time

import httpx

logger = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SELECT_FIELDS = (
    "id,title,publication_year,doi,authorships,primary_location,best_oa_location,"
    "open_access,abstract_inverted_index,cited_by_count"
)
TIMEOUT_SECONDS = 15.0
MAX_ATTEMPTS = 2
MAX_RETRY_WAIT_SECONDS = 3.0
MAX_AUTHORS = 20


class SearchError(RuntimeError):
    """A search request failed; the message is safe to show."""


class RateLimitedError(SearchError):
    pass


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi or None


def _abstract(inverted: dict | None) -> str | None:
    if not inverted:
        return None
    positions: list[tuple[int, str]] = []
    for word, indexes in inverted.items():
        positions.extend((i, word) for i in indexes)
    text = " ".join(word for _, word in sorted(positions))
    return text or None


def normalize_work(work: dict) -> dict | None:
    """One OpenAlex work -> {openalex_id, title, authors, year, venue, doi,
    doi_url, abstract, pdf_url, has_pdf, landing_url, cited_by_count}, or None
    when it has no title."""
    title = " ".join((work.get("title") or "").split())
    if not title:
        return None
    doi = normalize_doi(work.get("doi"))
    primary = work.get("primary_location") or {}
    best_oa = work.get("best_oa_location") or {}
    pdf_url = best_oa.get("pdf_url") or primary.get("pdf_url")
    doi_url = f"https://doi.org/{doi}" if doi else None
    authors = [
        (a.get("author") or {}).get("display_name")
        for a in (work.get("authorships") or [])[:MAX_AUTHORS]
    ]
    return {
        "openalex_id": (work.get("id") or "").rsplit("/", 1)[-1] or None,
        "title": title,
        "authors": [a for a in authors if a],
        "year": work.get("publication_year"),
        "venue": (primary.get("source") or {}).get("display_name"),
        "doi": doi,
        "doi_url": doi_url,
        "abstract": _abstract(work.get("abstract_inverted_index")),
        "pdf_url": pdf_url,
        "has_pdf": bool(pdf_url),
        "landing_url": primary.get("landing_page_url") or best_oa.get("landing_page_url") or doi_url,
        "cited_by_count": work.get("cited_by_count") or 0,
    }


def _retry_after(response: httpx.Response) -> float:
    try:
        return max(0.0, float(response.headers.get("Retry-After", "1")))
    except ValueError:
        return 1.0


def search_works(
    query: str, *, per_page: int = 10, mailto: str | None = None, client: httpx.Client | None = None
) -> list[dict]:
    """Normalised results for one query. Raises RateLimitedError / SearchError."""
    params = {"search": query, "per-page": per_page, "select": SELECT_FIELDS}
    if mailto:
        params["mailto"] = mailto  # OpenAlex "polite pool": faster, more reliable
    owns_client = client is None
    client = client or httpx.Client(timeout=TIMEOUT_SECONDS)
    try:
        last: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = client.get(OPENALEX_WORKS_URL, params=params)
            except httpx.HTTPError as exc:
                last = SearchError("OpenAlex could not be reached.")
                logger.warning("OpenAlex request failed (attempt %d): %s", attempt, type(exc).__name__)
            else:
                if response.status_code == 429:
                    last = RateLimitedError("OpenAlex rate limit reached. Try again in a moment.")
                    wait = _retry_after(response)
                    if attempt < MAX_ATTEMPTS and wait <= MAX_RETRY_WAIT_SECONDS:
                        time.sleep(wait)
                        continue
                    raise last
                if response.status_code >= 500:
                    last = SearchError(f"OpenAlex is unavailable right now (HTTP {response.status_code}).")
                elif response.status_code >= 400:
                    raise SearchError(f"OpenAlex rejected the search (HTTP {response.status_code}).")
                else:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise SearchError("OpenAlex returned an unreadable response.") from exc
                    works = (payload or {}).get("results") or []
                    return [n for n in (normalize_work(w) for w in works if isinstance(w, dict)) if n]
            if attempt < MAX_ATTEMPTS:
                time.sleep(0.5)
        raise last or SearchError("OpenAlex search failed.")
    finally:
        if owns_client:
            client.close()


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def merge_results(per_query: list[tuple[str, list[dict]]]) -> list[dict]:
    """De-duplicate across queries: by DOI, else OpenAlex id, else normalised
    title. The first record wins, gaps (abstract, PDF) are filled from later
    duplicates, and `matched_queries` lists the queries that found it (most
    widely matched first, ties keep search order)."""
    merged: dict[str, dict] = {}
    for query, results in per_query:
        for record in results:
            key = record["doi"] or record["openalex_id"] or _title_key(record["title"])
            existing = merged.get(key)
            if existing is None:
                merged[key] = {**record, "matched_queries": [query]}
                continue
            if query not in existing["matched_queries"]:
                existing["matched_queries"].append(query)
            for field in ("abstract", "pdf_url", "venue", "doi", "doi_url"):
                if not existing.get(field) and record.get(field):
                    existing[field] = record[field]
            existing["has_pdf"] = bool(existing.get("pdf_url"))
    return sorted(merged.values(), key=lambda r: -len(r["matched_queries"]))


def search_many(
    queries: list[str], *, per_page: int, mailto: str | None, client: httpx.Client | None = None
) -> tuple[list[dict], list[str]]:
    """Run every query; (merged results, warnings). Raises SearchError only
    when every query failed."""
    per_query: list[tuple[str, list[dict]]] = []
    warnings: list[str] = []
    last_error: SearchError | None = None
    for query in queries:
        try:
            per_query.append((query, search_works(query, per_page=per_page, mailto=mailto, client=client)))
        except SearchError as exc:
            last_error = exc
            warnings.append(f'"{query}": {exc}')
            if isinstance(exc, RateLimitedError):
                break  # the remaining queries would be limited too
    if not per_query and last_error is not None:
        raise last_error
    return merge_results(per_query), warnings
