"""Discover: query expansion (Gemini, mocked) + OpenAlex search (HTTP, mocked).

unittest + in-memory SQLite, same convention as the other tests. Neither
Gemini nor OpenAlex is ever really called.

Tests:
  1.  Expansion: cleans, de-duplicates and caps queries at 6
  2.  Expansion: local provider -> 409 "requires Gemini"; model failure -> 502
  3.  Normalisation: title/authors/year/venue/DOI/abstract/pdf from an OpenAlex work
  4.  A work without a title is skipped; has_pdf reflects a PDF link
  5.  Results are de-duplicated by DOI across queries (matched_queries merged)
  6.  Search results come only from the API (nothing from the Gemini output)
  7.  Rate limit (429): retried once, then a clear 429; partial results keep warnings
  8.  Server errors / network errors: retried, then a clear 502, never a 500
  9.  Save to library is idempotent and results show saved_id
  10. Saved papers are per user (list / delete)
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["LLM_PROVIDER"] = "local"
os.environ["GEMINI_API_KEY"] = ""

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.discovery import openalex, query_expander
from app.main import app
from app.models import SavedPaper, User
from app.routers import discover as discover_router


class FakeGemini:
    name = "gemini"

    def __init__(self, data=None, error=None):
        self.data, self.error = data, error
        self.calls = 0

    def generate_json(self, system, messages, schema):
        self.calls += 1
        if self.error:
            raise self.error
        return self.data


class FakeLocal:
    name = "local"

    def generate_json(self, system, messages, schema):
        raise NotImplementedError


def work(title="Graph neural networks", doi="https://doi.org/10.1000/ABC", oa_id="https://openalex.org/W1", pdf=None, **extra):
    return {
        "id": oa_id,
        "title": title,
        "publication_year": 2021,
        "doi": doi,
        "authorships": [{"author": {"display_name": "Ada Lovelace"}}, {"author": {"display_name": "Alan Turing"}}],
        "primary_location": {"source": {"display_name": "Journal of Tests"}, "landing_page_url": "https://example.org/p"},
        "best_oa_location": {"pdf_url": pdf} if pdf else None,
        "abstract_inverted_index": {"We": [0], "study": [1], "graphs": [2]},
        "cited_by_count": 7,
        **extra,
    }


def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


class ExpanderTests(unittest.TestCase):
    def test_cleans_dedupes_and_caps(self):
        llm = FakeGemini({
            "queries": [" graph learning ", "Graph Learning", "gnn survey", "", "a", "b", "c", "d", "e", 5],
            "related_terms": ["message passing", "message passing", "spectral"],
        })
        out = query_expander.expand_topic(llm, "graph neural networks")
        self.assertEqual(out["queries"], ["graph learning", "gnn survey", "a", "b", "c", "d"])
        self.assertEqual(out["related_terms"], ["message passing", "spectral"])

    def test_local_provider_requires_gemini(self):
        with self.assertRaises(query_expander.QueryExpansionUnavailableError):
            query_expander.expand_topic(FakeLocal(), "topic")

    def test_model_failure_and_empty_output(self):
        with self.assertRaises(query_expander.QueryExpansionError):
            query_expander.expand_topic(FakeGemini(error=RuntimeError("boom")), "topic")
        with self.assertRaises(query_expander.QueryExpansionError):
            query_expander.expand_topic(FakeGemini({"queries": [], "related_terms": []}), "topic")
        with self.assertRaises(query_expander.QueryExpansionError):
            query_expander.expand_topic(FakeGemini({"queries": ["x"]}), "   ")


class NormalizeTests(unittest.TestCase):
    def test_normalises_a_work(self):
        record = openalex.normalize_work(work(pdf="https://example.org/p.pdf"))
        self.assertEqual(record["title"], "Graph neural networks")
        self.assertEqual(record["authors"], ["Ada Lovelace", "Alan Turing"])
        self.assertEqual(record["year"], 2021)
        self.assertEqual(record["venue"], "Journal of Tests")
        self.assertEqual(record["doi"], "10.1000/abc")
        self.assertEqual(record["doi_url"], "https://doi.org/10.1000/abc")
        self.assertEqual(record["abstract"], "We study graphs")
        self.assertEqual(record["openalex_id"], "W1")
        self.assertTrue(record["has_pdf"])

    def test_no_title_skipped_and_no_pdf_flag(self):
        self.assertIsNone(openalex.normalize_work(work(title="  ")))
        record = openalex.normalize_work(work(doi=None))
        self.assertFalse(record["has_pdf"])
        self.assertIsNone(record["doi"])

    def test_dedupe_by_doi_merges_queries_and_fills_gaps(self):
        first = openalex.normalize_work(work(pdf=None))
        again = openalex.normalize_work(work(pdf="https://example.org/p.pdf", doi="10.1000/abc", oa_id="https://openalex.org/W9"))
        other = openalex.normalize_work(work(title="Other", doi="10.1000/zzz", oa_id="https://openalex.org/W2"))
        merged = openalex.merge_results([("q1", [other, first]), ("q2", [again])])
        self.assertEqual(len(merged), 2)
        top = merged[0]  # matched by two queries -> first
        self.assertEqual(top["doi"], "10.1000/abc")
        self.assertEqual(top["matched_queries"], ["q1", "q2"])
        self.assertTrue(top["has_pdf"])


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        def override_db():
            db = self.Session()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_db
        self.client = TestClient(app)
        self.sleep = patch("app.discovery.openalex.time.sleep").start()
        self.addCleanup(patch.stopall)
        self.addCleanup(app.dependency_overrides.clear)

    def use_openalex(self, handler):
        patch.object(discover_router, "_http_client", lambda: mock_client(handler)).start()

    def test_queries_endpoint(self):
        llm = FakeGemini({"queries": ["a b", "c d"], "related_terms": ["x"]})
        with patch.object(discover_router, "get_llm_provider", return_value=llm):
            r = self.client.post("/api/discover/queries", json={"topic": "t"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["queries"], ["a b", "c d"])
        with patch.object(discover_router, "get_llm_provider", return_value=FakeLocal()):
            r = self.client.post("/api/discover/queries", json={"topic": "t"})
        self.assertEqual(r.status_code, 409)
        self.assertIn("Gemini", r.json()["detail"])
        with patch.object(discover_router, "get_llm_provider", return_value=FakeGemini(error=RuntimeError("x"))):
            r = self.client.post("/api/discover/queries", json={"topic": "t"})
        self.assertEqual(r.status_code, 502)

    def test_results_come_only_from_openalex(self):
        gemini = FakeGemini({"queries": ["invented paper by Nobody 1999"], "related_terms": []})

        def handler(request):
            self.assertEqual(request.url.host, "api.openalex.org")
            return httpx.Response(200, json={"results": [work(), work(title="Second", doi="10.1/2", oa_id="https://openalex.org/W2")]})

        self.use_openalex(handler)
        r = self.client.post("/api/discover/search", json={"queries": ["graph networks", "gnn"]})
        self.assertEqual(r.status_code, 200)
        titles = [p["title"] for p in r.json()["results"]]
        self.assertEqual(sorted(titles), ["Graph neural networks", "Second"])  # deduped across both queries
        self.assertEqual(gemini.calls, 0)
        self.assertEqual(r.json()["results"][0]["matched_queries"], ["graph networks", "gnn"])

    def test_rate_limit_then_clear_429(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(429, headers={"Retry-After": "1"})

        self.use_openalex(handler)
        r = self.client.post("/api/discover/search", json={"queries": ["a", "b"]})
        self.assertEqual(r.status_code, 429)
        self.assertEqual(len(calls), 2)  # one retry for the first query, then stop

    def test_partial_failure_keeps_results_with_warning(self):
        def handler(request):
            if request.url.params["search"] == "bad":
                return httpx.Response(500)
            return httpx.Response(200, json={"results": [work()]})

        self.use_openalex(handler)
        r = self.client.post("/api/discover/search", json={"queries": ["good", "bad"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["results"]), 1)
        self.assertEqual(len(r.json()["warnings"]), 1)

    def test_server_and_network_errors_are_502(self):
        self.use_openalex(lambda request: httpx.Response(503))
        r = self.client.post("/api/discover/search", json={"queries": ["a"]})
        self.assertEqual(r.status_code, 502)

        def boom(request):
            raise httpx.ConnectError("down")

        self.use_openalex(boom)
        r = self.client.post("/api/discover/search", json={"queries": ["a"]})
        self.assertEqual(r.status_code, 502)
        self.assertEqual(self.client.post("/api/discover/search", json={"queries": ["  "]}).status_code, 400)

    def test_save_is_idempotent_and_flags_results(self):
        record = openalex.normalize_work(work(pdf="https://example.org/p.pdf"))
        payload = {k: record[k] for k in ("title", "authors", "year", "venue", "doi", "openalex_id", "abstract", "pdf_url", "landing_url")}
        first = self.client.post("/api/discover/saved", json=payload)
        second = self.client.post("/api/discover/saved", json={**payload, "doi": "https://doi.org/10.1000/ABC"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["id"], second.json()["id"])
        self.assertEqual(len(self.client.get("/api/discover/saved").json()), 1)

        self.use_openalex(lambda request: httpx.Response(200, json={"results": [work()]}))
        r = self.client.post("/api/discover/search", json={"queries": ["q"]})
        self.assertEqual(r.json()["results"][0]["saved_id"], first.json()["id"])

    def test_saved_papers_are_per_user(self):
        saved = self.client.post("/api/discover/saved", json={"title": "Mine"}).json()
        with self.Session() as db:
            other = User(email="other@example.com", name="Other")
            db.add(other)
            db.commit()
            db.add(SavedPaper(owner_id=other.id, title="Theirs"))
            db.commit()
        titles = [p["title"] for p in self.client.get("/api/discover/saved").json()]
        self.assertEqual(titles, ["Mine"])
        self.assertEqual(self.client.delete(f"/api/discover/saved/{saved['id']}").status_code, 204)
        self.assertEqual(self.client.delete(f"/api/discover/saved/{saved['id']}").status_code, 404)


if __name__ == "__main__":
    unittest.main()
