"""Phase 8 — Research gap detection tests.

unittest + temp-file SQLite (same convention as test_comparison.py). The
retriever and LLM are stubbed, so no external API is needed.

Covered: workspace ownership, empty/too-small workspaces, ResearchGap creation,
Gap→Claim and Gap→Evidence links, evidence/paper correctness, multi-paper
generation, repeated-limitation and contradiction detection, unsupported-gap
rejection, malformed LLM output, confidence calculation, re-analysis, the API
endpoints, and single-paper RAG / workspace-chat regressions.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["DATABASE_URL"] = "sqlite://"
# Env vars beat backend/.env: keep tests on the offline providers so they
# never call a real API, whatever the developer has configured.
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["LLM_PROVIDER"] = "local"
os.environ["GEMINI_API_KEY"] = ""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.database import Base
from app.models import (
    ChunkLevel,
    Claim,
    Document,
    DocumentChunk,
    Evidence,
    GapClaim,
    GapConfidence,
    GapEvidence,
    GapType,
    ProcessingStatus,
    RagStatus,
    ResearchGap,
    ResearchWorkspace,
    User,
    WorkspaceDocument,
)
from app.rag.interfaces import RetrievedChunk


class FakeRetriever:
    """Returns each paper's own seeded child chunk for every query."""

    def __init__(self, empty_docs=()):
        self.empty_docs = set(empty_docs)

    def retrieve(self, document_id, query, top_k):
        if document_id in self.empty_docs:
            return []
        n = document_id.split("-")[1]
        return [
            RetrievedChunk(
                chunk_id=f"chunk-child-{n}",
                parent_chunk_id=f"chunk-parent-{n}",
                document_id=document_id,
                page_number=2,
                section_id=None,
                section_title="Limitations",
                child_text=f"Paper {n} evaluates on a single small dataset.",
                context_text=f"Paper {n} context.",
                score=0.9,
            )
        ]


class FakeLLM:
    name = "fake"  # anything but "local" (gap detection refuses the extractive provider)

    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, system, messages):
        self.prompts.append(messages[0].content)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response if isinstance(self.response, str) else json.dumps(self.response)


def _cand(title="Limited cross-dataset evaluation", gap_type="dataset", papers=(0, 1, 2),
          aspect="limitations", **extra):
    c = {
        "title": title,
        "description": "Several papers evaluate on a single small dataset only.",
        "gap_type": gap_type,
        "supporting_evidence": [{"paper_index": i, "aspect": aspect} for i in papers],
        "research_direction": "Evaluate across multiple datasets.",
    }
    c.update(extra)
    return c


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.engine = create_engine(f"sqlite:///{self._tmp.name}")
        Base.metadata.create_all(bind=self.engine)
        self.db = sessionmaker(bind=self.engine)()
        self._seed()

        from app.gaps import gap_service
        self.gs = gap_service
        self._orig = (gap_service.build_retriever, gap_service.get_llm_provider)
        self.retriever = FakeRetriever()
        gap_service.build_retriever = lambda db, settings: self.retriever
        self.set_llm([])

    def tearDown(self):
        self.gs.build_retriever, self.gs.get_llm_provider = self._orig
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()
        os.unlink(self._tmp.name)

    def set_llm(self, response):
        self.llm = FakeLLM(response)
        self.gs.get_llm_provider = lambda settings: self.llm

    def _doc(self, i, owner="user-1", indexed=True):
        self.db.add(Document(
            id=f"doc-{i}", filename=f"p{i}.pdf", original_filename=f"Paper {i}.pdf",
            file_size=1, content_type="application/pdf", storage_path=f"u/p{i}.pdf",
            processing_status=ProcessingStatus.READY,
            rag_status=RagStatus.INDEXED if indexed else RagStatus.NOT_INDEXED,
            chunk_count=2 if indexed else 0, owner_id=owner, title=f"Paper {i}",
        ))
        if indexed:
            self.db.add(DocumentChunk(
                id=f"chunk-parent-{i}", document_id=f"doc-{i}", page_number=2,
                chunk_level=ChunkLevel.PARENT, text=f"Paper {i} context.", order_index=0))
            self.db.add(DocumentChunk(
                id=f"chunk-child-{i}", document_id=f"doc-{i}", page_number=2,
                parent_chunk_id=f"chunk-parent-{i}", chunk_level=ChunkLevel.CHILD,
                text=f"Paper {i} evaluates on a single small dataset.",
                embedding=[0.1] * 768, order_index=1))

    def _seed(self):
        self.db.add(User(id="user-1", email="a@x.com", name="A"))
        self.db.add(User(id="user-2", email="b@x.com", name="B"))
        for i in (1, 2, 3, 4):
            self._doc(i)
        self._doc(9, owner="user-2")
        self.db.add(ResearchWorkspace(id="ws-1", name="WS", owner_id="user-1"))
        self.db.add(ResearchWorkspace(id="ws-empty", name="Empty", owner_id="user-1"))
        self.db.add(ResearchWorkspace(id="ws-one", name="One", owner_id="user-1"))
        for i in (1, 2, 3, 4):
            self.db.add(WorkspaceDocument(workspace_id="ws-1", document_id=f"doc-{i}"))
        self.db.add(WorkspaceDocument(workspace_id="ws-one", document_id="doc-1"))
        self.db.commit()

    def run_analysis(self, user="user-1", ws="ws-1"):
        return self.gs.analyze_gaps(self.db, ws, user, settings=Settings())


# 1-2. Ownership / empty workspace ------------------------------------------------


class TestValidation(_Base):
    def test_other_user_cannot_analyze(self):
        from app.rag.workspace_retrieval import WorkspaceRetrievalError
        with self.assertRaises(WorkspaceRetrievalError):
            self.run_analysis(user="user-2")

    def test_workspace_without_papers(self):
        from app.rag.workspace_retrieval import WorkspaceRetrievalError
        with self.assertRaises(WorkspaceRetrievalError):
            self.run_analysis(ws="ws-empty")

    def test_single_paper_workspace_rejected(self):
        with self.assertRaises(self.gs.GapDetectionError):
            self.run_analysis(ws="ws-one")

    def test_other_user_cannot_list_gaps(self):
        from app.rag.workspace_retrieval import WorkspaceRetrievalError
        with self.assertRaises(WorkspaceRetrievalError):
            self.gs.list_gaps(self.db, "ws-1", "user-2")


# 3-8. Creation, links, multi-paper, repeated limitation ---------------------------


class TestGapCreation(_Base):
    def test_gap_created_with_claims_and_evidence(self):
        self.set_llm([_cand()])
        out = self.run_analysis()
        self.assertEqual(len(out.gaps), 1)
        self.assertEqual(out.papers_analyzed, 4)
        gap = self.db.get(ResearchGap, out.gaps[0].id)
        self.assertEqual(gap.workspace_id, "ws-1")
        self.assertEqual(gap.gap_type, GapType.DATASET)
        self.assertEqual(gap.supporting_paper_count, 3)
        self.assertEqual(len(gap.gap_evidence_items), 3)
        self.assertEqual(len(gap.gap_claim_items), 3)

    def test_gap_claim_relationship_reuses_claim_model(self):
        self.set_llm([_cand()])
        gap = self.run_analysis().gaps[0]
        for gc in gap.gap_claim_items:
            self.assertIsInstance(gc.claim, Claim)
            self.assertEqual(gc.claim.workspace_id, "ws-1")
            self.assertTrue(gc.claim.evidence_items)

    def test_gap_evidence_relationship_and_correct_paper(self):
        self.set_llm([_cand(papers=(0, 2))])
        gap = self.run_analysis().gaps[0]
        docs = {ge.evidence.document_id for ge in gap.gap_evidence_items}
        self.assertEqual(docs, {"doc-1", "doc-3"})
        for ge in gap.gap_evidence_items:
            ev = ge.evidence
            self.assertEqual(ev.claim.document_id, ev.document_id)
            self.assertEqual(ev.chunk_id, f"chunk-child-{ev.document_id.split('-')[1]}")
            self.assertEqual(ev.page_number, 2)

    def test_evidence_only_from_workspace_papers(self):
        self.set_llm([_cand(papers=(0, 1, 2, 3))])
        gap = self.run_analysis().gaps[0]
        self.assertNotIn("doc-9", {ge.evidence.document_id for ge in gap.gap_evidence_items})

    def test_prompt_labels_each_paper_and_marks_missing(self):
        self.retriever = FakeRetriever(empty_docs={"doc-2"})
        self.gs.build_retriever = lambda db, s: self.retriever
        self.set_llm([])
        self.run_analysis()
        prompt = self.llm.prompts[0]
        self.assertIn("[Paper 0, limitations]", prompt)
        self.assertIn("[Paper 1, limitations]: Not found in paper", prompt)

    def test_repeated_limitation_detected_across_papers(self):
        self.set_llm([_cand(title="Small datasets recur", papers=(0, 1, 2, 3))])
        gap = self.run_analysis().gaps[0]
        self.assertEqual(gap.supporting_paper_count, 4)
        self.assertEqual(gap.confidence, GapConfidence.HIGH)

    def test_contradictory_finding_represented_with_both_sources(self):
        self.set_llm([_cand(title="Conflicting results", gap_type="contradictory",
                            papers=(0, 1), aspect="main_results")])
        gap = self.run_analysis().gaps[0]
        self.assertEqual(gap.gap_type, GapType.CONTRADICTORY)
        self.assertEqual({ge.evidence.document_id for ge in gap.gap_evidence_items},
                         {"doc-1", "doc-2"})

    def test_contradiction_with_one_source_rejected(self):
        self.set_llm([_cand(gap_type="contradictory", papers=(0,), aspect="main_results")])
        self.assertEqual(self.run_analysis().gaps, [])

    def test_future_work_gap_supported(self):
        self.set_llm([_cand(title="Shared future work", gap_type="future_work",
                            papers=(0, 1), aspect="future_work")])
        gap = self.run_analysis().gaps[0]
        self.assertEqual(gap.gap_type, GapType.FUTURE_WORK)
        self.assertEqual({ge.role for ge in gap.gap_evidence_items}, {"future_work"})


# 9-10. Rejection ------------------------------------------------------------------


class TestRejection(_Base):
    def test_unsupported_gap_rejected(self):
        self.set_llm([_cand(papers=(7, 8))])  # nonexistent papers
        out = self.run_analysis()
        self.assertEqual(out.gaps, [])
        self.assertEqual(out.candidates_rejected, 1)
        self.assertEqual(self.db.query(ResearchGap).count(), 0)
        self.assertEqual(self.db.query(Evidence).count(), 0)

    def test_gap_without_retrieved_evidence_rejected(self):
        self.retriever = FakeRetriever(empty_docs={"doc-1", "doc-2", "doc-3", "doc-4"})
        self.gs.build_retriever = lambda db, s: self.retriever
        self.set_llm([_cand()])
        self.assertEqual(self.run_analysis().gaps, [])
        self.assertEqual(self.db.query(Claim).count(), 0)

    def test_malformed_candidates_dropped_valid_kept(self):
        good = _cand()
        bad = [{"title": "x"}, {"title": "Ok title", "description": "long enough description",
                                "gap_type": "made-up", "supporting_evidence": []}, "junk"]
        self.set_llm(bad + [good])
        self.assertEqual(len(self.run_analysis().gaps), 1)

    def test_non_json_output_rejected(self):
        self.set_llm("Here are some gaps: there are many!")
        with self.assertRaises(self.gs.GapDetectionError):
            self.run_analysis()

    def test_llm_failure_wrapped(self):
        self.set_llm(RuntimeError("boom"))
        with self.assertRaises(self.gs.GapDetectionError):
            self.run_analysis()

    def test_code_fenced_json_accepted(self):
        self.set_llm("```json\n" + json.dumps([_cand()]) + "\n```")
        self.assertEqual(len(self.run_analysis().gaps), 1)

    def test_llm_supplied_confidence_ignored(self):
        self.set_llm([_cand(papers=(0,), confidence="high")])
        self.assertEqual(self.run_analysis().gaps[0].confidence, GapConfidence.LOW)


# 11. Confidence -----------------------------------------------------------------


class TestConfidence(unittest.TestCase):
    def test_levels(self):
        from app.gaps.gap_service import compute_confidence
        self.assertEqual(compute_confidence(3, 5)[0], GapConfidence.HIGH)
        self.assertEqual(compute_confidence(2, 2)[0], GapConfidence.HIGH)
        self.assertEqual(compute_confidence(2, 4)[0], GapConfidence.MEDIUM)
        self.assertEqual(compute_confidence(1, 4)[0], GapConfidence.LOW)
        self.assertEqual(compute_confidence(0, 4)[0], GapConfidence.LOW)

    def test_reason_is_transparent_and_cautious(self):
        from app.gaps.gap_service import compute_confidence
        _, reason = compute_confidence(2, 4, ["limitations", "dataset"])
        self.assertIn("2 of 4", reason)
        self.assertIn("limitations", reason)
        self.assertIn("Novelty requires verification", reason)
        self.assertNotIn("novel research gap", reason.lower())


# Re-analysis + cascade -----------------------------------------------------------


class TestReanalysis(_Base):
    def test_reanalysis_replaces_old_gaps_and_their_claims(self):
        self.set_llm([_cand()])
        self.run_analysis()
        self.run_analysis()
        self.assertEqual(self.db.query(ResearchGap).count(), 1)
        self.assertEqual(self.db.query(GapClaim).count(), 3)
        self.assertEqual(self.db.query(Claim).count(), 3)
        self.assertEqual(self.db.query(Evidence).count(), 3)

    def test_failed_reanalysis_keeps_existing_gaps(self):
        self.set_llm([_cand()])
        self.run_analysis()
        self.set_llm("not json")
        with self.assertRaises(self.gs.GapDetectionError):
            self.run_analysis()
        self.assertEqual(self.db.query(ResearchGap).count(), 1)


# API -----------------------------------------------------------------------------


class TestGapAPI(_Base):
    def setUp(self):
        super().setUp()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.database import get_db
        from app.deps import get_current_user
        from app.routers import gaps

        app = FastAPI()
        app.include_router(gaps.router)
        self.current = "user-1"
        app.dependency_overrides[get_db] = lambda: self.db
        app.dependency_overrides[get_current_user] = lambda: self.db.get(User, self.current)
        self.client = TestClient(app)

    def test_analyze_list_detail_evidence_claims(self):
        self.set_llm([_cand(), _cand(title="Conflicts", gap_type="contradictory",
                                     papers=(0, 1), aspect="main_results")])
        r = self.client.post("/api/workspaces/ws-1/gaps/analyze")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["gaps_found"], 2)
        self.assertEqual(body["papers_analyzed"], 4)
        self.assertIn("Dataset Gap", body["gap_type_counts"])

        listed = self.client.get("/api/workspaces/ws-1/gaps").json()
        self.assertEqual(len(listed), 2)

        gid = body["gaps"][0]["id"]
        detail = self.client.get(f"/api/gaps/{gid}").json()
        self.assertTrue(detail["confidence_reasoning"])
        self.assertIn("Novelty requires verification", detail["novelty_notice"])
        self.assertTrue(detail["claims"])

        ev = self.client.get(f"/api/gaps/{gid}/evidence").json()
        self.assertTrue(all(e["evidence"]["page_number"] == 2 for e in ev))

        claims = self.client.get(f"/api/gaps/{gid}/claims").json()
        self.assertEqual(len(claims), len(ev))
        self.assertTrue(all(c["evidence"] for c in claims))

    def test_analyze_errors(self):
        self.assertEqual(self.client.post("/api/workspaces/nope/gaps/analyze").status_code, 404)
        self.assertEqual(self.client.post("/api/workspaces/ws-empty/gaps/analyze").status_code, 404)
        self.assertEqual(self.client.post("/api/workspaces/ws-one/gaps/analyze").status_code, 400)

    def test_other_user_isolated(self):
        self.set_llm([_cand()])
        gid = self.client.post("/api/workspaces/ws-1/gaps/analyze").json()["gaps"][0]["id"]
        self.current = "user-2"
        self.assertEqual(self.client.post("/api/workspaces/ws-1/gaps/analyze").status_code, 404)
        self.assertEqual(self.client.get("/api/workspaces/ws-1/gaps").status_code, 404)
        for path in (f"/api/gaps/{gid}", f"/api/gaps/{gid}/evidence", f"/api/gaps/{gid}/claims"):
            self.assertEqual(self.client.get(path).status_code, 404)


# Regression ----------------------------------------------------------------------


class TestRegression(unittest.TestCase):
    def test_existing_suites_importable(self):
        """Single-paper RAG and workspace chat modules still import; their
        behaviour is covered by test_evidence.py / test_workspace_chat.py /
        test_comparison.py, which must be run alongside this file."""
        import app.rag.service  # noqa: F401
        import app.rag.workspace_chat  # noqa: F401
        import app.comparison.comparison_service  # noqa: F401


if __name__ == "__main__":
    unittest.main()
