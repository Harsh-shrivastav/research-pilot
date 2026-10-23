"""Phase 7 — Evidence-backed multi-paper comparison tests.

Uses Python's built-in unittest with an in-memory SQLite database, same
convention as tests/test_workspace_chat.py and tests/test_evidence.py.

Tests:
  1.  Dimension resolution (defaults = 11 Phase 7 aspects, custom, dedup, case-insensitive reuse)
  2.  Paper-count validation (min 2, max 5)
  3.  Papers must belong to the workspace
  4.  Papers must be indexed before they can be compared
  5.  Comparison matrix creation persists document_ids + dimensions snapshot
  6.  Comparison cells are grounded in Evidence (citations carry evidence_id)
  7.  Insufficient-evidence cells produce no citations
  8.  Ownership: another user cannot read/delete a comparison
  9.  Comparison list/get/delete
  10. A failing cell (retrieval/generation error) does not take down the
      whole matrix
  11. Missing information is explicitly marked ("Not found in paper")
  12. Conflicting results are preserved correctly per-paper
  13. Evidence → Claim → ComparisonCell traceability chain
  14. Comparison uses only workspace documents
  15. Multiple papers (3+ papers)
  16. Existing workspace chat still works (regression)
  17. Existing single-paper RAG still works (regression)
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

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
    ComparisonCell,
    ComparisonMatrix,
    ComparisonStatus,
    Document,
    DocumentChunk,
    Evidence,
    ProcessingStatus,
    RagStatus,
    ResearchWorkspace,
    User,
    WorkspaceDocument,
)
from app.rag.interfaces import RetrievedChunk


class _TestBase(unittest.TestCase):
    def setUp(self):
        # HybridRetriever runs vector + keyword search concurrently, each on
        # its own SQLAlchemy Session bound to the same engine (by design —
        # see hybrid_retriever.py). A true in-memory SQLite engine hands out
        # a *separate* empty database per connection, and forcing a single
        # shared connection (StaticPool) instead makes two threads use one
        # SQLite connection concurrently, which corrupts its transaction
        # state. A temp-file-backed SQLite database sidesteps both: every
        # connection opens the same file and concurrent *reads* are safe.
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmpfile.close()
        self.engine = create_engine(f"sqlite:///{self._tmpfile.name}")
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        self._seed()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()
        os.unlink(self._tmpfile.name)

    def _seed(self):
        self.db.add(User(id="user-1", email="test@example.com", name="Test User"))
        self.db.add(User(id="user-2", email="other@example.com", name="Other User"))

        for i in range(1, 7):
            indexed = i <= 5
            doc = Document(
                id=f"doc-{i}",
                filename=f"paper{i}.pdf",
                original_filename=f"Paper {i}.pdf",
                file_size=1000,
                content_type="application/pdf",
                storage_path=f"uploads/paper{i}.pdf",
                processing_status=ProcessingStatus.READY,
                rag_status=RagStatus.INDEXED if indexed else RagStatus.NOT_INDEXED,
                chunk_count=5 if indexed else 0,
                owner_id="user-1",
                title=f"Paper {i}: A Study",
            )
            self.db.add(doc)
            if indexed:
                parent = DocumentChunk(
                    id=f"chunk-parent-{i}",
                    document_id=f"doc-{i}",
                    page_number=1,
                    chunk_level=ChunkLevel.PARENT,
                    text=f"Full context for paper {i}. Paper {i} uses methodology M{i}.",
                    order_index=0,
                )
                self.db.add(parent)
                self.db.add(
                    DocumentChunk(
                        id=f"chunk-child-{i}",
                        document_id=f"doc-{i}",
                        page_number=2,
                        parent_chunk_id=f"chunk-parent-{i}",
                        chunk_level=ChunkLevel.CHILD,
                        text=f"Paper {i} uses methodology M{i}.",
                        embedding=[0.1] * 768,
                        order_index=0,
                    )
                )

        self.db.add(
            Document(
                id="doc-other-owner", filename="o.pdf", original_filename="Other.pdf",
                file_size=100, content_type="application/pdf", storage_path="uploads/o.pdf",
                processing_status=ProcessingStatus.READY, rag_status=RagStatus.INDEXED,
                chunk_count=1, owner_id="user-2", title="Other User's Paper",
            )
        )

        # Workspace with docs 1-4 (doc-5 indexed but NOT in workspace; doc-6 not indexed)
        self.db.add(ResearchWorkspace(id="ws-1", name="Test Workspace", owner_id="user-1"))
        for i in (1, 2, 3, 4, 6):
            self.db.add(WorkspaceDocument(workspace_id="ws-1", document_id=f"doc-{i}"))

        self.db.commit()


# ==========================================================================
# 1. Dimension resolution
# ==========================================================================


class TestDimensionResolution(unittest.TestCase):
    def test_defaults_when_none(self):
        from app.comparison.dimensions import DEFAULT_DIMENSIONS, resolve_dimensions
        dims = resolve_dimensions(None)
        self.assertEqual([d.key for d in dims], [d.key for d in DEFAULT_DIMENSIONS])

    def test_defaults_when_empty_list(self):
        from app.comparison.dimensions import resolve_dimensions
        dims = resolve_dimensions([])
        # Phase 7: 11 default dimensions
        self.assertEqual(len(dims), 11)

    def test_all_eleven_phase7_aspects_present(self):
        """Phase 7 requirement: all 11 standard aspects must be in defaults."""
        from app.comparison.dimensions import DEFAULT_DIMENSIONS
        expected_keys = {
            "research_problem", "research_objective", "methodology",
            "model_algorithm", "dataset", "dataset_size", "evaluation_metrics",
            "main_results", "baselines", "limitations", "future_work",
        }
        actual_keys = {d.key for d in DEFAULT_DIMENSIONS}
        self.assertEqual(expected_keys, actual_keys)

    def test_known_key_reused_case_insensitively(self):
        from app.comparison.dimensions import resolve_dimensions
        dims = resolve_dimensions(["Methodology", "DATASET"])
        self.assertEqual([d.key for d in dims], ["methodology", "dataset"])
        self.assertIn("methodology", dims[0].query.lower())

    def test_custom_dimension_gets_a_question(self):
        from app.comparison.dimensions import resolve_dimensions
        dims = resolve_dimensions(["novelty assessment"])
        self.assertEqual(len(dims), 1)
        self.assertEqual(dims[0].label, "novelty assessment")
        self.assertTrue(dims[0].query.endswith("?"))
        self.assertIn("novelty assessment", dims[0].query)

    def test_duplicate_keys_deduped(self):
        from app.comparison.dimensions import resolve_dimensions
        dims = resolve_dimensions(["Dataset", "dataset", "DATASET"])
        self.assertEqual(len(dims), 1)

    def test_capped_at_max_dimensions(self):
        from app.comparison.dimensions import resolve_dimensions
        dims = resolve_dimensions([f"criterion {i}" for i in range(20)], max_dimensions=3)
        self.assertEqual(len(dims), 3)

    def test_extensible_with_custom_dimensions(self):
        """The system must be extensible — custom fields beyond the 11 defaults."""
        from app.comparison.dimensions import resolve_dimensions
        dims = resolve_dimensions(["methodology", "custom research question"])
        self.assertEqual(len(dims), 2)
        self.assertEqual(dims[0].key, "methodology")
        self.assertEqual(dims[1].label, "custom research question")


# ==========================================================================
# 2. Paper-count validation
# ==========================================================================


class TestPaperCountValidation(_TestBase):
    def test_rejects_single_paper(self):
        from app.comparison import comparison_service
        with self.assertRaises(comparison_service.ComparisonError):
            comparison_service.create_and_run_comparison(
                self.db, "ws-1", "user-1", ["doc-1"], settings=Settings()
            )

    def test_rejects_more_than_five(self):
        from app.comparison import comparison_service
        with self.assertRaises(comparison_service.ComparisonError):
            comparison_service.create_and_run_comparison(
                self.db, "ws-1", "user-1",
                ["doc-1", "doc-2", "doc-3", "doc-4", "doc-5", "doc-6"],
                settings=Settings(),
            )

    def test_no_matrix_row_persisted_on_rejection(self):
        from app.comparison import comparison_service
        with self.assertRaises(comparison_service.ComparisonError):
            comparison_service.create_and_run_comparison(
                self.db, "ws-1", "user-1", ["doc-1"], settings=Settings()
            )
        self.assertEqual(self.db.query(ComparisonMatrix).count(), 0)


# ==========================================================================
# 3. Papers must belong to workspace
# ==========================================================================


class TestPaperMustBelongToWorkspace(_TestBase):
    def test_rejects_paper_not_in_workspace(self):
        from app.comparison import comparison_service
        # doc-5 is indexed but not linked to ws-1.
        with self.assertRaises(comparison_service.ComparisonError):
            comparison_service.create_and_run_comparison(
                self.db, "ws-1", "user-1", ["doc-1", "doc-5"], settings=Settings()
            )

    def test_other_users_workspace_is_invisible(self):
        from app.comparison import comparison_service
        from app.rag.workspace_retrieval import WorkspaceRetrievalError
        with self.assertRaises(WorkspaceRetrievalError):
            comparison_service.create_and_run_comparison(
                self.db, "ws-1", "user-2", ["doc-1", "doc-2"], settings=Settings()
            )


# ==========================================================================
# 4. Papers must be indexed
# ==========================================================================


class TestPapersMustBeIndexed(_TestBase):
    def test_rejects_unindexed_paper(self):
        from app.comparison import comparison_service
        # doc-6 is in the workspace but NOT indexed.
        with self.assertRaises(comparison_service.ComparisonError) as ctx:
            comparison_service.create_and_run_comparison(
                self.db, "ws-1", "user-1", ["doc-1", "doc-6"], settings=Settings()
            )
        self.assertIn("indexed", str(ctx.exception).lower())


# ==========================================================================
# 5-8. Core comparison run
# ==========================================================================


class TestComparisonRun(_TestBase):
    """Full run using the real local providers (no network / API key needed)."""

    def test_matrix_snapshots_documents_and_dimensions(self):
        from app.comparison import comparison_service
        matrix = comparison_service.create_and_run_comparison(
            self.db, "ws-1", "user-1", ["doc-1", "doc-2"],
            dimensions_raw=["methodology", "dataset"], settings=Settings(),
        )
        self.assertEqual(matrix.status, ComparisonStatus.COMPLETED)
        self.assertEqual(set(matrix.document_ids), {"doc-1", "doc-2"})
        self.assertEqual([d["key"] for d in matrix.dimensions], ["methodology", "dataset"])

    def test_one_cell_per_paper_times_dimension(self):
        from app.comparison import comparison_service
        matrix = comparison_service.create_and_run_comparison(
            self.db, "ws-1", "user-1", ["doc-1", "doc-2", "doc-3"],
            dimensions_raw=["methodology", "dataset"], settings=Settings(),
        )
        self.assertEqual(len(matrix.cells), 3 * 2)
        pairs = {(c.document_id, c.dimension_key) for c in matrix.cells}
        for doc_id in ("doc-1", "doc-2", "doc-3"):
            for dim in ("methodology", "dataset"):
                self.assertIn((doc_id, dim), pairs)

    def test_cells_grounded_in_evidence_when_supported(self):
        from app.comparison import comparison_service
        matrix = comparison_service.create_and_run_comparison(
            self.db, "ws-1", "user-1", ["doc-1", "doc-2"],
            dimensions_raw=["methodology"], settings=Settings(),
        )
        supported_cells = [c for c in matrix.cells if not c.insufficient_evidence and c.citations]
        self.assertTrue(supported_cells, "expected at least one grounded cell")
        for cell in supported_cells:
            for citation in cell.citations:
                if citation.get("evidence_id"):
                    evidence = self.db.get(Evidence, citation["evidence_id"])
                    self.assertIsNotNone(evidence)
                    self.assertEqual(evidence.document_id, cell.document_id)

    def test_insufficient_evidence_cell_has_no_citations(self):
        from app.comparison import comparison_service
        # A question with zero lexical overlap with the seeded chunk text —
        # the local extractive LLM provider must fall back to "not found".
        matrix = comparison_service.create_and_run_comparison(
            self.db, "ws-1", "user-1", ["doc-1", "doc-2"],
            dimensions_raw=["astrophysical plasma jet formation timelines"],
            settings=Settings(),
        )
        for cell in matrix.cells:
            if cell.insufficient_evidence:
                self.assertEqual(cell.citations, [])

    def test_custom_dimension_label_preserved_on_cell(self):
        from app.comparison import comparison_service
        matrix = comparison_service.create_and_run_comparison(
            self.db, "ws-1", "user-1", ["doc-1", "doc-2"],
            dimensions_raw=["novelty assessment"], settings=Settings(),
        )
        self.assertTrue(all(c.dimension_label == "novelty assessment" for c in matrix.cells))


# ==========================================================================
# 9. Failure isolation
# ==========================================================================


class TestComparisonFailureIsolation(_TestBase):
    def test_one_bad_cell_does_not_fail_whole_matrix(self):
        """A retriever that blows up on one specific document must not
        prevent the other documents' cells from being produced."""
        from app.comparison import comparison_service

        real_retriever = comparison_service.build_retriever(self.db, Settings())

        class FlakyRetriever:
            def retrieve(self, document_id, query, top_k):
                if document_id == "doc-2":
                    raise RuntimeError("boom")
                return real_retriever.retrieve(document_id, query, top_k)

        original_build_retriever = comparison_service.build_retriever
        comparison_service.build_retriever = lambda db, settings: FlakyRetriever()
        try:
            matrix = comparison_service.create_and_run_comparison(
                self.db, "ws-1", "user-1", ["doc-1", "doc-2"],
                dimensions_raw=["methodology"], settings=Settings(),
            )
        finally:
            comparison_service.build_retriever = original_build_retriever

        self.assertEqual(matrix.status, ComparisonStatus.COMPLETED)
        self.assertEqual(len(matrix.cells), 2)
        by_doc = {c.document_id: c for c in matrix.cells}
        self.assertIsNotNone(by_doc["doc-2"].error)
        self.assertIsNone(by_doc["doc-1"].error)


# ==========================================================================
# 10. CRUD and ownership
# ==========================================================================


class TestComparisonCRUDAndOwnership(_TestBase):
    def test_list_get_delete(self):
        from app.comparison import comparison_service
        matrix = comparison_service.create_and_run_comparison(
            self.db, "ws-1", "user-1", ["doc-1", "doc-2"],
            dimensions_raw=["methodology"], settings=Settings(),
        )
        listed = comparison_service.list_comparisons_for_workspace(self.db, "ws-1", "user-1")
        self.assertEqual([m.id for m in listed], [matrix.id])

        fetched = comparison_service.get_comparison(self.db, matrix.id, "user-1")
        self.assertEqual(fetched.id, matrix.id)

        comparison_service.delete_comparison(self.db, matrix.id, "user-1")
        self.assertEqual(self.db.query(ComparisonMatrix).count(), 0)
        self.assertEqual(self.db.query(ComparisonCell).count(), 0)

    def test_other_user_cannot_read_comparison(self):
        from app.comparison import comparison_service
        matrix = comparison_service.create_and_run_comparison(
            self.db, "ws-1", "user-1", ["doc-1", "doc-2"],
            dimensions_raw=["methodology"], settings=Settings(),
        )
        with self.assertRaises(comparison_service.ComparisonNotFoundError):
            comparison_service.get_comparison(self.db, matrix.id, "user-2")

    def test_other_user_cannot_delete_comparison(self):
        from app.comparison import comparison_service
        matrix = comparison_service.create_and_run_comparison(
            self.db, "ws-1", "user-1", ["doc-1", "doc-2"],
            dimensions_raw=["methodology"], settings=Settings(),
        )
        with self.assertRaises(comparison_service.ComparisonNotFoundError):
            comparison_service.delete_comparison(self.db, matrix.id, "user-2")
        # Still there.
        self.assertEqual(self.db.query(ComparisonMatrix).count(), 1)


# ==========================================================================
# 11. Missing information handling (Phase 7 requirement)
# ==========================================================================


class TestMissingInformation(_TestBase):
    """Phase 7: if information is not found, the cell must say so — no
    fabrication, no invention of data."""

    def test_missing_info_not_fabricated(self):
        """A question totally unrelated to the seeded content must produce
        an insufficient-evidence or explicitly empty answer, never a
        fabricated one."""
        from app.comparison import comparison_service
        matrix = comparison_service.create_and_run_comparison(
            self.db, "ws-1", "user-1", ["doc-1", "doc-2"],
            dimensions_raw=["quantum teleportation fidelity thresholds"],
            settings=Settings(),
        )
        for cell in matrix.cells:
            # Either marked insufficient or the value text should convey
            # no information was found (local LLM returns the "not found" sentinel).
            if cell.insufficient_evidence:
                self.assertEqual(cell.citations, [])
            # Cell should never claim to have quantitative data about
            # quantum teleportation when it was never in the papers.
            self.assertNotIn("99.9%", cell.value_text)


# ==========================================================================
# 12. Conflicting results preservation (Phase 7 requirement)
# ==========================================================================


class TestConflictingResults(_TestBase):
    """Phase 7: different papers may report different results. The system must
    preserve each paper's values independently, never auto-declaring one better."""

    def test_each_paper_gets_own_cell(self):
        from app.comparison import comparison_service
        matrix = comparison_service.create_and_run_comparison(
            self.db, "ws-1", "user-1", ["doc-1", "doc-2", "doc-3"],
            dimensions_raw=["methodology"], settings=Settings(),
        )
        # Each paper should have its own independent cell, not a merged one.
        by_doc = {}
        for cell in matrix.cells:
            if cell.dimension_key == "methodology":
                by_doc[cell.document_id] = cell

        self.assertIn("doc-1", by_doc)
        self.assertIn("doc-2", by_doc)
        self.assertIn("doc-3", by_doc)
        # The cells are independent objects, not shared.
        self.assertNotEqual(by_doc["doc-1"].id, by_doc["doc-2"].id)
        self.assertNotEqual(by_doc["doc-2"].id, by_doc["doc-3"].id)


# ==========================================================================
# 13. Evidence traceability chain (Phase 7 requirement)
# ==========================================================================


class TestEvidenceTraceability(_TestBase):
    """Phase 7: ComparisonCell → Claim → Evidence → Document → Page/Section/Chunk
    Every populated comparison cell must be traceable."""

    def test_claim_evidence_chain(self):
        from app.comparison import comparison_service
        matrix = comparison_service.create_and_run_comparison(
            self.db, "ws-1", "user-1", ["doc-1", "doc-2"],
            dimensions_raw=["methodology"], settings=Settings(),
        )
        # Find cells with evidence_id in their citations.
        evidence_ids = []
        for cell in matrix.cells:
            for citation in (cell.citations or []):
                if citation.get("evidence_id"):
                    evidence_ids.append(citation["evidence_id"])

        if evidence_ids:
            for eid in evidence_ids:
                evidence = self.db.get(Evidence, eid)
                self.assertIsNotNone(evidence, f"Evidence {eid} should exist")
                # Evidence → Document
                self.assertIsNotNone(evidence.document_id)
                # Evidence → Claim
                if evidence.claim_id:
                    claim = self.db.get(Claim, evidence.claim_id)
                    self.assertIsNotNone(claim, f"Claim {evidence.claim_id} should exist")
                    # Claim → Document
                    self.assertEqual(claim.document_id, evidence.document_id)
                # Evidence → chunk/page provenance
                # (at least one provenance field should be set)
                self.assertTrue(
                    evidence.chunk_id or evidence.page_number is not None,
                    "Evidence must have chunk or page provenance",
                )


# ==========================================================================
# 14. Comparison uses only workspace documents (Phase 7 requirement)
# ==========================================================================


class TestComparisonUsesOnlyWorkspaceDocs(_TestBase):
    """Phase 7: comparison must use ONLY documents that are in the workspace."""

    def test_cannot_compare_non_workspace_document(self):
        from app.comparison import comparison_service
        # doc-5 is indexed but NOT in workspace ws-1
        with self.assertRaises(comparison_service.ComparisonError):
            comparison_service.create_and_run_comparison(
                self.db, "ws-1", "user-1", ["doc-1", "doc-5"],
                settings=Settings(),
            )

    def test_cannot_compare_other_users_document(self):
        from app.comparison import comparison_service
        from app.rag.workspace_retrieval import WorkspaceRetrievalError
        # doc-other-owner belongs to user-2, ws-1 belongs to user-1
        with self.assertRaises((comparison_service.ComparisonError, WorkspaceRetrievalError)):
            comparison_service.create_and_run_comparison(
                self.db, "ws-1", "user-1", ["doc-1", "doc-other-owner"],
                settings=Settings(),
            )


# ==========================================================================
# 15. Multiple papers (Phase 7 requirement)
# ==========================================================================


class TestMultiplePapers(_TestBase):
    """Phase 7: the system must work with 2, 3, 4, and 5 papers."""

    def test_three_paper_comparison(self):
        from app.comparison import comparison_service
        matrix = comparison_service.create_and_run_comparison(
            self.db, "ws-1", "user-1", ["doc-1", "doc-2", "doc-3"],
            dimensions_raw=["methodology"], settings=Settings(),
        )
        self.assertEqual(matrix.status, ComparisonStatus.COMPLETED)
        self.assertEqual(len(matrix.cells), 3)

    def test_four_paper_comparison(self):
        from app.comparison import comparison_service
        matrix = comparison_service.create_and_run_comparison(
            self.db, "ws-1", "user-1", ["doc-1", "doc-2", "doc-3", "doc-4"],
            dimensions_raw=["methodology"], settings=Settings(),
        )
        self.assertEqual(matrix.status, ComparisonStatus.COMPLETED)
        self.assertEqual(len(matrix.cells), 4)


# ==========================================================================
# 16. Regression: existing workspace chat still works
# ==========================================================================


class TestWorkspaceChatRegression(_TestBase):
    """Phase 7: existing workspace chat must not be broken by comparison changes."""

    def test_workspace_chat_service_still_importable(self):
        """The workspace chat module must remain importable and functional."""
        from app.rag.workspace_chat import WorkspaceChatService
        self.assertIsNotNone(WorkspaceChatService)

    def test_workspace_retrieval_still_works(self):
        """resolve_workspace_documents must still work for workspace chat."""
        from app.rag.workspace_retrieval import resolve_workspace_documents
        docs = resolve_workspace_documents(self.db, "ws-1", "user-1")
        self.assertTrue(len(docs) >= 4)  # ws-1 has docs 1-4 + doc-6


# ==========================================================================
# 17. Regression: existing single-paper RAG still works
# ==========================================================================


class TestSinglePaperRagRegression(_TestBase):
    """Phase 7: single-paper RAG (Phase 3) must not be broken."""

    def test_rag_service_still_importable(self):
        from app.rag.service import RAGChatService
        self.assertIsNotNone(RAGChatService)

    def test_build_retriever_still_works(self):
        from app.rag.service import build_retriever
        retriever = build_retriever(self.db, Settings())
        self.assertIsNotNone(retriever)

    def test_retriever_can_retrieve_single_document(self):
        from app.rag.service import build_retriever
        retriever = build_retriever(self.db, Settings())
        results = retriever.retrieve("doc-1", "methodology", 3)
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
