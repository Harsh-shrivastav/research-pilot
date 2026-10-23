"""Phase 5 Evidence Foundation tests.

Uses Python's built-in unittest with an in-memory SQLite database — no
external dependencies beyond what the project already requires.

Tests:
  1. Evidence creation
  2. Evidence validation (cross-document reference rejection)
  3. Claim creation
  4. Claim → Evidence relationship
  5. Evidence retrieval
  6. DOI URL generation
  7. arXiv URL generation
  8. Citation evidence_id serialization
  9. Existing RAG citation compatibility (no evidence_id)
"""
import sys
import os
import unittest
from datetime import datetime, timezone

# Ensure the backend root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force SQLite in-memory for tests — must be set before any app import
os.environ["DATABASE_URL"] = "sqlite://"
# Env vars beat backend/.env: keep tests on the offline providers so they
# never call a real API, whatever the developer has configured.
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["LLM_PROVIDER"] = "local"
os.environ["GEMINI_API_KEY"] = ""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base
from app.models import (
    Claim,
    ClaimType,
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
    Evidence,
    EvidenceSourceType,
    ResearchPaper,
    SupportStatus,
    User,
)
from app.evidence.evidence_service import (
    EvidenceNotFoundError,
    EvidenceValidationError,
    create_evidence,
    get_evidence,
    list_evidence_for_document,
    attach_evidence_to_claim,
    evidence_from_retrieved_chunk,
    persist_evidence_from_retrieved_chunks,
)
from app.evidence.claim_service import (
    ClaimNotFoundError,
    ClaimValidationError,
    create_claim,
    get_claim,
    list_claims_for_document,
)
from app.evidence.paper_service import (
    build_doi_url,
    build_arxiv_url,
    get_or_create_paper,
    get_paper_for_document,
)
from app.rag.interfaces import Citation, RetrievedChunk


def _make_engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return engine


def _seed(db: Session):
    """Create minimal test data: a user, two documents, a page, section,
    and chunk for document 1."""
    user = User(id="user-1", email="test@example.com", name="Test User")
    db.add(user)

    doc1 = Document(
        id="doc-1",
        filename="paper.pdf",
        original_filename="paper.pdf",
        file_size=1000,
        storage_path="/uploads/paper.pdf",
        owner_id="user-1",
        title="Test Paper Title",
        authors="Author A, Author B",
    )
    doc2 = Document(
        id="doc-2",
        filename="other.pdf",
        original_filename="other.pdf",
        file_size=2000,
        storage_path="/uploads/other.pdf",
        owner_id="user-1",
    )
    db.add_all([doc1, doc2])

    page = DocumentPage(id="page-1", document_id="doc-1", page_number=3)
    section = DocumentSection(
        id="sec-1", document_id="doc-1", title="Introduction", level=1
    )
    chunk = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        page_number=3,
        section_id="sec-1",
        chunk_level="child",
        text="Test chunk text for retrieval.",
        char_count=30,
    )
    # A page on doc-2 for cross-document validation tests
    page2 = DocumentPage(id="page-2", document_id="doc-2", page_number=1)
    section2 = DocumentSection(
        id="sec-2", document_id="doc-2", title="Abstract", level=1
    )
    chunk2 = DocumentChunk(
        id="chunk-2",
        document_id="doc-2",
        page_number=1,
        section_id="sec-2",
        chunk_level="child",
        text="Other doc chunk.",
        char_count=16,
    )
    db.add_all([page, section, chunk, page2, section2, chunk2])
    db.commit()


class EvidenceTestCase(unittest.TestCase):
    """Base class with fresh in-memory DB per test."""

    def setUp(self):
        self.engine = _make_engine()
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        _seed(self.db)

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()


# ========================================================================
# 1. Evidence creation
# ========================================================================
class TestEvidenceCreation(EvidenceTestCase):
    def test_create_evidence_basic(self):
        ev = create_evidence(
            self.db,
            document_id="doc-1",
            snippet="Important finding on page 3.",
            source_type=EvidenceSourceType.PARAGRAPH,
            chunk_id="chunk-1",
            page_id="page-1",
            section_id="sec-1",
            page_number=3,
            section_title="Introduction",
            retrieval_score=0.85,
        )
        self.assertIsNotNone(ev.id)
        self.assertEqual(ev.document_id, "doc-1")
        self.assertEqual(ev.snippet, "Important finding on page 3.")
        self.assertEqual(ev.source_type, EvidenceSourceType.PARAGRAPH)
        self.assertEqual(ev.page_number, 3)
        self.assertAlmostEqual(ev.retrieval_score, 0.85)

    def test_create_evidence_nonexistent_document(self):
        with self.assertRaises(EvidenceValidationError):
            create_evidence(
                self.db, document_id="no-such-doc", snippet="test"
            )


# ========================================================================
# 2. Evidence validation — cross-document rejection
# ========================================================================
class TestEvidenceValidation(EvidenceTestCase):
    def test_reject_chunk_from_different_document(self):
        """chunk-2 belongs to doc-2, so creating evidence for doc-1 with it must fail."""
        with self.assertRaises(EvidenceValidationError):
            create_evidence(
                self.db,
                document_id="doc-1",
                snippet="test",
                chunk_id="chunk-2",  # belongs to doc-2
            )

    def test_reject_page_from_different_document(self):
        with self.assertRaises(EvidenceValidationError):
            create_evidence(
                self.db,
                document_id="doc-1",
                snippet="test",
                page_id="page-2",  # belongs to doc-2
            )

    def test_reject_section_from_different_document(self):
        with self.assertRaises(EvidenceValidationError):
            create_evidence(
                self.db,
                document_id="doc-1",
                snippet="test",
                section_id="sec-2",  # belongs to doc-2
            )

    def test_reject_nonexistent_chunk(self):
        with self.assertRaises(EvidenceValidationError):
            create_evidence(
                self.db,
                document_id="doc-1",
                snippet="test",
                chunk_id="no-such-chunk",
            )

    def test_reject_invalid_source_type(self):
        with self.assertRaises(EvidenceValidationError):
            create_evidence(
                self.db,
                document_id="doc-1",
                snippet="test",
                source_type="invalid_type",
            )

    def test_reject_invalid_support_status(self):
        with self.assertRaises(EvidenceValidationError):
            create_evidence(
                self.db,
                document_id="doc-1",
                snippet="test",
                support_status="invalid_status",
            )


# ========================================================================
# 3. Claim creation
# ========================================================================
class TestClaimCreation(EvidenceTestCase):
    def test_create_claim_basic(self):
        claim = create_claim(
            self.db,
            document_id="doc-1",
            claim_text="The method achieves 95% accuracy.",
            claim_type=ClaimType.REPORTED_RESULT,
        )
        self.assertIsNotNone(claim.id)
        self.assertEqual(claim.document_id, "doc-1")
        self.assertEqual(claim.claim_text, "The method achieves 95% accuracy.")
        self.assertEqual(claim.claim_type, ClaimType.REPORTED_RESULT)
        self.assertEqual(claim.support_status, SupportStatus.UNCERTAIN)

    def test_reject_empty_claim_text(self):
        with self.assertRaises(ClaimValidationError):
            create_claim(self.db, document_id="doc-1", claim_text="   ")

    def test_reject_nonexistent_document(self):
        with self.assertRaises(ClaimValidationError):
            create_claim(self.db, document_id="no-such-doc", claim_text="text")

    def test_reject_invalid_claim_type(self):
        with self.assertRaises(ClaimValidationError):
            create_claim(
                self.db,
                document_id="doc-1",
                claim_text="text",
                claim_type="invalid",
            )


# ========================================================================
# 4. Claim → Evidence relationship
# ========================================================================
class TestClaimEvidenceRelationship(EvidenceTestCase):
    def test_attach_evidence_to_claim(self):
        claim = create_claim(
            self.db, document_id="doc-1", claim_text="Claim text."
        )
        ev = create_evidence(
            self.db,
            document_id="doc-1",
            snippet="Supporting evidence.",
            chunk_id="chunk-1",
        )
        self.assertIsNone(ev.claim_id)

        ev = attach_evidence_to_claim(self.db, ev.id, claim.id)
        self.assertEqual(ev.claim_id, claim.id)

    def test_reject_cross_document_attachment(self):
        claim = create_claim(
            self.db, document_id="doc-1", claim_text="Claim in doc-1."
        )
        ev = create_evidence(
            self.db, document_id="doc-2", snippet="Evidence in doc-2."
        )
        with self.assertRaises(EvidenceValidationError):
            attach_evidence_to_claim(self.db, ev.id, claim.id)

    def test_claim_evidence_items_list(self):
        claim = create_claim(
            self.db, document_id="doc-1", claim_text="Claim."
        )
        ev1 = create_evidence(
            self.db,
            document_id="doc-1",
            snippet="Evidence 1.",
            claim_id=claim.id,
        )
        ev2 = create_evidence(
            self.db,
            document_id="doc-1",
            snippet="Evidence 2.",
            claim_id=claim.id,
        )
        self.db.refresh(claim)
        self.assertEqual(len(claim.evidence_items), 2)


# ========================================================================
# 5. Evidence retrieval
# ========================================================================
class TestEvidenceRetrieval(EvidenceTestCase):
    def test_get_evidence_success(self):
        ev = create_evidence(
            self.db, document_id="doc-1", snippet="Retrievable."
        )
        fetched = get_evidence(self.db, ev.id)
        self.assertEqual(fetched.id, ev.id)
        self.assertEqual(fetched.snippet, "Retrievable.")

    def test_get_evidence_not_found(self):
        with self.assertRaises(EvidenceNotFoundError):
            get_evidence(self.db, "nonexistent-id")

    def test_list_evidence_for_document(self):
        create_evidence(self.db, document_id="doc-1", snippet="A.")
        create_evidence(self.db, document_id="doc-1", snippet="B.")
        create_evidence(self.db, document_id="doc-2", snippet="C.")

        items = list_evidence_for_document(self.db, "doc-1")
        self.assertEqual(len(items), 2)
        # All belong to doc-1
        self.assertTrue(all(e.document_id == "doc-1" for e in items))


# ========================================================================
# 6. DOI URL generation
# ========================================================================
class TestDOIUrl(EvidenceTestCase):
    def test_valid_doi(self):
        url = build_doi_url("10.1234/test.paper")
        self.assertEqual(url, "https://doi.org/10.1234/test.paper")

    def test_none_doi(self):
        self.assertIsNone(build_doi_url(None))

    def test_empty_doi(self):
        self.assertIsNone(build_doi_url(""))

    def test_invalid_doi(self):
        self.assertIsNone(build_doi_url("not-a-doi"))

    def test_already_url_doi(self):
        url = build_doi_url("https://doi.org/10.1234/already")
        self.assertEqual(url, "https://doi.org/10.1234/already")


# ========================================================================
# 7. arXiv URL generation
# ========================================================================
class TestArxivUrl(EvidenceTestCase):
    def test_valid_arxiv_id(self):
        url = build_arxiv_url("2301.12345")
        self.assertEqual(url, "https://arxiv.org/abs/2301.12345")

    def test_valid_arxiv_id_with_version(self):
        url = build_arxiv_url("2301.12345v2")
        self.assertEqual(url, "https://arxiv.org/abs/2301.12345v2")

    def test_none_arxiv(self):
        self.assertIsNone(build_arxiv_url(None))

    def test_empty_arxiv(self):
        self.assertIsNone(build_arxiv_url(""))

    def test_invalid_arxiv(self):
        self.assertIsNone(build_arxiv_url("not-an-arxiv-id"))

    def test_already_url_arxiv(self):
        url = build_arxiv_url("https://arxiv.org/abs/2301.12345")
        self.assertEqual(url, "https://arxiv.org/abs/2301.12345")


# ========================================================================
# 8. Citation evidence_id serialization
# ========================================================================
class TestCitationSerialization(unittest.TestCase):
    def test_citation_with_evidence_id(self):
        c = Citation(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=3,
            section_title="Intro",
            snippet="Some text",
            score=0.9,
            marker=1,
            evidence_id="ev-123",
        )
        d = c.to_dict()
        self.assertEqual(d["evidence_id"], "ev-123")
        self.assertEqual(d["chunk_id"], "chunk-1")
        self.assertEqual(d["marker"], 1)

    def test_citation_without_evidence_id(self):
        c = Citation(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=3,
            section_title=None,
            snippet="Some text",
            score=0.9,
        )
        d = c.to_dict()
        self.assertIsNone(d["evidence_id"])
        self.assertIsNone(d["marker"])


# ========================================================================
# 9. Existing RAG citation compatibility
# ========================================================================
class TestRagCitationCompatibility(unittest.TestCase):
    """Verify the Citation dataclass stays backward-compatible: code that
    creates Citations without evidence_id still works."""

    def test_citation_defaults(self):
        c = Citation(
            chunk_id="c1",
            document_id="d1",
            page_number=1,
            section_title=None,
            snippet="text",
            score=0.5,
        )
        self.assertIsNone(c.evidence_id)
        self.assertIsNone(c.marker)
        d = c.to_dict()
        self.assertIn("evidence_id", d)
        self.assertIsNone(d["evidence_id"])

    def test_score_rounding(self):
        c = Citation(
            chunk_id="c1",
            document_id="d1",
            page_number=1,
            section_title=None,
            snippet="text",
            score=0.123456789,
        )
        d = c.to_dict()
        self.assertEqual(d["score"], 0.1235)  # rounded to 4 decimal places


# ========================================================================
# Paper service
# ========================================================================
class TestPaperService(EvidenceTestCase):
    def test_get_or_create_paper(self):
        doc = self.db.get(Document, "doc-1")
        paper = get_or_create_paper(self.db, doc)
        self.assertIsNotNone(paper.id)
        self.assertEqual(paper.document_id, "doc-1")
        self.assertEqual(paper.title, "Test Paper Title")
        self.assertEqual(paper.authors, "Author A, Author B")
        self.assertEqual(paper.metadata_source, "document_parser")
        self.assertAlmostEqual(paper.metadata_confidence, 0.5)

    def test_get_or_create_paper_idempotent(self):
        doc = self.db.get(Document, "doc-1")
        paper1 = get_or_create_paper(self.db, doc)
        paper2 = get_or_create_paper(self.db, doc)
        self.assertEqual(paper1.id, paper2.id)

    def test_get_paper_for_document(self):
        doc = self.db.get(Document, "doc-1")
        get_or_create_paper(self.db, doc)
        paper = get_paper_for_document(self.db, "doc-1")
        self.assertIsNotNone(paper)
        self.assertEqual(paper.document_id, "doc-1")

    def test_get_paper_for_document_nonexistent(self):
        paper = get_paper_for_document(self.db, "doc-1")
        self.assertIsNone(paper)


# ========================================================================
# Evidence from RetrievedChunk conversion
# ========================================================================
class TestEvidenceFromRetrievedChunk(EvidenceTestCase):
    def test_evidence_from_retrieved_chunk(self):
        chunk = RetrievedChunk(
            chunk_id="chunk-1",
            parent_chunk_id=None,
            document_id="doc-1",
            page_number=3,
            section_id="sec-1",
            section_title="Introduction",
            child_text="Some retrieved text.",
            context_text="Larger context text.",
            score=0.85,
            vector_score=0.9,
            keyword_score=0.7,
            fusion_score=0.8,
            rerank_score=0.85,
        )
        kwargs = evidence_from_retrieved_chunk(chunk)
        self.assertEqual(kwargs["document_id"], "doc-1")
        self.assertEqual(kwargs["chunk_id"], "chunk-1")
        self.assertEqual(kwargs["page_number"], 3)
        self.assertEqual(kwargs["section_title"], "Introduction")
        self.assertIn("vector_score", kwargs["evidence_metadata"])

    def test_persist_evidence_from_retrieved_chunks(self):
        chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                parent_chunk_id=None,
                document_id="doc-1",
                page_number=3,
                section_id="sec-1",
                section_title="Intro",
                child_text="Text A.",
                context_text="Context A.",
                score=0.9,
            ),
        ]
        created = persist_evidence_from_retrieved_chunks(
            self.db, "doc-1", chunks
        )
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].document_id, "doc-1")
        self.assertIsNotNone(created[0].id)


if __name__ == "__main__":
    unittest.main()
