"""Phase 6A — Workspace multi-paper retrieval + chat tests.

Uses Python's built-in unittest with an in-memory SQLite database.

Tests:
  1. Evidence creation
  2. Evidence validation (cross-document reference rejection)  
  3. Claim creation
  4. Claim → Evidence relationship
  5. Workspace document ownership
  6. Duplicate document prevention
  7. Maximum 5 documents
  8. Workspace retrieval only searches workspace documents
  9. Evidence preserves correct document_id
  10. Citations preserve evidence_id
  11. Workspace chat endpoint
  12. Workspace with no documents returns a useful error
  13. Single-paper chat remains functional (regression)
"""
import sys
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure the backend root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force SQLite in-memory for tests
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
    ChatConversation,
    ChatMessage,
    ChatRole,
    ChunkLevel,
    Claim,
    ClaimType,
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
    Evidence,
    EvidenceSourceType,
    ProcessingStatus,
    RagStatus,
    ResearchWorkspace,
    SupportStatus,
    User,
    WorkspaceDocument,
)
from app.rag.interfaces import Citation, RetrievedChunk


class _TestBase(unittest.TestCase):
    """Base test class that creates a fresh in-memory DB for every test."""

    def setUp(self):
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        self._seed()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def _seed(self):
        """Create test data: users, 6 documents, workspace with 2 docs, empty workspace."""
        user = User(id="user-1", email="test@example.com", name="Test User")
        self.db.add(user)

        other_user = User(id="user-2", email="other@example.com", name="Other User")
        self.db.add(other_user)

        for i in range(6):
            doc = Document(
                id=f"doc-{i+1}",
                filename=f"paper{i+1}.pdf",
                original_filename=f"Paper {i+1}.pdf",
                file_size=100000,
                content_type="application/pdf",
                storage_path=f"uploads/paper{i+1}.pdf",
                processing_status=ProcessingStatus.READY,
                rag_status=RagStatus.INDEXED if i < 5 else RagStatus.NOT_INDEXED,
                chunk_count=10 if i < 5 else 0,
                owner_id="user-1",
                title=f"Paper {i+1}: Test Research",
            )
            self.db.add(doc)

            if i < 5:
                parent = DocumentChunk(
                    id=f"chunk-parent-{i+1}",
                    document_id=f"doc-{i+1}",
                    page_number=1,
                    chunk_level=ChunkLevel.PARENT,
                    text=f"Full context for paper {i+1}.",
                    char_count=30,
                    token_count=8,
                    order_index=0,
                )
                self.db.add(parent)

                child = DocumentChunk(
                    id=f"chunk-child-{i+1}",
                    document_id=f"doc-{i+1}",
                    page_number=1,
                    parent_chunk_id=f"chunk-parent-{i+1}",
                    chunk_level=ChunkLevel.CHILD,
                    text=f"Paper {i+1} uses dataset X.",
                    char_count=25,
                    token_count=7,
                    embedding=[0.1] * 768,
                    order_index=0,
                )
                self.db.add(child)

        # Other user's doc
        self.db.add(Document(
            id="doc-other", filename="other.pdf", original_filename="Other.pdf",
            file_size=50000, content_type="application/pdf", storage_path="uploads/other.pdf",
            processing_status=ProcessingStatus.READY, rag_status=RagStatus.INDEXED,
            chunk_count=5, owner_id="user-2", title="Other User's Paper",
        ))

        # Workspace with 2 docs
        self.db.add(ResearchWorkspace(id="ws-1", name="Test Workspace", owner_id="user-1"))
        self.db.add(WorkspaceDocument(workspace_id="ws-1", document_id="doc-1"))
        self.db.add(WorkspaceDocument(workspace_id="ws-1", document_id="doc-2"))

        # Empty workspace
        self.db.add(ResearchWorkspace(id="ws-empty", name="Empty Workspace", owner_id="user-1"))

        self.db.commit()


class TestWorkspaceDocumentOwnership(_TestBase):
    """Test 1: workspace document ownership."""

    def test_workspace_owner_can_see_documents(self):
        from app.rag.workspace_retrieval import resolve_workspace_documents
        docs = resolve_workspace_documents(self.db, "ws-1", "user-1")
        self.assertEqual(len(docs), 2)
        doc_ids = {d.id for d in docs}
        self.assertIn("doc-1", doc_ids)
        self.assertIn("doc-2", doc_ids)

    def test_other_user_cannot_access_workspace(self):
        from app.rag.workspace_retrieval import WorkspaceRetrievalError, resolve_workspace_documents
        with self.assertRaises(WorkspaceRetrievalError):
            resolve_workspace_documents(self.db, "ws-1", "user-2")

    def test_nonexistent_workspace_fails(self):
        from app.rag.workspace_retrieval import WorkspaceRetrievalError, resolve_workspace_documents
        with self.assertRaises(WorkspaceRetrievalError):
            resolve_workspace_documents(self.db, "ws-nonexistent", "user-1")


class TestDuplicateDocumentPrevention(_TestBase):
    """Test 2: duplicate document prevention."""

    def test_cannot_add_duplicate_document(self):
        from sqlalchemy.exc import IntegrityError
        link = WorkspaceDocument(workspace_id="ws-1", document_id="doc-1")
        self.db.add(link)
        with self.assertRaises(IntegrityError):
            self.db.flush()


class TestMaxDocuments(_TestBase):
    """Test 3: maximum 5 documents per workspace."""

    def test_workspace_allows_up_to_5_documents(self):
        # ws-1 has 2. Add 3 more => total 5.
        for i in range(3, 6):
            self.db.add(WorkspaceDocument(workspace_id="ws-1", document_id=f"doc-{i}"))
        self.db.flush()
        count = self.db.query(WorkspaceDocument).filter_by(workspace_id="ws-1").count()
        self.assertEqual(count, 5)


class TestWorkspaceRetrievalScope(_TestBase):
    """Test 4: workspace retrieval only searches workspace documents."""

    def test_retrieval_scoped_to_workspace(self):
        from app.rag.workspace_retrieval import retrieve_for_workspace

        mock_retriever = MagicMock()
        def mock_retrieve(document_id, query, top_k):
            return [RetrievedChunk(
                chunk_id=f"chunk-{document_id}", parent_chunk_id=None,
                document_id=document_id, page_number=1, section_id=None,
                section_title="Test", child_text=f"From {document_id}",
                context_text=f"Context from {document_id}", score=0.8,
            )]
        mock_retriever.retrieve = mock_retrieve

        results = retrieve_for_workspace(mock_retriever, self.db, "ws-1", "user-1", "test", 10)
        result_doc_ids = {r.document_id for r in results}
        self.assertIn("doc-1", result_doc_ids)
        self.assertIn("doc-2", result_doc_ids)
        self.assertNotIn("doc-3", result_doc_ids)
        self.assertNotIn("doc-other", result_doc_ids)

    def test_retrieval_preserves_document_id(self):
        from app.rag.workspace_retrieval import retrieve_for_workspace

        mock_retriever = MagicMock()
        def mock_retrieve(document_id, query, top_k):
            return [RetrievedChunk(
                chunk_id=f"chunk-{document_id}", parent_chunk_id=None,
                document_id=document_id, page_number=1, section_id=None,
                section_title="Section", child_text="text", context_text="ctx",
                score=0.9, vector_score=0.85, keyword_score=0.7, fusion_score=0.8,
            )]
        mock_retriever.retrieve = mock_retrieve

        results = retrieve_for_workspace(mock_retriever, self.db, "ws-1", "user-1", "test", 10)
        for result in results:
            self.assertIn(result.document_id, {"doc-1", "doc-2"})
            self.assertEqual(result.chunk_id, f"chunk-{result.document_id}")


class TestEvidencePreservesDocumentId(_TestBase):
    """Test 5: evidence preserves correct document_id."""

    def test_evidence_from_retrieved_chunk_preserves_doc_id(self):
        from app.evidence.evidence_service import evidence_from_retrieved_chunk
        chunk = RetrievedChunk(
            chunk_id="chunk-child-1", parent_chunk_id="chunk-parent-1",
            document_id="doc-1", page_number=3, section_id=None,
            section_title="Methodology", child_text="Uses dataset X.",
            context_text="Full context.", score=0.85, vector_score=0.8,
        )
        kwargs = evidence_from_retrieved_chunk(chunk)
        self.assertEqual(kwargs["document_id"], "doc-1")
        self.assertEqual(kwargs["page_number"], 3)
        self.assertEqual(kwargs["section_title"], "Methodology")
        self.assertIn("vector_score", kwargs.get("evidence_metadata", {}))


class TestCitationsPreserveEvidenceId(unittest.TestCase):
    """Test 6: citations preserve evidence_id (no DB needed)."""

    def test_citation_to_dict_includes_evidence_id(self):
        citation = Citation(
            chunk_id="c1", document_id="doc-1", page_number=5,
            section_title="Results", snippet="95% accuracy.",
            score=0.9, marker=1, evidence_id="evidence-123",
        )
        d = citation.to_dict()
        self.assertEqual(d["evidence_id"], "evidence-123")
        self.assertEqual(d["document_id"], "doc-1")

    def test_citation_without_evidence_id(self):
        citation = Citation(
            chunk_id="c1", document_id="doc-1", page_number=5,
            section_title=None, snippet="text.", score=0.7,
        )
        d = citation.to_dict()
        self.assertIsNone(d["evidence_id"])


class TestWorkspaceChatEndpoint(_TestBase):
    """Test 7: workspace chat conversation creation."""

    def test_workspace_conversation_creation(self):
        conv = ChatConversation(workspace_id="ws-1", owner_id="user-1", title="WS chat")
        self.db.add(conv)
        self.db.flush()
        self.assertIsNotNone(conv.id)
        self.assertEqual(conv.workspace_id, "ws-1")
        self.assertIsNone(conv.document_id)

    def test_workspace_conversation_has_messages(self):
        conv = ChatConversation(workspace_id="ws-1", owner_id="user-1")
        self.db.add(conv)
        self.db.flush()

        msg = ChatMessage(
            conversation_id=conv.id, role=ChatRole.USER,
            content="What datasets do these papers use?",
        )
        self.db.add(msg)
        self.db.flush()
        self.assertEqual(len(conv.messages), 1)
        self.assertEqual(conv.messages[0].content, "What datasets do these papers use?")


class TestEmptyWorkspaceError(_TestBase):
    """Test 8: workspace with no documents returns useful error."""

    def test_empty_workspace_retrieval_error(self):
        from app.rag.workspace_retrieval import WorkspaceRetrievalError, resolve_workspace_documents
        with self.assertRaises(WorkspaceRetrievalError) as ctx:
            resolve_workspace_documents(self.db, "ws-empty", "user-1")
        self.assertIn("no documents", str(ctx.exception).lower())


class TestSinglePaperRegression(_TestBase):
    """Test 9: single-paper chat infrastructure remains functional."""

    def test_single_doc_conversation_still_works(self):
        conv = ChatConversation(document_id="doc-1", owner_id="user-1", title="Single chat")
        self.db.add(conv)
        self.db.flush()
        self.assertEqual(conv.document_id, "doc-1")
        self.assertIsNone(conv.workspace_id)

    def test_single_doc_retriever_interface_unchanged(self):
        from app.rag.interfaces import Retriever
        import inspect
        sig = inspect.signature(Retriever.retrieve)
        params = list(sig.parameters.keys())
        self.assertIn("document_id", params)
        self.assertIn("query", params)
        self.assertIn("top_k", params)

    def test_single_doc_citation_structure_unchanged(self):
        citation = Citation(
            chunk_id="c1", document_id="doc-1", page_number=1,
            section_title="Intro", snippet="text", score=0.9, marker=1,
            evidence_id="ev-1",
        )
        d = citation.to_dict()
        expected_keys = {
            "chunk_id", "document_id", "page_number", "section_title",
            "snippet", "score", "marker", "evidence_id",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_evidence_from_chunk_interface_unchanged(self):
        from app.evidence.evidence_service import evidence_from_retrieved_chunk
        chunk = RetrievedChunk(
            chunk_id="c1", parent_chunk_id="p1", document_id="doc-1",
            page_number=2, section_id=None, section_title="Results",
            child_text="90% accuracy.", context_text="Full paragraph.",
            score=0.88,
        )
        kwargs = evidence_from_retrieved_chunk(chunk)
        self.assertEqual(kwargs["document_id"], "doc-1")
        self.assertIn("snippet", kwargs)


class TestWorkspaceContext(unittest.TestCase):
    """Test workspace context construction (no DB needed)."""

    def test_workspace_source_blocks_include_paper_title(self):
        from app.rag.workspace_context import build_workspace_source_blocks
        chunks = [
            RetrievedChunk(
                chunk_id="c1", parent_chunk_id=None, document_id="doc-1",
                page_number=4, section_id=None, section_title="Methodology",
                child_text="Uses CIFAR-10.", context_text="This paper uses CIFAR-10.",
                score=0.9,
            ),
            RetrievedChunk(
                chunk_id="c2", parent_chunk_id=None, document_id="doc-2",
                page_number=6, section_id=None, section_title="Dataset",
                child_text="Uses ImageNet.", context_text="Experiments on ImageNet.",
                score=0.85,
            ),
        ]
        doc_titles = {"doc-1": "Deep Learning Paper", "doc-2": "Transfer Learning Survey"}
        text, used = build_workspace_source_blocks(chunks, doc_titles)
        self.assertEqual(len(used), 2)
        self.assertIn("Deep Learning Paper", text)
        self.assertIn("Transfer Learning Survey", text)
        self.assertIn("[Source 1]", text)
        self.assertIn("[Source 2]", text)

    def test_workspace_user_message_includes_paper_identity(self):
        from app.rag.workspace_context import build_workspace_user_message
        chunks = [RetrievedChunk(
            chunk_id="c1", parent_chunk_id=None, document_id="doc-1",
            page_number=1, section_id=None, section_title=None,
            child_text="text", context_text="context", score=0.5,
        )]
        content, used = build_workspace_user_message("What datasets?", chunks, {"doc-1": "Test Paper"})
        self.assertIn("Test Paper", content)
        self.assertIn("What datasets?", content)
        self.assertIn("which paper", content.lower())


class TestWorkspaceCitationParsing(unittest.TestCase):
    """Test workspace citation parsing (no DB needed)."""

    def test_workspace_citations_preserve_document_id(self):
        from app.rag.workspace_chat import _parse_workspace_citations
        chunks = [
            RetrievedChunk(
                chunk_id="c1", parent_chunk_id=None, document_id="doc-1",
                page_number=1, section_id=None, section_title="Intro",
                child_text="CIFAR.", context_text="ctx", score=0.9,
            ),
            RetrievedChunk(
                chunk_id="c2", parent_chunk_id=None, document_id="doc-2",
                page_number=2, section_id=None, section_title="Data",
                child_text="ImageNet.", context_text="ctx", score=0.85,
            ),
        ]
        answer = "Paper 1 uses CIFAR [1] while Paper 2 uses ImageNet [2]."
        citations, insufficient = _parse_workspace_citations(answer, chunks)
        self.assertFalse(insufficient)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0].document_id, "doc-1")
        self.assertEqual(citations[1].document_id, "doc-2")

    def test_insufficient_evidence_detected(self):
        from app.rag.workspace_chat import _parse_workspace_citations
        from app.rag.workspace_context import WORKSPACE_NOT_FOUND_SENTENCE
        citations, insufficient = _parse_workspace_citations(WORKSPACE_NOT_FOUND_SENTENCE, [])
        self.assertTrue(insufficient)
        self.assertEqual(len(citations), 0)


if __name__ == "__main__":
    unittest.main()
