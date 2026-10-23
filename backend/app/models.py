"""SQLAlchemy ORM models.

Phase 1: users, documents, research_workspaces, workspace_documents.
Phase 2 adds the document-intelligence tables — document_pages,
document_sections, document_elements, document_references — plus the
processing-state and extracted-metadata columns on `documents`.

Auth-ready (owner_id foreign keys in place) even though there is no auth flow
yet."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

# Use a plain String(36) for UUIDs so the same models work with SQLite and PostgreSQL.
UUID = lambda **kw: String(36)

from app.database import Base
from app.rag.vector_type import EmbeddingVector

# Fixed embedding dimensionality for the whole app. Every configured
# EmbeddingProvider must project into this size (the local provider is a
# hashing embedder that targets it exactly; hosted providers are wrapped to
# truncate/pad — see app/rag/embeddings/*). Keeping one dimension keeps the
# vector column schema (and any pgvector index) static regardless of which
# provider is selected at runtime.
EMBEDDING_DIM = 768


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Processing states (Phase 2)
#
#   UPLOADED  -> file is stored, waiting for a worker to claim it
#   PROCESSING-> a worker has claimed it and is parsing
#   READY     -> structured representation is persisted and queryable
#   FAILED    -> parsing failed; processing_error holds the reason
# ---------------------------------------------------------------------------
class ProcessingStatus:
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"

    ALL = (UPLOADED, PROCESSING, READY, FAILED)
    TERMINAL = (READY, FAILED)


# Phase 1 wrote lowercase statuses. Map them forward on startup.
LEGACY_STATUS_MAP = {
    "pending": ProcessingStatus.UPLOADED,
    "uploaded": ProcessingStatus.UPLOADED,
    "processing": ProcessingStatus.PROCESSING,
    "ready": ProcessingStatus.READY,
    "done": ProcessingStatus.READY,
    "failed": ProcessingStatus.FAILED,
    "error": ProcessingStatus.FAILED,
}


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    documents: Mapped[list["Document"]] = relationship(back_populates="owner")
    workspaces: Mapped[list["ResearchWorkspace"]] = relationship(back_populates="owner")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(500))
    original_filename: Mapped[str] = mapped_column(String(500))
    file_size: Mapped[int] = mapped_column(Integer)
    content_type: Mapped[str] = mapped_column(String(120), default="application/pdf")
    storage_path: Mapped[str] = mapped_column(String(1000))
    # UPLOADED -> PROCESSING -> READY | FAILED  (see ProcessingStatus above)
    processing_status: Mapped[str] = mapped_column(
        String(50), default=ProcessingStatus.UPLOADED, index=True
    )
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # -- processing bookkeeping -------------------------------------------
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    processing_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processing_attempts: Mapped[int] = mapped_column(Integer, default=0)
    parser_backend: Mapped[str | None] = mapped_column(String(50), nullable=True)
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False)

    # -- extracted document metadata (all nullable: absent != invented) ----
    title: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    authors: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    word_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doc_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Denormalised counts so the library grid needs no aggregate queries.
    section_count: Mapped[int] = mapped_column(Integer, default=0)
    element_count: Mapped[int] = mapped_column(Integer, default=0)
    table_count: Mapped[int] = mapped_column(Integer, default=0)
    figure_count: Mapped[int] = mapped_column(Integer, default=0)
    reference_count: Mapped[int] = mapped_column(Integer, default=0)

    owner_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"))
    owner: Mapped["User"] = relationship(back_populates="documents")

    workspace_links: Mapped[list["WorkspaceDocument"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    pages: Mapped[list["DocumentPage"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentPage.page_number",
    )
    sections: Mapped[list["DocumentSection"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentSection.order_index",
    )
    elements: Mapped[list["DocumentElement"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentElement.order_index",
    )
    references: Mapped[list["DocumentReference"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentReference.order_index",
    )

    # -- Phase 3: RAG indexing state ---------------------------------------
    # Separate from `processing_status` (Phase 2 parsing) on purpose: a
    # document can be READY (structure extracted, viewable) while chat
    # indexing is still running, failed, or has not started, without that
    # ever regressing the Phase 2 pipeline's own state machine.
    rag_status: Mapped[str] = mapped_column(String(50), default="NOT_INDEXED", index=True)
    rag_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    rag_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    conversations: Mapped[list["ChatConversation"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    # -- Evidence & Research Object Layer (foundation for citation
    # verification, comparison, gap detection etc. — see app/evidence/) ----
    paper: Mapped["ResearchPaper | None"] = relationship(
        back_populates="document", uselist=False, cascade="all, delete-orphan"
    )
    evidence_items: Mapped[list["Evidence"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    claims: Mapped[list["Claim"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class RagStatus:
    NOT_INDEXED = "NOT_INDEXED"
    INDEXING = "INDEXING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"

    ALL = (NOT_INDEXED, INDEXING, INDEXED, FAILED)


class ResearchWorkspace(Base):
    __tablename__ = "research_workspaces"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    owner_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"))
    owner: Mapped["User"] = relationship(back_populates="workspaces")

    document_links: Mapped[list["WorkspaceDocument"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )


class WorkspaceDocument(Base):
    __tablename__ = "workspace_documents"
    __table_args__ = (
        UniqueConstraint("workspace_id", "document_id", name="uq_workspace_document"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("research_workspaces.id"), index=True
    )
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id"), index=True
    )
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    workspace: Mapped["ResearchWorkspace"] = relationship(back_populates="document_links")
    document: Mapped["Document"] = relationship(back_populates="workspace_links")


# ===========================================================================
# Phase 2 — document intelligence tables
#
# Every extracted row keeps enough provenance to point back at the exact place
# in the source PDF it came from: document_id, page_number, section and (when
# the parser supplies geometry) a bounding box in PDF points, top-left origin.
# ===========================================================================


class DocumentPage(Base):
    """One physical page of the source PDF."""

    __tablename__ = "document_pages"
    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_document_page"),
        Index("ix_document_pages_doc_page", "document_id", "page_number"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer)  # 1-based

    width: Mapped[float | None] = mapped_column(Float, nullable=True)
    height: Mapped[float | None] = mapped_column(Float, nullable=True)
    rotation: Mapped[int] = mapped_column(Integer, default=0)

    text: Mapped[str] = mapped_column(Text, default="")
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    # False means the PDF itself carried no extractable text on this page.
    has_text_layer: Mapped[bool] = mapped_column(Boolean, default=True)
    # True means the text above was recovered by OCR rather than read directly.
    ocr_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    image_count: Mapped[int] = mapped_column(Integer, default=0)

    document: Mapped["Document"] = relationship(back_populates="pages")


class DocumentSection(Base):
    """A logical section, derived from the document's heading hierarchy."""

    __tablename__ = "document_sections"
    __table_args__ = (
        Index("ix_document_sections_doc_order", "document_id", "order_index"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    parent_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("document_sections.id", ondelete="CASCADE"), nullable=True
    )

    title: Mapped[str] = mapped_column(String(1000))
    level: Mapped[int] = mapped_column(Integer, default=1)
    order_index: Mapped[int] = mapped_column(Integer, default=0)

    start_page: Mapped[int] = mapped_column(Integer, default=1)
    end_page: Mapped[int] = mapped_column(Integer, default=1)

    text: Mapped[str] = mapped_column(Text, default="")
    char_count: Mapped[int] = mapped_column(Integer, default=0)

    document: Mapped["Document"] = relationship(back_populates="sections")
    elements: Mapped[list["DocumentElement"]] = relationship(
        back_populates="section", order_by="DocumentElement.order_index"
    )

    # The parent/child hierarchy is exposed through `parent_id` rather than a
    # self-referential relationship: the API assembles the tree in one pass,
    # which avoids the N+1 queries a lazy self-join would cause here.


class DocumentElement(Base):
    """An atomic extracted element: paragraph, heading, table, figure, caption,
    list item or formula. This is the table the RAG phase will chunk from."""

    __tablename__ = "document_elements"
    __table_args__ = (
        Index("ix_document_elements_doc_order", "document_id", "order_index"),
        Index("ix_document_elements_doc_type", "document_id", "element_type"),
        Index("ix_document_elements_doc_page", "document_id", "page_number"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("document_pages.id", ondelete="CASCADE"), nullable=True
    )
    section_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("document_sections.id", ondelete="CASCADE"), nullable=True
    )

    page_number: Mapped[int] = mapped_column(Integer, index=True)
    element_type: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text, default="")
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    level: Mapped[int | None] = mapped_column(Integer, nullable=True)  # headings only

    # Bounding box in PDF points, top-left origin. Null when the backend could
    # not supply geometry — never zero-filled to fake precision we do not have.
    bbox_x0: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_y0: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_x1: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_y1: Mapped[float | None] = mapped_column(Float, nullable=True)

    element_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="elements")
    section: Mapped["DocumentSection | None"] = relationship(back_populates="elements")


class DocumentReference(Base):
    """One bibliography entry. Structured fields are best-effort: a field left
    as NULL means it was not confidently determinable, and `raw_text` always
    keeps the verbatim entry as it appeared in the PDF."""

    __tablename__ = "document_references"
    __table_args__ = (
        Index("ix_document_references_doc_order", "document_id", "order_index"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )

    order_index: Mapped[int] = mapped_column(Integer, default=0)
    raw_text: Mapped[str] = mapped_column(Text)

    authors: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    title: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True)
    url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="references")


# ===========================================================================
# Phase 3 — evidence-grounded paper chat
#
# document_chunks holds both "parent" (broad context) and "child" (precise,
# embedded, retrieved) chunks in one table, linked by parent_chunk_id. This
# mirrors the well-known "small-to-big" / parent-document retrieval pattern:
# similarity search runs over the small, topically-focused child chunks (more
# precise embeddings, less dilution), but the LLM is handed the wider parent
# chunk's text for context once a child chunk matches — see
# app/rag/retriever.py and app/rag/chunking.py.
# ===========================================================================


class ChunkLevel:
    PARENT = "parent"
    CHILD = "child"


class DocumentChunk(Base):
    """One chunk of cleaned document text, ready for embedding + retrieval.

    Required provenance fields (per spec): document_id, page_number,
    section_id, chunk_id (`id`), parent_chunk_id, text, embedding, metadata.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("ix_document_chunks_doc_level", "document_id", "chunk_level"),
        Index("ix_document_chunks_doc_page", "document_id", "page_number"),
        Index("ix_document_chunks_parent", "parent_chunk_id"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)  # chunk_id
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, index=True)
    section_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("document_sections.id", ondelete="SET NULL"), nullable=True
    )
    parent_chunk_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=True
    )

    chunk_level: Mapped[str] = mapped_column(String(10), default=ChunkLevel.CHILD)
    order_index: Mapped[int] = mapped_column(Integer, default=0)

    text: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    # Null for parent chunks (only children are embedded/searched directly).
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector(EMBEDDING_DIM), nullable=True)

    # section title, start/end page, contributing element ids + types,
    # approximate bbox of the first contributing element (for the frontend's
    # best-effort citation highlight) — see app/rag/chunking.py.
    chunk_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class ChatConversation(Base):
    __tablename__ = "chat_conversations"
    __table_args__ = (Index("ix_chat_conversations_doc", "document_id", "created_at"),)

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    document_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True, index=True
    )
    # Phase 6A: workspace-scoped conversations (multi-paper chat).
    # Null for single-document conversations; set for workspace chat.
    workspace_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("research_workspaces.id", ondelete="CASCADE"), nullable=True, index=True
    )
    owner_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"))
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    document: Mapped["Document"] = relationship(back_populates="conversations")
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatRole:
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (Index("ix_chat_messages_conv_created", "conversation_id", "created_at"),)

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("chat_conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))  # user | assistant
    content: Mapped[str] = mapped_column(Text)

    # List[dict] — see app/rag/interfaces.py:Citation. Empty list for user
    # messages and for assistant answers the paper did not support.
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    insufficient_evidence: Mapped[bool] = mapped_column(Boolean, default=False)

    llm_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # True when generation stopped partway (stream error / client gone):
    # `content` then holds only the partial answer, never shown as complete.
    incomplete: Mapped[bool] = mapped_column(Boolean, default=False)

    # Phase 4: full retrieval score breakdown for this turn (query, every
    # candidate's vector/keyword/fusion/rerank scores, which were selected,
    # config, timings) — see app/rag/interfaces.py:RetrievalDebug. Only
    # populated when DEBUG_MODE is on (app/config.py); None for user messages
    # and for assistant answers produced while debug mode was off. Never sent
    # to the frontend unless DEBUG_MODE is still on at read time — see
    # app/routers/chat.py's `_message_out`.
    retrieval_debug: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    conversation: Mapped["ChatConversation"] = relationship(back_populates="messages")


# ===========================================================================
# Evidence & Research Object Layer (foundation for the next generation of
# ResearchPilot: citation verification, multi-paper comparison, literature
# discovery, gap detection, ideation, writing assistance, similarity
# checking — see app/evidence/).
#
# Conceptual pipeline this layer sits on top of, without replacing any of it:
#
#   Document -> Document Structure -> Chunk -> Retrieval Result
#            -> Evidence -> Claim -> Answer / Citation
#
# Evidence never copies document/chunk text into a new source of truth: it
# references existing rows (document_id, chunk_id, page_id, section_id) and
# keeps only what retrieval/generation actually produced (snippet, scores,
# confidence) — the same provenance discipline `DocumentElement`/`DocumentChunk`
# already follow (nullable geometry, never zero-filled to fake precision).
# ===========================================================================


class SupportStatus:
    """How well a piece of Evidence backs the Claim it is attached to.

    Deliberately unopinionated at this stage: Task 4/5 only establish the
    data model + service layer. Real support-status *classification* (i.e.
    deciding SUPPORTED vs PARTIALLY_SUPPORTED for a given claim/evidence
    pair) is future citation-verification work, not part of this layer.
    """

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"

    ALL = (SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, UNCERTAIN)


class EvidenceSourceType:
    """What kind of source object an Evidence row points back at."""

    DOCUMENT = "document"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    REFERENCE = "reference"
    EXTERNAL_METADATA = "external_metadata"

    ALL = (DOCUMENT, PARAGRAPH, TABLE, FIGURE, REFERENCE, EXTERNAL_METADATA)


class ClaimType:
    REPORTED_RESULT = "reported_result"
    REPORTED_METHOD = "reported_method"
    REPORTED_FACT = "reported_fact"
    INFERENCE = "inference"
    HYPOTHESIS = "hypothesis"
    UNKNOWN = "unknown"

    ALL = (
        REPORTED_RESULT,
        REPORTED_METHOD,
        REPORTED_FACT,
        INFERENCE,
        HYPOTHESIS,
        UNKNOWN,
    )


class MetadataSource:
    """Where a ResearchPaper's metadata came from. Only `document_parser` is
    populated by this task — the others are the seam future phases (Semantic
    Scholar / OpenAlex / DOI-resolver lookups, explicitly out of scope here)
    will write into."""

    DOCUMENT_PARSER = "document_parser"
    REFERENCE_ENTRY = "reference_entry"
    MANUAL = "manual"
    EXTERNAL_LOOKUP = "external_lookup"

    ALL = (DOCUMENT_PARSER, REFERENCE_ENTRY, MANUAL, EXTERNAL_LOOKUP)


class ResearchPaper(Base):
    """Paper-level metadata for a Document, normalized into its own table
    rather than bolted onto `documents` further, so later phases (multi-paper
    comparison, literature discovery) can query/join papers without dragging
    in upload/processing bookkeeping columns.

    One row per Document (1:1). Seeded from Phase 2's extracted
    `Document.title` / `Document.authors` at creation time — see
    `app/evidence/paper_service.py`. Every other field is left NULL until a
    real source for it exists; nothing here is ever invented (no fabricated
    DOIs, authors, years, or URLs — see `app/evidence/paper_service.py:build_doi_url`).
    """

    __tablename__ = "research_papers"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("documents.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )

    title: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    authors: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(String(500), nullable=True)

    doi: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Derived, never hand-set: only populated when `doi` is a real value —
    # see build_doi_url(). Stored (rather than computed on every read) so the
    # API and the frontend never have to re-derive/guess the construction
    # rule, and so a bad DOI can never silently produce a bad link.
    doi_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    arxiv_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    arxiv_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    publisher_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # document_parser | reference_entry | manual | external_lookup
    metadata_source: Mapped[str] = mapped_column(
        String(50), default=MetadataSource.DOCUMENT_PARSER
    )
    metadata_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    document: Mapped["Document"] = relationship(back_populates="paper")


class Claim(Base):
    """One important statement generated or extracted by ResearchPilot
    (currently: an assistant chat answer — see `app/evidence/claim_service.py`
    and `app/rag/service.py`). Deliberately holds no verification logic of
    its own; `support_status` here is a foundation field for future
    citation-verification work, set to a simple default for now.
    """

    __tablename__ = "claims"
    __table_args__ = (Index("ix_claims_document_created", "document_id", "created_at"),)

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    workspace_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("research_workspaces.id", ondelete="SET NULL"), nullable=True
    )
    conversation_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("chat_conversations.id", ondelete="SET NULL"), nullable=True
    )
    message_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("chat_messages.id", ondelete="SET NULL"), nullable=True
    )

    claim_text: Mapped[str] = mapped_column(Text)
    # reported_result | reported_method | reported_fact | inference | hypothesis | unknown
    claim_type: Mapped[str] = mapped_column(String(32), default=ClaimType.UNKNOWN)
    # supported | partially_supported | unsupported | uncertain
    support_status: Mapped[str] = mapped_column(String(32), default=SupportStatus.UNCERTAIN)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    document: Mapped["Document"] = relationship(back_populates="claims")
    evidence_items: Mapped[list["Evidence"]] = relationship(
        back_populates="claim", cascade="all, delete-orphan", order_by="Evidence.created_at"
    )


class Evidence(Base):
    """One normalized piece of evidence, pointing back at an existing source
    object (a retrieved/indexed chunk, page, section) rather than duplicating
    its text. Converted from `RetrievedChunk` results produced by the
    existing hybrid retrieval pipeline — see
    `app/evidence/evidence_service.py:evidence_from_retrieved_chunk` — never
    from a second retrieval path.

    `claim_id` is nullable: evidence can be created standalone (e.g. surfaced
    to a document's evidence list) before ever being attached to a claim.
    """

    __tablename__ = "evidence"
    __table_args__ = (
        Index("ix_evidence_document_created", "document_id", "created_at"),
        Index("ix_evidence_claim", "claim_id"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)

    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    claim_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("claims.id", ondelete="SET NULL"), nullable=True, index=True
    )
    chunk_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("document_chunks.id", ondelete="SET NULL"), nullable=True
    )
    page_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("document_pages.id", ondelete="SET NULL"), nullable=True
    )
    section_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("document_sections.id", ondelete="SET NULL"), nullable=True
    )

    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Denormalised snapshots of the section/subsection title at the moment
    # this evidence was created — same rationale as `DocumentChunk.chunk_metadata`
    # keeping a section title copy: the citation must still read correctly
    # even if the section is later re-extracted (reprocess) or removed.
    section_title: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    subsection_title: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    snippet: Mapped[str] = mapped_column(Text)

    # Bounding box in PDF points, top-left origin — same convention as
    # `DocumentElement`. Null when no geometry is available (best-effort only).
    bbox_x0: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_y0: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_x1: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_y1: Mapped[float | None] = mapped_column(Float, nullable=True)

    retrieval_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # document | paragraph | table | figure | reference | external_metadata
    source_type: Mapped[str] = mapped_column(String(32), default=EvidenceSourceType.PARAGRAPH)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # supported | partially_supported | unsupported | uncertain
    support_status: Mapped[str] = mapped_column(String(32), default=SupportStatus.UNCERTAIN)

    # Free-form extras that don't warrant their own column yet (e.g. the
    # vector/keyword/fusion/rerank score breakdown for a retrieval-sourced
    # piece of evidence, or a reference id for a reference-sourced one).
    evidence_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    document: Mapped["Document"] = relationship(back_populates="evidence_items")
    claim: Mapped["Claim | None"] = relationship(back_populates="evidence_items")


# ===========================================================================
# Phase 6B — Evidence-backed multi-paper comparison
#
# Built entirely on top of the existing pipeline, never duplicating it:
#
#   Workspace -> Select 2-5 Papers -> Compare Papers
#            -> Targeted Retrieval    (existing Retriever, one query per
#                                      dimension, scoped to one document_id
#                                      at a time — see app/comparison/)
#            -> Evidence              (existing Evidence layer, unchanged)
#            -> Structured Extraction (existing LLMProvider.generate(), one
#                                      short grounded answer per paper x
#                                      dimension cell)
#            -> Comparison Matrix     (ComparisonMatrix + ComparisonCell)
#            -> View Evidence         (existing GET /api/evidence/{id})
#
# A ComparisonMatrix snapshots which documents and which dimensions it was
# run with (JSON, not live FKs to a mutable "current dimension list") so a
# saved comparison keeps reading the same way even if the workspace's
# documents or the app's default dimension set change later — the same
# provenance discipline `DocumentChunk.chunk_metadata` and
# `Evidence.section_title` already follow.
# ===========================================================================


class ComparisonStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    ALL = (PENDING, RUNNING, COMPLETED, FAILED)


class ComparisonMatrix(Base):
    """One "Compare Papers" run: a fixed set of 2-5 documents x a fixed set
    of comparison dimensions, evaluated once. Re-running with different
    papers or dimensions creates a new matrix rather than mutating this one,
    so a workspace can keep a history of comparisons (e.g. before/after
    adding a paper) — see `app/comparison/comparison_service.py`.
    """

    __tablename__ = "comparison_matrices"
    __table_args__ = (Index("ix_comparison_matrices_workspace", "workspace_id", "created_at"),)

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("research_workspaces.id", ondelete="CASCADE"), index=True
    )
    owner_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"))

    title: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Snapshots, not live relationships — see class docstring.
    document_ids: Mapped[list] = mapped_column(JSON)  # list[str], ordered (matrix columns)
    dimensions: Mapped[list] = mapped_column(JSON)  # list[{"key","label","query"}], ordered (matrix rows)

    # pending | running | completed | failed
    status: Mapped[str] = mapped_column(String(20), default=ComparisonStatus.PENDING, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Are the compared papers directly comparable (same dataset / metric /
    # task / split / setup)? Judged from the filled cells by
    # app/comparison/comparability.py: {overall, issues, caveat, checked_by}.
    # NULL = not checked, or the check failed (the matrix is still valid).
    comparability: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    workspace: Mapped["ResearchWorkspace"] = relationship()
    cells: Mapped[list["ComparisonCell"]] = relationship(
        back_populates="comparison",
        cascade="all, delete-orphan",
        order_by="ComparisonCell.order_index",
    )
    drafts: Mapped[list["LiteratureDraft"]] = relationship(
        back_populates="comparison",
        cascade="all, delete-orphan",
        order_by="LiteratureDraft.created_at",
    )


class ComparisonCell(Base):
    """One (paper, dimension) cell of a ComparisonMatrix: a short,
    citation-grounded extraction produced the same way a chat answer is —
    targeted retrieval scoped to `document_id`, then
    `LLMProvider.generate()` — never a second extraction mechanism. Citations
    are the same `Citation.to_dict()` shape chat uses, each optionally
    grounded in a persisted `Evidence` row via `evidence_id`, so "View
    Evidence" reuses the existing evidence endpoint/dialog verbatim.
    """

    __tablename__ = "comparison_cells"
    __table_args__ = (
        UniqueConstraint("comparison_id", "document_id", "dimension_key", name="uq_comparison_cell"),
        Index("ix_comparison_cells_comparison", "comparison_id", "order_index"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    comparison_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("comparison_matrices.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )

    dimension_key: Mapped[str] = mapped_column(String(100))
    dimension_label: Mapped[str] = mapped_column(String(200))
    order_index: Mapped[int] = mapped_column(Integer, default=0)

    value_text: Mapped[str] = mapped_column(Text, default="")
    insufficient_evidence: Mapped[bool] = mapped_column(Boolean, default=False)

    # list[Citation.to_dict()] — see app/rag/interfaces.py:Citation.
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)

    llm_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Structured comparison only (see app/comparison/paper_extractor.py): was
    # the quote the model gave for this cell found, verbatim, in the paper's
    # stored text? True = verified, False = the model quoted something the
    # paper does not contain. NULL = not applicable (per-cell answers, and
    # cells where the paper simply did not say).
    quote_verified: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    comparison: Mapped["ComparisonMatrix"] = relationship(back_populates="cells")


class LiteratureDraft(Base):
    """A related-work draft generated from a ComparisonMatrix
    (app/writing/review_writer.py). `content` is generated once and never
    edited: {paragraphs: [{sentences: [{text, evidence_ids, citation_labels,
    status}]}], caveat, dropped_count}. The user's edits live in
    `edited_text`."""

    __tablename__ = "literature_drafts"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    comparison_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("comparison_matrices.id", ondelete="CASCADE"), index=True
    )
    owner_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"))
    content: Mapped[dict] = mapped_column(JSON)
    edited_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    comparison: Mapped["ComparisonMatrix"] = relationship(back_populates="drafts")


# ===========================================================================
# Phase 8 — Research Gap Detection & Research Opportunity Engine
#
# Evidence-first gap detection: the system identifies potential research gaps
# by cross-referencing Claims and Evidence from multiple workspace papers,
# NOT by asking an LLM to hallucinate gaps. Every persisted ResearchGap
# must have at least one supporting Evidence row.
#
# ResearchGap → GapEvidence (association) → Evidence → Document/Chunk
# ResearchGap → GapClaim (association) → Claim → Evidence
# ===========================================================================


class GapType:
    """Taxonomy of research gap types."""
    METHODOLOGICAL = "methodological"
    DATASET = "dataset"
    EVALUATION = "evaluation"
    DOMAIN = "domain"
    POPULATION = "population"
    COMPARATIVE = "comparative"
    THEORETICAL = "theoretical"
    REPRODUCIBILITY = "reproducibility"
    CONTRADICTORY = "contradictory"
    FUTURE_WORK = "future_work"

    ALL = (
        METHODOLOGICAL, DATASET, EVALUATION, DOMAIN, POPULATION,
        COMPARATIVE, THEORETICAL, REPRODUCIBILITY, CONTRADICTORY,
        FUTURE_WORK,
    )

    LABELS = {
        METHODOLOGICAL: "Methodological Gap",
        DATASET: "Dataset Gap",
        EVALUATION: "Evaluation Gap",
        DOMAIN: "Domain Gap",
        POPULATION: "Population Gap",
        COMPARATIVE: "Comparative Gap",
        THEORETICAL: "Theoretical Gap",
        REPRODUCIBILITY: "Reproducibility Gap",
        CONTRADICTORY: "Contradictory Findings",
        FUTURE_WORK: "Future-Work Gap",
    }


class GapStatus:
    PENDING = "pending"
    IDENTIFIED = "identified"
    VERIFIED = "verified"
    DISMISSED = "dismissed"

    ALL = (PENDING, IDENTIFIED, VERIFIED, DISMISSED)


class GapConfidence:
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    ALL = (HIGH, MEDIUM, LOW)


class GapEvidence(Base):
    """Association table linking a ResearchGap to its supporting Evidence
    items. A gap can reference evidence from multiple papers; an evidence
    item can support multiple gaps."""

    __tablename__ = "gap_evidence"
    __table_args__ = (
        UniqueConstraint("gap_id", "evidence_id", name="uq_gap_evidence"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    gap_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("research_gaps.id", ondelete="CASCADE"), index=True
    )
    evidence_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("evidence.id", ondelete="CASCADE"), index=True
    )
    # Role of this evidence piece in supporting the gap
    # e.g. "limitation", "future_work", "methodology", "contradiction"
    role: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    gap: Mapped["ResearchGap"] = relationship(back_populates="gap_evidence_items")
    evidence: Mapped["Evidence"] = relationship()


class GapClaim(Base):
    """Association linking a ResearchGap to the (existing) Claim rows that
    support it. Claims are per-paper statements grounded in Evidence; a gap
    references claims from several papers. Reuses the Phase 5 Claim model —
    no parallel claim structure is created."""

    __tablename__ = "gap_claims"
    __table_args__ = (
        UniqueConstraint("gap_id", "claim_id", name="uq_gap_claim"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    gap_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("research_gaps.id", ondelete="CASCADE"), index=True
    )
    claim_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("claims.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    gap: Mapped["ResearchGap"] = relationship(back_populates="gap_claim_items")
    claim: Mapped["Claim"] = relationship()


class ResearchGap(Base):
    """A potential research gap identified from cross-paper analysis of
    Evidence and Claims within a workspace. Every gap must have at least
    one supporting Evidence row — unsupported gaps are never persisted.

    Wording is deliberately cautious: "Potential research gap identified
    from the analyzed papers" — the system cannot establish global novelty
    from a limited workspace.
    """

    __tablename__ = "research_gaps"
    __table_args__ = (
        Index("ix_research_gaps_workspace", "workspace_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("research_workspaces.id", ondelete="CASCADE"), index=True
    )
    owner_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"))

    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str] = mapped_column(Text)

    # See GapType.ALL
    gap_type: Mapped[str] = mapped_column(String(50))
    # See GapConfidence.ALL
    confidence: Mapped[str] = mapped_column(String(20), default=GapConfidence.MEDIUM)
    # Human-readable explanation of the confidence level
    confidence_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    # See GapStatus.ALL
    status: Mapped[str] = mapped_column(String(20), default=GapStatus.IDENTIFIED)

    # AI-generated research direction (clearly labeled as such)
    research_direction: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Which papers contributed to this gap (snapshot of document IDs)
    source_paper_ids: Mapped[list] = mapped_column(JSON, default=list)
    # Number of supporting papers
    supporting_paper_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    workspace: Mapped["ResearchWorkspace"] = relationship()
    gap_evidence_items: Mapped[list["GapEvidence"]] = relationship(
        back_populates="gap", cascade="all, delete-orphan",
        order_by="GapEvidence.created_at",
    )
    gap_claim_items: Mapped[list["GapClaim"]] = relationship(
        back_populates="gap", cascade="all, delete-orphan",
        order_by="GapClaim.created_at",
    )
