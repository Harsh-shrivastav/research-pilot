"""Pydantic v2 request/response schemas — the API contract for the frontend."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    """Summary view used by the library grid, dashboard and workspace header."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    original_filename: str
    file_size: int
    content_type: str
    # UPLOADED | PROCESSING | READY | FAILED
    processing_status: str
    uploaded_at: datetime
    file_url: str

    # -- Phase 2 extraction results (all optional: absent != invented) -----
    title: str | None = None
    authors: str | None = None
    page_count: int | None = None
    word_count: int | None = None
    parser_backend: str | None = None
    ocr_used: bool = False

    section_count: int = 0
    element_count: int = 0
    table_count: int = 0
    figure_count: int = 0
    reference_count: int = 0

    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None
    processing_duration_ms: int | None = None
    processing_error: str | None = None

    # -- Phase 3: RAG chat indexing (absent/zero until indexing runs) ------
    rag_status: str = "NOT_INDEXED"
    rag_error: str | None = None
    chunk_count: int = 0
    embedding_provider: str | None = None
    embedding_model: str | None = None


class ProcessingStatusOut(BaseModel):
    """Small payload the UI polls while a document is being processed."""

    id: str
    processing_status: str
    processing_error: str | None = None
    processing_duration_ms: int | None = None
    page_count: int | None = None
    section_count: int = 0
    element_count: int = 0
    table_count: int = 0
    figure_count: int = 0
    reference_count: int = 0


class BBoxOut(BaseModel):
    """Bounding box in PDF points, top-left origin."""

    x0: float
    y0: float
    x1: float
    y1: float


class ElementOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    page_number: int
    element_type: str
    content: str
    order_index: int
    level: int | None = None
    section_id: str | None = None
    bbox: BBoxOut | None = None
    metadata: dict | None = None


class SectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    parent_id: str | None = None
    title: str
    level: int
    order_index: int
    start_page: int
    end_page: int
    char_count: int
    preview: str = ""


class SectionDetailOut(SectionOut):
    """A single section with its full text and constituent elements."""

    text: str = ""
    elements: list[ElementOut] = []


class PageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    page_number: int
    width: float | None = None
    height: float | None = None
    rotation: int = 0
    char_count: int
    has_text_layer: bool
    ocr_applied: bool
    image_count: int


class PageDetailOut(PageOut):
    text: str = ""
    elements: list[ElementOut] = []


class ReferenceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    order_index: int
    raw_text: str
    authors: str | None = None
    title: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    page_number: int | None = None


class DocumentDetailOut(DocumentOut):
    """Everything the document workspace needs in one request."""

    doc_metadata: dict | None = None
    sections: list[SectionOut] = []
    pages: list[PageOut] = []
    warnings: list[str] = []


class WorkspaceCreate(BaseModel):
    name: str
    description: str | None = None
    document_id: str | None = None


class WorkspaceAddDocument(BaseModel):
    document_id: str


class WorkspaceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
    document_count: int = 0


class WorkspaceDetailOut(WorkspaceOut):
    documents: list[DocumentOut] = []


class WorkspaceChatAskIn(BaseModel):
    question: str
    conversation_id: str | None = None


class WorkspaceChatConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    workspace_id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ActivityItem(BaseModel):
    id: str
    type: str  # "document_uploaded" | "workspace_created" | "document_deleted"
    message: str
    timestamp: datetime


class DashboardStats(BaseModel):
    total_documents: int
    active_workspaces: int
    recent_documents: list[DocumentOut]
    recent_activity: list[ActivityItem]


# ---------------------------------------------------------------------------
# Phase 3: RAG chat
# ---------------------------------------------------------------------------


class RagStatusOut(BaseModel):
    """Polled by the UI while a document is being indexed for chat, the same
    way ProcessingStatusOut is polled during Phase 2 parsing."""

    id: str
    rag_status: str  # NOT_INDEXED | INDEXING | INDEXED | FAILED
    rag_error: str | None = None
    chunk_count: int = 0
    embedding_provider: str | None = None
    embedding_model: str | None = None


class CitationOut(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    section_title: str | None = None
    snippet: str
    score: float
    marker: int | None = None
    # Evidence & Research Object Layer: id of the persisted Evidence row this
    # citation was grounded in, if grounding succeeded — see app/evidence/.
    # Fetch it with GET /api/evidence/{evidence_id} for the "View Evidence" panel.
    evidence_id: str | None = None


class DebugCandidateOut(BaseModel):
    """One candidate's full score breakdown, for the Retrieval Debug panel
    only — see app/rag/interfaces.py:DebugCandidate."""

    chunk_id: str
    page_number: int
    section_title: str | None = None
    snippet: str
    vector_rank: int | None = None
    vector_score: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    matched_keyword_terms: list[str] = []
    fusion_score: float | None = None
    rerank_score: float | None = None
    selected: bool = False


class RetrievalDebugOut(BaseModel):
    """Development/admin-only payload: the preprocessed query, every
    candidate the hybrid pipeline considered, and the config/timings that
    produced this run. Never populated in an API response unless
    `DEBUG_MODE=true` on the server — see `app/routers/chat.py`."""

    query: dict
    mode: str
    candidates: list[DebugCandidateOut] = []
    config: dict
    timings_ms: dict[str, float] = {}


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    role: str
    content: str
    citations: list[CitationOut] = []
    insufficient_evidence: bool = False
    llm_provider: str | None = None
    llm_model: str | None = None
    error: str | None = None
    incomplete: bool = False
    created_at: datetime
    # Only ever non-null when DEBUG_MODE is on, both at the time the message
    # was answered (whether it was computed/stored at all — see
    # RAGChatService.answer) and at read time (whether it's serialized here —
    # see routers/chat.py's `_message_out`). Absent entirely for normal users.
    retrieval_debug: RetrievalDebugOut | None = None


class ChatConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ChatConversationDetailOut(ChatConversationOut):
    messages: list[ChatMessageOut] = []


class ChatAskIn(BaseModel):
    question: str
    conversation_id: str | None = None


class SuggestedQuestionsOut(BaseModel):
    questions: list[str]


# ---------------------------------------------------------------------------
# Evidence & Research Object Layer
# ---------------------------------------------------------------------------


class ResearchPaperOut(BaseModel):
    """Paper-level metadata for a document. Every URL/id field is either a
    real, non-fabricated value or null — see app/evidence/paper_service.py."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    title: str | None = None
    authors: str | None = None
    abstract: str | None = None
    publication_year: int | None = None
    venue: str | None = None
    doi: str | None = None
    doi_url: str | None = None
    arxiv_id: str | None = None
    arxiv_url: str | None = None
    publisher_url: str | None = None
    source_url: str | None = None
    metadata_source: str
    metadata_confidence: float | None = None
    created_at: datetime
    updated_at: datetime


class EvidenceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    claim_id: str | None = None
    chunk_id: str | None = None
    page_id: str | None = None
    section_id: str | None = None
    page_number: int | None = None
    section_title: str | None = None
    subsection_title: str | None = None
    snippet: str
    bbox: BBoxOut | None = None
    retrieval_score: float | None = None
    source_type: str
    confidence: float | None = None
    support_status: str
    created_at: datetime
    # Denormalised so a citation card never needs a second round trip just to
    # show the paper title/DOI link alongside the snippet.
    document_title: str | None = None
    paper: ResearchPaperOut | None = None


class ClaimOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    message_id: str | None = None
    claim_text: str
    claim_type: str
    support_status: str
    confidence: float | None = None
    created_at: datetime


class ClaimDetailOut(ClaimOut):
    evidence: list[EvidenceOut] = []


# ---------------------------------------------------------------------------
# Phase 6B: Evidence-backed multi-paper comparison
# ---------------------------------------------------------------------------


class ComparisonCreateIn(BaseModel):
    document_ids: list[str]
    # Free-text criteria, or the default set's keys/labels. Omitted/empty ->
    # the default five dimensions — see app/comparison/dimensions.py.
    dimensions: list[str] | None = None
    title: str | None = None


class ComparisonDimensionOut(BaseModel):
    key: str
    label: str
    query: str


class ComparisonCellOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    # Denormalised so the matrix never needs a second round trip per cell.
    document_title: str | None = None
    dimension_key: str
    dimension_label: str
    value_text: str
    insufficient_evidence: bool = False
    citations: list[CitationOut] = []
    llm_provider: str | None = None
    llm_model: str | None = None
    error: str | None = None
    # See ComparisonCell.quote_verified: true / false for structured cells
    # that carried a quote, null when not applicable.
    quote_verified: bool | None = None
    created_at: datetime


class ComparabilityCellRefOut(BaseModel):
    document_id: str
    dimension_key: str


class ComparabilityIssueOut(BaseModel):
    # dataset | metric | task | split | setup
    type: str
    # One plain sentence naming the papers and the difference.
    description: str
    document_ids: list[str] = []
    # The matrix cells the judgement is based on (evidence-first: every issue
    # points at the cells - and through them the quotes - behind it).
    cell_refs: list[ComparabilityCellRefOut] = []


class ComparabilityOut(BaseModel):
    # comparable | partially_comparable | not_comparable | unknown
    overall: str
    issues: list[ComparabilityIssueOut] = []
    # ONE plain sentence a student could paste into a literature review.
    caveat: str = ""
    # rules | gemini
    checked_by: str = "rules"


class ComparisonMatrixOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    workspace_id: str
    title: str | None = None
    # pending | running | completed | failed
    status: str
    error: str | None = None
    dimensions: list[ComparisonDimensionOut] = []
    documents: list[DocumentOut] = []
    cells: list[ComparisonCellOut] = []
    # null when the comparability check has not run or failed.
    comparability: ComparabilityOut | None = None
    created_at: datetime
    updated_at: datetime


class ComparisonSummaryOut(BaseModel):
    """List view — no cells, so listing a workspace's comparison history is
    cheap even once a matrix has dozens of cells."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    workspace_id: str
    title: str | None = None
    status: str
    document_count: int
    dimension_count: int
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Phase 8: Research Gap Detection
# ---------------------------------------------------------------------------


class GapEvidenceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    evidence_id: str
    role: str | None = None
    # Inline evidence detail so the frontend doesn't need N+1 calls
    evidence: EvidenceOut | None = None


class GapClaimOut(BaseModel):
    """A Claim (per-paper statement) supporting a gap, with its Evidence."""
    id: str
    claim_id: str
    document_id: str
    document_title: str | None = None
    claim_text: str
    claim_type: str
    support_status: str
    evidence: list[EvidenceOut] = []


class ResearchGapOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    workspace_id: str
    title: str
    description: str
    gap_type: str
    gap_type_label: str = ""
    confidence: str
    confidence_reasoning: str | None = None
    status: str
    research_direction: str | None = None
    source_paper_ids: list[str] = []
    supporting_paper_count: int = 0
    evidence_items: list[GapEvidenceOut] = []
    claims: list[GapClaimOut] = []
    # Fixed wording: the system cannot establish novelty from a workspace.
    novelty_notice: str = (
        "Potential research gap identified from the analyzed papers. "
        "Novelty requires verification against the broader literature."
    )
    created_at: datetime
    updated_at: datetime


class ResearchGapSummaryOut(BaseModel):
    """List view — no evidence items, just counts."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    workspace_id: str
    title: str
    description: str
    gap_type: str
    gap_type_label: str = ""
    confidence: str
    status: str
    supporting_paper_count: int = 0
    evidence_count: int = 0
    research_direction: str | None = None
    created_at: datetime
    updated_at: datetime


class GapAnalysisResult(BaseModel):
    """Response for the POST analyze endpoint."""
    papers_analyzed: int
    gaps_found: int
    candidates_rejected: int = 0
    gaps: list[ResearchGapSummaryOut]
    gap_type_counts: dict[str, int] = {}
