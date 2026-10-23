export type ProcessingStatus = "UPLOADED" | "PROCESSING" | "READY" | "FAILED";

export type ElementType =
  | "heading"
  | "paragraph"
  | "table"
  | "figure"
  | "caption"
  | "list_item"
  | "formula"
  | "reference";

export interface DocumentOut {
  id: string;
  filename: string;
  original_filename: string;
  file_size: number;
  content_type: string;
  processing_status: ProcessingStatus;
  uploaded_at: string;
  file_url: string;

  // Phase 2 extraction results — null/0 until processing completes.
  title: string | null;
  authors: string | null;
  page_count: number | null;
  word_count: number | null;
  parser_backend: string | null;
  ocr_used: boolean;

  section_count: number;
  element_count: number;
  table_count: number;
  figure_count: number;
  reference_count: number;

  processing_started_at: string | null;
  processing_completed_at: string | null;
  processing_duration_ms: number | null;
  processing_error: string | null;

  // Phase 3: RAG chat indexing — null/0 until indexing runs.
  rag_status: RagStatus;
  rag_error: string | null;
  chunk_count: number;
  embedding_provider: string | null;
  embedding_model: string | null;
}

export interface ProcessingStatusOut {
  id: string;
  processing_status: ProcessingStatus;
  processing_error: string | null;
  processing_duration_ms: number | null;
  page_count: number | null;
  section_count: number;
  element_count: number;
  table_count: number;
  figure_count: number;
  reference_count: number;
}

export interface BBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface ElementOut {
  id: string;
  document_id: string;
  page_number: number;
  element_type: ElementType;
  content: string;
  order_index: number;
  level: number | null;
  section_id: string | null;
  bbox: BBox | null;
  metadata: Record<string, unknown> | null;
}

export interface SectionOut {
  id: string;
  document_id: string;
  parent_id: string | null;
  title: string;
  level: number;
  order_index: number;
  start_page: number;
  end_page: number;
  char_count: number;
  preview: string;
}

export interface SectionDetailOut extends SectionOut {
  text: string;
  elements: ElementOut[];
}

export interface PageOut {
  id: string;
  document_id: string;
  page_number: number;
  width: number | null;
  height: number | null;
  rotation: number;
  char_count: number;
  has_text_layer: boolean;
  ocr_applied: boolean;
  image_count: number;
}

export interface PageDetailOut extends PageOut {
  text: string;
  elements: ElementOut[];
}

export interface ReferenceOut {
  id: string;
  document_id: string;
  order_index: number;
  raw_text: string;
  authors: string | null;
  title: string | null;
  year: number | null;
  venue: string | null;
  doi: string | null;
  url: string | null;
  page_number: number | null;
}

export interface DocumentDetailOut extends DocumentOut {
  doc_metadata: Record<string, unknown> | null;
  sections: SectionOut[];
  pages: PageOut[];
  warnings: string[];
}

export interface ProcessingInfo {
  backends_available: string[];
  backend_order: string[];
  configured_backend: string;
  ocr_enabled: boolean;
  ocr_available: boolean;
  ocr_dpi: number;
  max_ocr_pages: number;
  worker_count: number;
  max_upload_mb: number;
}

export interface WorkspaceOut {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  document_count: number;
}

export interface WorkspaceDetailOut extends WorkspaceOut {
  documents: DocumentOut[];
}

export interface ActivityItem {
  id: string;
  type: "document_uploaded" | "workspace_created" | "document_deleted";
  message: string;
  timestamp: string;
}

export interface DashboardStats {
  total_documents: number;
  active_workspaces: number;
  recent_documents: DocumentOut[];
  recent_activity: ActivityItem[];
}

export const TERMINAL_STATUSES: ProcessingStatus[] = ["READY", "FAILED"];

export function isProcessing(status: ProcessingStatus): boolean {
  return status === "UPLOADED" || status === "PROCESSING";
}

// ---------------------------------------------------------------------------
// Phase 3: RAG chat
// ---------------------------------------------------------------------------

export type RagStatus = "NOT_INDEXED" | "INDEXING" | "INDEXED" | "FAILED";

export interface RagStatusOut {
  id: string;
  rag_status: RagStatus;
  rag_error: string | null;
  chunk_count: number;
  embedding_provider: string | null;
  embedding_model: string | null;
}

export function isIndexing(status: RagStatus): boolean {
  return status === "NOT_INDEXED" || status === "INDEXING";
}

export interface CitationOut {
  chunk_id: string;
  document_id: string;
  page_number: number;
  section_title: string | null;
  snippet: string;
  score: number;
  marker: number | null;
  // Evidence & Research Object Layer: id of the persisted Evidence row this
  // citation was grounded in. Fetch via GET /api/evidence/{evidence_id}.
  evidence_id: string | null;
  // Citation verification (POST .../verify). Absent/false until checked.
  support_status?: SupportStatus | null;
  support_reason?: string | null;
  support_checked?: boolean;
}

export interface DraftSentenceOut {
  text: string;
  evidence_ids: string[];
  // Built by the backend, parallel to evidence_ids.
  citation_labels: string[];
  // "dropped" sentences were not grounded and must not be displayed.
  status: "ok" | "dropped";
}

export interface DraftContentOut {
  paragraphs: { sentences: DraftSentenceOut[] }[];
  caveat: string | null;
  dropped_count: number;
}

export interface LiteratureDraftOut {
  id: string;
  comparison_id: string;
  content: DraftContentOut;
  edited_text: string | null;
  created_at: string;
  updated_at: string;
}

export interface DraftVerificationOut {
  sentences: {
    paragraph: number;
    sentence: number;
    status: SupportStatus;
    confidence: number;
    reason: string;
  }[];
}

export type SupportStatus = "supported" | "partially_supported" | "unsupported" | "uncertain";

// Result of POST /api/comparisons/{id}/verify and /api/chat/conversations/{id}/verify.
export interface VerificationSummaryOut {
  total: number;
  supported: number;
  partially_supported: number;
  unsupported: number;
  uncertain: number;
}

// ---------------------------------------------------------------------------
// Phase 5: Evidence & Research Object Layer
// ---------------------------------------------------------------------------

export interface ResearchPaperOut {
  id: string;
  document_id: string;
  title: string | null;
  authors: string | null;
  abstract: string | null;
  publication_year: number | null;
  venue: string | null;
  doi: string | null;
  doi_url: string | null;
  arxiv_id: string | null;
  arxiv_url: string | null;
  publisher_url: string | null;
  source_url: string | null;
  metadata_source: string;
  metadata_confidence: number | null;
  created_at: string;
  updated_at: string;
}

export interface EvidenceOut {
  id: string;
  document_id: string;
  claim_id: string | null;
  chunk_id: string | null;
  page_id: string | null;
  section_id: string | null;
  page_number: number | null;
  section_title: string | null;
  subsection_title: string | null;
  snippet: string;
  bbox: BBox | null;
  retrieval_score: number | null;
  source_type: string;
  confidence: number | null;
  support_status: string;
  support_reason?: string | null;
  support_checked?: boolean;
  created_at: string;
  document_title: string | null;
  paper: ResearchPaperOut | null;
}

export interface ClaimOut {
  id: string;
  document_id: string;
  workspace_id: string | null;
  conversation_id: string | null;
  message_id: string | null;
  claim_text: string;
  claim_type: string;
  support_status: string;
  confidence: number | null;
  created_at: string;
}

export type ChatRole = "user" | "assistant";

export interface ChatMessageOut {
  id: string;
  conversation_id: string;
  role: ChatRole;
  content: string;
  citations: CitationOut[];
  insufficient_evidence: boolean;
  llm_provider: string | null;
  llm_model: string | null;
  error: string | null;
  // Generation stopped partway; `content` is only the partial answer.
  incomplete?: boolean;
  created_at: string;
  // Only ever present when the server has DEBUG_MODE on — see RagInfo.debug_mode.
  retrieval_debug?: RetrievalDebugOut | null;
}

export interface ChatConversationOut {
  id: string;
  document_id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ChatConversationDetailOut extends ChatConversationOut {
  messages: ChatMessageOut[];
}

export interface SuggestedQuestionsOut {
  questions: string[];
}

export interface RagInfo {
  embedding_provider: string;
  embedding_configured: boolean;
  embedding_dimensions: number;
  llm_provider: string;
  llm_configured: boolean;
  available_embedding_providers: string[];
  available_llm_providers: string[];

  // Phase 4: hybrid retrieval
  retrieval_mode: "hybrid" | "vector_only";
  reranking_enabled: boolean;
  rerank_provider: string;
  available_rerank_providers: string[];
  // Whether the Retrieval Debug panel is available in this deployment at
  // all. The UI hides the debug toggle entirely when this is false — and
  // the API omits debug payloads regardless of what the UI asks for.
  debug_mode: boolean;
}

// ---------------------------------------------------------------------------
// Phase 4: hybrid retrieval debug
// ---------------------------------------------------------------------------

export interface DebugCandidateOut {
  chunk_id: string;
  page_number: number;
  section_title: string | null;
  snippet: string;
  vector_rank: number | null;
  vector_score: number | null;
  keyword_rank: number | null;
  keyword_score: number | null;
  matched_keyword_terms: string[];
  fusion_score: number | null;
  rerank_score: number | null;
  selected: boolean;
}

export interface RetrievalDebugOut {
  query: {
    raw: string;
    normalized: string;
    tokens: string[];
    keywords: string[];
  };
  mode: string;
  candidates: DebugCandidateOut[];
  config: Record<string, unknown>;
  timings_ms: Record<string, number>;
}

// A chat message as rendered in the UI while it may still be streaming — a
// superset of ChatMessageOut with client-only bookkeeping fields.
export interface ChatUiMessage {
  id: string;
  role: ChatRole;
  content: string;
  citations: CitationOut[];
  insufficient_evidence: boolean;
  error: string | null;
  streaming: boolean;
  createdAt: string;
  // Phase 4: populated from the SSE "debug" event when the server has
  // DEBUG_MODE on; undefined otherwise (including for every user message).
  retrievalDebug?: RetrievalDebugOut | null;
}

// ---------------------------------------------------------------------------
// Phase 6A: Workspace chat
// ---------------------------------------------------------------------------

export interface WorkspaceChatConversationOut {
  id: string;
  workspace_id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
}

// ---------------------------------------------------------------------------
// Phase 6B: Evidence-backed multi-paper comparison
// ---------------------------------------------------------------------------

export type ComparisonStatus = "pending" | "running" | "completed" | "failed";

export interface ComparisonDimensionOut {
  key: string;
  label: string;
  query: string;
}

export interface ComparisonCellOut {
  id: string;
  document_id: string;
  document_title: string | null;
  dimension_key: string;
  dimension_label: string;
  value_text: string;
  insufficient_evidence: boolean;
  citations: CitationOut[];
  llm_provider: string | null;
  llm_model: string | null;
  error: string | null;
  // Structured comparison: was the supporting quote found verbatim in the
  // paper? null when not applicable (per-cell answers, "not found" cells).
  quote_verified?: boolean | null;
  // Is the claim backed by the cited text? Separate from quote_verified.
  support_status?: SupportStatus | null;
  support_reason?: string | null;
  support_checked?: boolean;
  created_at: string;
}

// Comparability check: are the compared papers directly comparable (same task,
// dataset, metric, split/setup)? Computed after the matrix is filled.
export type ComparabilityOverall =
  | "comparable"
  | "partially_comparable"
  | "not_comparable"
  | "unknown";

export type ComparabilityIssueType = "dataset" | "metric" | "task" | "split" | "setup";

/** A matrix cell is identified by (document_id, dimension_key). */
export interface ComparabilityCellRef {
  document_id: string;
  dimension_key: string;
}

export interface ComparabilityIssue {
  type: ComparabilityIssueType;
  /** One plain sentence naming the papers and the difference. */
  description: string;
  /** The papers involved (ids of matrix.documents[*].id). */
  document_ids: string[];
  /** The matrix cells the judgement is based on. */
  cell_refs: ComparabilityCellRef[];
}

export interface Comparability {
  overall: ComparabilityOverall;
  /** May be empty (e.g. when overall is "comparable" or "unknown"). */
  issues: ComparabilityIssue[];
  /** ONE plain sentence a student could paste into a review. */
  caveat: string;
  checked_by: "rules" | "gemini";
}

export interface ComparisonMatrixOut {
  id: string;
  workspace_id: string;
  title: string | null;
  status: ComparisonStatus;
  error: string | null;
  dimensions: ComparisonDimensionOut[];
  documents: DocumentOut[];
  cells: ComparisonCellOut[];
  // null when the check could not run; absent on older backends.
  comparability?: Comparability | null;
  created_at: string;
  updated_at: string;
}

export interface ComparisonSummaryOut {
  id: string;
  workspace_id: string;
  title: string | null;
  status: ComparisonStatus;
  document_count: number;
  dimension_count: number;
  created_at: string;
  updated_at: string;
}

export const COMPARISON_MIN_PAPERS = 2;
export const COMPARISON_MAX_PAPERS = 5;

export interface ComparisonCreateIn {
  document_ids: string[];
  dimensions?: string[] | null;
  title?: string | null;
}

// ---------------------------------------------------------------------------
// Phase 8: Research Gap Detection & Research Opportunity Engine
// ---------------------------------------------------------------------------

export type GapType =
  | "methodological"
  | "dataset"
  | "evaluation"
  | "domain"
  | "population"
  | "comparative"
  | "theoretical"
  | "reproducibility"
  | "contradictory"
  | "future_work";

export type GapConfidence = "high" | "medium" | "low";

export type GapStatus = "pending" | "identified" | "verified" | "dismissed";

export const GAP_TYPE_LABELS: Record<GapType, string> = {
  methodological: "Methodological Gap",
  dataset: "Dataset Gap",
  evaluation: "Evaluation Gap",
  domain: "Domain Gap",
  population: "Population Gap",
  comparative: "Comparative Gap",
  theoretical: "Theoretical Gap",
  reproducibility: "Reproducibility Gap",
  contradictory: "Contradictory Findings",
  future_work: "Future-Work Gap",
};

export const GAP_CONFIDENCE_COLORS: Record<GapConfidence, { bg: string; text: string; border: string }> = {
  high: { bg: "bg-green-500/10", text: "text-green-600", border: "border-green-500/20" },
  medium: { bg: "bg-amber-500/10", text: "text-amber-600", border: "border-amber-500/20" },
  low: { bg: "bg-slate-500/10", text: "text-slate-500", border: "border-slate-500/20" },
};

export const GAP_TYPE_COLORS: Record<GapType, string> = {
  methodological: "bg-blue-500/10 text-blue-600 border-blue-500/20",
  dataset: "bg-purple-500/10 text-purple-600 border-purple-500/20",
  evaluation: "bg-teal-500/10 text-teal-600 border-teal-500/20",
  domain: "bg-orange-500/10 text-orange-600 border-orange-500/20",
  population: "bg-rose-500/10 text-rose-600 border-rose-500/20",
  comparative: "bg-cyan-500/10 text-cyan-600 border-cyan-500/20",
  theoretical: "bg-indigo-500/10 text-indigo-600 border-indigo-500/20",
  reproducibility: "bg-yellow-500/10 text-yellow-600 border-yellow-500/20",
  contradictory: "bg-red-500/10 text-red-600 border-red-500/20",
  future_work: "bg-emerald-500/10 text-emerald-600 border-emerald-500/20",
};

export interface GapEvidenceOut {
  id: string;
  evidence_id: string;
  role: string | null;
  evidence: EvidenceOut | null;
}

export interface GapClaimOut {
  id: string;
  claim_id: string;
  document_id: string;
  document_title: string | null;
  claim_text: string;
  claim_type: string;
  support_status: string;
  evidence: EvidenceOut[];
}

export interface ResearchGapOut {
  id: string;
  workspace_id: string;
  title: string;
  description: string;
  gap_type: GapType;
  gap_type_label: string;
  confidence: GapConfidence;
  confidence_reasoning: string | null;
  status: GapStatus;
  research_direction: string | null;
  source_paper_ids: string[];
  supporting_paper_count: number;
  evidence_items: GapEvidenceOut[];
  claims: GapClaimOut[];
  novelty_notice: string;
  created_at: string;
  updated_at: string;
}

export interface ResearchGapSummaryOut {
  id: string;
  workspace_id: string;
  title: string;
  description: string;
  gap_type: GapType;
  gap_type_label: string;
  confidence: GapConfidence;
  status: GapStatus;
  supporting_paper_count: number;
  evidence_count: number;
  research_direction: string | null;
  created_at: string;
  updated_at: string;
}

export interface GapAnalysisResult {
  papers_analyzed: number;
  gaps_found: number;
  candidates_rejected: number;
  gaps: ResearchGapSummaryOut[];
  gap_type_counts: Record<string, number>;
}

export const GAP_MIN_PAPERS = 2;

