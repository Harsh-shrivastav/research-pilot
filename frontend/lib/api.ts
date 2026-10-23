import type {
  ChatConversationDetailOut,
  ChatConversationOut,
  ClaimOut,
  ComparisonCreateIn,
  ComparisonMatrixOut,
  ComparisonSummaryOut,
  ComparisonDimensionOut,
  DashboardStats,
  DocumentDetailOut,
  DocumentOut,
  ElementOut,
  ElementType,
  EvidenceOut,
  GapAnalysisResult,
  GapClaimOut,
  GapEvidenceOut,
  PageDetailOut,
  PageOut,
  ProcessingInfo,
  ProcessingStatusOut,
  RagInfo,
  RagStatusOut,
  ReferenceOut,
  ResearchGapOut,
  ResearchGapSummaryOut,
  ResearchPaperOut,
  RetrievalDebugOut,
  SectionDetailOut,
  SectionOut,
  SuggestedQuestionsOut,
  WorkspaceDetailOut,
  WorkspaceOut,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...init?.headers,
    },
    cache: "no-store",
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // ignore non-JSON error bodies
    }
    throw new ApiError(detail, res.status);
  }

  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export function fileUrl(path: string): string {
  return `${API_URL}${path}`;
}

export const api = {
  getDashboard: () => request<DashboardStats>("/api/dashboard"),

  listDocuments: () => request<DocumentOut[]>("/api/documents"),

  uploadDocument: (file: File, onProgress?: (pct: number) => void) => {
    return new Promise<DocumentOut>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const formData = new FormData();
      formData.append("file", file);

      xhr.open("POST", `${API_URL}/api/documents/upload`);

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          onProgress(Math.round((event.loaded / event.total) * 100));
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          let detail = xhr.statusText;
          try {
            detail = JSON.parse(xhr.responseText).detail || detail;
          } catch {
            // ignore
          }
          reject(new ApiError(detail, xhr.status));
        }
      };

      xhr.onerror = () => reject(new ApiError("Network error during upload", 0));
      xhr.send(formData);
    });
  },

  deleteDocument: (id: string) =>
    request<void>(`/api/documents/${id}`, { method: "DELETE" }),

  // --- Phase 2: extraction ------------------------------------------------

  getDocument: (id: string) => request<DocumentDetailOut>(`/api/documents/${id}`),

  getDocumentStatus: (id: string) =>
    request<ProcessingStatusOut>(`/api/documents/${id}/status`),

  reprocessDocument: (id: string) =>
    request<DocumentOut>(`/api/documents/${id}/reprocess`, { method: "POST" }),

  listSections: (id: string) => request<SectionOut[]>(`/api/documents/${id}/sections`),

  getSection: (documentId: string, sectionId: string) =>
    request<SectionDetailOut>(`/api/documents/${documentId}/sections/${sectionId}`),

  listPages: (id: string) => request<PageOut[]>(`/api/documents/${id}/pages`),

  getPage: (id: string, pageNumber: number) =>
    request<PageDetailOut>(`/api/documents/${id}/pages/${pageNumber}`),

  listElements: (
    id: string,
    opts?: { elementType?: ElementType; pageNumber?: number; limit?: number }
  ) => {
    const params = new URLSearchParams();
    if (opts?.elementType) params.set("element_type", opts.elementType);
    if (opts?.pageNumber) params.set("page_number", String(opts.pageNumber));
    if (opts?.limit) params.set("limit", String(opts.limit));
    const qs = params.toString();
    return request<ElementOut[]>(
      `/api/documents/${id}/elements${qs ? `?${qs}` : ""}`
    );
  },

  listReferences: (id: string) =>
    request<ReferenceOut[]>(`/api/documents/${id}/references`),

  getProcessingInfo: () => request<ProcessingInfo>("/api/system/processing"),

  listWorkspaces: () => request<WorkspaceOut[]>("/api/workspaces"),

  createWorkspace: (payload: { name: string; description?: string; document_id?: string }) =>
    request<WorkspaceOut>("/api/workspaces", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getWorkspace: (id: string) => request<WorkspaceDetailOut>(`/api/workspaces/${id}`),

  getWorkspaceByDocument: (documentId: string) =>
    request<WorkspaceOut | null>(`/api/workspaces/by-document/${documentId}`),

  // --- Phase 3: RAG chat ---------------------------------------------------

  getRagStatus: (id: string) => request<RagStatusOut>(`/api/documents/${id}/rag-status`),

  reindexDocument: (id: string) =>
    request<RagStatusOut>(`/api/documents/${id}/reindex`, { method: "POST" }),

  getSuggestedQuestions: (id: string) =>
    request<SuggestedQuestionsOut>(`/api/documents/${id}/chat/suggested-questions`),

  listConversations: (documentId: string) =>
    request<ChatConversationOut[]>(`/api/documents/${documentId}/chat/conversations`),

  createConversation: (documentId: string) =>
    request<ChatConversationOut>(`/api/documents/${documentId}/chat/conversations`, {
      method: "POST",
    }),

  getConversation: (conversationId: string) =>
    request<ChatConversationDetailOut>(`/api/chat/conversations/${conversationId}`),

  deleteConversation: (conversationId: string) =>
    request<void>(`/api/chat/conversations/${conversationId}`, { method: "DELETE" }),

  getRagInfo: () => request<RagInfo>("/api/system/rag"),

  // --- Phase 5: Evidence & Research Object Layer ---------------------------

  getEvidence: (evidenceId: string) =>
    request<EvidenceOut>(`/api/evidence/${evidenceId}`),

  listDocumentEvidence: (documentId: string, limit = 100, offset = 0) =>
    request<EvidenceOut[]>(
      `/api/documents/${documentId}/evidence?limit=${limit}&offset=${offset}`
    ),

  getClaim: (claimId: string) =>
    request<ClaimOut>(`/api/claims/${claimId}`),

  listClaimEvidence: (claimId: string) =>
    request<EvidenceOut[]>(`/api/claims/${claimId}/evidence`),

  listDocumentClaims: (documentId: string, limit = 100, offset = 0) =>
    request<ClaimOut[]>(
      `/api/documents/${documentId}/claims?limit=${limit}&offset=${offset}`
    ),

  getDocumentPaper: (documentId: string) =>
    request<ResearchPaperOut>(`/api/documents/${documentId}/paper`),

  // --- Phase 6A: Workspace document management + chat ----------------------

  addDocumentToWorkspace: (workspaceId: string, documentId: string) =>
    request<WorkspaceDetailOut>(`/api/workspaces/${workspaceId}/documents`, {
      method: "POST",
      body: JSON.stringify({ document_id: documentId }),
    }),

  removeDocumentFromWorkspace: (workspaceId: string, documentId: string) =>
    request<WorkspaceDetailOut>(`/api/workspaces/${workspaceId}/documents/${documentId}`, {
      method: "DELETE",
    }),

  listWorkspaceConversations: (workspaceId: string) =>
    request<import("./types").WorkspaceChatConversationOut[]>(
      `/api/workspaces/${workspaceId}/chat/conversations`
    ),

  // --- Phase 6B: Evidence-backed multi-paper comparison ---------------------

  getDefaultComparisonDimensions: () =>
    request<ComparisonDimensionOut[]>("/api/comparisons/default-dimensions"),

  createComparison: (workspaceId: string, payload: ComparisonCreateIn) =>
    request<ComparisonMatrixOut>(`/api/workspaces/${workspaceId}/comparisons`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  listWorkspaceComparisons: (workspaceId: string) =>
    request<ComparisonSummaryOut[]>(`/api/workspaces/${workspaceId}/comparisons`),

  getComparison: (comparisonId: string) =>
    request<ComparisonMatrixOut>(`/api/comparisons/${comparisonId}`),

  deleteComparison: (comparisonId: string) =>
    request<void>(`/api/comparisons/${comparisonId}`, { method: "DELETE" }),

  // Re-runs the comparability check; returns the whole updated matrix.
  rerunComparability: (comparisonId: string) =>
    request<ComparisonMatrixOut>(`/api/comparisons/${comparisonId}/comparability`, {
      method: "POST",
    }),

  // --- Phase 8: Research Gap Detection ------------------------------------

  analyzeGaps: (workspaceId: string) =>
    request<GapAnalysisResult>(`/api/workspaces/${workspaceId}/gaps/analyze`, {
      method: "POST",
    }),

  listWorkspaceGaps: (workspaceId: string) =>
    request<ResearchGapSummaryOut[]>(`/api/workspaces/${workspaceId}/gaps`),

  getGap: (gapId: string) =>
    request<ResearchGapOut>(`/api/gaps/${gapId}`),

  getGapEvidence: (gapId: string) =>
    request<GapEvidenceOut[]>(`/api/gaps/${gapId}/evidence`),

  getGapClaims: (gapId: string) =>
    request<GapClaimOut[]>(`/api/gaps/${gapId}/claims`),
};

export { ApiError };

// ---------------------------------------------------------------------------
// Phase 3: streaming chat (Server-Sent Events over a POST fetch stream)
// ---------------------------------------------------------------------------

export type ChatStreamEvent =
  | { type: "meta"; conversation_id: string }
  | { type: "token"; text: string }
  | { type: "debug"; debug: RetrievalDebugOut }
  | { type: "citations"; citations: import("./types").CitationOut[]; insufficient_evidence: boolean; message_id: string }
  | { type: "done"; message_id: string }
  | { type: "error"; message: string };

/**
 * Consumes the backend's `text/event-stream` chat response. Not implemented
 * with `EventSource` because that API only supports GET — the ask endpoint
 * is a POST carrying the question body, so this parses the SSE wire format
 * (`event: ...\ndata: ...\n\n` frames) directly off a fetch() ReadableStream.
 */
export async function streamChatAsk(
  documentId: string,
  payload: { question: string; conversation_id?: string | null },
  onEvent: (event: ChatStreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${API_URL}/api/documents/${documentId}/chat/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // ignore
    }
    throw new ApiError(detail || "Chat request failed.", res.status);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false; // saw a terminal done/error event

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");

      let eventName = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      try {
        const parsed = JSON.parse(data);
        if (eventName === "done" || eventName === "error") finished = true;
        onEvent({ type: eventName, ...parsed } as ChatStreamEvent);
      } catch {
        // malformed frame — skip rather than crash the stream
      }
    }
  }
  if (!finished) {
    // Stream closed without done/error (server crash, proxy cut): never leave
    // the answer looking complete.
    onEvent({ type: "error", message: "The connection closed before the answer finished. The partial answer is incomplete." });
  }
}

/**
 * Workspace chat streaming (SSE over POST). Same protocol as single-doc chat
 * but targets the workspace chat endpoint.
 */
export async function streamWorkspaceChatAsk(
  workspaceId: string,
  payload: { question: string; conversation_id?: string | null },
  onEvent: (event: ChatStreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${API_URL}/api/workspaces/${workspaceId}/chat/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // ignore
    }
    throw new ApiError(detail || "Workspace chat request failed.", res.status);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false; // saw a terminal done/error event

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");

      let eventName = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      try {
        const parsed = JSON.parse(data);
        if (eventName === "done" || eventName === "error") finished = true;
        onEvent({ type: eventName, ...parsed } as ChatStreamEvent);
      } catch {
        // malformed frame — skip
      }
    }
  }
  if (!finished) {
    // Stream closed without done/error (server crash, proxy cut): never leave
    // the answer looking complete.
    onEvent({ type: "error", message: "The connection closed before the answer finished. The partial answer is incomplete." });
  }
}
