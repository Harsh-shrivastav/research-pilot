"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  BookOpenText,
  Bug,
  ExternalLink,
  Eye,
  FileText,
  Loader2,
  RotateCcw,
  Send,
  Sparkles,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { api, ApiError, streamChatAsk } from "@/lib/api";
import { cn } from "@/lib/utils";
import { RetrievalDebugPanel } from "./retrieval-debug-panel";
import { ViewEvidenceDialog } from "./evidence-dialog";
import type {
  ChatUiMessage,
  CitationOut,
  DocumentDetailOut,
  RagStatus,
} from "@/lib/types";
import { isIndexing } from "@/lib/types";

const RAG_POLL_INTERVAL_MS = 1500;

function uid(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

// ---------------------------------------------------------------------------
// Source card — one citation, clickable to jump the PDF viewer to that page.
// ---------------------------------------------------------------------------

function SourceCard({
  citation,
  onOpen,
  onViewEvidence,
}: {
  citation: CitationOut;
  onOpen: (citation: CitationOut) => void;
  onViewEvidence: (evidenceId: string) => void;
}) {
  return (
    <div className="flex w-full flex-col gap-1 rounded-lg border border-border bg-muted/40 px-3 py-2 text-left text-xs transition-colors hover:border-primary/40 hover:bg-muted">
      <button
        onClick={() => onOpen(citation)}
        className="flex w-full flex-col gap-1 text-left"
      >
        <div className="flex items-center gap-1.5 font-medium text-foreground">
          <BookOpenText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          {citation.marker !== null && (
            <span className="text-muted-foreground">[{citation.marker}]</span>
          )}
          <span>Page {citation.page_number}</span>
          {citation.section_title && (
            <>
              <span className="text-muted-foreground">&middot;</span>
              <span className="truncate text-muted-foreground">
                Section: {citation.section_title}
              </span>
            </>
          )}
        </div>
        <p className="line-clamp-3 text-muted-foreground">{citation.snippet}</p>
      </button>
      {citation.evidence_id && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onViewEvidence(citation.evidence_id!);
          }}
          className="mt-1 inline-flex items-center gap-1 self-start rounded-md border border-primary/30 bg-primary/5 px-2 py-1 text-[11px] font-medium text-primary transition-colors hover:bg-primary/10"
        >
          <Eye className="h-3 w-3" />
          View Evidence
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// One message bubble
// ---------------------------------------------------------------------------

function MessageBubble({
  message,
  onOpenCitation,
  onViewEvidence,
  onRetry,
  debugMode,
}: {
  message: ChatUiMessage;
  onOpenCitation: (citation: CitationOut) => void;
  onViewEvidence: (evidenceId: string) => void;
  onRetry: (message: ChatUiMessage) => void;
  debugMode: boolean;
}) {
  const isUser = message.role === "user";
  const [showDebug, setShowDebug] = useState(false);
  const hasDebug = debugMode && !!message.retrievalDebug;

  return (
    <div className={cn("flex flex-col gap-2", isUser ? "items-end" : "items-start")}>
      <div
        className={cn(
          "max-w-[92%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
          isUser
            ? "bg-primary text-primary-foreground"
            : message.error
              ? "border border-destructive/30 bg-destructive/10 text-destructive"
              : "border border-border bg-card text-card-foreground"
        )}
      >
        {message.error ? (
          <>
            {message.content && (
              // Partial answer from a stream that failed midway: shown, but
              // never as a complete answer (no citations are attached to it).
              <span className="mb-2 block whitespace-pre-wrap text-card-foreground/80">{message.content}</span>
            )}
            <div className="flex items-start gap-2">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{message.error}</span>
            </div>
          </>
        ) : (
          <>
            <span className="whitespace-pre-wrap">{message.content}</span>
            {message.streaming && (
              <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-current align-middle" />
            )}
          </>
        )}
      </div>

      {!isUser && message.error && (
        <Button variant="outline" size="sm" className="h-7 gap-1.5 text-xs" onClick={() => onRetry(message)}>
          <RotateCcw className="h-3 w-3" />
          Retry
        </Button>
      )}

      {!isUser && !message.streaming && !message.error && message.insufficient_evidence && (
        <Badge variant="warning" className="gap-1">
          <AlertCircle className="h-3 w-3" />
          Not found in this paper
        </Badge>
      )}

      {!isUser && !message.streaming && message.citations.length > 0 && (
        <div className="flex w-full max-w-[92%] flex-col gap-1.5">
          {message.citations.map((c) => (
            <SourceCard key={`${message.id}-${c.chunk_id}-${c.marker ?? "x"}`} citation={c} onOpen={onOpenCitation} onViewEvidence={onViewEvidence} />
          ))}
        </div>
      )}

      {/* Retrieval Debug: development/admin-only, never shown unless the
          server has DEBUG_MODE on (see ChatPanel's ragInfo fetch below) —
          normal users never see this button, regardless of what happened
          server-side, because `debugMode` here reflects that same flag. */}
      {!isUser && hasDebug && (
        <Button
          variant="ghost"
          size="sm"
          className="h-6 gap-1.5 px-2 text-[11px] text-muted-foreground"
          onClick={() => setShowDebug((v) => !v)}
        >
          <Bug className="h-3 w-3" />
          {showDebug ? "Hide retrieval debug" : "Retrieval debug"}
        </Button>
      )}
      {!isUser && hasDebug && showDebug && message.retrievalDebug && (
        <RetrievalDebugPanel debug={message.retrievalDebug} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function ChatPanel({
  document: doc,
  onOpenCitation,
}: {
  document: DocumentDetailOut | null;
  onOpenCitation: (page: number, citation: CitationOut) => void;
}) {
  const [ragStatus, setRagStatus] = useState<RagStatus | null>(null);
  const [ragError, setRagError] = useState<string | null>(null);
  const [chunkCount, setChunkCount] = useState(0);
  const [debugMode, setDebugMode] = useState(false);

  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatUiMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [suggested, setSuggested] = useState<string[]>([]);
  const [clearing, setClearing] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const documentId = doc?.id ?? null;
  const documentReady = doc?.processing_status === "READY";

  // -- RAG indexing status polling (mirrors the Phase 2 status poll) -------
  useEffect(() => {
    const clear = () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
    if (!documentId || !documentReady) {
      clear();
      return;
    }

    let cancelled = false;
    const check = async () => {
      try {
        const status = await api.getRagStatus(documentId);
        if (cancelled) return;
        setRagStatus(status.rag_status);
        setRagError(status.rag_error);
        setChunkCount(status.chunk_count);
        if (!isIndexing(status.rag_status)) clear();
      } catch {
        // transient — next tick retries
      }
    };

    check();
    if (pollRef.current) return clear;
    pollRef.current = setInterval(check, RAG_POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clear();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documentId, documentReady]);

  // -- whether this deployment exposes the Retrieval Debug panel at all ----
  // (server-side DEBUG_MODE — see app/config.py / app/rag/factory.py). Fetched
  // once; the chat SSE stream simply won't send "debug" events when it's off,
  // so this only controls whether the UI *offers* to show the panel.
  useEffect(() => {
    api
      .getRagInfo()
      .then((info) => setDebugMode(Boolean(info.debug_mode)))
      .catch(() => setDebugMode(false));
  }, []);

  // -- suggested questions --------------------------------------------------
  useEffect(() => {
    if (!documentId) return;
    api
      .getSuggestedQuestions(documentId)
      .then((res) => setSuggested(res.questions))
      .catch(() => setSuggested([]));
  }, [documentId]);

  // -- reset chat state when the document changes ---------------------------
  useEffect(() => {
    setConversationId(null);
    setMessages([]);
    setInput("");
  }, [documentId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const chatReady = ragStatus === "INDEXED" && chunkCount > 0;

  const send = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || !documentId || sending) return;

      setInput("");
      const userMsg: ChatUiMessage = {
        id: uid(),
        role: "user",
        content: trimmed,
        citations: [],
        insufficient_evidence: false,
        error: null,
        streaming: false,
        createdAt: new Date().toISOString(),
      };
      const assistantId = uid();
      const assistantMsg: ChatUiMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        citations: [],
        insufficient_evidence: false,
        error: null,
        streaming: true,
        createdAt: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setSending(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await streamChatAsk(
          documentId,
          { question: trimmed, conversation_id: conversationId },
          (event) => {
            if (event.type === "meta") {
              setConversationId(event.conversation_id);
            } else if (event.type === "token") {
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + event.text } : m))
              );
            } else if (event.type === "debug") {
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantId ? { ...m, retrievalDebug: event.debug } : m))
              );
            } else if (event.type === "citations") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, citations: event.citations, insufficient_evidence: event.insufficient_evidence }
                    : m
                )
              );
            } else if (event.type === "error") {
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantId ? { ...m, error: event.message, streaming: false } : m))
              );
            } else if (event.type === "done") {
              setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, streaming: false } : m)));
            }
          },
          controller.signal
        );
      } catch (e) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  streaming: false,
                  error: e instanceof ApiError ? e.message : "Could not reach the chat service.",
                }
              : m
          )
        );
      } finally {
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, streaming: false } : m)));
        setSending(false);
      }
    },
    [documentId, conversationId, sending]
  );

  const handleRetry = useCallback(
    (message: ChatUiMessage) => {
      // Find the user question immediately preceding this failed answer.
      const idx = messages.findIndex((m) => m.id === message.id);
      const priorUser = [...messages.slice(0, idx)].reverse().find((m) => m.role === "user");
      if (!priorUser) return;
      setMessages((prev) => prev.filter((m) => m.id !== message.id));
      send(priorUser.content);
    },
    [messages, send]
  );

  const handleClear = useCallback(async () => {
    setClearing(true);
    try {
      if (conversationId) {
        await api.deleteConversation(conversationId).catch(() => undefined);
      }
      abortRef.current?.abort();
      setConversationId(null);
      setMessages([]);
    } finally {
      setClearing(false);
    }
  }, [conversationId]);

  const handleOpenCitation = useCallback(
    (citation: CitationOut) => onOpenCitation(citation.page_number, citation),
    [onOpenCitation]
  );

  // -- Phase 5: View Evidence dialog ----------------------------------------
  const [evidenceDialogOpen, setEvidenceDialogOpen] = useState(false);
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);

  const handleViewEvidence = useCallback((evidenceId: string) => {
    setSelectedEvidenceId(evidenceId);
    setEvidenceDialogOpen(true);
  }, []);

  const handleOpenPageFromEvidence = useCallback(
    (page: number) => {
      // Reuse the existing onOpenCitation with a synthetic citation-like call
      onOpenCitation(page, {
        chunk_id: "",
        document_id: documentId || "",
        page_number: page,
        section_title: null,
        snippet: "",
        score: 0,
        marker: null,
        evidence_id: null,
      });
    },
    [documentId, onOpenCitation]
  );

  // -- empty / not-ready states ----------------------------------------------

  if (!doc) {
    return (
      <Card className="flex h-full items-center justify-center p-6 text-center text-sm text-muted-foreground">
        Attach a document to this workspace to chat with it.
      </Card>
    );
  }

  if (!documentReady) {
    return (
      <Card className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        <p className="text-sm text-muted-foreground">
          Waiting for document processing to finish before the paper can be indexed for chat.
        </p>
      </Card>
    );
  }

  if (!chatReady) {
    return (
      <Card className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
        {ragStatus === "FAILED" ? (
          <>
            <AlertCircle className="h-5 w-5 text-destructive" />
            <p className="text-sm font-medium text-destructive">Indexing failed</p>
            {ragError && <p className="text-xs text-muted-foreground">{ragError}</p>}
            <Button
              size="sm"
              variant="outline"
              className="mt-2"
              onClick={() => documentId && api.reindexDocument(documentId).then(() => setRagStatus("INDEXING"))}
            >
              Retry indexing
            </Button>
          </>
        ) : (
          <>
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              Indexing this paper for chat (cleaning text, chunking, generating embeddings)&hellip;
            </p>
          </>
        )}
      </Card>
    );
  }

  return (
    <Card className="flex h-full flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">AI Research Assistant</h3>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 gap-1.5 text-xs text-muted-foreground"
          onClick={handleClear}
          disabled={clearing || messages.length === 0}
        >
          <Trash2 className="h-3.5 w-3.5" />
          Clear
        </Button>
      </div>

      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
            <div className="rounded-full bg-primary/10 p-3">
              <Sparkles className="h-5 w-5 text-primary" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-medium">Ask this paper anything</p>
              <p className="text-xs text-muted-foreground">
                Answers are grounded in the document, with page-level citations.
              </p>
            </div>
            {suggested.length > 0 && (
              <div className="flex flex-wrap justify-center gap-2 px-2">
                {suggested.map((q) => (
                  <button
                    key={q}
                    onClick={() => send(q)}
                    className="rounded-full border border-border bg-muted/50 px-3 py-1.5 text-xs text-foreground transition-colors hover:border-primary/40 hover:bg-muted"
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          messages.map((m) => (
            <MessageBubble
              key={m.id}
              message={m}
              onOpenCitation={handleOpenCitation}
              onViewEvidence={handleViewEvidence}
              onRetry={handleRetry}
              debugMode={debugMode}
            />
          ))
        )}
      </div>

      <form
        className="flex items-end gap-2 border-t border-border p-3"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(input);
            }
          }}
          placeholder="Ask a question about this paper..."
          rows={1}
          disabled={sending}
          className="flex-1 resize-none rounded-md border border-border bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
        />
        <Button type="submit" size="icon" disabled={sending || !input.trim()} aria-label="Send">
          {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
        </Button>
      </form>

      {/* Phase 5: View Evidence dialog */}
      {selectedEvidenceId && (
        <ViewEvidenceDialog
          evidenceId={selectedEvidenceId}
          open={evidenceDialogOpen}
          onOpenChange={setEvidenceDialogOpen}
          onOpenPage={handleOpenPageFromEvidence}
        />
      )}
    </Card>
  );
}
