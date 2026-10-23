"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  ArrowLeft,
  FileSearch,
  Lightbulb,
  MessagesSquare,
  Plus,
  Trash2,
  FileText,
  Send,
  Loader2,
  BookOpen,
  AlertCircle,
  Columns3,
} from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { Topbar } from "@/components/layout/topbar";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PdfPanel } from "@/components/workspace/pdf-panel";
import { DocInfoPanel } from "@/components/workspace/doc-info-panel";
import { ProcessingPanel } from "@/components/workspace/processing-panel";
import { ExtractionPanel } from "@/components/workspace/extraction-panel";
import { ChatPanel } from "@/components/workspace/chat-panel";
import { ComparisonPanel } from "@/components/workspace/comparison-panel";
import { GapsPanel } from "@/components/workspace/gaps-panel";
import { api, ApiError, streamWorkspaceChatAsk, ChatStreamEvent } from "@/lib/api";
import { isProcessing } from "@/lib/types";
import type { CitationOut, DocumentDetailOut, DocumentOut, WorkspaceDetailOut, ChatUiMessage } from "@/lib/types";

const POLL_INTERVAL_MS = 2000;

export default function WorkspacePage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();

  const [workspace, setWorkspace] = useState<WorkspaceDetailOut | null>(null);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [document, setDocument] = useState<DocumentDetailOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [reprocessing, setReprocessing] = useState(false);
  const [viewerPage, setViewerPage] = useState(1);
  const [rightTab, setRightTab] = useState<"chat" | "workspace-chat" | "compare" | "explore" | "gaps">(
    "workspace-chat"
  );
  const [highlight, setHighlight] = useState<{ token: number; label: string } | undefined>(undefined);

  // Document management state
  const [allDocuments, setAllDocuments] = useState<DocumentOut[]>([]);
  const [showAddDoc, setShowAddDoc] = useState(false);
  const [addingDoc, setAddingDoc] = useState(false);
  const [removingDocId, setRemovingDocId] = useState<string | null>(null);

  // Workspace chat state
  const [wsMessages, setWsMessages] = useState<ChatUiMessage[]>([]);
  const [wsQuestion, setWsQuestion] = useState("");
  const [wsStreaming, setWsStreaming] = useState(false);
  const [wsConversationId, setWsConversationId] = useState<string | null>(null);
  const wsAbortRef = useRef<AbortController | null>(null);
  const wsChatEndRef = useRef<HTMLDivElement | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadDocument = useCallback(async (documentId: string) => {
    const detail = await api.getDocument(documentId);
    setDocument(detail);
    return detail;
  }, []);

  const loadWorkspace = useCallback(async (wsId: string) => {
    const ws = await api.getWorkspace(wsId);
    setWorkspace(ws);
    return ws;
  }, []);

  useEffect(() => {
    if (!params.id) return;
    let cancelled = false;

    (async () => {
      try {
        const ws = await loadWorkspace(params.id);
        if (cancelled) return;

        // Load all user documents for the "Add" picker
        const docs = await api.listDocuments();
        if (!cancelled) setAllDocuments(docs);

        // Select first document for PDF viewer
        const primary = ws.documents[0];
        if (primary) {
          setSelectedDocId(primary.id);
          await loadDocument(primary.id);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof ApiError ? e.message : "Failed to load workspace.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [params.id, loadDocument, loadWorkspace]);

  // Poll while the selected document is being processed
  useEffect(() => {
    const clear = () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };

    if (!document || !isProcessing(document.processing_status)) {
      clear();
      return;
    }
    if (pollRef.current) return;

    const documentId = document.id;
    pollRef.current = setInterval(async () => {
      try {
        const status = await api.getDocumentStatus(documentId);
        if (!isProcessing(status.processing_status)) {
          await loadDocument(documentId);
        } else {
          setDocument((prev) =>
            prev && prev.processing_status !== status.processing_status
              ? { ...prev, processing_status: status.processing_status }
              : prev
          );
        }
      } catch {
        // Transient failures are fine
      }
    }, POLL_INTERVAL_MS);

    return clear;
  }, [document, loadDocument]);

  const handleReprocess = async () => {
    if (!document) return;
    setReprocessing(true);
    setError(null);
    try {
      await api.reprocessDocument(document.id);
      await loadDocument(document.id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not requeue this document.");
    } finally {
      setReprocessing(false);
    }
  };

  const handleOpenCitation = useCallback((page: number, citation: CitationOut) => {
    // If the citation is from a different document, switch to it
    if (citation.document_id && citation.document_id !== selectedDocId) {
      setSelectedDocId(citation.document_id);
      loadDocument(citation.document_id).then(() => {
        setViewerPage(page);
        setHighlight({
          token: Date.now(),
          label: citation.section_title
            ? `Citation: page ${page} · ${citation.section_title}`
            : `Citation: page ${page}`,
        });
      });
    } else {
      setViewerPage(page);
      setHighlight({
        token: Date.now(),
        label: citation.section_title
          ? `Citation: page ${page} · ${citation.section_title}`
          : `Citation: page ${page}`,
      });
    }
  }, [selectedDocId, loadDocument]);

  // Select a document from the sidebar
  const handleSelectDocument = async (docId: string) => {
    if (docId === selectedDocId) return;
    setSelectedDocId(docId);
    setViewerPage(1);
    try {
      await loadDocument(docId);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load document.");
    }
  };

  // Add document to workspace
  const handleAddDocument = async (docId: string) => {
    if (!workspace) return;
    setAddingDoc(true);
    setError(null);
    try {
      const updated = await api.addDocumentToWorkspace(workspace.id, docId);
      setWorkspace(updated);
      setShowAddDoc(false);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to add document.");
    } finally {
      setAddingDoc(false);
    }
  };

  // Remove document from workspace
  const handleRemoveDocument = async (docId: string) => {
    if (!workspace) return;
    setRemovingDocId(docId);
    setError(null);
    try {
      const updated = await api.removeDocumentFromWorkspace(workspace.id, docId);
      setWorkspace(updated);
      // If we removed the selected doc, select the first remaining
      if (docId === selectedDocId) {
        const first = updated.documents[0];
        if (first) {
          setSelectedDocId(first.id);
          await loadDocument(first.id);
        } else {
          setSelectedDocId(null);
          setDocument(null);
        }
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to remove document.");
    } finally {
      setRemovingDocId(null);
    }
  };

  // Workspace chat
  const handleWorkspaceAsk = async () => {
    if (!workspace || !wsQuestion.trim() || wsStreaming) return;

    const question = wsQuestion.trim();
    setWsQuestion("");

    // Add user message
    const userMsg: ChatUiMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: question,
      citations: [],
      insufficient_evidence: false,
      error: null,
      streaming: false,
      createdAt: new Date().toISOString(),
    };

    const assistantMsg: ChatUiMessage = {
      id: `assistant-${Date.now()}`,
      role: "assistant",
      content: "",
      citations: [],
      insufficient_evidence: false,
      error: null,
      streaming: true,
      createdAt: new Date().toISOString(),
    };

    setWsMessages((prev) => [...prev, userMsg, assistantMsg]);
    setWsStreaming(true);

    const controller = new AbortController();
    wsAbortRef.current = controller;

    try {
      await streamWorkspaceChatAsk(
        workspace.id,
        { question, conversation_id: wsConversationId },
        (event: ChatStreamEvent) => {
          if (event.type === "meta") {
            setWsConversationId((event as any).conversation_id);
          } else if (event.type === "token") {
            setWsMessages((prev) => {
              const msgs = [...prev];
              const last = msgs[msgs.length - 1];
              if (last && last.role === "assistant" && last.streaming) {
                msgs[msgs.length - 1] = {
                  ...last,
                  content: last.content + (event as any).text,
                };
              }
              return msgs;
            });
          } else if (event.type === "citations") {
            const citEvent = event as any;
            setWsMessages((prev) => {
              const msgs = [...prev];
              const last = msgs[msgs.length - 1];
              if (last && last.role === "assistant") {
                msgs[msgs.length - 1] = {
                  ...last,
                  citations: citEvent.citations || [],
                  insufficient_evidence: citEvent.insufficient_evidence || false,
                  streaming: false,
                };
              }
              return msgs;
            });
          } else if (event.type === "done") {
            setWsMessages((prev) => {
              const msgs = [...prev];
              const last = msgs[msgs.length - 1];
              if (last && last.role === "assistant") {
                msgs[msgs.length - 1] = { ...last, streaming: false, id: (event as any).message_id || last.id };
              }
              return msgs;
            });
          } else if (event.type === "error") {
            setWsMessages((prev) => {
              const msgs = [...prev];
              const last = msgs[msgs.length - 1];
              if (last && last.role === "assistant") {
                msgs[msgs.length - 1] = {
                  ...last,
                  error: (event as any).message || "An error occurred.",
                  streaming: false,
                };
              }
              return msgs;
            });
          }
        },
        controller.signal
      );
    } catch (e) {
      if ((e as any)?.name !== "AbortError") {
        setWsMessages((prev) => {
          const msgs = [...prev];
          const last = msgs[msgs.length - 1];
          if (last && last.role === "assistant") {
            msgs[msgs.length - 1] = {
              ...last,
              error: e instanceof ApiError ? e.message : "Chat request failed.",
              streaming: false,
            };
          }
          return msgs;
        });
      }
    } finally {
      setWsStreaming(false);
      wsAbortRef.current = null;
    }
  };

  // Auto-scroll workspace chat
  useEffect(() => {
    wsChatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [wsMessages]);

  // Documents NOT already in workspace (for the add picker)
  const availableDocs = allDocuments.filter(
    (d) => !workspace?.documents.some((wd) => wd.id === d.id)
  );

  // Lookup doc title by id for citations
  const docTitleMap: Record<string, string> = {};
  workspace?.documents.forEach((d) => {
    docTitleMap[d.id] = d.title || d.original_filename;
  });

  return (
    <AppShell>
      <Topbar
        title={loading ? "Loading workspace..." : workspace?.name || "Workspace"}
        subtitle={`Multi-paper research workspace · ${workspace?.documents.length || 0} papers`}
        action={
          <Button variant="ghost" size="sm" onClick={() => router.push("/documents")}>
            <ArrowLeft className="h-4 w-4" /> Back to documents
          </Button>
        }
      />

      <main className="flex-1 overflow-hidden p-6">
        {error && (
          <div className="mb-4 rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
            <button className="ml-2 underline" onClick={() => setError(null)}>dismiss</button>
          </div>
        )}

        {loading ? (
          <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-[1fr_440px]">
            <Skeleton className="h-full" />
            <div className="space-y-4">
              <Skeleton className="h-40" />
              <Skeleton className="h-64" />
            </div>
          </div>
        ) : (
          <div className="grid h-full min-h-0 grid-cols-1 gap-6 lg:grid-cols-[1fr_480px]">
            {/* LEFT: Paper list + PDF viewer */}
            <div className="flex min-h-[400px] flex-col gap-3">
              {/* Paper list sidebar */}
              <div className="rounded-lg border bg-card p-3">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-semibold text-foreground flex items-center gap-1.5">
                    <BookOpen className="h-4 w-4" />
                    Papers ({workspace?.documents.length || 0}/5)
                  </h3>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 text-xs"
                    onClick={() => setShowAddDoc(!showAddDoc)}
                    disabled={(workspace?.documents.length || 0) >= 5}
                  >
                    <Plus className="h-3 w-3 mr-1" /> Add
                  </Button>
                </div>

                {/* Add document picker */}
                {showAddDoc && (
                  <div className="mb-2 rounded-md border bg-muted/50 p-2 max-h-40 overflow-y-auto">
                    {availableDocs.length === 0 ? (
                      <p className="text-xs text-muted-foreground py-1">
                        No more documents available. Upload new papers first.
                      </p>
                    ) : (
                      availableDocs.map((d) => (
                        <button
                          key={d.id}
                          className="w-full text-left px-2 py-1.5 text-xs rounded hover:bg-accent transition-colors flex items-center gap-1.5"
                          onClick={() => handleAddDocument(d.id)}
                          disabled={addingDoc}
                        >
                          <FileText className="h-3 w-3 flex-shrink-0 text-muted-foreground" />
                          <span className="truncate">{d.title || d.original_filename}</span>
                          {d.rag_status === "INDEXED" && (
                            <span className="ml-auto text-[10px] text-green-600">indexed</span>
                          )}
                        </button>
                      ))
                    )}
                  </div>
                )}

                {/* Document list */}
                <div className="space-y-1">
                  {workspace?.documents.map((doc) => (
                    <div
                      key={doc.id}
                      className={`flex items-center gap-2 rounded-md px-2 py-1.5 text-xs cursor-pointer transition-colors ${
                        doc.id === selectedDocId
                          ? "bg-primary/10 text-primary border border-primary/20"
                          : "hover:bg-accent"
                      }`}
                      onClick={() => handleSelectDocument(doc.id)}
                    >
                      <FileText className="h-3.5 w-3.5 flex-shrink-0" />
                      <span className="truncate flex-1">{doc.title || doc.original_filename}</span>
                      <span className={`text-[10px] px-1 py-0.5 rounded ${
                        doc.rag_status === "INDEXED"
                          ? "bg-green-500/10 text-green-600"
                          : "bg-yellow-500/10 text-yellow-600"
                      }`}>
                        {doc.rag_status === "INDEXED" ? "indexed" : doc.rag_status.toLowerCase()}
                      </span>
                      <button
                        className="p-0.5 rounded hover:bg-destructive/10 hover:text-destructive transition-colors"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRemoveDocument(doc.id);
                        }}
                        disabled={removingDocId === doc.id}
                        title="Remove from workspace"
                      >
                        {removingDocId === doc.id ? (
                          <Loader2 className="h-3 w-3 animate-spin" />
                        ) : (
                          <Trash2 className="h-3 w-3" />
                        )}
                      </button>
                    </div>
                  ))}
                  {(!workspace?.documents || workspace.documents.length === 0) && (
                    <p className="text-xs text-muted-foreground py-2 text-center">
                      No papers yet. Click &quot;Add&quot; to add documents.
                    </p>
                  )}
                </div>
              </div>

              {/* PDF viewer */}
              <div className="flex-1 min-h-[300px]">
                <PdfPanel
                  document={document}
                  page={viewerPage}
                  pageCount={document?.page_count ?? null}
                  onPageChange={setViewerPage}
                  highlightToken={highlight?.token}
                  highlightLabel={highlight?.label}
                />
              </div>
            </div>

            {/* RIGHT: Workspace Chat / Single-doc Chat / Explore */}
            <div className="flex min-h-0 flex-col">
              <Tabs
                value={rightTab}
                onValueChange={(v) => setRightTab(v as "workspace-chat" | "chat" | "compare" | "explore" | "gaps")}
                className="flex min-h-0 flex-1 flex-col"
              >
                <TabsList className="grid w-full grid-cols-5">
                  <TabsTrigger value="workspace-chat" className="gap-1.5 text-xs">
                    <MessagesSquare className="h-3.5 w-3.5" />
                    Research Chat
                  </TabsTrigger>
                  <TabsTrigger value="chat" className="gap-1.5 text-xs">
                    <FileText className="h-3.5 w-3.5" />
                    Paper Chat
                  </TabsTrigger>
                  <TabsTrigger value="compare" className="gap-1.5 text-xs">
                    <Columns3 className="h-3.5 w-3.5" />
                    Compare
                  </TabsTrigger>
                  <TabsTrigger value="explore" className="gap-1.5 text-xs">
                    <FileSearch className="h-3.5 w-3.5" />
                    Explore
                  </TabsTrigger>
                  <TabsTrigger value="gaps" className="gap-1.5 text-xs">
                    <Lightbulb className="h-3.5 w-3.5" />
                    Gaps
                  </TabsTrigger>
                </TabsList>

                {/* Workspace Chat */}
                <TabsContent value="workspace-chat" className="mt-3 min-h-0 flex-1 flex flex-col">
                  <div className="flex-1 overflow-y-auto rounded-lg border bg-card p-3 mb-3 space-y-3">
                    {wsMessages.length === 0 ? (
                      <div className="flex flex-col items-center justify-center h-full text-center py-12">
                        <MessagesSquare className="h-10 w-10 text-muted-foreground/30 mb-3" />
                        <h4 className="text-sm font-medium text-muted-foreground mb-1">
                          Research Chat
                        </h4>
                        <p className="text-xs text-muted-foreground max-w-[260px]">
                          Ask questions across all papers in this workspace.
                          Citations will identify which paper each answer comes from.
                        </p>
                        <div className="mt-4 space-y-1.5">
                          {["What datasets do these papers use?",
                            "What methods do the papers employ?",
                            "What are the key findings?",
                          ].map((q) => (
                            <button
                              key={q}
                              className="block w-full text-left text-xs px-3 py-1.5 rounded-md border hover:bg-accent transition-colors"
                              onClick={() => {
                                setWsQuestion(q);
                              }}
                            >
                              {q}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : (
                      wsMessages.map((msg) => (
                        <div key={msg.id} className={`text-sm ${msg.role === "user" ? "ml-8" : ""}`}>
                          <div
                            className={`rounded-lg px-3 py-2 ${
                              msg.role === "user"
                                ? "bg-primary text-primary-foreground ml-auto max-w-[85%]"
                                : "bg-muted max-w-full"
                            }`}
                          >
                            {msg.error ? (
                              <>
                                {msg.content && (
                                  <div className="whitespace-pre-wrap text-xs leading-relaxed opacity-80 mb-2">{msg.content}</div>
                                )}
                                <div className="flex items-start gap-2 text-destructive">
                                  <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                                  <span className="text-xs">{msg.error}</span>
                                </div>
                              </>
                            ) : (
                              <div className="whitespace-pre-wrap text-xs leading-relaxed">
                                {msg.content}
                                {msg.streaming && (
                                  <span className="inline-block w-1.5 h-3.5 bg-foreground/60 ml-0.5 animate-pulse" />
                                )}
                              </div>
                            )}
                          </div>

                          {/* Citations */}
                          {msg.role === "assistant" && msg.citations.length > 0 && !msg.streaming && (
                            <div className="mt-2 space-y-1">
                              {msg.citations.map((cit, idx) => (
                                <button
                                  key={idx}
                                  className="w-full text-left text-[11px] p-2 rounded border bg-background hover:bg-accent transition-colors"
                                  onClick={() => handleOpenCitation(cit.page_number, cit)}
                                >
                                  <div className="flex items-center gap-1.5 mb-0.5">
                                    {cit.marker && (
                                      <span className="text-[10px] font-mono bg-primary/10 text-primary px-1 rounded">
                                        [{cit.marker}]
                                      </span>
                                    )}
                                    <span className="font-medium text-primary truncate">
                                      {docTitleMap[cit.document_id] || "Unknown Paper"}
                                    </span>
                                    <span className="text-muted-foreground">
                                      · p.{cit.page_number}
                                      {cit.section_title && ` · ${cit.section_title}`}
                                    </span>
                                  </div>
                                  <p className="text-muted-foreground line-clamp-2">{cit.snippet}</p>
                                </button>
                              ))}
                            </div>
                          )}

                          {msg.role === "assistant" && msg.insufficient_evidence && !msg.streaming && (
                            <p className="mt-1 text-[11px] text-yellow-600 flex items-center gap-1">
                              <AlertCircle className="h-3 w-3" />
                              Insufficient evidence in workspace papers
                            </p>
                          )}
                        </div>
                      ))
                    )}
                    <div ref={wsChatEndRef} />
                  </div>

                  {/* Input */}
                  <div className="flex gap-2">
                    <input
                      type="text"
                      className="flex-1 rounded-md border bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/40"
                      placeholder={
                        (workspace?.documents.length || 0) === 0
                          ? "Add papers to start chatting..."
                          : "Ask about all workspace papers..."
                      }
                      value={wsQuestion}
                      onChange={(e) => setWsQuestion(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleWorkspaceAsk()}
                      disabled={wsStreaming || (workspace?.documents.length || 0) === 0}
                    />
                    <Button
                      size="sm"
                      onClick={handleWorkspaceAsk}
                      disabled={wsStreaming || !wsQuestion.trim() || (workspace?.documents.length || 0) === 0}
                    >
                      {wsStreaming ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Send className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </TabsContent>

                {/* Single-document Chat (preserved from Phase 3) */}
                <TabsContent value="chat" className="mt-3 min-h-0 flex-1">
                  <ChatPanel document={document} onOpenCitation={handleOpenCitation} />
                </TabsContent>

                {/* Phase 6B: evidence-backed multi-paper comparison */}
                <TabsContent value="compare" className="mt-3 min-h-0 flex-1">
                  <ComparisonPanel workspace={workspace} onOpenCitation={handleOpenCitation} />
                </TabsContent>

                {/* Explore (preserved from Phase 2) */}
                <TabsContent value="explore" className="mt-3 min-h-0 flex-1 space-y-4 overflow-y-auto pb-2">
                  {document ? (
                    <>
                      <ProcessingPanel
                        document={document}
                        onReprocess={handleReprocess}
                        reprocessing={reprocessing}
                      />
                      <DocInfoPanel document={document} />
                      <div className="min-h-[420px]">
                        <ExtractionPanel document={document} onJumpToPage={setViewerPage} />
                      </div>
                    </>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      Select a document to explore its structure.
                    </p>
                  )}
                </TabsContent>

                {/* Phase 8: Research Gap Detection */}
                <TabsContent value="gaps" className="mt-3 min-h-0 flex-1">
                  <GapsPanel workspace={workspace} onOpenCitation={handleOpenCitation} />
                </TabsContent>
              </Tabs>
            </div>
          </div>
        )}
      </main>
    </AppShell>
  );
}
