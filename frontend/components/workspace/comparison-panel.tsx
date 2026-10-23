"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  Check,
  ChevronDown,
  ChevronUp,
  Columns3,
  History,
  Loader2,
  Plus,
  ShieldCheck,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ComparabilityBanner,
  cellKey,
  issueCellKeys,
} from "@/components/workspace/comparability-banner";
import { ViewEvidenceDialog } from "@/components/workspace/evidence-dialog";
import { DraftSection } from "@/components/workspace/draft-section";
import { QuoteBadge, SupportBadge } from "@/components/workspace/support-badge";
import { api, ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  COMPARISON_MAX_PAPERS,
  COMPARISON_MIN_PAPERS,
} from "@/lib/types";
import type {
  CitationOut,
  ComparisonCellOut,
  ComparisonDimensionOut,
  ComparisonMatrixOut,
  ComparisonSummaryOut,
  WorkspaceDetailOut,
} from "@/lib/types";

// ---------------------------------------------------------------------------
// One matrix cell: the extracted value, an insufficient-evidence badge if
// applicable, and small numbered source chips that open the shared "View
// Evidence" dialog — the same one single-document and workspace chat use.
// ---------------------------------------------------------------------------

function MatrixCell({
  cell,
  onOpenCitation,
  onViewEvidence,
}: {
  cell: ComparisonCellOut | undefined;
  onOpenCitation: (page: number, citation: CitationOut) => void;
  onViewEvidence: (citation: CitationOut) => void;
}) {
  if (!cell) {
    return <span className="text-xs text-muted-foreground">—</span>;
  }

  if (cell.error) {
    return (
      <div className="flex items-start gap-1.5 text-xs text-destructive">
        <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <span>{cell.error}</span>
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      <p
        className={cn(
          "text-xs leading-relaxed",
          cell.insufficient_evidence ? "italic text-muted-foreground" : "text-foreground"
        )}
      >
        {cell.value_text}
      </p>
      {(cell.quote_verified != null || cell.citations.length > 0) && (
        <div className="flex flex-wrap gap-1">
          <QuoteBadge verified={cell.quote_verified} />
          {cell.citations.length > 0 && (
            <SupportBadge
              status={cell.support_status}
              checked={cell.support_checked}
              reason={cell.support_reason}
            />
          )}
        </div>
      )}
      {cell.citations.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {cell.citations.map((c) => (
            <button
              key={`${cell.id}-${c.chunk_id}-${c.marker ?? "x"}`}
              onClick={() => {
                onOpenCitation(c.page_number, c);
                if (c.evidence_id) onViewEvidence(c);
              }}
              className="rounded border border-border bg-muted/50 px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary"
              title={`Page ${c.page_number}${c.section_title ? ` · ${c.section_title}` : ""}`}
            >
              [{c.marker ?? "?"}] p.{c.page_number}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function ComparisonPanel({
  workspace,
  onOpenCitation,
}: {
  workspace: WorkspaceDetailOut | null;
  onOpenCitation: (page: number, citation: CitationOut) => void;
}) {
  const [defaultDimensions, setDefaultDimensions] = useState<ComparisonDimensionOut[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [selectedDimensionKeys, setSelectedDimensionKeys] = useState<string[]>([]);
  const [customDimension, setCustomDimension] = useState("");

  const [history, setHistory] = useState<ComparisonSummaryOut[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [matrix, setMatrix] = useState<ComparisonMatrixOut | null>(null);

  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewingCitation, setViewingCitation] = useState<CitationOut | null>(null);

  // Comparability banner: which issue is selected (index into
  // matrix.comparability.issues), and the state of the "re-check" action.
  const [selectedIssue, setSelectedIssue] = useState<number | null>(null);
  const [rechecking, setRechecking] = useState(false);
  const [recheckError, setRecheckError] = useState<string | null>(null);
  // Citation verification ("Verify citations").
  const [verifying, setVerifying] = useState(false);
  const [verifyError, setVerifyError] = useState<string | null>(null);
  const matrixScrollRef = useRef<HTMLDivElement | null>(null);
  // Lets an async re-check tell whether the user has moved on to another matrix.
  const matrixIdRef = useRef<string | null>(null);
  matrixIdRef.current = matrix?.id ?? null;

  const workspaceId = workspace?.id ?? null;

  useEffect(() => {
    api
      .getDefaultComparisonDimensions()
      .then((dims) => {
        setDefaultDimensions(dims);
        setSelectedDimensionKeys(dims.map((d) => d.key));
      })
      .catch(() => setDefaultDimensions([]));
  }, []);

  const loadHistory = useCallback(() => {
    if (!workspaceId) return;
    api
      .listWorkspaceComparisons(workspaceId)
      .then(setHistory)
      .catch(() => undefined);
  }, [workspaceId]);

  useEffect(() => {
    setMatrix(null);
    setSelectedDocIds([]);
    setError(null);
    loadHistory();
  }, [workspaceId, loadHistory]);

  const toggleDoc = (docId: string) => {
    setSelectedDocIds((prev) => {
      if (prev.includes(docId)) return prev.filter((id) => id !== docId);
      if (prev.length >= COMPARISON_MAX_PAPERS) return prev;
      return [...prev, docId];
    });
  };

  const toggleDimension = (key: string) => {
    setSelectedDimensionKeys((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    );
  };

  const addCustomDimension = () => {
    const text = customDimension.trim();
    if (!text) return;
    if (!selectedDimensionKeys.includes(text)) {
      setSelectedDimensionKeys((prev) => [...prev, text]);
    }
    setCustomDimension("");
  };

  const removeCustomDimension = (key: string) => {
    setSelectedDimensionKeys((prev) => prev.filter((k) => k !== key));
  };

  const canRun =
    !!workspaceId &&
    selectedDocIds.length >= COMPARISON_MIN_PAPERS &&
    selectedDocIds.length <= COMPARISON_MAX_PAPERS &&
    selectedDimensionKeys.length > 0 &&
    !running;

  const handleRun = useCallback(async () => {
    if (!workspaceId || !canRun) return;
    setRunning(true);
    setError(null);
    try {
      const result = await api.createComparison(workspaceId, {
        document_ids: selectedDocIds,
        dimensions: selectedDimensionKeys,
      });
      setMatrix(result);
      loadHistory();
      if (result.status === "failed" && result.error) {
        setError(result.error);
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not run this comparison.");
    } finally {
      setRunning(false);
    }
  }, [workspaceId, canRun, selectedDocIds, selectedDimensionKeys, loadHistory]);

  const handleOpenHistoryItem = useCallback(async (id: string) => {
    setError(null);
    try {
      const detail = await api.getComparison(id);
      setMatrix(detail);
      setShowHistory(false);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not load this comparison.");
    }
  }, []);

  const handleDeleteHistoryItem = useCallback(
    async (id: string) => {
      try {
        await api.deleteComparison(id);
        setHistory((prev) => prev.filter((m) => m.id !== id));
        if (matrix?.id === id) setMatrix(null);
      } catch {
        // best-effort — leave it in the list if delete failed
      }
    },
    [matrix?.id]
  );

  // A different matrix (loaded, created, deleted) or a fresh comparability
  // result invalidates the selected issue and any re-check error.
  const comparability = matrix?.comparability ?? null;
  useEffect(() => {
    setSelectedIssue(null);
    setRecheckError(null);
  }, [matrix?.id, comparability]);

  // Cells named by the selected issue's cell_refs (unknown papers/criteria skipped).
  const highlightedCells = useMemo(() => {
    if (!matrix || !comparability || selectedIssue === null) return new Set<string>();
    const issues = Array.isArray(comparability.issues) ? comparability.issues : [];
    return new Set(issueCellKeys(issues[selectedIssue], matrix));
  }, [matrix, comparability, selectedIssue]);
  const highlightSignature = Array.from(highlightedCells).join("|");

  // The matrix scrolls horizontally: bring the first highlighted cell into view.
  useEffect(() => {
    if (!highlightSignature) return;
    const first = matrixScrollRef.current?.querySelector<HTMLElement>('td[data-highlighted="true"]');
    first?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [highlightSignature]);

  const handleRecheck = useCallback(async () => {
    const id = matrix?.id;
    if (!id) return;
    setRechecking(true);
    setRecheckError(null);
    try {
      const updated = await api.rerunComparability(id);
      // Ignore the result if the user switched to another matrix meanwhile.
      if (matrixIdRef.current === id) setMatrix(updated);
    } catch (e) {
      if (matrixIdRef.current === id) {
        setRecheckError(e instanceof ApiError ? e.message : "Could not re-check comparability.");
      }
    } finally {
      setRechecking(false);
    }
  }, [matrix?.id]);

  const handleVerify = useCallback(async () => {
    const id = matrix?.id;
    if (!id) return;
    setVerifying(true);
    setVerifyError(null);
    try {
      await api.verifyComparison(id);
      const updated = await api.getComparison(id);
      if (matrixIdRef.current === id) setMatrix(updated);
    } catch (e) {
      if (matrixIdRef.current === id) {
        setVerifyError(e instanceof ApiError ? e.message : "Could not verify citations.");
      }
    } finally {
      setVerifying(false);
    }
  }, [matrix?.id]);

  if (!workspace) {
    return (
      <Card className="flex h-full items-center justify-center p-6 text-center text-sm text-muted-foreground">
        Load a workspace to compare papers.
      </Card>
    );
  }

  const documents = workspace.documents;
  const cellFor = (documentId: string, dimensionKey: string) =>
    matrix?.cells.find((c) => c.document_id === documentId && c.dimension_key === dimensionKey);

  return (
    <Card className="flex h-full flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <Columns3 className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">Compare Papers</h3>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 gap-1.5 text-xs text-muted-foreground"
          onClick={() => setShowHistory((v) => !v)}
        >
          <History className="h-3.5 w-3.5" />
          History
          {showHistory ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        </Button>
      </div>

      {showHistory && (
        <div className="border-b border-border bg-muted/30 px-4 py-2">
          {history.length === 0 ? (
            <p className="py-1 text-xs text-muted-foreground">No comparisons run yet.</p>
          ) : (
            <div className="max-h-32 space-y-1 overflow-y-auto">
              {history.map((h) => (
                <div
                  key={h.id}
                  className="flex items-center justify-between gap-2 rounded px-2 py-1 text-xs hover:bg-accent"
                >
                  <button className="flex-1 truncate text-left" onClick={() => handleOpenHistoryItem(h.id)}>
                    {h.document_count} papers × {h.dimension_count} criteria
                    <span className="ml-1.5 text-muted-foreground">
                      · {new Date(h.created_at).toLocaleString()}
                    </span>
                    {h.status !== "completed" && (
                      <Badge variant={h.status === "failed" ? "destructive" : "secondary"} className="ml-1.5">
                        {h.status}
                      </Badge>
                    )}
                  </button>
                  <button
                    className="rounded p-0.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                    onClick={() => handleDeleteHistoryItem(h.id)}
                    title="Delete comparison"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-4">
        {error && (
          <div className="mb-4 flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Paper selection */}
        <div className="mb-4">
          <p className="mb-1.5 text-xs font-medium text-muted-foreground">
            Select {COMPARISON_MIN_PAPERS}–{COMPARISON_MAX_PAPERS} papers ({selectedDocIds.length} selected)
          </p>
          {documents.length === 0 ? (
            <p className="text-xs text-muted-foreground">Add papers to this workspace first.</p>
          ) : (
            <div className="space-y-1">
              {documents.map((doc) => {
                const checked = selectedDocIds.includes(doc.id);
                const disabled =
                  doc.rag_status !== "INDEXED" ||
                  (!checked && selectedDocIds.length >= COMPARISON_MAX_PAPERS);
                return (
                  <button
                    key={doc.id}
                    onClick={() => toggleDoc(doc.id)}
                    disabled={disabled}
                    className={cn(
                      "flex w-full items-center gap-2 rounded-md border px-2.5 py-1.5 text-left text-xs transition-colors",
                      checked
                        ? "border-primary/40 bg-primary/10"
                        : "border-border hover:bg-accent",
                      disabled && !checked && "cursor-not-allowed opacity-50"
                    )}
                  >
                    <span
                      className={cn(
                        "flex h-4 w-4 shrink-0 items-center justify-center rounded border",
                        checked ? "border-primary bg-primary text-primary-foreground" : "border-border"
                      )}
                    >
                      {checked && <Check className="h-3 w-3" />}
                    </span>
                    <span className="flex-1 truncate">{doc.title || doc.original_filename}</span>
                    {doc.rag_status !== "INDEXED" && (
                      <Badge variant="secondary" className="shrink-0">
                        not indexed
                      </Badge>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Dimension selection */}
        <div className="mb-4">
          <p className="mb-1.5 text-xs font-medium text-muted-foreground">Compare on</p>
          <div className="flex flex-wrap gap-1.5">
            {defaultDimensions.map((dim) => {
              const checked = selectedDimensionKeys.includes(dim.key);
              return (
                <button
                  key={dim.key}
                  onClick={() => toggleDimension(dim.key)}
                  className={cn(
                    "rounded-full border px-2.5 py-1 text-xs transition-colors",
                    checked
                      ? "border-primary/40 bg-primary/10 text-primary"
                      : "border-border text-muted-foreground hover:bg-accent"
                  )}
                >
                  {dim.label}
                </button>
              );
            })}
            {selectedDimensionKeys
              .filter((key) => !defaultDimensions.some((d) => d.key === key))
              .map((custom) => (
                <span
                  key={custom}
                  className="flex items-center gap-1 rounded-full border border-primary/40 bg-primary/10 px-2.5 py-1 text-xs text-primary"
                >
                  {custom}
                  <button onClick={() => removeCustomDimension(custom)} aria-label={`Remove ${custom}`}>
                    <X className="h-3 w-3" />
                  </button>
                </span>
              ))}
          </div>
          <div className="mt-2 flex gap-1.5">
            <input
              value={customDimension}
              onChange={(e) => setCustomDimension(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), addCustomDimension())}
              placeholder="Add a custom criterion (e.g. evaluation metrics)"
              className="flex-1 rounded-md border border-border bg-background px-2.5 py-1 text-xs placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
            <Button variant="outline" size="sm" className="h-7 px-2" onClick={addCustomDimension}>
              <Plus className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>

        <Button size="sm" className="mb-4 gap-1.5" onClick={handleRun} disabled={!canRun}>
          {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
          Compare Papers
        </Button>

        {/* Comparability verdict: above the matrix, below the Compare button */}
        {matrix && (
          <ComparabilityBanner
            className="mb-3"
            comparability={comparability}
            matrix={matrix}
            selectedIssue={selectedIssue}
            onSelectIssue={setSelectedIssue}
            onRecheck={handleRecheck}
            rechecking={rechecking}
            recheckError={recheckError}
          />
        )}

        {/* Verification header: how many cited cells have been checked */}
        {matrix && (
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs text-muted-foreground">
              {matrix.cells.filter((c) => c.citations.length > 0 && c.support_checked).length} of{" "}
              {matrix.cells.filter((c) => c.citations.length > 0).length} cells verified
              {verifyError && <span className="ml-2 text-destructive">{verifyError}</span>}
            </span>
            <Button
              variant="outline"
              size="sm"
              className="h-7 gap-1.5 text-xs"
              onClick={handleVerify}
              disabled={verifying || !matrix.cells.some((c) => c.citations.length > 0)}
            >
              {verifying ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <ShieldCheck className="h-3.5 w-3.5" />
              )}
              Verify citations
            </Button>
          </div>
        )}

        {/* Matrix */}
        {matrix && (
          <div ref={matrixScrollRef} className="overflow-x-auto rounded-lg border border-border">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-border bg-muted/40">
                  <th className="w-40 min-w-[10rem] border-r border-border p-2 text-left text-xs font-medium text-muted-foreground">
                    Criterion
                  </th>
                  {matrix.documents.map((doc) => (
                    <th
                      key={doc.id}
                      className="min-w-[14rem] border-r border-border p-2 text-left text-xs font-medium last:border-r-0"
                    >
                      {doc.title || doc.original_filename}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.dimensions.map((dim) => (
                  <tr key={dim.key} className="border-b border-border last:border-b-0">
                    <td className="border-r border-border bg-muted/20 p-2 align-top text-xs font-medium">
                      {dim.label}
                    </td>
                    {matrix.documents.map((doc) => {
                      const highlighted = highlightedCells.has(cellKey(doc.id, dim.key));
                      return (
                        <td
                          key={doc.id}
                          data-doc-id={doc.id}
                          data-dimension-key={dim.key}
                          data-highlighted={highlighted ? "true" : undefined}
                          className={cn(
                            "border-r border-border p-2 align-top transition-colors last:border-r-0",
                            highlighted && "bg-amber-500/10 ring-1 ring-inset ring-amber-500/60"
                          )}
                        >
                          <MatrixCell
                            cell={cellFor(doc.id, dim.key)}
                            onOpenCitation={onOpenCitation}
                            onViewEvidence={setViewingCitation}
                          />
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {matrix && (
          <DraftSection
            matrix={matrix}
            onViewEvidence={(evidenceId) =>
              setViewingCitation({
                chunk_id: "",
                document_id: "",
                page_number: 1,
                section_title: null,
                snippet: "",
                score: 0,
                marker: null,
                evidence_id: evidenceId,
              })
            }
          />
        )}

        {!matrix && !running && (
          <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border py-10 text-center">
            <Columns3 className="h-6 w-6 text-muted-foreground/50" />
            <p className="max-w-[240px] text-xs text-muted-foreground">
              Select papers and criteria, then run a comparison to see an evidence-backed matrix.
            </p>
          </div>
        )}
      </div>

      {viewingCitation?.evidence_id && (
        <ViewEvidenceDialog
          evidenceId={viewingCitation.evidence_id}
          open={!!viewingCitation}
          onOpenChange={(open) => !open && setViewingCitation(null)}
          onOpenPage={(page) => onOpenCitation(page, viewingCitation)}
        />
      )}
    </Card>
  );
}
