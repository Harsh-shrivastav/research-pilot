"use client";

import { useCallback, useEffect, useState } from "react";
import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  BookOpen,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  FileText,
  Lightbulb,
  Loader2,
  Search,
  Sparkles,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { api, ApiError } from "@/lib/api";
import type {
  CitationOut,
  EvidenceOut,
  GapAnalysisResult,
  ResearchGapOut,
  ResearchGapSummaryOut,
  WorkspaceDetailOut,
} from "@/lib/types";
import {
  GAP_CONFIDENCE_COLORS,
  GAP_MIN_PAPERS,
  GAP_TYPE_COLORS,
  GapConfidence,
  GapType,
} from "@/lib/types";

interface GapsPanelProps {
  workspace: WorkspaceDetailOut | null;
  onOpenCitation?: (page: number, citation: CitationOut) => void;
}

// ---------------------------------------------------------------------------
// Gap card component
// ---------------------------------------------------------------------------

function GapCard({
  gap,
  onViewDetail,
}: {
  gap: ResearchGapSummaryOut;
  onViewDetail: (id: string) => void;
}) {
  const confidenceColors = GAP_CONFIDENCE_COLORS[gap.confidence as GapConfidence] ?? GAP_CONFIDENCE_COLORS.medium;
  const typeColor = GAP_TYPE_COLORS[gap.gap_type as GapType] ?? "bg-slate-500/10 text-slate-500 border-slate-500/20";

  return (
    <div className="rounded-lg border bg-card p-4 hover:border-primary/20 transition-colors">
      <div className="flex items-start justify-between gap-3 mb-2">
        <h4 className="text-sm font-semibold text-foreground leading-snug flex-1">
          {gap.title}
        </h4>
        <span
          className={`inline-flex items-center whitespace-nowrap rounded-full border px-2 py-0.5 text-[10px] font-medium ${confidenceColors.bg} ${confidenceColors.text} ${confidenceColors.border}`}
        >
          {gap.confidence.toUpperCase()}
        </span>
      </div>

      <span
        className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium mb-2 ${typeColor}`}
      >
        {gap.gap_type_label}
      </span>

      <p className="text-xs text-muted-foreground leading-relaxed line-clamp-3 mb-3">
        {gap.description}
      </p>

      {/* Research direction (AI-generated) */}
      {gap.research_direction && (
        <div className="mb-3 rounded-md border border-dashed border-amber-500/30 bg-amber-500/5 px-3 py-2">
          <div className="flex items-center gap-1.5 mb-1">
            <Sparkles className="h-3 w-3 text-amber-500" />
            <span className="text-[10px] font-semibold text-amber-600 uppercase tracking-wide">
              AI-Generated Research Direction
            </span>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {gap.research_direction}
          </p>
        </div>
      )}

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <FileText className="h-3 w-3" />
            {gap.supporting_paper_count} paper{gap.supporting_paper_count !== 1 ? "s" : ""}
          </span>
          <span className="flex items-center gap-1">
            <BookOpen className="h-3 w-3" />
            {gap.evidence_count} evidence item{gap.evidence_count !== 1 ? "s" : ""}
          </span>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 text-xs gap-1"
          onClick={() => onViewDetail(gap.id)}
        >
          View Evidence <ArrowRight className="h-3 w-3" />
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Gap detail panel (expanded/side-panel view)
// ---------------------------------------------------------------------------

function EvidenceItem({
  ev,
  role,
  onOpenCitation,
}: {
  ev: EvidenceOut;
  role: string | null;
  onOpenCitation?: (page: number, citation: CitationOut) => void;
}) {
  return (
    <div className="rounded-md border bg-card p-2.5 hover:border-primary/20 transition-colors">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-[10px] text-muted-foreground">
          {ev.page_number ? `p.${ev.page_number}` : "page n/a"}
        </span>
        {ev.section_title && (
          <span className="text-[10px] text-muted-foreground truncate">· {ev.section_title}</span>
        )}
        {role && (
          <span className="ml-auto text-[10px] font-medium rounded-full bg-primary/10 text-primary px-2 py-0.5">
            {role.replace("_", " ")}
          </span>
        )}
      </div>
      <p className="text-xs text-muted-foreground leading-relaxed line-clamp-4">{ev.snippet}</p>
      {ev.page_number && onOpenCitation && (
        <button
          className="mt-1.5 text-[10px] text-primary hover:underline flex items-center gap-1"
          onClick={() =>
            onOpenCitation(ev.page_number!, {
              chunk_id: ev.chunk_id || "",
              document_id: ev.document_id,
              page_number: ev.page_number!,
              section_title: ev.section_title,
              snippet: ev.snippet,
              score: ev.retrieval_score || 0,
              marker: null,
              evidence_id: ev.id,
            })
          }
        >
          <ExternalLink className="h-2.5 w-2.5" />
          Open in PDF
        </button>
      )}
    </div>
  );
}

function GapDetailView({
  gapId,
  onClose,
  onOpenCitation,
}: {
  gapId: string;
  onClose: () => void;
  onOpenCitation?: (page: number, citation: CitationOut) => void;
}) {
  const [gap, setGap] = useState<ResearchGapOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .getGap(gapId)
      .then((data) => {
        if (!cancelled) setGap(data);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof ApiError ? e.message : "Failed to load gap details.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [gapId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error || !gap) {
    return (
      <div className="rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
        {error || "Gap not found."}
      </div>
    );
  }

  const confidenceColors = GAP_CONFIDENCE_COLORS[gap.confidence as GapConfidence] ?? GAP_CONFIDENCE_COLORS.medium;
  const typeColor = GAP_TYPE_COLORS[gap.gap_type as GapType] ?? "bg-slate-500/10 text-slate-500 border-slate-500/20";

  const roleByEvidence: Record<string, string | null> = {};
  const groups = new Map<string, { title: string; count: number }>();
  for (const ge of gap.evidence_items) {
    roleByEvidence[ge.evidence_id] = ge.role;
    const ev = ge.evidence;
    if (!ev) continue;
    const g = groups.get(ev.document_id) ?? { title: ev.document_title || "Unknown Paper", count: 0 };
    g.count += 1;
    groups.set(ev.document_id, g);
  }
  const paperGroups = Array.from(groups.entries());

  return (
    <div className="space-y-4 overflow-y-auto">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1">
          <h3 className="text-sm font-semibold text-foreground mb-1">{gap.title}</h3>
          <div className="flex items-center gap-2 flex-wrap">
            <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium ${typeColor}`}>
              {gap.gap_type_label}
            </span>
            <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium ${confidenceColors.bg} ${confidenceColors.text} ${confidenceColors.border}`}>
              {gap.confidence.toUpperCase()} confidence
            </span>
          </div>
        </div>
        <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Description */}
      <div>
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
          Description
        </h4>
        <p className="text-xs text-foreground leading-relaxed">{gap.description}</p>
      </div>

      {/* Confidence reasoning */}
      {gap.confidence_reasoning && (
        <div className="rounded-md border bg-muted/50 px-3 py-2">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
            Confidence Reasoning
          </h4>
          <p className="text-xs text-foreground leading-relaxed">{gap.confidence_reasoning}</p>
        </div>
      )}

      {/* Research direction */}
      {gap.research_direction && (
        <div className="rounded-md border border-dashed border-amber-500/30 bg-amber-500/5 px-3 py-2">
          <div className="flex items-center gap-1.5 mb-1">
            <Sparkles className="h-3 w-3 text-amber-500" />
            <span className="text-[10px] font-semibold text-amber-600 uppercase tracking-wide">
              AI-Generated Research Direction
            </span>
          </div>
          <p className="text-xs text-foreground leading-relaxed">{gap.research_direction}</p>
          <p className="text-[10px] text-amber-600/70 mt-1 italic">
            This is an AI interpretation based on the evidence. Novelty verification against broader literature is recommended.
          </p>
        </div>
      )}

      {/* Supporting papers */}
      {paperGroups.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
            Supporting Papers ({paperGroups.length})
          </h4>
          <div className="flex flex-wrap gap-1.5">
            {paperGroups.map(([id, g]) => (
              <span
                key={id}
                className="inline-flex items-center gap-1 rounded-full border bg-muted/50 px-2 py-0.5 text-[10px] text-foreground"
              >
                <FileText className="h-2.5 w-2.5 text-primary" />
                {g.title} · {g.count} item{g.count !== 1 ? "s" : ""}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Supporting claims → evidence */}
      <div>
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
          {gap.claims.length > 0
            ? `Supporting Claims & Evidence (${gap.claims.length})`
            : `Supporting Evidence (${gap.evidence_items.length})`}
        </h4>
        {gap.claims.length > 0 ? (
          <div className="space-y-3">
            {gap.claims.map((c) => (
              <div key={c.id} className="rounded-md border bg-background p-3">
                <div className="flex items-center gap-2 mb-1.5">
                  <FileText className="h-3 w-3 text-primary flex-shrink-0" />
                  <span className="text-[11px] font-medium text-primary truncate">
                    {c.document_title || "Unknown Paper"}
                  </span>
                  <span className="ml-auto text-[10px] rounded-full bg-muted px-2 py-0.5 text-muted-foreground">
                    {c.support_status === "uncertain" ? "unverified claim" : c.support_status.replace("_", " ")}
                  </span>
                </div>
                <div className="space-y-2">
                  {c.evidence.map((ev) => (
                    <EvidenceItem
                      key={ev.id}
                      ev={ev}
                      role={roleByEvidence[ev.id] ?? null}
                      onOpenCitation={onOpenCitation}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : gap.evidence_items.length === 0 ? (
          <p className="text-xs text-muted-foreground italic">No evidence items linked.</p>
        ) : (
          <div className="space-y-2">
            {gap.evidence_items.map((ge) =>
              ge.evidence ? (
                <EvidenceItem
                  key={ge.id}
                  ev={ge.evidence}
                  role={ge.role}
                  onOpenCitation={onOpenCitation}
                />
              ) : null,
            )}
          </div>
        )}
      </div>

      {/* Provenance disclaimer */}
      <div className="rounded-md border border-muted bg-muted/30 px-3 py-2">
        <p className="text-[10px] text-muted-foreground italic leading-relaxed">
          {gap.novelty_notice}
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Gaps Panel
// ---------------------------------------------------------------------------

export function GapsPanel({
  workspace,
  onOpenCitation,
}: GapsPanelProps) {
  const [gaps, setGaps] = useState<ResearchGapSummaryOut[]>([]);
  const [typeCounts, setTypeCounts] = useState<Record<string, number>>({});
  const [papersAnalyzed, setPapersAnalyzed] = useState(0);
  const [rejected, setRejected] = useState(0);
  const [analyzing, setAnalyzing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedGapId, setSelectedGapId] = useState<string | null>(null);
  const [hasLoaded, setHasLoaded] = useState(false);

  // Count indexed papers
  const indexedPaperCount =
    workspace?.documents.filter((d) => d.rag_status === "INDEXED").length || 0;

  // Load existing gaps on mount
  const loadGaps = useCallback(async () => {
    if (!workspace) return;
    setLoading(true);
    setError(null);
    try {
      const existing = await api.listWorkspaceGaps(workspace.id);
      setGaps(existing);
      // Build type counts from existing
      const counts: Record<string, number> = {};
      for (const g of existing) {
        const label = g.gap_type_label;
        counts[label] = (counts[label] || 0) + 1;
      }
      setTypeCounts(counts);
      setHasLoaded(true);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load existing gaps.");
    } finally {
      setLoading(false);
    }
  }, [workspace]);

  useEffect(() => {
    loadGaps();
  }, [loadGaps]);

  const handleAnalyze = async () => {
    if (!workspace || analyzing) return;
    setAnalyzing(true);
    setError(null);
    setSelectedGapId(null);

    try {
      const result: GapAnalysisResult = await api.analyzeGaps(workspace.id);
      setGaps(result.gaps);
      setTypeCounts(result.gap_type_counts);
      setPapersAnalyzed(result.papers_analyzed);
      setRejected(result.candidates_rejected);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Gap analysis failed.");
    } finally {
      setAnalyzing(false);
    }
  };

  // Detail view
  if (selectedGapId) {
    return (
      <div className="flex flex-col h-full overflow-y-auto rounded-lg border bg-card p-4">
        <GapDetailView
          gapId={selectedGapId}
          onClose={() => setSelectedGapId(null)}
          onOpenCitation={onOpenCitation}
        />
      </div>
    );
  }

  // Not enough papers
  if (indexedPaperCount < GAP_MIN_PAPERS) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center py-12 rounded-lg border bg-card p-4">
        <Search className="h-10 w-10 text-muted-foreground/30 mb-3" />
        <h4 className="text-sm font-medium text-muted-foreground mb-1">
          Research Gap Detection
        </h4>
        <p className="text-xs text-muted-foreground max-w-[280px] mb-4">
          At least {GAP_MIN_PAPERS} indexed papers are required for gap detection.
          This workspace has {indexedPaperCount} indexed paper{indexedPaperCount !== 1 ? "s" : ""}.
        </p>
        <div className="flex items-center gap-1.5 text-xs text-amber-600 bg-amber-500/10 rounded-md px-3 py-1.5">
          <AlertTriangle className="h-3.5 w-3.5" />
          Add and index more papers to enable gap detection.
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden rounded-lg border bg-card">
      {/* Header */}
      <div className="border-b px-4 py-3">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-foreground flex items-center gap-1.5">
            <Lightbulb className="h-4 w-4" />
            Research Gaps
          </h3>
          <Button
            size="sm"
            className="h-7 text-xs gap-1.5"
            disabled={analyzing}
            onClick={handleAnalyze}
          >
            {analyzing ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin" />
                Analyzing…
              </>
            ) : (
              <>
                <Search className="h-3 w-3" />
                {gaps.length > 0 ? "Re-Analyze" : "Analyze Research Gaps"}
              </>
            )}
          </Button>
        </div>

        {/* Summary row */}
        {gaps.length > 0 && (
          <div className="flex items-center gap-4 text-[11px] text-muted-foreground">
            <span>
              <strong className="text-foreground">{gaps.length}</strong> potential gap{gaps.length !== 1 ? "s" : ""} found
            </span>
            <span>
              <strong className="text-foreground">{papersAnalyzed || indexedPaperCount}</strong> paper
              {(papersAnalyzed || indexedPaperCount) !== 1 ? "s" : ""} analyzed
            </span>
            {rejected > 0 && (
              <span title="Candidates discarded because no verified evidence supported them">
                <strong className="text-foreground">{rejected}</strong> unsupported candidate{rejected !== 1 ? "s" : ""} rejected
              </span>
            )}
          </div>
        )}

        {/* Type breakdown */}
        {Object.keys(typeCounts).length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-2">
            {Object.entries(typeCounts).map(([label, count]) => (
              <span
                key={label}
                className="inline-flex items-center rounded-full border bg-muted/50 px-2 py-0.5 text-[10px] text-muted-foreground"
              >
                {label}: <strong className="ml-0.5 text-foreground">{count}</strong>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mx-4 mt-3 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive flex items-start gap-2">
          <AlertCircle className="h-3.5 w-3.5 mt-0.5 flex-shrink-0" />
          <span>{error}</span>
          <button className="ml-auto underline text-[10px]" onClick={() => setError(null)}>dismiss</button>
        </div>
      )}

      {/* Gap list or empty state */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : gaps.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-12">
            <Search className="h-10 w-10 text-muted-foreground/30 mb-3" />
            <h4 className="text-sm font-medium text-muted-foreground mb-1">
              No Research Gaps Yet
            </h4>
            <p className="text-xs text-muted-foreground max-w-[260px]">
              Click &ldquo;Analyze Research Gaps&rdquo; to identify potential research gaps
              across your workspace papers based on their limitations, future work,
              methodologies, and findings.
            </p>
          </div>
        ) : (
          gaps.map((gap) => (
            <GapCard
              key={gap.id}
              gap={gap}
              onViewDetail={setSelectedGapId}
            />
          ))
        )}
      </div>

      {/* Provenance footer */}
      {gaps.length > 0 && (
        <div className="border-t px-4 py-2">
          <p className="text-[10px] text-muted-foreground italic leading-relaxed">
            Potential research gaps identified from the analyzed papers. This system cannot establish
            global novelty. Novelty verification against the broader literature is recommended.
          </p>
        </div>
      )}
    </div>
  );
}
