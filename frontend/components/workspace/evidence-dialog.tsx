"use client";

import { useEffect, useState } from "react";
import {
  AlertCircle,
  BookOpenText,
  ExternalLink,
  FileText,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { EvidenceOut } from "@/lib/types";

// ---------------------------------------------------------------------------
// View Evidence dialog — shows full evidence detail from the API.
//
// Shared by single-document chat, workspace chat, and Phase 6B comparison
// cells: every one of them attaches an `evidence_id` to its citations via
// the same Evidence & Research Object Layer (see app/evidence/), so there is
// exactly one "View Evidence" experience across the whole app rather than a
// copy per feature.
// ---------------------------------------------------------------------------

export function ViewEvidenceDialog({
  evidenceId,
  open,
  onOpenChange,
  onOpenPage,
}: {
  evidenceId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onOpenPage?: (page: number) => void;
}) {
  const [evidence, setEvidence] = useState<EvidenceOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !evidenceId) return;
    setLoading(true);
    setError(null);
    api
      .getEvidence(evidenceId)
      .then(setEvidence)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load evidence"))
      .finally(() => setLoading(false));
  }, [open, evidenceId]);

  const statusColor = (status: string) => {
    switch (status) {
      case "supported":
        return "bg-emerald-500/15 text-emerald-700 border-emerald-500/30";
      case "partially_supported":
        return "bg-amber-500/15 text-amber-700 border-amber-500/30";
      case "unsupported":
        return "bg-red-500/15 text-red-700 border-red-500/30";
      default:
        return "bg-muted text-muted-foreground border-border";
    }
  };

  const statusLabel = (status: string) =>
    status.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="h-4.5 w-4.5 text-primary" />
            Evidence Detail
          </DialogTitle>
          <DialogDescription>Source evidence from the document</DialogDescription>
        </DialogHeader>

        {loading && (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </div>
        )}

        {evidence && !loading && (
          <div className="space-y-4">
            {/* Document / Paper title */}
            <div>
              <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                Document
              </p>
              <p className="mt-0.5 text-sm font-semibold text-foreground">
                {evidence.paper?.title || evidence.document_title || "Untitled document"}
              </p>
              {evidence.paper?.authors && (
                <p className="mt-0.5 text-xs text-muted-foreground">
                  {evidence.paper.authors}
                </p>
              )}
            </div>

            {/* Location info */}
            <div className="flex flex-wrap items-center gap-2">
              {evidence.page_number != null && (
                <Badge variant="outline" className="gap-1 text-xs">
                  Page {evidence.page_number}
                </Badge>
              )}
              {evidence.section_title && (
                <Badge variant="outline" className="gap-1 text-xs">
                  § {evidence.section_title}
                </Badge>
              )}
              {evidence.subsection_title && (
                <Badge variant="outline" className="gap-1 text-xs">
                  § {evidence.subsection_title}
                </Badge>
              )}
              <Badge variant="outline" className="text-xs capitalize">
                {evidence.source_type}
              </Badge>
            </div>

            {/* Snippet */}
            <div>
              <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                Evidence snippet
              </p>
              <div className="mt-1 rounded-lg border border-border bg-muted/40 p-3 text-sm leading-relaxed text-foreground">
                {evidence.snippet}
              </div>
            </div>

            {/* Scores & Status */}
            <div className="grid grid-cols-3 gap-3">
              {evidence.retrieval_score != null && (
                <div className="rounded-lg border border-border bg-muted/30 px-3 py-2">
                  <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                    Retrieval score
                  </p>
                  <p className="mt-0.5 text-sm font-semibold tabular-nums">
                    {evidence.retrieval_score.toFixed(4)}
                  </p>
                </div>
              )}
              {evidence.confidence != null && (
                <div className="rounded-lg border border-border bg-muted/30 px-3 py-2">
                  <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                    Confidence
                  </p>
                  <p className="mt-0.5 text-sm font-semibold tabular-nums">
                    {(evidence.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              )}
              <div className="rounded-lg border border-border bg-muted/30 px-3 py-2">
                <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                  Support status
                </p>
                <p className="mt-0.5">
                  <span
                    className={cn(
                      "inline-block rounded-full border px-2 py-0.5 text-[11px] font-medium",
                      statusColor(evidence.support_status)
                    )}
                  >
                    {statusLabel(evidence.support_status)}
                  </span>
                </p>
              </div>
            </div>

            {/* DOI / arXiv links */}
            {(evidence.paper?.doi_url || evidence.paper?.arxiv_url) && (
              <div className="flex flex-wrap gap-2">
                {evidence.paper?.doi_url && (
                  <a
                    href={evidence.paper.doi_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
                  >
                    <ExternalLink className="h-3 w-3" />
                    DOI
                  </a>
                )}
                {evidence.paper?.arxiv_url && (
                  <a
                    href={evidence.paper.arxiv_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
                  >
                    <ExternalLink className="h-3 w-3" />
                    arXiv
                  </a>
                )}
              </div>
            )}

            {/* Open page action */}
            {evidence.page_number != null && onOpenPage && (
              <Button
                variant="outline"
                size="sm"
                className="w-full gap-1.5"
                onClick={() => {
                  onOpenPage(evidence.page_number!);
                  onOpenChange(false);
                }}
              >
                <BookOpenText className="h-3.5 w-3.5" />
                Open Page {evidence.page_number}
              </Button>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
