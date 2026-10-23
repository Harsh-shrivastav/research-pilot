"use client";

import { AlertTriangle, CheckCircle2, Info, Loader2, RefreshCw } from "lucide-react";
import { badgeVariants } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Comparability, ComparabilityIssue, ComparisonMatrixOut } from "@/lib/types";

// ---------------------------------------------------------------------------
// Comparability banner: shown above the comparison matrix. It says whether the
// compared papers are directly comparable and, when they are not, exactly why.
// Each issue is a button; selecting one highlights the matrix cells it is based
// on (the panel owns the selection and does the highlighting).
//
//   not_comparable / partially_comparable -> amber warning, caveat + issue list
//   comparable                            -> small green note
//   unknown                               -> grey note
//   null / undefined                      -> nothing
// ---------------------------------------------------------------------------

type MatrixShape = Pick<ComparisonMatrixOut, "documents" | "dimensions">;

/** Stable key for one matrix cell: a cell is identified by (document, criterion). */
export function cellKey(documentId: string, dimensionKey: string): string {
  return `${documentId}\u0000${dimensionKey}`;
}

/**
 * The matrix cells an issue names. References to a paper or criterion that is
 * not in this matrix are skipped (the backend may be ahead of / behind the
 * matrix the user is looking at), and duplicates are collapsed. Never throws.
 */
export function issueCellKeys(
  issue: ComparabilityIssue | null | undefined,
  matrix: MatrixShape
): string[] {
  if (!issue || !Array.isArray(issue.cell_refs)) return [];
  const docIds = new Set(matrix.documents.map((d) => d.id));
  const dimKeys = new Set(matrix.dimensions.map((d) => d.key));
  const keys: string[] = [];
  for (const ref of issue.cell_refs) {
    if (!ref || !docIds.has(ref.document_id) || !dimKeys.has(ref.dimension_key)) continue;
    const key = cellKey(ref.document_id, ref.dimension_key);
    if (!keys.includes(key)) keys.push(key);
  }
  return keys;
}

/** Display names of the papers an issue involves, skipping ids not in the matrix. */
function issuePaperNames(issue: ComparabilityIssue, matrix: MatrixShape): string[] {
  const ids =
    Array.isArray(issue.document_ids) && issue.document_ids.length > 0
      ? issue.document_ids
      : Array.isArray(issue.cell_refs)
        ? issue.cell_refs.map((r) => r?.document_id)
        : [];
  const names: string[] = [];
  for (const id of ids) {
    const doc = matrix.documents.find((d) => d.id === id);
    const name = doc ? doc.title || doc.original_filename : null;
    if (name && !names.includes(name)) names.push(name);
  }
  return names;
}

function checkedByLabel(checkedBy: Comparability["checked_by"] | undefined): string | null {
  if (checkedBy === "rules") return "Checked by rules";
  if (checkedBy === "gemini") return "Checked by Gemini";
  return null;
}

/** Muted "checked by" text + the small re-check control (+ inline error). */
function BannerFooter({
  checkedBy,
  onRecheck,
  rechecking,
  recheckError,
  className,
}: {
  checkedBy: Comparability["checked_by"] | undefined;
  onRecheck: () => void;
  rechecking: boolean;
  recheckError: string | null;
  className?: string;
}) {
  const label = checkedByLabel(checkedBy);
  return (
    <div className={className}>
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <span className="text-[11px] text-muted-foreground">{label}</span>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-6 gap-1 px-1.5 text-[11px] text-muted-foreground"
          onClick={onRecheck}
          disabled={rechecking}
          aria-label="Re-check comparability"
        >
          {rechecking ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <RefreshCw className="h-3 w-3" />
          )}
          {rechecking ? "Checking…" : "Re-check"}
        </Button>
      </div>
      {recheckError && (
        <p role="alert" className="mt-1 flex items-start gap-1 text-[11px] text-destructive">
          <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
          <span className="min-w-0 [overflow-wrap:anywhere]">{recheckError}</span>
        </p>
      )}
    </div>
  );
}

export function ComparabilityBanner({
  comparability,
  matrix,
  selectedIssue,
  onSelectIssue,
  onRecheck,
  rechecking,
  recheckError,
  className,
}: {
  comparability: Comparability | null | undefined;
  matrix: MatrixShape;
  /** Index into comparability.issues, or null. */
  selectedIssue: number | null;
  onSelectIssue: (index: number | null) => void;
  onRecheck: () => void;
  rechecking: boolean;
  recheckError: string | null;
  className?: string;
}) {
  if (!comparability) return null;

  const caveat = typeof comparability.caveat === "string" ? comparability.caveat.trim() : "";
  const issues = Array.isArray(comparability.issues) ? comparability.issues : [];
  const footer = (extra?: string) => (
    <BannerFooter
      checkedBy={comparability.checked_by}
      onRecheck={onRecheck}
      rechecking={rechecking}
      recheckError={recheckError}
      className={extra}
    />
  );

  // --- amber: not comparable / partially comparable -------------------------
  if (comparability.overall === "not_comparable" || comparability.overall === "partially_comparable") {
    const heading =
      comparability.overall === "not_comparable" ? "Not directly comparable" : "Only partially comparable";
    return (
      <section
        aria-label="Comparability check"
        className={cn(
          "rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2.5 text-xs text-amber-900",
          className
        )}
      >
        <div className="flex items-start gap-2">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
          <div className="min-w-0 flex-1">
            <p className="font-semibold">{heading}</p>
            {caveat && <p className="mt-0.5 leading-relaxed [overflow-wrap:anywhere]">{caveat}</p>}
          </div>
        </div>

        {issues.length > 0 && (
          <p className="mt-2 text-[11px] text-amber-800">Select an issue to highlight the cells it is based on.</p>
        )}
        {issues.length > 0 && (
          <ul className="mt-1.5 space-y-1.5" aria-label="Comparability issues">
            {issues.map((issue, index) => {
              const cellCount = issueCellKeys(issue, matrix).length;
              const selected = selectedIssue === index;
              const papers = issuePaperNames(issue, matrix);
              return (
                <li key={index}>
                  <button
                    type="button"
                    aria-pressed={selected}
                    disabled={cellCount === 0}
                    onClick={() => onSelectIssue(selected ? null : index)}
                    className={cn(
                      "block w-full rounded-md border px-2.5 py-1.5 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                      selected
                        ? "border-amber-600/70 bg-amber-500/20 ring-1 ring-amber-500/60"
                        : "border-amber-500/30 bg-background/70 hover:bg-amber-500/15",
                      cellCount === 0 && "cursor-default opacity-70 hover:bg-background/70"
                    )}
                  >
                    <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      {/* Badge styles on a <span>: Badge itself renders a <div>, which is invalid inside a <button>. */}
                      <span
                        className={cn(
                          badgeVariants({ variant: "outline" }),
                          "border-amber-500/40 bg-amber-500/15 px-2 py-0 text-[10px] font-semibold uppercase tracking-wide text-amber-800"
                        )}
                      >
                        {issue.type}
                      </span>
                      {papers.length > 0 && (
                        <span className="min-w-0 text-[11px] text-muted-foreground [overflow-wrap:anywhere]">
                          {papers.join(" · ")}
                        </span>
                      )}
                    </span>
                    <span className="mt-1 block text-xs leading-relaxed text-foreground [overflow-wrap:anywhere]">
                      {issue.description}
                    </span>
                    {(cellCount === 0 || selected) && (
                      <span className="mt-0.5 block text-[11px] text-muted-foreground">
                        {cellCount === 0
                          ? "No matching cells in this matrix"
                          : `Highlighting ${cellCount} ${cellCount === 1 ? "cell" : "cells"} in the matrix · click to clear`}
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        )}

        {footer("mt-2 border-t border-amber-500/20 pt-1.5")}
      </section>
    );
  }

  // --- green: comparable ----------------------------------------------------
  if (comparability.overall === "comparable") {
    return (
      <section
        aria-label="Comparability check"
        className={cn(
          "rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-900",
          className
        )}
      >
        <div className="flex items-start gap-2">
          <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-600" />
          <p className="min-w-0 flex-1 leading-relaxed [overflow-wrap:anywhere]">
            <span className="font-semibold">Directly comparable.</span>
            {caveat && <> {caveat}</>}
          </p>
        </div>
        {footer("mt-1 pl-5")}
      </section>
    );
  }

  // --- grey: unknown (and any value this UI does not recognise) --------------
  return (
    <section
      aria-label="Comparability check"
      className={cn("rounded-md border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground", className)}
    >
      <div className="flex items-start gap-2">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <div className="min-w-0 flex-1">
          <p className="font-medium text-foreground/80">Not enough evidence to judge comparability</p>
          {caveat && (
            <p className="mt-0.5 text-[11px] leading-relaxed [overflow-wrap:anywhere]">{caveat}</p>
          )}
        </div>
      </div>
      {footer("mt-1 pl-5")}
    </section>
  );
}
