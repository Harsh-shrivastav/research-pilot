"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, ExternalLink, FileWarning } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { fileUrl } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { DocumentOut } from "@/lib/types";

/**
 * Basic PDF viewer.
 *
 * Renders the PDF served by the backend in an iframe and drives the browser's
 * built-in viewer with the `#page=N` fragment, so selecting a section jumps
 * to the page that section starts on. Changing the fragment alone does not
 * always re-navigate an iframe, so the key is bumped to force a reload when
 * the requested page changes.
 *
 * Deliberately not pdf.js: that would be a new frontend dependency, and the
 * native viewer already gives zoom, search, text selection and printing.
 *
 * Citation highlighting: the native browser PDF viewer inside the iframe
 * gives this page no API to read its render geometry (zoom level, scroll
 * offset), so an exact bounding-box overlay over the cited passage is not
 * achievable without switching to a pdf.js-based viewer. `highlightToken`
 * instead drives a best-effort visual cue — the panel border pulses and a
 * "Jumped to page N from a citation" chip appears briefly — so a citation
 * click is still clearly, visibly acknowledged even though the exact region
 * on the page cannot be boxed.
 */
export function PdfPanel({
  document: doc,
  page,
  pageCount,
  onPageChange,
  highlightToken,
  highlightLabel,
}: {
  document: DocumentOut | null;
  page?: number;
  pageCount?: number | null;
  onPageChange?: (page: number) => void;
  /** Bump this (e.g. Date.now()) to trigger the best-effort highlight flash. */
  highlightToken?: number;
  highlightLabel?: string;
}) {
  const [current, setCurrent] = useState(page ?? 1);
  const [flashing, setFlashing] = useState(false);
  const loadedRef = useRef(0);
  const flashTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (page && page !== current) setCurrent(page);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page]);

  useEffect(() => {
    if (!highlightToken) return;
    setFlashing(true);
    if (flashTimeoutRef.current) clearTimeout(flashTimeoutRef.current);
    flashTimeoutRef.current = setTimeout(() => setFlashing(false), 2200);
    return () => {
      if (flashTimeoutRef.current) clearTimeout(flashTimeoutRef.current);
    };
  }, [highlightToken]);

  if (!doc) {
    return (
      <Card className="flex h-full items-center justify-center">
        <CardContent className="flex flex-col items-center gap-2 pt-6 text-center text-muted-foreground">
          <FileWarning className="h-8 w-8" />
          <p className="text-sm">No document attached to this workspace yet.</p>
        </CardContent>
      </Card>
    );
  }

  const total = pageCount ?? null;
  const src = `${fileUrl(doc.file_url)}#page=${current}&view=FitH`;

  const go = (next: number) => {
    const clamped = Math.max(1, total ? Math.min(next, total) : next);
    setCurrent(clamped);
    loadedRef.current += 1;
    onPageChange?.(clamped);
  };

  return (
    <Card
      className={cn(
        "relative flex h-full flex-col overflow-hidden transition-shadow duration-300",
        flashing && "ring-2 ring-primary shadow-[0_0_0_4px_hsl(var(--primary)/0.15)]"
      )}
    >
      <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => go(current - 1)}
            disabled={current <= 1}
            aria-label="Previous page"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="min-w-[5.5rem] text-center text-xs text-muted-foreground">
            Page {current}
            {total ? ` of ${total}` : ""}
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => go(current + 1)}
            disabled={total ? current >= total : false}
            aria-label="Next page"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
        <a
          href={fileUrl(doc.file_url)}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          <ExternalLink className="h-3.5 w-3.5" />
          Open
        </a>
      </div>
      <div className="relative min-h-0 flex-1">
        <iframe
          key={`${doc.id}-${current}-${loadedRef.current}`}
          title={doc.original_filename}
          src={src}
          className="h-full w-full"
        />
        {flashing && (
          <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-center p-3">
            <div className="animate-in fade-in slide-in-from-top-1 rounded-full border border-primary/30 bg-primary text-primary-foreground px-3 py-1 text-xs font-medium shadow-lg">
              {highlightLabel || `Jumped to page ${current} from a citation`}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
