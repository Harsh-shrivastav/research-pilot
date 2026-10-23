"use client";

import { AlertTriangle, CheckCircle2, Loader2, RefreshCw } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/dashboard/status-badge";
import type { DocumentDetailOut } from "@/lib/types";

/** Processing status, extraction counts and any parser warnings. */
export function ProcessingPanel({
  document: doc,
  onReprocess,
  reprocessing,
}: {
  document: DocumentDetailOut;
  onReprocess: () => void;
  reprocessing: boolean;
}) {
  const inFlight =
    doc.processing_status === "UPLOADED" || doc.processing_status === "PROCESSING";

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base">Processing</CardTitle>
        <StatusBadge status={doc.processing_status} />
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        {inFlight && (
          <div className="flex items-center gap-2 rounded-md bg-secondary px-3 py-2 text-xs text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            {doc.processing_status === "UPLOADED"
              ? "Queued — waiting for a worker to pick this up."
              : "Extracting pages, sections, tables and references…"}
          </div>
        )}

        {doc.processing_status === "FAILED" && (
          <div className="space-y-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2">
            <div className="flex items-center gap-2 text-xs font-medium text-destructive">
              <AlertTriangle className="h-3.5 w-3.5" />
              Processing failed
            </div>
            {doc.processing_error && (
              <p className="break-words font-mono text-[11px] leading-relaxed text-destructive">
                {doc.processing_error}
              </p>
            )}
          </div>
        )}

        {doc.processing_status === "READY" && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
            Extracted with <span className="font-medium">{doc.parser_backend}</span>
            {doc.ocr_used && " (OCR applied)"}
            {doc.processing_duration_ms != null &&
              ` in ${(doc.processing_duration_ms / 1000).toFixed(1)}s`}
          </div>
        )}

        {doc.warnings.length > 0 && (
          <div className="space-y-1 rounded-md border border-amber-300 bg-amber-50 px-3 py-2">
            <div className="flex items-center gap-2 text-xs font-medium text-amber-800">
              <AlertTriangle className="h-3.5 w-3.5" />
              Extraction warnings
            </div>
            <ul className="list-inside list-disc space-y-0.5 text-[11px] text-amber-800">
              {doc.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}

        <Button
          variant="outline"
          size="sm"
          className="w-full"
          onClick={onReprocess}
          disabled={reprocessing || doc.processing_status === "PROCESSING"}
        >
          <RefreshCw className={`h-4 w-4 ${reprocessing ? "animate-spin" : ""}`} />
          {reprocessing ? "Requeuing…" : "Reprocess document"}
        </Button>
      </CardContent>
    </Card>
  );
}
