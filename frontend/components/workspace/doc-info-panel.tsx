import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { DocumentDetailOut } from "@/lib/types";
import { formatBytes, formatRelativeTime } from "@/lib/utils";

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3 text-sm">
      <span className="shrink-0 text-muted-foreground">{label}</span>
      <span className="max-w-[62%] break-words text-right font-medium">{value}</span>
    </div>
  );
}

/** Document metadata — extracted values where available, file facts otherwise. */
export function DocInfoPanel({ document: doc }: { document: DocumentDetailOut }) {
  const ocrPages = doc.pages.filter((p) => p.ocr_applied).length;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Document metadata</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        {doc.title && <Row label="Title" value={doc.title} />}
        {doc.authors && <Row label="Authors" value={doc.authors} />}
        <Row label="File" value={doc.original_filename} />
        <Row label="Size" value={formatBytes(doc.file_size)} />
        <Row label="Uploaded" value={formatRelativeTime(doc.uploaded_at)} />

        {doc.processing_status === "READY" && (
          <>
            <div className="!mt-4 border-t border-border pt-3" />
            <Row label="Pages" value={doc.page_count ?? "—"} />
            <Row
              label="Words"
              value={doc.word_count != null ? doc.word_count.toLocaleString() : "—"}
            />
            <Row label="Sections" value={doc.section_count} />
            <Row label="Elements" value={doc.element_count} />
            <Row label="Tables" value={doc.table_count} />
            <Row label="Figures" value={doc.figure_count} />
            <Row label="References" value={doc.reference_count} />
            {ocrPages > 0 && (
              <Row label="OCR pages" value={`${ocrPages} of ${doc.page_count ?? "?"}`} />
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
