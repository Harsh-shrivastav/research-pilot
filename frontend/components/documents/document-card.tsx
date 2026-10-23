"use client";

import {
  FileText,
  MoreVertical,
  Trash2,
  FolderOpen,
  RefreshCw,
  Table2,
  ImageIcon,
  BookMarked,
  ScanLine,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/dashboard/status-badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import type { DocumentOut } from "@/lib/types";
import { formatBytes, formatRelativeTime } from "@/lib/utils";

export function DocumentCard({
  document,
  onOpenWorkspace,
  onDelete,
  onReprocess,
  isOpeningWorkspace,
}: {
  document: DocumentOut;
  onOpenWorkspace: (doc: DocumentOut) => void;
  onDelete: (doc: DocumentOut) => void;
  onReprocess: (doc: DocumentOut) => void;
  isOpeningWorkspace: boolean;
}) {
  const ready = document.processing_status === "READY";
  const failed = document.processing_status === "FAILED";

  return (
    <Card className="group relative flex flex-col">
      <CardContent className="flex flex-1 flex-col pt-6">
        <div className="flex items-start justify-between">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-100 text-blue-700">
            <FileText className="h-5 w-5" />
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <MoreVertical className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuItem onClick={() => onOpenWorkspace(document)}>
                <FolderOpen className="h-4 w-4" /> Open workspace
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => onReprocess(document)}>
                <RefreshCw className="h-4 w-4" /> Reprocess
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => onDelete(document)} destructive>
                <Trash2 className="h-4 w-4" /> Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        <p
          className="mt-4 truncate text-sm font-semibold"
          title={document.title || document.original_filename}
        >
          {document.title || document.original_filename}
        </p>
        {document.title && (
          <p className="truncate text-xs text-muted-foreground" title={document.original_filename}>
            {document.original_filename}
          </p>
        )}
        <p className="mt-1 text-xs text-muted-foreground">
          {formatBytes(document.file_size)}
          {ready && document.page_count != null && ` · ${document.page_count} pages`}
          {" · "}
          {formatRelativeTime(document.uploaded_at)}
        </p>

        {ready && (
          <div className="mt-3 flex flex-wrap gap-2 text-xs text-muted-foreground">
            {document.section_count > 0 && (
              <Stat icon={FolderOpen} value={document.section_count} label="sections" />
            )}
            {document.table_count > 0 && (
              <Stat icon={Table2} value={document.table_count} label="tables" />
            )}
            {document.figure_count > 0 && (
              <Stat icon={ImageIcon} value={document.figure_count} label="figures" />
            )}
            {document.reference_count > 0 && (
              <Stat icon={BookMarked} value={document.reference_count} label="refs" />
            )}
            {document.ocr_used && (
              <span className="flex items-center gap-1 text-amber-700" title="Text recovered by OCR">
                <ScanLine className="h-3 w-3" /> OCR
              </span>
            )}
          </div>
        )}

        {failed && document.processing_error && (
          <p
            className="mt-3 line-clamp-2 break-words font-mono text-[11px] text-destructive"
            title={document.processing_error}
          >
            {document.processing_error}
          </p>
        )}

        <div className="mt-auto flex items-center justify-between pt-4">
          <StatusBadge status={document.processing_status} />
          <Button
            variant="outline"
            size="sm"
            onClick={() => onOpenWorkspace(document)}
            disabled={isOpeningWorkspace}
          >
            {isOpeningWorkspace ? "Opening..." : "Open workspace"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function Stat({
  icon: Icon,
  value,
  label,
}: {
  icon: React.ComponentType<{ className?: string }>;
  value: number;
  label: string;
}) {
  return (
    <span className="flex items-center gap-1" title={`${value} ${label}`}>
      <Icon className="h-3 w-3" />
      {value}
    </span>
  );
}
