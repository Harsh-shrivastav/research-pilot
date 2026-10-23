"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { FileX2 } from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { Topbar } from "@/components/layout/topbar";
import { UploadDropzone } from "@/components/documents/upload-dropzone";
import { DocumentCard } from "@/components/documents/document-card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { api, ApiError } from "@/lib/api";
import { isProcessing } from "@/lib/types";
import type { DocumentOut } from "@/lib/types";

const POLL_INTERVAL_MS = 2500;

export default function DocumentsPage() {
  const router = useRouter();
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<DocumentOut | null>(null);
  const [openingWorkspaceFor, setOpeningWorkspaceFor] = useState<string | null>(null);

  const loadDocuments = useCallback(() => {
    setLoading(true);
    api
      .listDocuments()
      .then(setDocuments)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Failed to load documents."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  // Refresh the list while anything is still queued or processing, so cards
  // move from "Queued" to "Ready" (with their extraction counts) on their own.
  const anyProcessing = documents.some((d) => isProcessing(d.processing_status));
  useEffect(() => {
    if (!anyProcessing) return;
    const timer = setInterval(() => {
      api
        .listDocuments()
        .then(setDocuments)
        .catch(() => {
          /* transient: the next tick retries */
        });
    }, POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [anyProcessing]);

  const handleUpload = async (file: File, onProgress: (pct: number) => void) => {
    const doc = await api.uploadDocument(file, onProgress);
    setDocuments((prev) => [doc, ...prev]);
  };

  const handleOpenWorkspace = async (doc: DocumentOut) => {
    setOpeningWorkspaceFor(doc.id);
    try {
      const existing = await api.getWorkspaceByDocument(doc.id);
      if (existing) {
        router.push(`/workspace/${existing.id}`);
        return;
      }
      const workspace = await api.createWorkspace({
        name: doc.original_filename.replace(/\.pdf$/i, ""),
        document_id: doc.id,
      });
      router.push(`/workspace/${workspace.id}`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not open workspace.");
      setOpeningWorkspaceFor(null);
    }
  };

  const handleReprocess = async (doc: DocumentOut) => {
    setError(null);
    try {
      const updated = await api.reprocessDocument(doc.id);
      setDocuments((prev) => prev.map((d) => (d.id === updated.id ? updated : d)));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not requeue that document.");
    }
  };

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    const target = pendingDelete;
    setPendingDelete(null);
    setDocuments((prev) => prev.filter((d) => d.id !== target.id));
    try {
      await api.deleteDocument(target.id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to delete document.");
      loadDocuments();
    }
  };

  return (
    <AppShell>
      <Topbar title="Documents" subtitle="Your uploaded research papers" />

      <main className="flex-1 overflow-y-auto p-6">
        {error && (
          <div className="mb-6 rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}

        <UploadDropzone onUploadFile={handleUpload} />

        <div className="mt-8">
          {loading ? (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-48" />
              ))}
            </div>
          ) : documents.length === 0 ? (
            <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border py-16 text-center">
              <FileX2 className="mb-3 h-8 w-8 text-muted-foreground" />
              <p className="text-sm font-medium">No documents yet</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Upload a PDF above to get started.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {documents.map((doc) => (
                <DocumentCard
                  key={doc.id}
                  document={doc}
                  onOpenWorkspace={handleOpenWorkspace}
                  onDelete={setPendingDelete}
                  onReprocess={handleReprocess}
                  isOpeningWorkspace={openingWorkspaceFor === doc.id}
                />
              ))}
            </div>
          )}
        </div>
      </main>

      <Dialog open={!!pendingDelete} onOpenChange={(open: boolean) => !open && setPendingDelete(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete document?</DialogTitle>
            <DialogDescription>
              This will permanently delete &ldquo;{pendingDelete?.original_filename}&rdquo; and
              remove it from any workspace. This can&rsquo;t be undone.
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setPendingDelete(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmDelete}>
              Delete
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </AppShell>
  );
}
