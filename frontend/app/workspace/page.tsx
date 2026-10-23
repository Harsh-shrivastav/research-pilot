"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { FileText, FolderKanban, Loader2, Plus } from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { PageMain } from "@/components/layout/page-layout";
import { Topbar } from "@/components/layout/topbar";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { api, ApiError } from "@/lib/api";
import type { WorkspaceOut } from "@/lib/types";
import { formatRelativeTime } from "@/lib/utils";

const TABS = ["chat", "compare", "gaps"];

export default function WorkspacesPage() {
  const router = useRouter();
  const [workspaces, setWorkspaces] = useState<WorkspaceOut[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const tab = new URLSearchParams(window.location.search).get("tab");
    api
      .listWorkspaces()
      .then((list) => {
        const sorted = [...list].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
        // A sidebar tool link (?tab=...) goes straight to the latest workspace.
        if (tab && TABS.includes(tab) && sorted.length > 0) {
          router.replace(`/workspace/${sorted[0].id}?tab=${tab}`);
          return;
        }
        setWorkspaces(sorted);
      })
      .catch((e) => {
        setError(e instanceof ApiError ? e.message : "Could not load workspaces.");
        setWorkspaces([]);
      });
  }, [router]);

  return (
    <AppShell>
      <Topbar
        title="Research Workspaces"
        subtitle="Group papers to chat, compare and find gaps across them"
        action={
          <Button asChild>
            <Link href="/documents">
              <Plus className="h-4 w-4" /> New workspace
            </Link>
          </Button>
        }
      />
      <PageMain>
        {error && (
          <div className="mb-6 rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}
        {workspaces === null ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading workspaces...
          </div>
        ) : workspaces.length === 0 ? (
          <Card className="flex flex-col items-center gap-3 border-dashed p-10 text-center">
            <FolderKanban className="h-8 w-8 text-muted-foreground/50" />
            <p className="max-w-sm text-sm text-muted-foreground">
              No workspaces yet. Open a document from the Documents page to start one.
            </p>
            <Button asChild variant="outline" size="sm">
              <Link href="/documents">Go to Documents</Link>
            </Button>
          </Card>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {workspaces.map((ws) => (
              <Link key={ws.id} href={`/workspace/${ws.id}`} className="group">
                <Card className="h-full p-4 transition-colors group-hover:border-primary/40 group-hover:bg-muted/30">
                  <div className="flex items-start gap-3">
                    <span className="rounded-lg bg-primary/10 p-2 text-primary">
                      <FolderKanban className="h-4 w-4" />
                    </span>
                    <div className="min-w-0">
                      <h3 className="truncate text-sm font-semibold">{ws.name}</h3>
                      {ws.description && (
                        <p className="mt-0.5 line-clamp-2 text-xs text-muted-foreground">{ws.description}</p>
                      )}
                    </div>
                  </div>
                  <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <FileText className="h-3.5 w-3.5" />
                      {ws.document_count} paper{ws.document_count === 1 ? "" : "s"}
                    </span>
                    <span>Updated {formatRelativeTime(ws.updated_at)}</span>
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        )}
      </PageMain>
    </AppShell>
  );
}
