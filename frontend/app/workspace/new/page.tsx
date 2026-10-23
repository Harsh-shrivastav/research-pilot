"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { PageMain } from "@/components/layout/page-layout";
import { Topbar } from "@/components/layout/topbar";
import { Button } from "@/components/ui/button";
import { api, ApiError } from "@/lib/api";

/**
 * "Research Workspace" in the sidebar lands here: it opens a blank workspace
 * (all tools, no papers yet - add them with the Papers picker). An existing
 * empty workspace is reused so repeated visits don't pile up blank ones.
 */
export default function NewWorkspacePage() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const existing = await api.listWorkspaces();
        const blank = existing
          .filter((w) => w.document_count === 0)
          .sort((a, b) => b.updated_at.localeCompare(a.updated_at))[0];
        const ws = blank ?? (await api.createWorkspace({ name: "Untitled workspace" }));
        if (!cancelled) router.replace(`/workspace/${ws.id}`);
      } catch (e) {
        if (!cancelled) setError(e instanceof ApiError ? e.message : "Could not open a workspace.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  return (
    <AppShell>
      <Topbar title="Research Workspace" subtitle="Opening a blank workspace..." />
      <PageMain>
        {error ? (
          <div className="space-y-3">
            <div className="rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              {error}
            </div>
            <Button asChild variant="outline" size="sm">
              <Link href="/workspace">All workspaces</Link>
            </Button>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> Preparing your workspace...
          </div>
        )}
      </PageMain>
    </AppShell>
  );
}
