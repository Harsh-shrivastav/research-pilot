"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Clock,
  Columns3,
  FileText,
  FolderKanban,
  Plus,
  ShieldCheck,
} from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { PageMain } from "@/components/layout/page-layout";
import { Topbar } from "@/components/layout/topbar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Donut, StackedBar, UploadsChart } from "@/components/dashboard/charts";
import { StatCard } from "@/components/dashboard/stat-card";
import { StatusBadge } from "@/components/dashboard/status-badge";
import { api, ApiError } from "@/lib/api";
import type { DashboardStats } from "@/lib/types";
import { formatBytes, formatRelativeTime } from "@/lib/utils";

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .getDashboard()
      .then(setStats)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Failed to load dashboard."))
      .finally(() => setLoading(false));
  }, []);

  const citations = stats?.citation_counts ?? {};
  const claimTotal = Object.values(citations).reduce((sum, n) => sum + n, 0);
  const verified = citations.supported ?? 0;
  const checked = claimTotal - (citations.uncertain ?? 0);

  return (
    <AppShell>
      <Topbar
        title="Dashboard"
        subtitle="Your research at a glance"
        action={
          <Button asChild>
            <Link href="/documents">
              <Plus className="h-4 w-4" /> New Research Workspace
            </Link>
          </Button>
        }
      />

      <PageMain>
        {error && (
          <div className="mb-6 rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Could not reach the API ({error}). Is the backend running?
          </div>
        )}

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {loading ? (
            <>
              <Skeleton className="h-24" />
              <Skeleton className="h-24" />
              <Skeleton className="h-24" />
              <Skeleton className="h-24" />
            </>
          ) : (
            <>
              <StatCard
                label="Total documents"
                value={stats?.total_documents ?? 0}
                icon={FileText}
                accent="bg-blue-100 text-blue-700"
                hint={`${stats?.status_counts?.READY ?? 0} ready to use`}
              />
              <StatCard
                label="Active research workspaces"
                value={stats?.active_workspaces ?? 0}
                icon={FolderKanban}
                accent="bg-violet-100 text-violet-700"
              />
              <StatCard
                label="Comparisons run"
                value={stats?.total_comparisons ?? 0}
                icon={Columns3}
                accent="bg-amber-100 text-amber-700"
              />
              <StatCard
                label="Citations verified"
                value={verified}
                icon={ShieldCheck}
                accent="bg-emerald-100 text-emerald-700"
                hint={checked > 0 ? `${Math.round((verified / checked) * 100)}% of ${checked} checked` : "Run Verify citations"}
              />
            </>
          )}
        </div>

        {!loading && (
          <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-3">
            <Card className="lg:col-span-2">
              <CardHeader className="flex-row items-center justify-between space-y-0">
                <CardTitle className="text-base">Uploads, last 14 days</CardTitle>
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent className="pt-0">
                <UploadsChart data={stats?.uploads_by_day ?? []} />
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex-row items-center justify-between space-y-0">
                <CardTitle className="text-base">Document status</CardTitle>
                <FileText className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent className="pt-0">
                <Donut
                  centerLabel="Documents by status"
                  slices={[
                    { label: "Ready", value: stats?.status_counts?.READY ?? 0, color: "#10b981" },
                    { label: "Processing", value: (stats?.status_counts?.PROCESSING ?? 0) + (stats?.status_counts?.UPLOADED ?? 0), color: "#f59e0b" },
                    { label: "Failed", value: stats?.status_counts?.FAILED ?? 0, color: "#ef4444" },
                  ]}
                />
              </CardContent>
            </Card>
          </div>
        )}

        {!loading && (
          <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader className="flex-row items-center justify-between space-y-0">
                <CardTitle className="text-base">Citation verification</CardTitle>
                <ShieldCheck className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent className="pt-0">
                {claimTotal > 0 ? (
                  <StackedBar
                    slices={[
                      { label: "Supported", value: citations.supported ?? 0, color: "#10b981" },
                      { label: "Weak", value: citations.partially_supported ?? 0, color: "#f59e0b" },
                      { label: "Unsupported", value: citations.unsupported ?? 0, color: "#ef4444" },
                      { label: "Unchecked", value: citations.uncertain ?? 0, color: "#94a3b8" },
                    ]}
                  />
                ) : (
                  <EmptyRow text="No cited claims yet - ask a question or run a comparison." />
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex-row items-center justify-between space-y-0">
                <CardTitle className="text-base">Recent workspaces</CardTitle>
                <FolderKanban className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent className="pt-0">
                {stats?.recent_workspaces && stats.recent_workspaces.length > 0 ? (
                  <ul className="divide-y divide-border">
                    {stats.recent_workspaces.map((ws) => (
                      <li key={ws.id}>
                        <Link
                          href={`/workspace/${ws.id}`}
                          className="group -mx-2 flex items-center justify-between gap-3 rounded-md px-2 py-2.5 transition-colors hover:bg-muted/50"
                        >
                          <div className="min-w-0">
                            <p className="truncate text-sm font-medium">{ws.name}</p>
                            <p className="text-xs text-muted-foreground">
                              {ws.document_count} paper{ws.document_count === 1 ? "" : "s"} Â· {formatRelativeTime(ws.updated_at)}
                            </p>
                          </div>
                          <ArrowRight className="h-4 w-4 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
                        </Link>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <EmptyRow
                    text="No workspaces yet."
                    cta={
                      <Link href="/workspace/new" className="text-primary underline-offset-4 hover:underline">
                        Open a blank workspace
                      </Link>
                    }
                  />
                )}
              </CardContent>
            </Card>
          </div>
        )}

        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader className="flex-row items-center justify-between space-y-0">
              <CardTitle className="text-base">Recent documents</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent className="pt-0">
              {loading ? (
                <div className="space-y-3">
                  <Skeleton className="h-10" />
                  <Skeleton className="h-10" />
                </div>
              ) : stats && stats.recent_documents.length > 0 ? (
                <ul className="divide-y divide-border">
                  {stats.recent_documents.map((doc) => (
                    <li key={doc.id} className="flex items-center justify-between py-3">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium">{doc.original_filename}</p>
                        <p className="text-xs text-muted-foreground">
                          {formatBytes(doc.file_size)} · {formatRelativeTime(doc.uploaded_at)}
                        </p>
                      </div>
                      <StatusBadge status={doc.processing_status} />
                    </li>
                  ))}
                </ul>
              ) : (
                <EmptyRow
                  text="No documents yet."
                  cta={
                    <Link href="/documents" className="text-primary underline-offset-4 hover:underline">
                      Upload your first paper
                    </Link>
                  }
                />
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex-row items-center justify-between space-y-0">
              <CardTitle className="text-base">Recent activity</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent className="pt-0">
              {loading ? (
                <div className="space-y-3">
                  <Skeleton className="h-8" />
                  <Skeleton className="h-8" />
                </div>
              ) : stats && stats.recent_activity.length > 0 ? (
                <ul className="space-y-3">
                  {stats.recent_activity.map((item) => (
                    <li key={item.id} className="flex items-start gap-3 text-sm">
                      <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                      <div>
                        <p>{item.message}</p>
                        <p className="text-xs text-muted-foreground">
                          {formatRelativeTime(item.timestamp)}
                        </p>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <EmptyRow text="No activity yet — upload a document to get started." />
              )}
            </CardContent>
          </Card>
        </div>
      </PageMain>
    </AppShell>
  );
}

function EmptyRow({ text, cta }: { text: string; cta?: React.ReactNode }) {
  return (
    <div className="rounded-md border border-dashed border-border py-8 text-center text-sm text-muted-foreground">
      <p>{text}</p>
      {cta && <p className="mt-1">{cta}</p>}
    </div>
  );
}
