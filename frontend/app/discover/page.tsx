"use client";

import { useCallback, useEffect, useState } from "react";
import {
  AlertCircle,
  BookmarkCheck,
  BookmarkPlus,
  ExternalLink,
  FileDown,
  Loader2,
  Search,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { PageMain } from "@/components/layout/page-layout";
import { Topbar } from "@/components/layout/topbar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api, ApiError } from "@/lib/api";
import type { DiscoverPaperOut, SavedPaperOut } from "@/lib/types";
import { cn } from "@/lib/utils";

function errorText(e: unknown, fallback: string): string {
  return e instanceof ApiError ? e.message : fallback;
}

function byline(p: { authors: string[]; year: number | null; venue: string | null }): string {
  const authors = p.authors.length > 3 ? `${p.authors.slice(0, 3).join(", ")} et al.` : p.authors.join(", ");
  return [authors, p.year, p.venue].filter(Boolean).join(" · ");
}

function PaperCard({
  paper,
  saved,
  saving,
  onSave,
}: {
  paper: DiscoverPaperOut;
  saved: boolean;
  saving: boolean;
  onSave: () => void;
}) {
  const [open, setOpen] = useState(false);
  const link = paper.doi_url || paper.landing_url;
  return (
    <Card className="space-y-2 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold leading-snug">{paper.title}</h3>
          <p className="mt-0.5 text-xs text-muted-foreground">{byline(paper) || "Unknown authors"}</p>
        </div>
        <Button
          size="sm"
          variant={saved ? "secondary" : "outline"}
          className="h-8 shrink-0 gap-1.5 text-xs"
          onClick={onSave}
          disabled={saved || saving}
        >
          {saving ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : saved ? (
            <BookmarkCheck className="h-3.5 w-3.5" />
          ) : (
            <BookmarkPlus className="h-3.5 w-3.5" />
          )}
          {saved ? "Saved" : "Save"}
        </Button>
      </div>

      {paper.abstract && (
        <div>
          <p className={cn("text-xs leading-relaxed text-muted-foreground", !open && "line-clamp-3")}>
            {paper.abstract}
          </p>
          <button className="mt-0.5 text-[11px] text-primary hover:underline" onClick={() => setOpen(!open)}>
            {open ? "Show less" : "Show more"}
          </button>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 pt-1">
        <Badge variant={paper.has_pdf ? "success" : "secondary"} className="text-[10px]">
          {paper.has_pdf ? "PDF available" : "No PDF link"}
        </Badge>
        {paper.cited_by_count > 0 && (
          <Badge variant="outline" className="text-[10px]">
            {paper.cited_by_count} citations
          </Badge>
        )}
        {paper.matched_queries.length > 1 && (
          <Badge variant="outline" className="text-[10px]">
            Found by {paper.matched_queries.length} queries
          </Badge>
        )}
        <span className="flex-1" />
        {paper.pdf_url && (
          <a
            href={paper.pdf_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
          >
            <FileDown className="h-3.5 w-3.5" /> Open PDF
          </a>
        )}
        {link && (
          <a
            href={link}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
          >
            <ExternalLink className="h-3.5 w-3.5" /> {paper.doi ? "DOI" : "Source"}
          </a>
        )}
      </div>
    </Card>
  );
}

export default function DiscoverPage() {
  const [topic, setTopic] = useState("");
  const [queries, setQueries] = useState<string[]>([]);
  const [related, setRelated] = useState<string[]>([]);
  const [newQuery, setNewQuery] = useState("");
  const [expanding, setExpanding] = useState(false);
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<DiscoverPaperOut[] | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<SavedPaperOut[]>([]);
  const [savingKey, setSavingKey] = useState<string | null>(null);

  const refreshSaved = useCallback(async () => {
    try {
      setSaved(await api.listSavedPapers());
    } catch {
      // the saved list is secondary; search still works without it
    }
  }, []);

  useEffect(() => {
    refreshSaved();
  }, [refreshSaved]);

  const paperKey = (p: DiscoverPaperOut) => p.doi ?? p.openalex_id ?? p.title;
  const isSaved = (p: DiscoverPaperOut) =>
    !!p.saved_id ||
    saved.some((s) => (p.doi && s.doi === p.doi) || (p.openalex_id && s.openalex_id === p.openalex_id));

  const handleExpand = async () => {
    if (!topic.trim()) return;
    setExpanding(true);
    setError(null);
    try {
      const out = await api.discoverQueries(topic.trim());
      setQueries(out.queries);
      setRelated(out.related_terms);
    } catch (e) {
      setError(errorText(e, "Could not expand the topic."));
    } finally {
      setExpanding(false);
    }
  };

  const runSearch = async (list: string[]) => {
    if (list.length === 0) return;
    setSearching(true);
    setError(null);
    setWarnings([]);
    try {
      const out = await api.discoverSearch(list);
      setResults(out.results);
      setWarnings(out.warnings);
    } catch (e) {
      setError(errorText(e, "Search failed."));
    } finally {
      setSearching(false);
    }
  };

  const addQuery = () => {
    const q = newQuery.trim();
    if (q && !queries.includes(q) && queries.length < 6) setQueries([...queries, q]);
    setNewQuery("");
  };

  const handleSave = async (paper: DiscoverPaperOut) => {
    setSavingKey(paperKey(paper));
    try {
      const row = await api.savePaper(paper);
      setSaved((prev) => [row, ...prev.filter((s) => s.id !== row.id)]);
      setResults((prev) => prev?.map((p) => (paperKey(p) === paperKey(paper) ? { ...p, saved_id: row.id } : p)) ?? prev);
    } catch (e) {
      setError(errorText(e, "Could not save this paper."));
    } finally {
      setSavingKey(null);
    }
  };

  const handleRemove = async (id: string) => {
    try {
      await api.deleteSavedPaper(id);
      setSaved((prev) => prev.filter((s) => s.id !== id));
      setResults((prev) => prev?.map((p) => (p.saved_id === id ? { ...p, saved_id: null } : p)) ?? prev);
    } catch (e) {
      setError(errorText(e, "Could not remove this paper."));
    }
  };

  return (
    <AppShell>
      <Topbar title="Discover" subtitle="Find papers by topic - results come from OpenAlex" />
      <PageMain>
        <div className="mx-auto max-w-4xl space-y-5">
          {error && (
            <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span className="flex-1">{error}</span>
              <button onClick={() => setError(null)} aria-label="Dismiss">
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          <Card className="space-y-4 p-4">
            <form
              className="flex flex-col gap-2 sm:flex-row"
              onSubmit={(e) => {
                e.preventDefault();
                handleExpand();
              }}
            >
              <Input
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="Describe your research topic, e.g. detecting AI-generated text"
                className="flex-1"
              />
              <Button type="submit" disabled={expanding || !topic.trim()} className="gap-1.5">
                {expanding ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                Expand with AI
              </Button>
              <Button
                type="button"
                variant="outline"
                disabled={searching || !topic.trim()}
                onClick={() => runSearch([topic.trim()])}
                className="gap-1.5"
              >
                <Search className="h-4 w-4" />
                Search directly
              </Button>
            </form>

            {queries.length > 0 && (
              <div className="space-y-3 border-t border-border pt-4">
                <p className="text-xs font-medium text-muted-foreground">
                  Search queries (remove or add your own, up to 6)
                </p>
                <div className="flex flex-wrap gap-2">
                  {queries.map((q) => (
                    <span
                      key={q}
                      className="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/50 py-1 pl-3 pr-1.5 text-xs"
                    >
                      {q}
                      <button
                        onClick={() => setQueries(queries.filter((x) => x !== q))}
                        aria-label={`Remove ${q}`}
                        className="rounded-full p-0.5 text-muted-foreground hover:bg-secondary hover:text-foreground"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </span>
                  ))}
                </div>
                {related.length > 0 && (
                  <p className="text-xs text-muted-foreground">
                    Related terms:{" "}
                    {related.map((term) => (
                      <button
                        key={term}
                        className="mr-1.5 text-primary hover:underline"
                        onClick={() => !queries.includes(term) && queries.length < 6 && setQueries([...queries, term])}
                        title="Add as a query"
                      >
                        {term}
                      </button>
                    ))}
                  </p>
                )}
                <div className="flex flex-col gap-2 sm:flex-row">
                  <Input
                    value={newQuery}
                    onChange={(e) => setNewQuery(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && addQuery()}
                    placeholder="Add a query"
                    className="h-8 flex-1 text-xs"
                  />
                  <Button size="sm" variant="outline" onClick={addQuery} disabled={!newQuery.trim() || queries.length >= 6}>
                    Add
                  </Button>
                  <Button size="sm" onClick={() => runSearch(queries)} disabled={searching} className="gap-1.5">
                    {searching ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Search className="h-3.5 w-3.5" />}
                    Search {queries.length} queries
                  </Button>
                </div>
              </div>
            )}
          </Card>

          <Tabs defaultValue="results">
            <TabsList>
              <TabsTrigger value="results">Results{results ? ` (${results.length})` : ""}</TabsTrigger>
              <TabsTrigger value="saved">Saved ({saved.length})</TabsTrigger>
            </TabsList>

            <TabsContent value="results" className="mt-4 space-y-3">
              {warnings.length > 0 && (
                <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                  {warnings.map((w) => (
                    <p key={w}>{w}</p>
                  ))}
                </div>
              )}
              {searching && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" /> Searching OpenAlex...
                </div>
              )}
              {!searching && results === null && (
                <Card className="border-dashed p-8 text-center text-sm text-muted-foreground">
                  Enter a topic, expand it into queries, then search. Papers come only from OpenAlex.
                </Card>
              )}
              {!searching && results?.length === 0 && (
                <Card className="border-dashed p-8 text-center text-sm text-muted-foreground">
                  No papers found. Try different or broader queries.
                </Card>
              )}
              {results?.map((paper) => (
                <PaperCard
                  key={paperKey(paper)}
                  paper={paper}
                  saved={isSaved(paper)}
                  saving={savingKey === paperKey(paper)}
                  onSave={() => handleSave(paper)}
                />
              ))}
            </TabsContent>

            <TabsContent value="saved" className="mt-4 space-y-3">
              {saved.length === 0 && (
                <Card className="border-dashed p-8 text-center text-sm text-muted-foreground">
                  Nothing saved yet. Save a result to keep it in your library.
                </Card>
              )}
              {saved.map((paper) => (
                <Card key={paper.id} className="flex items-start justify-between gap-3 p-4">
                  <div className="min-w-0">
                    <h3 className="text-sm font-semibold leading-snug">{paper.title}</h3>
                    <p className="mt-0.5 text-xs text-muted-foreground">{byline(paper)}</p>
                    <div className="mt-2 flex gap-3 text-xs">
                      {paper.pdf_url && (
                        <a href={paper.pdf_url} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
                          Open PDF
                        </a>
                      )}
                      {(paper.doi || paper.landing_url) && (
                        <a
                          href={paper.doi ? `https://doi.org/${paper.doi}` : paper.landing_url!}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary hover:underline"
                        >
                          {paper.doi ? "DOI" : "Source"}
                        </a>
                      )}
                    </div>
                  </div>
                  <Button size="sm" variant="ghost" className="h-8 w-8 shrink-0 p-0" onClick={() => handleRemove(paper.id)} aria-label="Remove">
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </Card>
              ))}
            </TabsContent>
          </Tabs>
        </div>
      </PageMain>
    </AppShell>
  );
}
