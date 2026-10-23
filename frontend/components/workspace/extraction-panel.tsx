"use client";

import { useCallback, useEffect, useState } from "react";
import {
  BookMarked,
  FileStack,
  ImageIcon,
  Loader2,
  ListTree,
  Table2,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { ElementView, MarkdownTable } from "@/components/workspace/element-view";
import { api, ApiError } from "@/lib/api";
import type {
  DocumentDetailOut,
  ElementOut,
  PageDetailOut,
  ReferenceOut,
  SectionDetailOut,
} from "@/lib/types";

/**
 * The extraction browser: outline, tables & figures, references, and raw page
 * text. Selecting a section loads its real extracted content from the API and
 * jumps the PDF viewer to the page it starts on.
 */
export function ExtractionPanel({
  document: doc,
  onJumpToPage,
}: {
  document: DocumentDetailOut;
  onJumpToPage: (page: number) => void;
}) {
  const [selectedSectionId, setSelectedSectionId] = useState<string | null>(null);
  const [section, setSection] = useState<SectionDetailOut | null>(null);
  const [sectionLoading, setSectionLoading] = useState(false);

  const [media, setMedia] = useState<ElementOut[] | null>(null);
  const [references, setReferences] = useState<ReferenceOut[] | null>(null);
  const [page, setPage] = useState<PageDetailOut | null>(null);
  const [pageLoading, setPageLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ready = doc.processing_status === "READY";

  const selectSection = useCallback(
    async (sectionId: string, startPage: number) => {
      setSelectedSectionId(sectionId);
      setSectionLoading(true);
      setError(null);
      onJumpToPage(startPage);
      try {
        setSection(await api.getSection(doc.id, sectionId));
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Could not load that section.");
        setSection(null);
      } finally {
        setSectionLoading(false);
      }
    },
    [doc.id, onJumpToPage]
  );

  // Load tables + figures + captions together for the media tab.
  useEffect(() => {
    if (!ready || media !== null) return;
    Promise.all([
      api.listElements(doc.id, { elementType: "table", limit: 500 }),
      api.listElements(doc.id, { elementType: "figure", limit: 500 }),
      api.listElements(doc.id, { elementType: "caption", limit: 500 }),
    ])
      .then(([tables, figures, captions]) =>
        setMedia(
          [...tables, ...figures, ...captions].sort(
            (a, b) => a.order_index - b.order_index
          )
        )
      )
      .catch(() => setMedia([]));
  }, [doc.id, ready, media]);

  useEffect(() => {
    if (!ready || references !== null) return;
    api
      .listReferences(doc.id)
      .then(setReferences)
      .catch(() => setReferences([]));
  }, [doc.id, ready, references]);

  const loadPage = useCallback(
    async (pageNumber: number) => {
      setPageLoading(true);
      onJumpToPage(pageNumber);
      try {
        setPage(await api.getPage(doc.id, pageNumber));
      } catch {
        setPage(null);
      } finally {
        setPageLoading(false);
      }
    },
    [doc.id, onJumpToPage]
  );

  if (!ready) {
    return (
      <Card className="flex flex-col">
        <CardHeader>
          <CardTitle className="text-base">Extracted content</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center gap-3 py-10 pt-0 text-center">
          {doc.processing_status === "FAILED" ? (
            <p className="max-w-xs text-sm text-muted-foreground">
              Extraction failed for this document, so there is nothing to show. The
              error is in the Processing panel — you can fix the source file or hit
              Reprocess.
            </p>
          ) : (
            <>
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              <p className="max-w-xs text-sm text-muted-foreground">
                Waiting for processing to finish. Extracted sections, tables,
                figures and references will appear here.
              </p>
            </>
          )}
        </CardContent>
      </Card>
    );
  }

  const tables = (media ?? []).filter((e) => e.element_type === "table");
  const figures = (media ?? []).filter((e) => e.element_type === "figure");
  const captions = (media ?? []).filter((e) => e.element_type === "caption");

  return (
    <Card className="flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Extracted content</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col pt-0">
        {error && (
          <div className="mb-3 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {error}
          </div>
        )}

        <Tabs defaultValue="sections" className="flex flex-col">
          <TabsList className="grid w-full grid-cols-4 [&>*]:px-1.5">
            <TabsTrigger value="sections" className="gap-1 text-xs">
              <ListTree className="h-3.5 w-3.5" />
              Sections
            </TabsTrigger>
            <TabsTrigger value="media" className="gap-1 text-xs">
              <Table2 className="h-3.5 w-3.5" />
              Tables
            </TabsTrigger>
            <TabsTrigger value="references" className="gap-1 text-xs">
              <BookMarked className="h-3.5 w-3.5" />
              Refs
            </TabsTrigger>
            <TabsTrigger value="pages" className="gap-1 text-xs">
              <FileStack className="h-3.5 w-3.5" />
              Pages
            </TabsTrigger>
          </TabsList>

          {/* ---------------- Sections ---------------- */}
          <TabsContent value="sections">
            {doc.sections.length === 0 ? (
              <Empty>No sections were detected in this document.</Empty>
            ) : (
              <div className="space-y-1">
                {doc.sections.map((s) => {
                  const active = s.id === selectedSectionId;
                  return (
                    <div key={s.id}>
                      <button
                        type="button"
                        onClick={() => selectSection(s.id, s.start_page)}
                        style={{ paddingLeft: `${(s.level - 1) * 12 + 8}px` }}
                        className={`w-full rounded-md py-1.5 pr-2 text-left text-sm transition-colors hover:bg-secondary ${
                          active ? "bg-secondary font-medium" : ""
                        }`}
                      >
                        <span className="block truncate">{s.title}</span>
                        <span className="text-[11px] text-muted-foreground">
                          p.{s.start_page}
                          {s.end_page !== s.start_page && `–${s.end_page}`} ·{" "}
                          {s.char_count.toLocaleString()} chars
                        </span>
                      </button>

                      {active && (
                        <div className="mb-2 ml-2 mt-1 space-y-3 rounded-md border border-border bg-muted/30 p-3">
                          {sectionLoading ? (
                            <>
                              <Skeleton className="h-4 w-full" />
                              <Skeleton className="h-4 w-5/6" />
                              <Skeleton className="h-4 w-4/6" />
                            </>
                          ) : section && section.elements.length > 0 ? (
                            section.elements.map((el) => (
                              <ElementView
                                key={el.id}
                                element={el}
                                onJumpToPage={onJumpToPage}
                              />
                            ))
                          ) : section && section.text ? (
                            <p className="whitespace-pre-wrap text-sm leading-relaxed">
                              {section.text}
                            </p>
                          ) : (
                            <p className="text-xs text-muted-foreground">
                              This section heading has no body content of its own.
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </TabsContent>

          {/* ---------------- Tables & figures ---------------- */}
          <TabsContent value="media">
            {media === null ? (
              <Skeleton className="h-24 w-full" />
            ) : media.length === 0 ? (
              <Empty>No tables or figures were detected.</Empty>
            ) : (
              <div className="space-y-5">
                <Group icon={Table2} label={`Tables (${tables.length})`}>
                  {tables.length === 0 ? (
                    <Empty small>None detected.</Empty>
                  ) : (
                    tables.map((t) => (
                      <div key={t.id} className="space-y-1">
                        <PageLink page={t.page_number} onJump={onJumpToPage} />
                        <MarkdownTable content={t.content} />
                      </div>
                    ))
                  )}
                </Group>

                <Group icon={ImageIcon} label={`Figures (${figures.length})`}>
                  {figures.length === 0 ? (
                    <Empty small>None detected.</Empty>
                  ) : (
                    figures.map((f) => (
                      <ElementView key={f.id} element={f} onJumpToPage={onJumpToPage} />
                    ))
                  )}
                </Group>

                {captions.length > 0 && (
                  <Group icon={ImageIcon} label={`Captions (${captions.length})`}>
                    {captions.map((c) => (
                      <ElementView key={c.id} element={c} onJumpToPage={onJumpToPage} />
                    ))}
                  </Group>
                )}
              </div>
            )}
          </TabsContent>

          {/* ---------------- References ---------------- */}
          <TabsContent value="references">
            {references === null ? (
              <Skeleton className="h-24 w-full" />
            ) : references.length === 0 ? (
              <Empty>
                No bibliography was detected. Not every document has one, and
                unusual reference formats can defeat detection.
              </Empty>
            ) : (
              <ol className="space-y-3">
                {references.map((r) => (
                  <li key={r.id} className="border-b border-border pb-3 last:border-0">
                    <div className="flex items-baseline gap-2">
                      <span className="text-xs text-muted-foreground">
                        [{r.order_index + 1}]
                      </span>
                      <div className="min-w-0 flex-1">
                        {r.title ? (
                          <p className="text-sm font-medium">{r.title}</p>
                        ) : (
                          <p className="text-sm">{r.raw_text}</p>
                        )}
                        <p className="mt-0.5 text-xs text-muted-foreground">
                          {[r.authors, r.venue, r.year ? String(r.year) : null]
                            .filter(Boolean)
                            .join(" · ")}
                        </p>
                        {(r.doi || r.url) && (
                          <a
                            href={r.doi ? `https://doi.org/${r.doi}` : r.url!}
                            target="_blank"
                            rel="noreferrer"
                            className="mt-0.5 inline-block break-all text-xs text-blue-600 hover:underline"
                          >
                            {r.doi ? `doi:${r.doi}` : r.url}
                          </a>
                        )}
                      </div>
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </TabsContent>

          {/* ---------------- Pages ---------------- */}
          <TabsContent value="pages">
            <div className="mb-3 flex flex-wrap gap-1">
              {doc.pages.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => loadPage(p.page_number)}
                  title={
                    p.ocr_applied
                      ? "Text recovered by OCR"
                      : p.has_text_layer
                        ? "Embedded text layer"
                        : "No text found on this page"
                  }
                  className={`h-7 min-w-[1.75rem] rounded border px-1.5 text-xs transition-colors hover:bg-secondary ${
                    page?.page_number === p.page_number
                      ? "border-foreground bg-secondary font-medium"
                      : "border-border"
                  } ${p.ocr_applied ? "text-amber-700" : ""}`}
                >
                  {p.page_number}
                </button>
              ))}
            </div>

            {pageLoading ? (
              <Skeleton className="h-32 w-full" />
            ) : page ? (
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground">
                  {page.char_count.toLocaleString()} chars ·{" "}
                  {page.ocr_applied
                    ? "recovered by OCR"
                    : page.has_text_layer
                      ? "embedded text layer"
                      : "no text found"}
                  {page.image_count > 0 && ` · ${page.image_count} image(s)`}
                </p>
                <p className="whitespace-pre-wrap text-sm leading-relaxed">
                  {page.text || "(no text on this page)"}
                </p>
              </div>
            ) : (
              <Empty small>Select a page to see its extracted text.</Empty>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

function Empty({ children, small }: { children: React.ReactNode; small?: boolean }) {
  return (
    <p className={`text-muted-foreground ${small ? "text-xs" : "py-8 text-center text-sm"}`}>
      {children}
    </p>
  );
}

function Group({
  icon: Icon,
  label,
  children,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
        <Icon className="h-3.5 w-3.5" />
        {label}
      </div>
      {children}
    </div>
  );
}

function PageLink({ page, onJump }: { page: number; onJump: (p: number) => void }) {
  return (
    <button
      type="button"
      onClick={() => onJump(page)}
      className="text-[11px] uppercase tracking-wide text-muted-foreground hover:text-foreground hover:underline"
    >
      page {page}
    </button>
  );
}
