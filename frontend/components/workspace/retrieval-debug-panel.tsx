"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { DebugCandidateOut, RetrievalDebugOut } from "@/lib/types";

function ScoreCell({ value, dim }: { value: number | null; dim?: boolean }) {
  if (value === null) {
    return <span className="text-muted-foreground/50">&mdash;</span>;
  }
  return <span className={cn("tabular-nums", dim && "text-muted-foreground")}>{value.toFixed(4)}</span>;
}

function RankCell({ value }: { value: number | null }) {
  if (value === null) {
    return <span className="text-muted-foreground/50">&mdash;</span>;
  }
  return <span className="tabular-nums text-muted-foreground">#{value}</span>;
}

function CandidateRow({ candidate }: { candidate: DebugCandidateOut }) {
  return (
    <tr className={cn("border-b border-border/60 last:border-0", candidate.selected && "bg-primary/5")}>
      <td className="whitespace-nowrap px-2 py-1.5 align-top">
        <div className="flex items-center gap-1.5">
          {candidate.selected && (
            <span
              className="inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-primary"
              title="Selected into final Top-K"
            />
          )}
          <span className="text-muted-foreground">p.{candidate.page_number}</span>
        </div>
        {candidate.section_title && (
          <div className="max-w-[10rem] truncate text-[11px] text-muted-foreground/70">
            {candidate.section_title}
          </div>
        )}
      </td>
      <td className="px-2 py-1.5 align-top">
        <p className="line-clamp-2 max-w-[22rem] text-muted-foreground">{candidate.snippet}</p>
        {candidate.matched_keyword_terms.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {candidate.matched_keyword_terms.slice(0, 6).map((t) => (
              <span key={t} className="rounded bg-muted px-1 py-0.5 text-[10px] text-muted-foreground">
                {t}
              </span>
            ))}
          </div>
        )}
      </td>
      <td className="px-2 py-1.5 text-right align-top">
        <ScoreCell value={candidate.vector_score} />
        <div>
          <RankCell value={candidate.vector_rank} />
        </div>
      </td>
      <td className="px-2 py-1.5 text-right align-top">
        <ScoreCell value={candidate.keyword_score} />
        <div>
          <RankCell value={candidate.keyword_rank} />
        </div>
      </td>
      <td className="px-2 py-1.5 text-right align-top font-medium">
        <ScoreCell value={candidate.fusion_score} />
      </td>
      <td className="px-2 py-1.5 text-right align-top font-medium text-primary">
        <ScoreCell value={candidate.rerank_score} />
      </td>
    </tr>
  );
}

export function RetrievalDebugPanel({ debug }: { debug: RetrievalDebugOut }) {
  const selectedCount = debug.candidates.filter((c) => c.selected).length;

  return (
    <div className="w-full max-w-[92%] overflow-hidden rounded-lg border border-dashed border-primary/30 bg-muted/20">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-border/60 px-3 py-2 text-[11px]">
        <span className="font-semibold uppercase tracking-wide text-primary">Retrieval Debug</span>
        <Badge variant="outline" className="h-5 px-1.5 text-[10px]">
          {debug.mode}
        </Badge>
        <span className="text-muted-foreground">
          {debug.candidates.length} candidates &middot; {selectedCount} selected
        </span>
        {typeof debug.timings_ms.total_ms === "number" && (
          <span className="text-muted-foreground">{debug.timings_ms.total_ms.toFixed(1)} ms total</span>
        )}
        {typeof debug.config.reranker === "string" && (
          <span className="text-muted-foreground">reranker: {debug.config.reranker as string}</span>
        )}
      </div>

      <div className="border-b border-border/60 px-3 py-2 text-[11px]">
        <div className="text-muted-foreground">
          Query: <span className="text-foreground">{debug.query.normalized}</span>
        </div>
        {debug.query.keywords.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {debug.query.keywords.map((k) => (
              <span key={k} className="rounded bg-muted px-1 py-0.5 text-[10px] text-muted-foreground">
                {k}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-[11px]">
          <thead>
            <tr className="border-b border-border/60 text-left text-[10px] uppercase tracking-wide text-muted-foreground">
              <th className="px-2 py-1.5 font-medium">Chunk</th>
              <th className="px-2 py-1.5 font-medium">Text</th>
              <th className="px-2 py-1.5 text-right font-medium">Vector</th>
              <th className="px-2 py-1.5 text-right font-medium">Keyword</th>
              <th className="px-2 py-1.5 text-right font-medium">Fusion</th>
              <th className="px-2 py-1.5 text-right font-medium">Reranker</th>
            </tr>
          </thead>
          <tbody>
            {debug.candidates.map((c) => (
              <CandidateRow key={c.chunk_id} candidate={c} />
            ))}
          </tbody>
        </table>
      </div>

      {debug.candidates.length === 0 && (
        <p className="px-3 py-2 text-[11px] text-muted-foreground">No candidates were retrieved for this query.</p>
      )}
    </div>
  );
}
