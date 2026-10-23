"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, Check, Copy, FileText, Loader2, Save, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SupportBadge } from "@/components/workspace/support-badge";
import { api, ApiError } from "@/lib/api";
import type { ComparisonMatrixOut, LiteratureDraftOut, SupportStatus } from "@/lib/types";

const MIN_PAPERS_WITH_EVIDENCE = 2;

// Markdown of the generated draft, with the backend-built citation labels.
function draftToMarkdown(draft: LiteratureDraftOut): string {
  const paragraphs = draft.content.paragraphs.map((p) =>
    p.sentences
      .filter((s) => s.status === "ok" && s.evidence_ids.length > 0)
      .map((s) => `${s.text} ${s.citation_labels.join(" ")}`.trim())
      .join(" ")
  );
  if (draft.content.caveat) paragraphs.push(draft.content.caveat);
  return paragraphs.filter(Boolean).join("\n\n");
}

export function DraftSection({
  matrix,
  onViewEvidence,
}: {
  matrix: ComparisonMatrixOut;
  onViewEvidence: (evidenceId: string) => void;
}) {
  const [draft, setDraft] = useState<LiteratureDraftOut | null>(null);
  const [text, setText] = useState("");
  const [generating, setGenerating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [copied, setCopied] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [verdicts, setVerdicts] = useState<Map<string, { status: SupportStatus; reason: string }>>(new Map());
  const [error, setError] = useState<string | null>(null);

  const matrixId = matrix.id;

  const adopt = useCallback((d: LiteratureDraftOut | null) => {
    setDraft(d);
    setText(d ? (d.edited_text ?? draftToMarkdown(d)) : "");
    setVerdicts(new Map());
    setSaved(false);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    api
      .getLatestDraft(matrixId)
      .then((d) => !cancelled && adopt(d))
      .catch(() => !cancelled && adopt(null));
    return () => {
      cancelled = true;
    };
  }, [matrixId, adopt]);

  // Enough grounded cells from at least two papers (the server re-checks).
  const papersWithEvidence = useMemo(() => {
    const ids = new Set<string>();
    for (const c of matrix.cells) {
      if (
        c.value_text &&
        !c.insufficient_evidence &&
        !c.error &&
        c.quote_verified !== false &&
        c.support_status !== "unsupported" &&
        c.citations.some((x) => x.evidence_id)
      ) {
        ids.add(c.document_id);
      }
    }
    return ids.size;
  }, [matrix.cells]);
  const canDraft = papersWithEvidence >= MIN_PAPERS_WITH_EVIDENCE;

  const handleGenerate = async () => {
    setGenerating(true);
    setError(null);
    try {
      adopt(await api.createDraft(matrixId));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not generate a draft.");
    } finally {
      setGenerating(false);
    }
  };

  const handleSave = async () => {
    if (!draft) return;
    setSaving(true);
    setError(null);
    try {
      const updated = await api.saveDraft(draft.id, text);
      setDraft(updated);
      setSaved(true);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not save the draft.");
    } finally {
      setSaving(false);
    }
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setError("Could not copy to the clipboard.");
    }
  };

  const handleVerify = async () => {
    if (!draft) return;
    setVerifying(true);
    setError(null);
    try {
      const result = await api.verifyDraft(draft.id);
      setVerdicts(
        new Map(result.sentences.map((s) => [`${s.paragraph}:${s.sentence}`, { status: s.status, reason: s.reason }]))
      );
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not verify the draft.");
    } finally {
      setVerifying(false);
    }
  };

  return (
    <div className="mt-4 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-xs font-medium">
          <FileText className="h-3.5 w-3.5 text-primary" />
          Related work
        </span>
        <span title={canDraft ? undefined : "Needs grounded evidence from at least two papers."}>
          <Button
            variant="outline"
            size="sm"
            className="h-7 gap-1.5 text-xs"
            onClick={handleGenerate}
            disabled={!canDraft || generating}
          >
            {generating && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            {draft ? "Regenerate draft" : "Draft related work"}
          </Button>
        </span>
      </div>

      {error && (
        <div className="flex items-start gap-1.5 text-xs text-destructive">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {draft && (
        <div className="space-y-3 rounded-lg border border-border p-3">
          <div className="space-y-2 text-xs leading-relaxed">
            {draft.content.paragraphs.map((paragraph, pi) => (
              <p key={pi}>
                {paragraph.sentences.map((s, si) => {
                  if (s.status !== "ok" || s.evidence_ids.length === 0) return null;
                  const verdict = verdicts.get(`${pi}:${si}`);
                  return (
                    <span key={si}>
                      {s.text}{" "}
                      {s.evidence_ids.map((evidenceId, ci) => (
                        <button
                          key={evidenceId}
                          onClick={() => onViewEvidence(evidenceId)}
                          className="mr-1 rounded border border-border bg-muted/50 px-1 py-0.5 font-mono text-[10px] text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary"
                          title="View evidence"
                        >
                          {s.citation_labels[ci] ?? "[source]"}
                        </button>
                      ))}
                      {verdict && (
                        <SupportBadge status={verdict.status} checked reason={verdict.reason} className="mr-1" />
                      )}
                    </span>
                  );
                })}
              </p>
            ))}
          </div>

          {draft.content.dropped_count > 0 && (
            <p className="text-[11px] text-muted-foreground">
              {draft.content.dropped_count} sentence{draft.content.dropped_count === 1 ? " was" : "s were"} removed
              because {draft.content.dropped_count === 1 ? "it was" : "they were"} not grounded.
            </p>
          )}

          {draft.content.caveat && (
            <div className="rounded-md border border-amber-200 bg-amber-50 px-2.5 py-1.5 text-xs text-amber-900">
              {draft.content.caveat}
            </div>
          )}

          <div className="space-y-1.5">
            <label className="text-[11px] font-medium text-muted-foreground" htmlFor={`draft-text-${draft.id}`}>
              Editable text (Markdown)
            </label>
            <textarea
              id={`draft-text-${draft.id}`}
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                setSaved(false);
              }}
              rows={8}
              className="w-full rounded-md border border-border bg-background p-2 text-xs leading-relaxed"
            />
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" className="h-7 gap-1.5 text-xs" onClick={handleSave} disabled={saving}>
                {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                {saved ? "Saved" : "Save"}
              </Button>
              <Button variant="outline" size="sm" className="h-7 gap-1.5 text-xs" onClick={handleCopy}>
                {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                Copy as Markdown
              </Button>
              <Button variant="outline" size="sm" className="h-7 gap-1.5 text-xs" onClick={handleVerify} disabled={verifying}>
                {verifying ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ShieldCheck className="h-3.5 w-3.5" />}
                Verify draft
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
