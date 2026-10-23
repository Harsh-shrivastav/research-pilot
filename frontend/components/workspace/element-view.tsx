import type { ElementOut } from "@/lib/types";

/** Renders a markdown pipe-table (what the parser emits for tables). */
export function MarkdownTable({ content }: { content: string }) {
  const rows = content
    .trim()
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("|"))
    .map((line) =>
      line
        .slice(1, line.endsWith("|") ? -1 : undefined)
        .split("|")
        .map((cell) => cell.trim())
    );

  // Row 1 is the markdown separator (---|---), not data.
  const body = rows.filter(
    (row) => !row.every((cell) => /^:?-{2,}:?$/.test(cell) || cell === "")
  );

  if (body.length === 0) {
    return <pre className="whitespace-pre-wrap text-xs">{content}</pre>;
  }

  const [header, ...dataRows] = body;

  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="w-full border-collapse text-xs">
        <thead className="bg-muted">
          <tr>
            {header.map((cell, i) => (
              <th key={i} className="border-b border-border px-2 py-1.5 text-left font-semibold">
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {dataRows.map((row, r) => (
            <tr key={r} className="even:bg-muted/40">
              {row.map((cell, c) => (
                <td key={c} className="border-b border-border px-2 py-1.5">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const TYPE_LABEL: Record<string, string> = {
  heading: "Heading",
  paragraph: "Paragraph",
  table: "Table",
  figure: "Figure",
  caption: "Caption",
  list_item: "List item",
  formula: "Formula",
  reference: "Reference",
};

/** One extracted element, rendered according to its type, with provenance. */
export function ElementView({
  element,
  onJumpToPage,
}: {
  element: ElementOut;
  onJumpToPage?: (page: number) => void;
}) {
  const meta = element.metadata || {};
  const label = (meta.label as string) || TYPE_LABEL[element.element_type] || element.element_type;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-muted-foreground">
        <span className="rounded bg-secondary px-1.5 py-0.5 font-medium">
          {TYPE_LABEL[element.element_type] || element.element_type}
        </span>
        <button
          type="button"
          className="hover:text-foreground hover:underline"
          onClick={() => onJumpToPage?.(element.page_number)}
        >
          page {element.page_number}
        </button>
        {element.bbox && (
          <span
            className="font-mono text-[10px] normal-case"
            title="Bounding box in PDF points (top-left origin)"
          >
            [{element.bbox.x0.toFixed(0)}, {element.bbox.y0.toFixed(0)},{" "}
            {element.bbox.x1.toFixed(0)}, {element.bbox.y1.toFixed(0)}]
          </span>
        )}
      </div>

      {element.element_type === "table" ? (
        <MarkdownTable content={element.content} />
      ) : element.element_type === "figure" ? (
        <div className="rounded-md border border-dashed border-border px-3 py-2 text-xs text-muted-foreground">
          {label}
          {typeof meta.width === "number" && typeof meta.height === "number" && (
            <span className="ml-1 font-mono">
              ({Math.round(meta.width as number)}×{Math.round(meta.height as number)} pt)
            </span>
          )}
        </div>
      ) : element.element_type === "heading" ? (
        <p className="text-sm font-semibold">{element.content}</p>
      ) : element.element_type === "formula" ? (
        <pre className="overflow-x-auto rounded bg-muted px-2 py-1 font-mono text-xs">
          {element.content}
        </pre>
      ) : (
        <p className="whitespace-pre-wrap text-sm leading-relaxed">{element.content}</p>
      )}
    </div>
  );
}
