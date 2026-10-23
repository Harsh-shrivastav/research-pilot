import { cn } from "@/lib/utils";

// Small dependency-free charts (SVG / CSS) for the dashboard.

// ---------------------------------------------------------------------------
// Uploads per day (bars)
// ---------------------------------------------------------------------------

export function UploadsChart({ data }: { data: { date: string; count: number }[] }) {
  const max = Math.max(1, ...data.map((d) => d.count));
  const total = data.reduce((sum, d) => sum + d.count, 0);
  return (
    <div>
      <div className="flex h-32 items-end gap-1.5" role="img" aria-label={`${total} uploads in the last 14 days`}>
        {data.map((d) => (
          <div key={d.date} className="group relative flex h-full flex-1 items-end" title={`${d.date}: ${d.count}`}>
            <div
              className={cn(
                "w-full rounded-t-sm transition-colors",
                d.count > 0 ? "bg-primary/80 group-hover:bg-primary" : "bg-muted"
              )}
              style={{ height: d.count > 0 ? `${Math.max(8, (d.count / max) * 100)}%` : "4px" }}
            />
          </div>
        ))}
      </div>
      <div className="mt-2 flex justify-between text-[10px] text-muted-foreground">
        <span>{data[0] ? shortDate(data[0].date) : ""}</span>
        <span>{total} upload{total === 1 ? "" : "s"} in 14 days</span>
        <span>{data.length ? shortDate(data[data.length - 1].date) : ""}</span>
      </div>
    </div>
  );
}

function shortDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

// ---------------------------------------------------------------------------
// Donut with legend
// ---------------------------------------------------------------------------

export interface Slice {
  label: string;
  value: number;
  color: string; // CSS color
}

export function Donut({ slices, centerLabel }: { slices: Slice[]; centerLabel: string }) {
  const total = slices.reduce((sum, s) => sum + s.value, 0);
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;
  return (
    <div className="flex items-center gap-6">
      <svg viewBox="0 0 100 100" className="h-32 w-32 shrink-0 -rotate-90" role="img" aria-label={centerLabel}>
        <circle cx="50" cy="50" r={radius} fill="none" stroke="hsl(var(--muted))" strokeWidth="12" />
        {total > 0 &&
          slices
            .filter((s) => s.value > 0)
            .map((s) => {
              const length = (s.value / total) * circumference;
              const circle = (
                <circle
                  key={s.label}
                  cx="50"
                  cy="50"
                  r={radius}
                  fill="none"
                  stroke={s.color}
                  strokeWidth="12"
                  strokeDasharray={`${length} ${circumference - length}`}
                  strokeDashoffset={-offset}
                />
              );
              offset += length;
              return circle;
            })}
        <g className="rotate-90 origin-center">
          <text x="50" y="50" textAnchor="middle" dominantBaseline="central" className="fill-foreground text-[18px] font-semibold">
            {total}
          </text>
        </g>
      </svg>
      <ul className="min-w-0 flex-1 space-y-1.5 text-xs">
        {slices.map((s) => (
          <li key={s.label} className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: s.color }} />
            <span className="flex-1 truncate text-muted-foreground">{s.label}</span>
            <span className="font-medium tabular-nums">{s.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stacked horizontal bar with legend
// ---------------------------------------------------------------------------

export function StackedBar({ slices }: { slices: Slice[] }) {
  const total = slices.reduce((sum, s) => sum + s.value, 0);
  return (
    <div className="space-y-3">
      <div className="flex h-3 w-full overflow-hidden rounded-full bg-muted">
        {total > 0 &&
          slices
            .filter((s) => s.value > 0)
            .map((s) => (
              <div
                key={s.label}
                style={{ width: `${(s.value / total) * 100}%`, backgroundColor: s.color }}
                title={`${s.label}: ${s.value}`}
              />
            ))}
      </div>
      <ul className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
        {slices.map((s) => (
          <li key={s.label} className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: s.color }} />
            <span className="flex-1 truncate text-muted-foreground">{s.label}</span>
            <span className="font-medium tabular-nums">{s.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
