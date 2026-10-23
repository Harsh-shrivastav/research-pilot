import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Shared page-layout primitives. Every page sits inside `AppShell` (a fixed
 * viewport-height frame) under a `Topbar`; these decide how the rest scrolls,
 * so no page needs its own height / overflow arithmetic.
 *
 *   PageMain  - the scrolling body of a page (padding scales with the screen)
 *   SplitView - two columns on desktop, one stacked column below `lg`
 *               (give its parent `lg:flex lg:flex-col` so it fills the space left under any banner)
 *   SplitPrimary / SplitSecondary - the two columns
 *
 * Rule of thumb: below `lg` the PAGE scrolls and panels get a sensible height;
 * from `lg` up the page is a fixed two-column frame and each column manages
 * its own scrolling. Panels that scroll on purpose (chat history, comparison
 * table) do so inside a bounded region; nothing else clips.
 */

export function PageMain({ className, ...props }: React.HTMLAttributes<HTMLElement>) {
  return <main className={cn("min-h-0 flex-1 overflow-y-auto p-4 sm:p-6", className)} {...props} />;
}

export function SplitView({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "grid grid-cols-1 gap-6 lg:min-h-0 lg:flex-1 lg:grid-cols-[minmax(0,1fr)_minmax(400px,480px)]",
        className
      )}
      {...props}
    />
  );
}

/** Left column: grows naturally when stacked, scrolls on its own on desktop. */
export function SplitPrimary({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("flex min-w-0 flex-col gap-3 lg:min-h-0 lg:overflow-y-auto", className)} {...props} />;
}

/**
 * Right column (tabbed panels). Stacked, it gets a viewport-relative height so
 * chat-style panels have a bounded area to scroll inside; on desktop it simply
 * fills the row.
 */
export function SplitSecondary({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("flex min-w-0 flex-col h-[min(85dvh,760px)] min-h-[520px] lg:h-auto lg:min-h-0", className)}
      {...props}
    />
  );
}
