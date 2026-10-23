"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  FileText,
  FolderKanban,
  MessageSquare,
  Columns3,
  Lightbulb,
  Settings,
  Telescope,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface NavItem {
  label: string;
  href: string;
  icon: React.ElementType;
}

interface NavGroup {
  title: string;
  items: NavItem[];
}

// Only features that exist. The research tools open the most recent workspace
// on the matching tab (see app/workspace/page.tsx).
const navGroups: NavGroup[] = [
  {
    title: "Overview",
    items: [
      { label: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
      { label: "Documents", href: "/documents", icon: FileText },
      { label: "Research Workspace", href: "/workspace", icon: FolderKanban },
    ],
  },
  {
    title: "Research tools",
    items: [
      { label: "Paper Chat", href: "/workspace?tab=chat", icon: MessageSquare },
      { label: "Compare Papers", href: "/workspace?tab=compare", icon: Columns3 },
      { label: "Research Gaps", href: "/workspace?tab=gaps", icon: Lightbulb },
    ],
  },
  { title: "Account", items: [{ label: "Settings", href: "/settings", icon: Settings }] },
];

const navItems: NavItem[] = navGroups.flatMap((g) => g.items);

function isActive(pathname: string, item: NavItem): boolean {
  // Tool links share the /workspace path with the workspace page itself, so
  // only the plain "Research Workspace" entry is highlighted there.
  return !item.href.includes("?") && pathname.startsWith(item.href);
}

/** Below `md` the sidebar is hidden; this strip keeps navigation reachable. */
export function MobileNav() {
  const pathname = usePathname();

  return (
    <nav className="flex shrink-0 items-center gap-1 overflow-x-auto border-b border-border bg-background px-3 py-2 md:hidden">
      <Telescope className="mr-1 h-5 w-5 shrink-0 text-primary" aria-hidden />
      {navItems.map((item) => {
        const Icon = item.icon;
        return (
          <Link
            key={item.label}
            href={item.href}
            className={cn(
              "flex shrink-0 items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
              isActive(pathname, item)
                ? "bg-secondary text-foreground"
                : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
            )}
          >
            <Icon className="h-4 w-4" />
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r border-border bg-background md:flex">
      <div className="flex h-16 items-center gap-2.5 border-b border-border px-6">
        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <Telescope className="h-4 w-4" />
        </span>
        <span className="text-base font-semibold tracking-tight">ResearchPilot</span>
      </div>
      <nav className="flex-1 space-y-5 overflow-y-auto p-3">
        {navGroups.map((group) => (
          <div key={group.title} className="space-y-0.5">
            <p className="px-3 pb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/70">
              {group.title}
            </p>
            {group.items.map((item) => {
              const Icon = item.icon;
              const active = isActive(pathname, item);
              return (
                <Link
                  key={item.label}
                  href={item.href}
                  className={cn(
                    "relative flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                    active
                      ? "bg-secondary text-foreground"
                      : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
                  )}
                >
                  {active && <span className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-primary" />}
                  <Icon className={cn("h-4 w-4", active && "text-primary")} />
                  {item.label}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>
      <div className="border-t border-border p-3">
        <div className="rounded-lg bg-secondary/60 p-3 text-xs leading-relaxed text-muted-foreground">
          Chat with papers, compare them side by side, and verify every citation against the source.
        </div>
      </div>
    </aside>
  );
}
