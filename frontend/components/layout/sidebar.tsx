"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  FileText,
  FolderKanban,
  PanelLeftClose,
  PanelLeftOpen,
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

// Chat, comparison and gap detection live inside the workspace, so they are
// reached through "Research Workspace" (a blank workspace, or pick an existing
// one from there) rather than listed here.
const navGroups: NavGroup[] = [
  {
    title: "Overview",
    items: [
      { label: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
      { label: "Documents", href: "/documents", icon: FileText },
      { label: "Research Workspace", href: "/workspace/new", icon: FolderKanban },
    ],
  },
  { title: "Account", items: [{ label: "Settings", href: "/settings", icon: Settings }] },
];

const navItems: NavItem[] = navGroups.flatMap((g) => g.items);

function isActive(pathname: string, item: NavItem): boolean {
  // Every /workspace/... page (list, blank, a specific workspace) belongs to the workspace entry.
  const base = item.href.startsWith("/workspace") ? "/workspace" : item.href;
  return pathname.startsWith(base);
}

const COLLAPSED_KEY = "researchpilot.sidebar.collapsed";

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
  const [collapsed, setCollapsed] = useState(false);

  // Remembered per browser; the first paint is expanded, then restored.
  useEffect(() => {
    try {
      setCollapsed(localStorage.getItem(COLLAPSED_KEY) === "1");
    } catch {
      // storage unavailable: stay expanded
    }
  }, []);

  const toggle = () => {
    setCollapsed((prev) => {
      try {
        localStorage.setItem(COLLAPSED_KEY, prev ? "0" : "1");
      } catch {
        // ignore
      }
      return !prev;
    });
  };

  return (
    <aside
      className={cn(
        "hidden shrink-0 flex-col border-r border-border bg-background transition-[width] duration-200 md:flex",
        collapsed ? "w-16" : "w-64"
      )}
    >
      <div className={cn("flex h-16 items-center border-b border-border", collapsed ? "justify-center" : "gap-2.5 px-6")}>
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <Telescope className="h-4 w-4" />
        </span>
        {!collapsed && <span className="flex-1 truncate text-base font-semibold tracking-tight">ResearchPilot</span>}
        {!collapsed && (
          <button
            onClick={toggle}
            aria-label="Collapse sidebar"
            title="Collapse sidebar"
            className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
          >
            <PanelLeftClose className="h-4 w-4" />
          </button>
        )}
      </div>
      <nav className="flex-1 space-y-5 overflow-y-auto overflow-x-hidden p-3">
        {navGroups.map((group) => (
          <div key={group.title} className="space-y-0.5">
            {!collapsed && (
              <p className="px-3 pb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/70">
                {group.title}
              </p>
            )}
            {group.items.map((item) => {
              const Icon = item.icon;
              const active = isActive(pathname, item);
              return (
                <Link
                  key={item.label}
                  href={item.href}
                  title={collapsed ? item.label : undefined}
                  aria-label={item.label}
                  className={cn(
                    "relative flex items-center rounded-md py-2 text-sm font-medium transition-colors",
                    collapsed ? "justify-center px-0" : "gap-3 px-3",
                    active
                      ? "bg-secondary text-foreground"
                      : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
                  )}
                >
                  {active && <span className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-primary" />}
                  <Icon className={cn("h-4 w-4 shrink-0", active && "text-primary")} />
                  {!collapsed && item.label}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>
      <div className="border-t border-border p-3">
        {collapsed ? (
          <button
            onClick={toggle}
            aria-label="Expand sidebar"
            title="Expand sidebar"
            className="flex w-full items-center justify-center rounded-md p-2 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
          >
            <PanelLeftOpen className="h-4 w-4" />
          </button>
        ) : (
          <div className="rounded-lg bg-secondary/60 p-3 text-xs leading-relaxed text-muted-foreground">
            Chat with papers, compare them side by side, and verify every citation against the source.
          </div>
        )}
      </div>
    </aside>
  );
}
