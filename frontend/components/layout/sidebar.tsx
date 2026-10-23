"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  FileText,
  FolderKanban,
  MessageSquare,
  Lightbulb,
  Library,
  Sparkles,
  PenLine,
  BookMarked,
  Settings,
  Telescope,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface NavItem {
  label: string;
  href: string;
  icon: React.ElementType;
  enabled: boolean;
}

const navItems: NavItem[] = [
  { label: "Dashboard", href: "/dashboard", icon: LayoutDashboard, enabled: true },
  { label: "Documents", href: "/documents", icon: FileText, enabled: true },
  { label: "Research Workspace", href: "/documents", icon: FolderKanban, enabled: true },
  { label: "Paper Chat", href: "#", icon: MessageSquare, enabled: false },
  { label: "Research Gaps", href: "#", icon: Lightbulb, enabled: false },
  { label: "Literature", href: "#", icon: Library, enabled: false },
  { label: "Research Ideas", href: "#", icon: Sparkles, enabled: false },
  { label: "Writing Studio", href: "#", icon: PenLine, enabled: false },
  { label: "References", href: "#", icon: BookMarked, enabled: false },
  { label: "Settings", href: "/settings", icon: Settings, enabled: true },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r border-border bg-background md:flex">
      <div className="flex h-16 items-center gap-2 border-b border-border px-6">
        <Telescope className="h-5 w-5 text-primary" />
        <span className="text-base font-semibold tracking-tight">ResearchPilot</span>
      </div>
      <nav className="flex-1 space-y-0.5 overflow-y-auto p-3">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = item.enabled && pathname.startsWith(item.href) && item.href !== "#";

          if (!item.enabled) {
            return (
              <div
                key={item.label}
                title="Coming in a later phase"
                className="flex cursor-not-allowed items-center gap-3 rounded-md px-3 py-2 text-sm text-muted-foreground/50"
              >
                <Icon className="h-4 w-4" />
                <span className="flex-1">{item.label}</span>
                <span className="rounded-full border border-border px-1.5 py-0.5 text-[10px] uppercase tracking-wide">
                  soon
                </span>
              </div>
            );
          }

          return (
            <Link
              key={item.label}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive
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
      <div className="border-t border-border p-3">
        <div className="rounded-lg bg-secondary/60 p-3 text-xs text-muted-foreground">
          Phase 1 foundation — AI features unlock in later phases.
        </div>
      </div>
    </aside>
  );
}
