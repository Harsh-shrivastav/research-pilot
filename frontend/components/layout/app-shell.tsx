import { MobileNav, Sidebar } from "./sidebar";

/**
 * The fixed viewport-height frame every page renders in: sidebar (desktop) or
 * a nav strip (mobile) + a column holding the page's Topbar and body. The
 * frame itself never scrolls - the page body does (see `PageMain`).
 * `h-dvh` tracks the visible height on mobile browsers, where `h-screen`
 * (100vh) includes the collapsing address bar and pushes content off screen.
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-dvh overflow-hidden">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <MobileNav />
        {children}
      </div>
    </div>
  );
}
