import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ResearchPilot — AI Research Workspace",
  description: "AI-Powered Research Intelligence & Writing Workspace",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans antialiased">{children}</body>
    </html>
  );
}
