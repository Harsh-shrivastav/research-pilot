import Link from "next/link";
import {
  ArrowRight,
  FileSearch,
  GitCompareArrows,
  Lightbulb,
  Quote,
  Telescope,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const features = [
  {
    icon: FileSearch,
    title: "Evidence-based Q&A",
    description: "Ask questions about your papers and get answers grounded in page-level citations.",
  },
  {
    icon: GitCompareArrows,
    title: "Compare papers",
    description: "Line up methodology, datasets, and results across multiple papers side by side.",
  },
  {
    icon: Lightbulb,
    title: "Find research gaps",
    description: "Surface open questions and promising directions your literature hasn't covered yet.",
  },
  {
    icon: Quote,
    title: "Citation-aware writing",
    description: "Draft and refine your paper with references and evidence managed alongside it.",
  },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-6">
        <div className="flex items-center gap-2">
          <Telescope className="h-5 w-5 text-primary" />
          <span className="text-base font-semibold tracking-tight">ResearchPilot</span>
        </div>
        <nav className="flex items-center gap-3">
          <Button variant="ghost" asChild>
            <Link href="/dashboard">Dashboard</Link>
          </Button>
          <Button asChild>
            <Link href="/documents">
              Get started <ArrowRight className="h-4 w-4" />
            </Link>
          </Button>
        </nav>
      </header>

      <main>
        <section className="mx-auto max-w-4xl px-6 pb-20 pt-16 text-center">
          {/* <div className="mx-auto mb-6 inline-flex items-center gap-2 rounded-full border border-border bg-secondary/60 px-3 py-1 text-xs font-medium text-muted-foreground">
            Phase 1 — Product foundation
          </div> */}
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
            The research workspace that thinks with you
          </h1>
          <p className="mx-auto mt-5 max-w-2xl text-lg text-muted-foreground">
            Upload papers, retrieve evidence with citations, compare methodologies, and take a
            paper from open questions to a finished draft — all in one workspace built for the
            full research lifecycle.
          </p>
          <div className="mt-8 flex items-center justify-center gap-3">
            <Button size="lg" asChild>
              <Link href="/documents">
                Upload your first paper <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <Link href="/dashboard">View dashboard</Link>
            </Button>
          </div>
        </section>

        <section className="mx-auto max-w-6xl px-6 pb-24">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {features.map((f) => (
              <Card key={f.title} className="border-border/80">
                <CardContent className="pt-6">
                  <f.icon className="h-6 w-6 text-primary" />
                  <h3 className="mt-4 font-semibold">{f.title}</h3>
                  <p className="mt-1.5 text-sm text-muted-foreground">{f.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
          <p className="mt-8 text-center text-xs text-muted-foreground">
            AI analysis, chat, and gap-finding are being built in upcoming phases — this
            foundation release focuses on document management and the workspace shell.
          </p>
        </section>
      </main>
    </div>
  );
}
