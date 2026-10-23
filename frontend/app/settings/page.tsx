"use client";

import { useState } from "react";
import { Info } from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { Topbar } from "@/components/layout/topbar";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

export default function SettingsPage() {
  const [name, setName] = useState("Dev Researcher");
  const [email, setEmail] = useState("dev@researchpilot.local");
  const [saved, setSaved] = useState(false);

  return (
    <AppShell>
      <Topbar title="Settings" subtitle="Manage your account and preferences" />

      <main className="flex-1 overflow-y-auto p-6">
        <div className="mx-auto max-w-2xl space-y-6">
          <div className="flex items-start gap-2 rounded-md border border-border bg-secondary/50 p-3 text-xs text-muted-foreground">
            <Info className="mt-0.5 h-4 w-4 shrink-0" />
            <span>
              Phase 1 has no authentication backend yet — these fields are stored locally in this
              browser tab only and reset on reload. Full account management arrives with the
              auth phase.
            </span>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Profile</CardTitle>
              <CardDescription>Basic account information.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 pt-0">
              <div className="space-y-1.5">
                <Label htmlFor="name">Name</Label>
                <Input
                  id="name"
                  value={name}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setName(e.target.value)}
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEmail(e.target.value)}
                />
              </div>
              <Separator />
              <div className="flex items-center gap-3">
                <Button
                  onClick={() => {
                    setSaved(true);
                    setTimeout(() => setSaved(false), 2000);
                  }}
                >
                  Save changes
                </Button>
                {saved && <span className="text-sm text-muted-foreground">Saved locally.</span>}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Workspace defaults</CardTitle>
              <CardDescription>Upload limits and processing preferences.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 pt-0 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Max upload size</span>
                <span className="font-medium">50 MB</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Allowed file types</span>
                <span className="font-medium">PDF</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">AI processing</span>
                <span className="font-medium text-muted-foreground">Not yet enabled</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </AppShell>
  );
}
