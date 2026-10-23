import { Badge } from "@/components/ui/badge";
import { Loader2 } from "lucide-react";
import type { ProcessingStatus } from "@/lib/types";

const STATUS_CONFIG: Record<
  string,
  {
    label: string;
    variant: "success" | "warning" | "destructive" | "secondary";
    spinning?: boolean;
  }
> = {
  UPLOADED: { label: "Queued", variant: "warning" },
  PROCESSING: { label: "Processing", variant: "secondary", spinning: true },
  READY: { label: "Ready", variant: "success" },
  FAILED: { label: "Failed", variant: "destructive" },
};

export function StatusBadge({ status }: { status: ProcessingStatus | string }) {
  const config =
    STATUS_CONFIG[status] || { label: status, variant: "secondary" as const };
  return (
    <Badge variant={config.variant} className="gap-1">
      {config.spinning && <Loader2 className="h-3 w-3 animate-spin" />}
      {config.label}
    </Badge>
  );
}
