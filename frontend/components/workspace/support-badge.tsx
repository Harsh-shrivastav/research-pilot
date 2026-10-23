import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// Citation-verification badge: does the cited text back the claim? Separate
// from "quote found" (QuoteBadge), which only says the quote is in the paper.

const SUPPORT_DISPLAY = {
  supported: { label: "Verified", variant: "success" },
  partially_supported: { label: "Weak", variant: "warning" },
  unsupported: { label: "Unsupported", variant: "destructive" },
} as const;

export function SupportBadge({
  status,
  checked,
  reason,
  className,
}: {
  status?: string | null;
  checked?: boolean;
  reason?: string | null;
  className?: string;
}) {
  const known = checked ? SUPPORT_DISPLAY[status as keyof typeof SUPPORT_DISPLAY] : undefined;
  const label = known?.label ?? "Unchecked";
  const title = reason || (checked ? "Verification was inconclusive." : "Not verified yet.");
  return (
    <Badge
      variant={known?.variant ?? "secondary"}
      className={cn("px-1.5 py-0 text-[10px]", !known && "text-muted-foreground", className)}
      title={title}
    >
      {label}
    </Badge>
  );
}

export function QuoteBadge({ verified, className }: { verified?: boolean | null; className?: string }) {
  if (verified === null || verified === undefined) return null;
  return (
    <Badge
      variant={verified ? "outline" : "destructive"}
      className={cn("px-1.5 py-0 text-[10px]", className)}
      title={verified ? "The quote was found in the paper's text." : "The quoted text was not found in the paper."}
    >
      {verified ? "Quote found" : "Quote not found"}
    </Badge>
  );
}
