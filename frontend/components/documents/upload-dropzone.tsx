"use client";

import { useCallback, useRef, useState } from "react";
import { UploadCloud, FileWarning } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

const MAX_SIZE_MB = 50;

interface UploadTask {
  id: string;
  fileName: string;
  progress: number;
  error?: string;
}

export function UploadDropzone({
  onUploadFile,
}: {
  onUploadFile: (file: File, onProgress: (pct: number) => void) => Promise<void>;
}) {
  const [dragActive, setDragActive] = useState(false);
  const [tasks, setTasks] = useState<UploadTask[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const validate = (file: File): string | null => {
    if (file.type !== "application/pdf") return "Only PDF files are supported.";
    if (file.size > MAX_SIZE_MB * 1024 * 1024) return `File exceeds the ${MAX_SIZE_MB}MB limit.`;
    return null;
  };

  const handleFiles = useCallback((fileList: FileList | null) => {
    if (!fileList) return;
    Array.from(fileList).forEach((file) => {
      const taskId = `${file.name}-${Date.now()}-${Math.random()}`;
      const validationError = validate(file);

      setTasks((prev) => [...prev, { id: taskId, fileName: file.name, progress: 0, error: validationError || undefined }]);

      if (validationError) return;

      onUploadFile(file, (pct) => {
        setTasks((prev) => prev.map((t) => (t.id === taskId ? { ...t, progress: pct } : t)));
      })
        .then(() => {
          setTimeout(() => {
            setTasks((prev) => prev.filter((t) => t.id !== taskId));
          }, 1200);
        })
        .catch((e: Error) => {
          setTasks((prev) => prev.map((t) => (t.id === taskId ? { ...t, error: e.message } : t)));
        });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div>
      <div
        onDragOver={(e: React.DragEvent<HTMLDivElement>) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e: React.DragEvent<HTMLDivElement>) => {
          e.preventDefault();
          setDragActive(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 text-center transition-colors",
          dragActive ? "border-primary bg-secondary/60" : "border-border hover:bg-secondary/30"
        )}
      >
        <UploadCloud className="mb-3 h-8 w-8 text-muted-foreground" />
        <p className="text-sm font-medium">Drag and drop a PDF here, or click to browse</p>
        <p className="mt-1 text-xs text-muted-foreground">PDF only, up to {MAX_SIZE_MB}MB</p>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          multiple
          className="hidden"
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleFiles(e.target.files)}
        />
      </div>

      {tasks.length > 0 && (
        <div className="mt-4 space-y-3">
          {tasks.map((task) => (
            <div key={task.id} className="rounded-md border border-border p-3">
              <div className="flex items-center justify-between text-sm">
                <span className="truncate font-medium">{task.fileName}</span>
                {task.error ? (
                  <span className="flex items-center gap-1 text-xs text-destructive">
                    <FileWarning className="h-3.5 w-3.5" /> {task.error}
                  </span>
                ) : (
                  <span className="text-xs text-muted-foreground">{task.progress}%</span>
                )}
              </div>
              {!task.error && <Progress value={task.progress} className="mt-2" />}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
