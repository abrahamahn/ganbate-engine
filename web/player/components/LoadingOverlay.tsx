interface LoadingOverlayProps {
  readonly phase: string;
  readonly progress: number;
}

export function LoadingOverlay({ phase, progress }: LoadingOverlayProps) {
  if (!phase) return null;
  const pct = Math.round(Math.max(0, Math.min(1, progress)) * 100);
  return (
    <div id="loading" className="panel">
      <div id="phase">{phase}</div>
      <div className="bar">
        <div id="progress" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
