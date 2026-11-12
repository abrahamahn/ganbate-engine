interface HudOverlayProps {
  readonly status: string;
  readonly adapter: string;
  readonly helpOpen: boolean;
  readonly showLauncherToggle: boolean;
  readonly onToggleHelp: () => void;
  readonly onOpenLauncher: () => void;
}

export function HudOverlay({
  status,
  adapter,
  helpOpen,
  showLauncherToggle,
  onToggleHelp,
  onOpenLauncher,
}: HudOverlayProps) {
  return (
    <div id="hud" className={`panel${helpOpen ? ' open' : ''}`}>
      <div className="hud-header">
        <div className="hud-title">
          <b>Ganbate</b>
          <span id="status">{status}</span>
        </div>
        <div className="hud-actions">
          {showLauncherToggle && (
            <button
              type="button"
              className="small"
              onClick={onOpenLauncher}
              title="Open scene or GPU self-test launcher"
            >
              Open…
            </button>
          )}
          <button
            type="button"
            className="small"
            id="help-toggle"
            aria-expanded={helpOpen}
            onClick={onToggleHelp}
            title="Toggle controls help (H)"
          >
            ?
          </button>
        </div>
      </div>
      {adapter && <div id="adapter">{adapter}</div>}
      {helpOpen && (
        <div id="help">
          W A S D fly · Q / E down / up · arrows or drag to look · Shift faster · 1–9 views · T / G
          time · C day cycle · Y weather · P pause motion.
          <br />
          Touch: drag with one finger to look, with two to move; the buttons below change time,
          weather and views.
        </div>
      )}
    </div>
  );
}
