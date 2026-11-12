interface ErrorDrawerProps {
  readonly errors: readonly string[];
  readonly onDismiss: () => void;
}

export function ErrorDrawer({ errors, onDismiss }: ErrorDrawerProps) {
  if (errors.length === 0) return null;
  return (
    <div id="errors" className="panel" role="alert">
      <div className="errors-header">
        <h2>Errors</h2>
        <button type="button" className="small" onClick={onDismiss}>
          Clear
        </button>
      </div>
      <pre id="error-list">{errors.join('\n')}</pre>
    </div>
  );
}
