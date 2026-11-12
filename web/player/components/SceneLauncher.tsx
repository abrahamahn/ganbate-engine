import { useRef, useState, type DragEvent, type FormEvent } from 'react';
import type { GanbateApi } from '../../sdk/src/index.ts';
import type { LaunchRequest } from '../model/player-types.ts';

interface SceneLauncherProps {
  readonly initialApi: GanbateApi;
  readonly canCancel: boolean;
  readonly onLaunch: (request: LaunchRequest) => void;
  readonly onCancel: () => void;
}

export function SceneLauncher({
  initialApi,
  canCancel,
  onLaunch,
  onCancel,
}: SceneLauncherProps) {
  const [sceneUrl, setSceneUrl] = useState('');
  const [lookUrl, setLookUrl] = useState('');
  const [api, setApi] = useState<GanbateApi>(initialApi);
  const [worldFile, setWorldFile] = useState<File | null>(null);
  const [lookFile, setLookFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const worldInputRef = useRef<HTMLInputElement>(null);
  const lookInputRef = useRef<HTMLInputElement>(null);

  const assignDroppedFiles = (files: FileList) => {
    for (const file of Array.from(files)) {
      if (file.name.endsWith('.gbew')) {
        setWorldFile(file);
      } else if (file.name.endsWith('.json')) {
        setLookFile(file);
      }
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files.length > 0) {
      assignDroppedFiles(e.dataTransfer.files);
    }
  };

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const ownedBlobUrls: string[] = [];
    let resolvedScene = sceneUrl.trim();
    let resolvedLook = lookUrl.trim();

    if (worldFile) {
      resolvedScene = URL.createObjectURL(worldFile);
      ownedBlobUrls.push(resolvedScene);
    }
    if (lookFile) {
      resolvedLook = URL.createObjectURL(lookFile);
      ownedBlobUrls.push(resolvedLook);
    }
    if (!resolvedScene) return;

    onLaunch({
      mode: 'player',
      scene: resolvedScene,
      look: resolvedLook,
      api,
      query: window.location.search,
      ownedBlobUrls,
    });
  };

  const handleSelfTest = () => {
    onLaunch({
      mode: 'selftest',
      scene: '',
      look: '',
      api,
      query: window.location.search,
      ownedBlobUrls: [],
    });
  };

  const readyToLoad = Boolean(worldFile || sceneUrl.trim());

  return (
    <div
      id="launcher"
      className={`panel${dragActive ? ' drop-active' : ''}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragActive(true);
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={handleDrop}
    >
      <div className="launcher-header">
        <h1>Ganbate Web Player</h1>
        <p>
          Open a cooked <code>.gbew</code> world with WebGPU or WebGL 2, or run the GPU backend
          self-test.
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="launcher-section">
          <div className="dropzone">
            <div>
              <strong>Drop</strong> a <code>.gbew</code> world (and optional <code>.look.json</code>
              ) here, or choose local files:
            </div>
            <div className="file-badges">
              <button
                type="button"
                className="small"
                onClick={() => worldInputRef.current?.click()}
              >
                {worldFile ? `World: ${worldFile.name}` : 'Choose .gbew…'}
              </button>
              <button
                type="button"
                className="small"
                onClick={() => lookInputRef.current?.click()}
              >
                {lookFile ? `Look: ${lookFile.name}` : 'Choose .look.json…'}
              </button>
              {(worldFile || lookFile) && (
                <button
                  type="button"
                  className="small"
                  onClick={() => {
                    setWorldFile(null);
                    setLookFile(null);
                  }}
                >
                  Reset files
                </button>
              )}
            </div>
            <input
              ref={worldInputRef}
              type="file"
              accept=".gbew,application/octet-stream"
              hidden
              onChange={(e) => {
                const file = e.target.files?.[0] ?? null;
                if (file) setWorldFile(file);
              }}
            />
            <input
              ref={lookInputRef}
              type="file"
              accept=".json,application/json"
              hidden
              onChange={(e) => {
                const file = e.target.files?.[0] ?? null;
                if (file) setLookFile(file);
              }}
            />
          </div>

          <label className="launcher-field">
            <span>Or world URL</span>
            <input
              type="text"
              placeholder="world.gbew"
              value={sceneUrl}
              disabled={Boolean(worldFile)}
              onChange={(e) => setSceneUrl(e.target.value)}
            />
          </label>

          <label className="launcher-field">
            <span>Optional Look URL</span>
            <input
              type="text"
              placeholder="look.json"
              value={lookUrl}
              disabled={Boolean(lookFile)}
              onChange={(e) => setLookUrl(e.target.value)}
            />
          </label>

          <label className="launcher-field">
            <span>Graphics Backend</span>
            <select value={api} onChange={(e) => setApi(e.target.value as GanbateApi)}>
              <option value="auto">Auto (WebGPU with WebGL 2 fallback)</option>
              <option value="webgpu">WebGPU only</option>
              <option value="webgl2">WebGL 2 only</option>
            </select>
          </label>
        </div>

        <div className="launcher-row">
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <button type="submit" className="primary" disabled={!readyToLoad}>
              Load Scene
            </button>
            <button type="button" onClick={handleSelfTest}>
              Run GPU Self-Test
            </button>
          </div>
          {canCancel && (
            <button type="button" onClick={onCancel}>
              Cancel
            </button>
          )}
        </div>
      </form>

      <div className="launcher-hint">
        URL query support: <code>?scene=world.gbew&amp;look=look.json&amp;api=auto</code> ·{' '}
        <code>?selftest=1</code>
      </div>
    </div>
  );
}
