import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  startPlayer,
  startSelfTest,
  type GanbatePlayerController,
} from '../sdk/src/index.ts';
import { ErrorDrawer } from './components/ErrorDrawer.tsx';
import { HudOverlay } from './components/HudOverlay.tsx';
import { LoadingOverlay } from './components/LoadingOverlay.tsx';
import { SceneLauncher } from './components/SceneLauncher.tsx';
import { TouchControls } from './components/TouchControls.tsx';
import {
  applyFixedCanvasStyle,
  createCanvasSizer,
  parsePlayerUrlConfig,
} from './model/canvas-sizing.ts';
import type { LaunchRequest } from './model/player-types.ts';

function isEditableElement(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || target.isContentEditable;
}

export function App() {
  const config = useMemo(
    () => parsePlayerUrlConfig(window.location.search, import.meta.env.DEV),
    [],
  );

  const [sessionId, setSessionId] = useState(1);
  const [activeRequest, setActiveRequest] = useState<LaunchRequest | null>(() => {
    if (config.mode === 'selftest') {
      return {
        mode: 'selftest',
        scene: '',
        look: '',
        api: config.api,
        query: config.query,
        ownedBlobUrls: [],
      };
    }
    if (config.mode === 'player') {
      return {
        mode: 'player',
        scene: config.scene,
        look: config.look,
        api: config.api,
        query: config.query,
        ownedBlobUrls: [],
      };
    }
    return null;
  });

  const [launcherOpen, setLauncherOpen] = useState(config.mode === 'launcher');
  const [helpOpen, setHelpOpen] = useState(false);
  const [status, setStatus] = useState(
    config.mode === 'launcher' ? 'ready' : 'starting…',
  );
  const [adapter, setAdapter] = useState('');
  const [phase, setPhase] = useState(config.mode === 'launcher' ? '' : 'Loading…');
  const [progress, setProgress] = useState(0);
  const [errors, setErrors] = useState<string[]>([]);
  const [player, setPlayer] = useState<GanbatePlayerController | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  const reportError = useCallback((rawText: string) => {
    const text = String(rawText);
    console.error(text);
    setErrors((prev) => {
      if (prev.includes(text) || prev.length >= 50) return prev;
      return [...prev, text];
    });
  }, []);

  useEffect(() => {
    if (config.forceTouch) {
      document.body.classList.add('touch');
    }
  }, [config.forceTouch]);

  useEffect(() => {
    const onError = (e: ErrorEvent) => {
      if (e.message) reportError(e.message);
    };
    const onUnhandledRejection = (e: PromiseRejectionEvent) => {
      const reason = e.reason as { message?: string } | undefined;
      reportError(String(reason?.message ?? e.reason));
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'KeyH' && !e.repeat && !isEditableElement(e.target)) {
        setHelpOpen((open) => !open);
      }
    };
    window.addEventListener('error', onError);
    window.addEventListener('unhandledrejection', onUnhandledRejection);
    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('error', onError);
      window.removeEventListener('unhandledrejection', onUnhandledRejection);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [reportError]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    applyFixedCanvasStyle(canvas, config.fixed);
    const sizeCanvas = createCanvasSizer(canvas, config);
    sizeCanvas();

    if (!activeRequest) return;

    let cancelled = false;
    let activeController: GanbatePlayerController | null = null;

    setPlayer(null);
    setAdapter('');
    setProgress(0);
    setStatus('starting…');
    setPhase('Loading…');

    const run = async () => {
      if (activeRequest.mode === 'selftest') {
        try {
          const result = await startSelfTest({
            canvas,
            wasm: config.wasm,
            sizeCanvas,
            api: activeRequest.api,
            onPhase(text) {
              if (!cancelled) setPhase(text);
            },
            onLog(level, text) {
              if (level >= 2) {
                reportError(text);
              } else {
                console.log(text);
              }
            },
            onError: reportError,
          });
          if (cancelled) return;
          setPhase('');
          setStatus(`self-test done, ${result.errors().length} errors`);
          setAdapter(`${result.adapter} · ${result.format}`);
          window.ganbate = { selftest: result };
        } catch (err) {
          if (cancelled) return;
          const message = err instanceof Error ? err.message : String(err);
          setPhase('');
          reportError(message);
          window.ganbate = { failed: message };
        }
        return;
      }

      try {
        const controller = await startPlayer({
          canvas,
          wasm: config.wasm,
          api: activeRequest.api,
          scene: activeRequest.scene,
          look: activeRequest.look,
          query: activeRequest.query,
          sizeCanvas,
          onPhase(text, p) {
            if (cancelled) return;
            setPhase(text);
            setProgress(p);
          },
          onLog(level, text) {
            if (level >= 2) {
              reportError(text);
            } else if (level === 1) {
              console.warn(text);
            } else {
              console.log(text);
            }
          },
          onStatus(text) {
            if (!cancelled) setStatus(text);
          },
          onError: reportError,
        });

        if (cancelled) {
          controller.stop();
          return;
        }

        activeController = controller;
        setPlayer(controller);
        setPhase('');
        setAdapter(`${controller.adapter} · ${controller.format}`);
        window.ganbate = controller;
        canvas.focus();
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : String(err);
        setPhase('');
        reportError(message);
        window.ganbate = { failed: message };
      }
    };

    void run();

    return () => {
      cancelled = true;
      if (activeController) {
        activeController.stop();
      }
      for (const blobUrl of activeRequest.ownedBlobUrls) {
        URL.revokeObjectURL(blobUrl);
      }
    };
  }, [activeRequest, config, reportError, sessionId]);

  const handleLaunch = (request: LaunchRequest) => {
    setErrors([]);
    setLauncherOpen(false);
    setSessionId((n) => n + 1);
    setActiveRequest(request);
  };

  return (
    <>
      <canvas
        key={sessionId}
        ref={canvasRef}
        id="view"
        tabIndex={0}
        aria-label="Ganbate render view"
      />
      <HudOverlay
        status={status}
        adapter={adapter}
        helpOpen={helpOpen}
        showLauncherToggle={!launcherOpen}
        onToggleHelp={() => setHelpOpen((open) => !open)}
        onOpenLauncher={() => setLauncherOpen(true)}
      />
      <LoadingOverlay phase={phase} progress={progress} />
      {launcherOpen && (
        <SceneLauncher
          initialApi={activeRequest?.api ?? config.api}
          canCancel={Boolean(activeRequest)}
          onLaunch={handleLaunch}
          onCancel={() => setLauncherOpen(false)}
        />
      )}
      <ErrorDrawer errors={errors} onDismiss={() => setErrors([])} />
      <TouchControls player={player} />
    </>
  );
}
