import { useState, type PointerEvent } from 'react';
import type { GanbatePlayerController } from '../../sdk/ganbate.js';

interface TouchControlsProps {
  readonly player: GanbatePlayerController | null;
}

interface HoldButtonProps {
  readonly label: string;
  readonly keyCode: number;
  readonly title: string;
  readonly player: GanbatePlayerController;
}

function HoldButton({ label, keyCode, title, player }: HoldButtonProps) {
  const [active, setActive] = useState(false);

  const release = () => {
    setActive((wasActive) => {
      if (wasActive) player.key(keyCode, false);
      return false;
    });
  };

  const press = (e: PointerEvent<HTMLButtonElement>) => {
    e.preventDefault();
    setActive(true);
    player.key(keyCode, true);
  };

  return (
    <button
      type="button"
      className={active ? 'on' : undefined}
      title={title}
      onPointerDown={press}
      onPointerUp={release}
      onPointerCancel={release}
      onPointerLeave={release}
    >
      {label}
    </button>
  );
}

interface TapButtonProps {
  readonly label: string;
  readonly keyCode: number;
  readonly title: string;
  readonly player: GanbatePlayerController;
}

function TapButton({ label, keyCode, title, player }: TapButtonProps) {
  return (
    <button
      type="button"
      title={title}
      onClick={() => {
        player.key(keyCode, true);
        player.key(keyCode, false);
      }}
    >
      {label}
    </button>
  );
}

const code = (ch: string): number => ch.charCodeAt(0);

export function TouchControls({ player }: TouchControlsProps) {
  if (!player) return null;
  const views = player.views.slice(0, 9);

  return (
    <div id="touch" className="panel" aria-label="Controls">
      <HoldButton label="◀ time" keyCode={code('g')} title="Earlier (G)" player={player} />
      <HoldButton label="time ▶" keyCode={code('t')} title="Later (T)" player={player} />
      <TapButton label="cycle" keyCode={code('c')} title="Day cycle (C)" player={player} />
      <TapButton label="weather" keyCode={code('y')} title="Next weather (Y)" player={player} />
      <TapButton label="pause" keyCode={code('p')} title="Pause motion (P)" player={player} />
      <div className="sep" />
      <HoldButton label="▲" keyCode={code('e')} title="Up (E)" player={player} />
      <HoldButton label="▼" keyCode={code('q')} title="Down (Q)" player={player} />
      {views.length > 0 && <div className="sep" />}
      {views.map((name, index) => {
        const num = String(index + 1);
        return (
          <TapButton
            key={`${num}-${name}`}
            label={`${num} ${name}`}
            keyCode={code(num)}
            title={`View ${num}`}
            player={player}
          />
        );
      })}
    </div>
  );
}
