import { useEffect, useMemo, useState } from "react";
import { fingerCount, guitarFingerings, pianoKeys, type Fingering } from "../lib/music/voicings";
import type { Song } from "../lib/db";
import { keyPrefersFlats, parseChord } from "../lib/music/chords";

export type HelperInstrument = "guitar" | "piano";

export function defaultHelperInstrument(song: Song): HelperInstrument {
  return song.instrument === "piano" ? "piano" : "guitar";
}

/**
 * Diagrams for every chord in the song, in the order they first appear. Guitar shows the shapes
 * you play (capo applied); piano shows the sounding chord. Tap a guitar diagram for another voicing.
 */
export function ChordHelper({ song, guitarChords, pianoChords, onClose }: {
  song: Song;
  /** Chord names as played on guitar (after transpose and capo). */
  guitarChords: string[];
  /** Chord names as they sound (after transpose only). */
  pianoChords: string[];
  onClose: () => void;
}) {
  const [inst, setInst] = useState<HelperInstrument>(() => defaultHelperInstrument(song));
  useEffect(() => setInst(defaultHelperInstrument(song)), [song.id, song.instrument]); // eslint-disable-line react-hooks/exhaustive-deps
  const chords = inst === "guitar" ? guitarChords : pianoChords;

  return (
    <aside className="chord-helper no-print" data-no-swipe aria-label="Chord diagrams">
      <div className="chord-helper-head">
        <div className="seg" role="group" aria-label="Instrument">
          <button className={inst === "guitar" ? "on" : ""} onClick={() => setInst("guitar")}>Guitar</button>
          <button className={inst === "piano" ? "on" : ""} onClick={() => setInst("piano")}>Piano</button>
        </div>
        <button className="btn small ghost icon" onClick={onClose} aria-label="Hide chord diagrams">✕</button>
      </div>
      {chords.length === 0 ? (
        <div className="small dim" style={{ padding: 8 }}>No chords in this chart.</div>
      ) : (
        <div className="chord-grid">
          {chords.map((c) => (inst === "guitar" ? <GuitarCard key={c} name={c} /> : <PianoCard key={c} name={c} />))}
        </div>
      )}
    </aside>
  );
}

function GuitarCard({ name }: { name: string }) {
  const options = useMemo(() => guitarFingerings(name), [name]);
  const [i, setI] = useState(0);
  const f = options[i % Math.max(1, options.length)];
  return (
    <button
      className="chord-card"
      onClick={() => options.length > 1 && setI(i + 1)}
      title={options.length > 1 ? "Tap for another way to play it" : undefined}
    >
      <div className="chord-name">{name}</div>
      {f ? <GuitarDiagram f={f} /> : <div className="small dim" style={{ height: 92, display: "grid", placeItems: "center" }}>?</div>}
      {options.length > 1 && <div className="chord-alt">{(i % options.length) + 1}/{options.length}</div>}
    </button>
  );
}

const W = 78, H = 92, LEFT = 10, TOP = 16, STRING_GAP = 11, FRET_GAP = 14, FRETS = 5;

export function GuitarDiagram({ f }: { f: Fingering }) {
  const fretted = f.filter((x) => x > 0);
  const maxF = fretted.length ? Math.max(...fretted) : 0;
  const minF = fretted.length ? Math.min(...fretted) : 0;
  const start = maxF <= FRETS ? 1 : minF; // first fret shown
  // Draw a barre only where players use one: across the bass note (F, Bm) or when there
  // aren't enough fingers otherwise — not for A (x02220) or D (xx0232)
  const { barre: possible } = fingerCount(f);
  const lowestSounding = f.findIndex((v) => v >= 0);
  const barre = possible && (f[lowestSounding] === possible || fretted.length > 4) ? possible : null;
  const x = (s: number) => LEFT + s * STRING_GAP;
  const y = (fret: number) => TOP + (fret - start + 0.5) * FRET_GAP;
  const barreStrings = barre ? f.map((v, s) => (v === barre ? s : -1)).filter((s) => s >= 0) : [];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="guitar-diagram" aria-hidden>
      {/* nut or starting fret */}
      {start === 1
        ? <rect x={x(0)} y={TOP - 3} width={x(5) - x(0)} height={3} className="dg-nut" />
        : <text x={x(5) + 4} y={TOP + FRET_GAP * 0.65} className="dg-fretno">{start}</text>}
      {Array.from({ length: FRETS + 1 }, (_, k) => (
        <line key={`f${k}`} x1={x(0)} x2={x(5)} y1={TOP + k * FRET_GAP} y2={TOP + k * FRET_GAP} className="dg-line" />
      ))}
      {Array.from({ length: 6 }, (_, s) => (
        <line key={`s${s}`} x1={x(s)} x2={x(s)} y1={TOP} y2={TOP + FRETS * FRET_GAP} className="dg-line" />
      ))}
      {f.map((v, s) => v <= 0 && (
        v === 0
          ? <circle key={`o${s}`} cx={x(s)} cy={TOP - 8} r={2.8} className="dg-open" />
          : <text key={`m${s}`} x={x(s)} y={TOP - 5} className="dg-mute">×</text>
      ))}
      {barreStrings.length > 1 && (
        <rect x={x(barreStrings[0]) - 4} y={y(barre!) - 4} width={x(barreStrings[barreStrings.length - 1]) - x(barreStrings[0]) + 8} height={8} rx={4} className="dg-dot" />
      )}
      {f.map((v, s) => v > 0 && !(barreStrings.length > 1 && v === barre) && (
        <circle key={`d${s}`} cx={x(s)} cy={y(v)} r={4.2} className="dg-dot" />
      ))}
    </svg>
  );
}

const SHARP_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
const FLAT_NAMES = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"];
/** Spell notes the way the chord's own key would (E → G♯, Bb → D, F → A C). */
function noteNamesFor(chord: string) {
  const c = parseChord(chord);
  const key = c ? c.root + (/^m(?!aj)/.test(c.quality) ? "m" : "") : "C";
  return keyPrefersFlats(key) || /^[A-G]b/.test(chord) ? FLAT_NAMES : SHARP_NAMES;
}
const BLACK = new Set([1, 3, 6, 8, 10]);

function PianoCard({ name }: { name: string }) {
  const v = useMemo(() => pianoKeys(name), [name]);
  return (
    <div className="chord-card piano">
      <div className="chord-name">{name}</div>
      {v ? <PianoDiagram keys={v.keys} bass={v.bass} /> : <div className="small dim">?</div>}
      {v && (
        <div className="chord-alt">
          {(v.bass !== null ? [v.bass, ...v.keys] : v.keys).map((k) => noteNamesFor(name)[((k % 12) + 12) % 12]).join(" ")}
        </div>
      )}
    </div>
  );
}

/** Keyboard from C (two octaves, or three with a slash bass) with the chord's keys lit. */
export function PianoDiagram({ keys, bass }: { keys: number[]; bass: number | null }) {
  const from = bass !== null ? -12 : 0;
  const to = 23;
  const whites: number[] = [];
  for (let k = from; k <= to; k++) if (!BLACK.has(((k % 12) + 12) % 12)) whites.push(k);
  const ww = 9, wh = 40, bw = 6, bh = 25;
  const width = whites.length * ww;
  const xOfWhite = (k: number) => whites.indexOf(k) * ww;
  const lit = new Set(keys);
  return (
    <svg viewBox={`0 0 ${width + 1} ${wh + 1}`} width={Math.min(200, width + 1)} className="piano-diagram" aria-hidden>
      {whites.map((k) => (
        <rect key={k} x={xOfWhite(k) + 0.5} y={0.5} width={ww} height={wh} rx={1.5}
          className={k === bass ? "pk-white bass" : lit.has(k) ? "pk-white on" : "pk-white"} />
      ))}
      {Array.from({ length: to - from + 1 }, (_, i) => from + i).filter((k) => BLACK.has(((k % 12) + 12) % 12)).map((k) => (
        <rect key={k} x={xOfWhite(k - 1) + ww - bw / 2 + 0.5} y={0.5} width={bw} height={bh} rx={1}
          className={k === bass ? "pk-black bass" : lit.has(k) ? "pk-black on" : "pk-black"} />
      ))}
    </svg>
  );
}
