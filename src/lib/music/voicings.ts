// Chord tones, guitar fingerings and piano voicings for the chord helper.
import { noteIndex, parseChord } from "./chords";

/** Semitones above the root for a chord quality ("m7", "sus4", "maj9", "7#9", "dim7", "6/9"…). */
export function chordIntervals(quality: string): number[] {
  let q = quality.replace(/[()]/g, "").replace(/♯/g, "#").replace(/♭/g, "b").replace(/\s+/g, "");
  q = q.replace(/^(min|-)(?!\d*b5)/, "m").replace(/^(M|Δ|\^)(?=\d|$)/, "maj").replace(/^°/, "dim").replace(/^o(?=7|$)/, "dim");
  const has = (re: RegExp) => re.test(q);
  const tones = new Set<number>();

  // Triad
  if (has(/^(ø|m7b5)/)) [0, 3, 6, 10].forEach((t) => tones.add(t));
  else if (has(/^dim/)) [0, 3, 6].forEach((t) => tones.add(t));
  else if (has(/^(aug|\+)/)) [0, 4, 8].forEach((t) => tones.add(t));
  else if (has(/^m(?!aj)/)) [0, 3, 7].forEach((t) => tones.add(t));
  else if (q === "5") [0, 7].forEach((t) => tones.add(t));
  else [0, 4, 7].forEach((t) => tones.add(t));

  if (has(/sus2/)) { tones.delete(3); tones.delete(4); tones.add(2); }
  else if (has(/sus(4)?(?!2)/)) { tones.delete(3); tones.delete(4); tones.add(5); }

  // Sevenths and extensions (9/11/13 imply the 7th unless "add")
  const maj = has(/maj/);
  const ext = /(?:^|[^b#d\d])(13|11|9|7)/.exec(q.replace(/add\d+/g, "").replace(/6\/?9/, ""))?.[1];
  if (has(/^dim7|^dim.*7/) && !has(/^(ø|m7b5)/)) tones.add(9);
  else if (ext) {
    tones.add(maj ? 11 : 10);
    if (ext === "9" || ext === "11" || ext === "13") tones.add(14);
    if (ext === "11" || ext === "13") tones.add(17);
    if (ext === "13") tones.add(21);
  }
  if (has(/6\/?9/)) { tones.add(9); tones.add(14); }
  else if (has(/(?:^|m|maj)6(?!\d)/) || q === "6") tones.add(9);
  if (has(/add(9|2)/)) tones.add(14);
  if (has(/add(11|4)/)) tones.add(17);

  // Alterations
  if (has(/b5/) && !has(/^(ø|m7b5)/)) { tones.delete(7); tones.add(6); }
  if (has(/#5/)) { tones.delete(7); tones.add(8); }
  if (has(/b9/)) { tones.delete(14); tones.add(13); }
  if (has(/#9/)) { tones.delete(14); tones.add(15); }
  if (has(/#11/)) { tones.delete(17); tones.add(18); }
  if (has(/b13/)) { tones.delete(21); tones.add(20); }
  return [...tones].sort((a, b) => a - b);
}

export interface ChordNotes {
  root: number;
  /** Pitch class that should sound lowest (slash bass, or the root). */
  bass: number;
  intervals: number[];
  /** Pitch classes in the chord, bass included. */
  pcs: number[];
}

export function chordNotes(name: string): ChordNotes | null {
  const c = parseChord(name);
  if (!c) return null;
  const root = noteIndex(c.root);
  if (root < 0) return null;
  const intervals = chordIntervals(c.quality);
  const bass = c.bass ? noteIndex(c.bass) : root;
  const pcs = [...new Set([bass, ...intervals.map((i) => (root + i) % 12)])];
  return { root, bass: bass < 0 ? root : bass, intervals, pcs };
}

// ---------------------------------------------------------------- guitar

/** Standard tuning, low E to high E, as pitch classes. */
const TUNING = [4, 9, 2, 7, 11, 4];

/** Frets low E → high E; -1 = muted, 0 = open. */
export type Fingering = number[];

function fingerCount(f: Fingering): { fingers: number; barre: number | null } {
  const fretted = f.filter((x) => x > 0);
  if (!fretted.length) return { fingers: 0, barre: null };
  const min = Math.min(...fretted);
  const at = f.map((x, i) => (x === min ? i : -1)).filter((i) => i >= 0);
  // One finger can lie across every string from the first to the last at the lowest fret
  // if nothing in between is open or muted.
  const canBarre = at.length > 1 && f.slice(at[0], at[at.length - 1] + 1).every((x) => x >= min);
  if (canBarre) return { fingers: 1 + fretted.filter((x) => x > min).length, barre: min };
  return { fingers: fretted.length, barre: null };
}

/**
 * Playable fingerings for a chord, best first: open shapes, then barre and higher shapes.
 * Searches every 4-fret window, keeps strings that sound contiguous, the right bass note,
 * all the chord's important tones (the 5th may be left out) and at most four fingers.
 */
export function guitarFingerings(name: string, max = 3): Fingering[] {
  const n = chordNotes(name);
  if (!n) return [];
  const pcs = new Set(n.pcs);
  const fifth = (n.root + 7) % 12;
  const isThirteen = n.intervals.includes(21);
  const optional = new Set<number>();
  if (n.pcs.length >= 4) optional.add(fifth);
  if (isThirteen) { optional.add((n.root + 2) % 12); optional.add((n.root + 5) % 12); }
  if (n.pcs.length >= 5 && !isThirteen) optional.add((n.root + 5) % 12); // 11th chords: drop the 3rd-clashing 11 last
  const required = n.pcs.filter((p) => !optional.has(p));

  const found: { f: Fingering; score: number; pos: number }[] = [];
  for (let pos = 0; pos <= 10; pos++) {
    const lo = Math.max(1, pos), hi = Math.max(4, pos + 3);
    const options = TUNING.map((open) => {
      const o: number[] = [-1];
      if (pos <= 3 && pcs.has(open)) o.push(0);
      for (let fr = lo; fr <= hi; fr++) if (pcs.has((open + fr) % 12)) o.push(fr);
      return o;
    });
    const cur: number[] = [];
    const walk = (s: number) => {
      if (s === 6) { consider(cur.slice(), pos); return; }
      for (const fr of options[s]) { cur.push(fr); walk(s + 1); cur.pop(); }
    };
    const consider = (f: Fingering, p: number) => {
      const sounding = f.map((x, i) => (x >= 0 ? i : -1)).filter((i) => i >= 0);
      if (sounding.length < (n.pcs.length <= 3 ? 4 : 4)) return;
      // Strings that sound must be next to each other (no muted string in the middle)
      if (sounding[sounding.length - 1] - sounding[0] + 1 !== sounding.length) return;
      const notes = sounding.map((i) => (TUNING[i] + f[i]) % 12);
      if (notes[0] !== n.bass) return;
      if (!required.every((r) => notes.includes(r))) return;
      const fretted = f.filter((x) => x > 0);
      const span = fretted.length ? Math.max(...fretted) - Math.min(...fretted) : 0;
      if (span > 3) return;
      const { fingers, barre } = fingerCount(f);
      if (fingers > 4) return;
      // A position-0 shape must actually use an open string; otherwise it's the same as position 1
      const usesOpen = f.includes(0);
      if (p === 0 && !usesOpen) return;
      if (p > 0 && usesOpen) return;
      const minFret = fretted.length ? Math.min(...fretted) : 0;
      const muted = 6 - sounding.length;
      const topMuted = 5 - sounding[sounding.length - 1];
      // Awkward open shapes: an open string right above a fretted bass note (F as 10321x), or open
      // strings mixed with notes high up the neck (Bm as x20402) — the barre shape is the usual one
      const lowest = sounding[0];
      const openAboveBass = f[lowest] > 0 && f[lowest + 1] === 0 ? 2.5 : 0;
      const openWithHigh = usesOpen && fretted.some((x) => x >= 4) ? 2.5 : 0;
      const score =
        (usesOpen ? 0 : 2 + minFret * 0.4) + // open chords first, then lower barres
        openAboveBass + openWithHigh +
        muted * 1.4 + topMuted * 2 + fingers * 0.7 + span * 0.5 + (barre ? 1 : 0) +
        (notes.filter((x) => x === n.bass).length > 2 ? 0.3 : 0);
      found.push({ f, score, pos: usesOpen ? 0 : minFret });
    };
    walk(0);
  }
  found.sort((a, b) => a.score - b.score);
  // Different positions on the neck for the alternatives
  const out: Fingering[] = [];
  const usedPos: number[] = [];
  for (const c of found) {
    if (usedPos.some((p) => Math.abs(p - c.pos) < 2)) continue;
    out.push(c.f);
    usedPos.push(c.pos);
    if (out.length >= max) break;
  }
  return out;
}

export { fingerCount };

// ---------------------------------------------------------------- piano

/**
 * Keys to press, as semitones from the C below middle C (0 = C3): a close voicing
 * starting at the root, with a slash bass an octave below.
 */
export function pianoKeys(name: string): { keys: number[]; bass: number | null } | null {
  const n = chordNotes(name);
  if (!n) return null;
  const base = n.root; // root in the first octave (C3–B3)
  const keys = n.intervals.map((i) => base + (i > 12 ? i - 12 : i)).filter((k, i, a) => a.indexOf(k) === i).sort((a, b) => a - b);
  const bass = n.bass !== n.root ? n.bass - 12 : null; // C2 octave
  return { keys, bass };
}
