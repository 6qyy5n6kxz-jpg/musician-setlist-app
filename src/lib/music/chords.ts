// Chord parsing, transposition, capo shapes and Nashville numbers.

const SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];

/** Keys conventionally written with flats (major and minor). */
const FLAT_KEYS = new Set(["F", "Bb", "Eb", "Ab", "Db", "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm"]);

export const ALL_KEYS = [
  "C", "C#", "Db", "D", "Eb", "E", "F", "F#", "Gb", "G", "Ab", "A", "Bb", "B",
  "Cm", "C#m", "Dm", "D#m", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm",
];

export interface Chord {
  root: string;     // e.g. "F#"
  quality: string;  // everything between root and slash, e.g. "m7", "sus4", "maj7(#11)"
  bass?: string;    // slash bass note, e.g. "B" in "G/B"
}

const CHORD_RE = /^([A-G])([#b♯♭]?)([^/\s]*?)(?:\/([A-G])([#b♯♭]?))?$/;
// Qualities we accept after the root. Keeps words like "Am I" or "Do" from parsing as chords.
const QUALITY_RE = /^(?:m|min|maj|M|dim|aug|sus|add|no|alt|°|ø|\+|-|\d|\(|\)|#|b|♯|♭|\^|Δ|o)*$/;

function normAccidental(a: string): string {
  return a === "♯" ? "#" : a === "♭" ? "b" : a;
}

export function parseChord(text: string): Chord | null {
  const t = text.trim();
  const m = CHORD_RE.exec(t);
  if (!m) return null;
  const quality = m[3] ?? "";
  if (!QUALITY_RE.test(quality)) return null;
  return {
    root: m[1] + normAccidental(m[2] ?? ""),
    quality,
    bass: m[4] ? m[4] + normAccidental(m[5] ?? "") : undefined,
  };
}

/** True for tokens that may appear on a chord line besides chords (bars, repeats, N.C., etc.). */
export function isChordLineFiller(token: string): boolean {
  return /^(\||\|\||\/|\\|-+|\.+|%|x\d+|\(x?\d+x?\)|\d+x|N\.?C\.?|\(|\)|\*|:\|\|?|\|\|?:)$/i.test(token);
}

export function noteIndex(note: string): number {
  const n = note.charAt(0).toUpperCase() + note.slice(1);
  let i = SHARPS.indexOf(n);
  if (i < 0) i = FLATS.indexOf(n);
  if (i < 0) {
    // Cb, Fb, E#, B#
    const base = SHARPS.indexOf(n.charAt(0));
    if (base < 0) return -1;
    i = (base + (n.endsWith("#") ? 1 : n.endsWith("b") ? -1 : 0) + 12) % 12;
  }
  return i;
}

export function noteName(index: number, preferFlats: boolean): string {
  const i = ((index % 12) + 12) % 12;
  return (preferFlats ? FLATS : SHARPS)[i];
}

export function keyPrefersFlats(key: string | null | undefined): boolean {
  if (!key) return false;
  return FLAT_KEYS.has(normalizeKey(key));
}

/** "g" -> "G", "f#min" -> "F#m", "Bb minor" -> "Bbm". */
export function normalizeKey(key: string): string {
  const m = /^\s*([A-Ga-g])([#b♯♭]?)\s*(m|min|minor|-)?\b/i.exec(key);
  if (!m) return key.trim();
  const root = m[1].toUpperCase() + normAccidental(m[2] ?? "");
  const minor = m[3] && !/^maj/i.test(m[3]) && (m[3] === "m" || /^min/i.test(m[3]) || m[3] === "-");
  return root + (minor ? "m" : "");
}

export function isMinorKey(key: string): boolean {
  return normalizeKey(key).endsWith("m");
}

export function keyRoot(key: string): string {
  return normalizeKey(key).replace(/m$/, "");
}

export function transposeKey(key: string, semitones: number): string {
  const k = normalizeKey(key);
  const minor = k.endsWith("m");
  const idx = noteIndex(keyRoot(k));
  if (idx < 0) return key;
  // pick the conventional spelling for the resulting key
  const sharpName = noteName(idx + semitones, false) + (minor ? "m" : "");
  const flatName = noteName(idx + semitones, true) + (minor ? "m" : "");
  return FLAT_KEYS.has(flatName) ? flatName : sharpName;
}

/** Semitones to move from one key to another (shortest direction, -5..+6). */
export function keyDistance(from: string, to: string): number {
  const a = noteIndex(keyRoot(from));
  const b = noteIndex(keyRoot(to));
  if (a < 0 || b < 0) return 0;
  let d = (b - a + 12) % 12;
  if (d > 6) d -= 12;
  return d;
}

export function chordToString(c: Chord): string {
  return c.root + c.quality + (c.bass ? "/" + c.bass : "");
}

export function transposeChord(text: string, semitones: number, preferFlats: boolean): string {
  const c = parseChord(text);
  if (!c) return text;
  if (semitones === 0) return text; // keep the chart's own spelling
  const root = noteName(noteIndex(c.root) + semitones, preferFlats);
  const bass = c.bass ? noteName(noteIndex(c.bass) + semitones, preferFlats) : undefined;
  return chordToString({ root, quality: c.quality, bass });
}

const DEGREE_NAMES = ["1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"];
const MINOR_DEGREE_NAMES = ["1", "b2", "2", "3", "#3", "4", "b5", "5", "6", "#6", "7", "#7"];

/**
 * Nashville number for a chord in a key. Minor chords get "m" (6m), other qualities are kept (5sus, 1maj7).
 * In minor keys the tonic is 1 (Am in A minor -> 1m), using natural-minor degrees.
 */
export function toNashville(text: string, key: string): string {
  const c = parseChord(text);
  if (!c || !key) return text;
  const tonic = noteIndex(keyRoot(key));
  if (tonic < 0) return text;
  const names = isMinorKey(key) ? MINOR_DEGREE_NAMES : DEGREE_NAMES;
  const deg = (r: string) => names[(noteIndex(r) - tonic + 12) % 12];
  // "27sus4" would read as twenty-seven: wrap qualities that start with a digit
  const quality = /^\d/.test(c.quality) ? `(${c.quality})` : c.quality;
  return deg(c.root) + quality + (c.bass ? "/" + deg(c.bass) : "");
}

/** Guess a song's key from its chords: tonic of the first chord, weighted by the last chord. */
export function guessKey(chords: string[]): string | null {
  const parsed = chords.map(parseChord).filter((c): c is Chord => !!c);
  if (!parsed.length) return null;
  const first = parsed[0];
  const minor = /^m(?!aj)/.test(first.quality);
  return first.root + (minor ? "m" : "");
}

/** Shape (as played) for a sounding chord with a capo: capo 2 turns a sounding A into a G shape. */
export function capoShape(text: string, capo: number, preferFlats = false): string {
  if (!capo) return text;
  return transposeChord(text, -capo, preferFlats);
}
