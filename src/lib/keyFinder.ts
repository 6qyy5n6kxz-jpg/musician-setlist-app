// Best key for a singer: shift the song's melody range (lowest → highest sung note, in the written
// key) until it sits comfortably inside the singer's range, slightly above the middle where voices
// carry best. Guitar songs also get a capo suggestion when the best key has awkward shapes.
import { isMinorKey, keyRoot, noteIndex, normalizeKey, transposeKey } from "./music/chords";
import { midiName } from "./pitch";

export interface Range { low: number; high: number }
export type VocalRanges = Partial<Record<"kendra" | "devin", Range>>;

export interface KeyOption {
  key: string;
  /** Semitones from the written key (−6 … +6). */
  semitones: number;
  /** Melody range in this key. */
  low: number;
  high: number;
  fit: "comfortable" | "tight" | "stretch";
  /** Semitones of room left below the lowest and above the highest note (worst singer). */
  roomLow: number;
  roomHigh: number;
  /** For guitar: play these shapes with a capo (when the key itself is awkward). */
  capo?: { fret: number; shapes: string };
  score: number;
}

const GUITAR_MAJOR = new Set(["C", "D", "E", "G", "A"]);
const GUITAR_MINOR = new Set(["Am", "Em", "Dm"]);
export const guitarFriendly = (key: string) => {
  const k = normalizeKey(key);
  return isMinorKey(k) ? GUITAR_MINOR.has(k) : GUITAR_MAJOR.has(keyRoot(k));
};

/** Lowest capo position that turns `key` into open-chord shapes. */
export function capoFor(key: string): { fret: number; shapes: string } | undefined {
  if (guitarFriendly(key)) return undefined;
  for (let fret = 1; fret <= 7; fret++) {
    const shapes = transposeKey(key, -fret);
    if (guitarFriendly(shapes)) return { fret, shapes };
  }
  return undefined;
}

/**
 * Keys ranked best first for one or more singers (a duet must fit both).
 * `instrument` "acoustic"/"electric" adds guitar-friendliness; piano has no preference.
 */
export function recommendKeys(writtenKey: string, melody: Range, singers: Range[], instrument?: string | null): KeyOption[] {
  if (!singers.length || noteIndex(keyRoot(normalizeKey(writtenKey))) < 0) return [];
  const guitar = instrument === "acoustic" || instrument === "electric";
  const out: KeyOption[] = [];
  for (let t = -6; t <= 6; t++) {
    const low = melody.low + t, high = melody.high + t;
    let score = 0, roomLow = Infinity, roomHigh = Infinity;
    for (const s of singers) {
      const below = Math.max(0, s.low - low);
      const above = Math.max(0, high - s.high);
      // Notes the singer can't reach are the worst; high misses hurt a bit more than low ones
      score += below * 9 + above * 11;
      // Sit a little above the middle of the voice (where it carries), not at either edge
      const target = (s.low + s.high) / 2 + 1;
      score += Math.abs((low + high) / 2 - target) * 0.9;
      roomLow = Math.min(roomLow, low - s.low);
      roomHigh = Math.min(roomHigh, s.high - high);
    }
    // Keep at least a semitone in reserve on top — a voice tires over a night, and nerves push sharp
    if (roomHigh >= 0 && roomHigh < 1) score += 3;
    score += Math.abs(t) * 0.25; // all else equal, stay close to the original
    const key = transposeKey(writtenKey, t);
    const capo = guitar ? capoFor(key) : undefined;
    if (guitar && !guitarFriendly(key)) score += capo ? 0.6 : 1.5;
    const fit = roomLow >= 0 && roomHigh >= 1 ? "comfortable" : roomLow >= -1 && roomHigh >= -1 ? "tight" : "stretch";
    out.push({ key, semitones: t, low, high, fit, roomLow, roomHigh, capo, score: Math.round(score * 10) / 10 });
  }
  // Same key reached by +6 and −6: keep the better one
  const best = new Map<string, KeyOption>();
  for (const o of out.sort((a, b) => a.score - b.score)) if (!best.has(o.key)) best.set(o.key, o);
  return [...best.values()];
}

export function describeOption(o: KeyOption): string {
  const range = `${midiName(o.low)}–${midiName(o.high)}`;
  if (o.fit === "comfortable") return `${range} · comfortable (${o.roomHigh} semitone${o.roomHigh === 1 ? "" : "s"} to spare on top)`;
  if (o.fit === "tight") return `${range} · right at the edge of the range`;
  const miss = [o.roomLow < 0 ? `${-o.roomLow} below` : "", o.roomHigh < 0 ? `${-o.roomHigh} above` : ""].filter(Boolean).join(", ");
  return `${range} · out of range (${miss})`;
}
