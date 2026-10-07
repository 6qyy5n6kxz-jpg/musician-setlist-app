// Reorders the songs inside each set (breaks stay put) for a better live flow:
// no back-to-back songs in the same key, slow songs spread out, a strong opener and closer,
// an energy arc that dips in the middle and builds to the end, fewer instrument switches,
// and the same singer or artist not stacked up.
import { keyRoot, noteIndex, normalizeKey, isMinorKey } from "./music/chords";

export interface OptSong {
  id: string;
  title: string;
  artist: string;
  /** Key it will be performed in (setlist override, singer key, or the song's key). */
  key: string | null;
  tempo: number | null;
  tags: string[];
  time_signature: string | null;
  instrument: string | null;
  lead_vocal: string | null;
}

export interface OptOptions {
  keepOpener: boolean;
  keepCloser: boolean;
  /** Weight given to avoiding instrument changes between songs. */
  fewerSwitches: boolean;
}

const UP_TAGS = /^(upbeat|dance|party|singalong|sing-along|rock|fast|uptempo|high energy|crowd pleaser|fun)$/i;
const DOWN_TAGS = /^(mellow|ballad|slow|worship|love|soft|quiet|acoustic ballad|chill)$/i;

/** 0 (slowest, quietest) … 1 (biggest), from tempo and tags. */
export function songEnergy(s: OptSong): number {
  const tempo = s.tempo ?? 100;
  let e = Math.max(0, Math.min(1, (tempo - 65) / 85));
  // 6/8 and 12/8 feel slower than the counted tempo
  if (/^(6|12)\/8$/.test(s.time_signature ?? "")) e *= 0.75;
  if (s.tags.some((t) => UP_TAGS.test(t))) e = Math.min(1, e + 0.2);
  if (s.tags.some((t) => DOWN_TAGS.test(t))) e = Math.max(0, e - 0.25);
  return Math.round(e * 100) / 100;
}

export const isSlow = (e: number) => e < 0.35;

/** Same key, or relative major/minor (C and Am). */
export function keyRelation(a: string | null, b: string | null): "same" | "relative" | "different" | "unknown" {
  if (!a || !b) return "unknown";
  const na = normalizeKey(a), nb = normalizeKey(b);
  const ra = noteIndex(keyRoot(na)), rb = noteIndex(keyRoot(nb));
  if (ra < 0 || rb < 0) return "unknown";
  const ma = isMinorKey(na), mb = isMinorKey(nb);
  if (ra === rb && ma === mb) return "same";
  // Relative pairs share every note: the minor tonic is 3 semitones below the major tonic
  const tonicMajor = (r: number, minor: boolean) => (minor ? (r + 3) % 12 : r);
  if (ma !== mb && tonicMajor(ra, ma) === tonicMajor(rb, mb)) return "relative";
  return "different";
}

/** Energy the crowd should feel at a point in the set (0 = start, 1 = end): open strong, breathe in the middle, finish biggest. */
export function arcTarget(t: number): number {
  return 0.6 + 0.2 * Math.cos(2 * Math.PI * t) + 0.12 * t;
}

export interface FlowReport {
  sameKey: number;
  relativeKey: number;
  slowStacked: number;
  weakOpenOrClose: number;
  switches: number;
  vocalRuns: number;
  sameArtist: number;
  /** Lower is better. */
  cost: number;
}

/** Score one set's running order (lower = better) and count the problems a person would notice. */
export function scoreSet(order: OptSong[], opts: OptOptions, energies = order.map(songEnergy)): FlowReport {
  const r: FlowReport = { sameKey: 0, relativeKey: 0, slowStacked: 0, weakOpenOrClose: 0, switches: 0, vocalRuns: 0, sameArtist: 0, cost: 0 };
  const n = order.length;
  let cost = 0;
  let vocalRun = 1;
  for (let i = 0; i < n; i++) {
    const s = order[i], e = energies[i];
    if (n > 2) cost += (e - arcTarget(n === 1 ? 0 : i / (n - 1))) ** 2 * 10;
    if (i === 0 || i === n - 1) if (n > 2 && isSlow(e)) { r.weakOpenOrClose++; cost += 14; }
    if (i === n - 1 && n > 3 && e < Math.max(...energies) - 0.25) cost += 4; // close near the top of the set's energy
    if (i === 0) continue;
    const p = order[i - 1], pe = energies[i - 1];
    const rel = keyRelation(p.key, s.key);
    if (rel === "same") { r.sameKey++; cost += 10; }
    else if (rel === "relative") { r.relativeKey++; cost += 3; }
    if (i >= 2 && keyRelation(order[i - 2].key, s.key) === "same") cost += 2;
    if (isSlow(pe) && isSlow(e)) { r.slowStacked++; cost += 12; }
    if (Math.abs(e - pe) > 0.6) cost += 2; // jarring cliff
    if (p.instrument && s.instrument && p.instrument !== s.instrument) { r.switches++; cost += opts.fewerSwitches ? 4 : 1.5; }
    if (p.artist && s.artist && p.artist.toLowerCase() === s.artist.toLowerCase()) { r.sameArtist++; cost += 6; }
    vocalRun = s.lead_vocal && s.lead_vocal === p.lead_vocal ? vocalRun + 1 : 1;
    if (vocalRun === 4) { r.vocalRuns++; cost += 3; } else if (vocalRun > 4) cost += 2;
  }
  r.cost = Math.round(cost * 10) / 10;
  return r;
}

/** Small seeded RNG so the same set gives the same answer each time. */
function rng(seed: number) {
  let s = seed >>> 0 || 1;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

/**
 * Best running order found for one set: simulated annealing over swaps and moves, a few restarts.
 * Locked opener / closer stay where they are.
 */
export function optimizeSet(songs: OptSong[], opts: OptOptions, seed = 7): OptSong[] {
  const n = songs.length;
  if (n < 3) return songs.slice();
  const energy = new Map(songs.map((s) => [s.id, songEnergy(s)]));
  const cost = (o: OptSong[]) => scoreSet(o, opts, o.map((s) => energy.get(s.id)!)).cost;
  const lo = opts.keepOpener ? 1 : 0;
  const hi = opts.keepCloser ? n - 2 : n - 1;
  if (hi - lo < 1) return songs.slice();
  const rand = rng(seed + n * 31);
  let best = songs.slice();
  let bestCost = cost(best);
  const iterations = Math.min(40000, 1500 * n);
  for (let restart = 0; restart < 4; restart++) {
    let cur = restart === 0 ? songs.slice() : best.slice();
    if (restart > 0) for (let k = 0; k < n; k++) { // shake it up
      const a = lo + Math.floor(rand() * (hi - lo + 1)), b = lo + Math.floor(rand() * (hi - lo + 1));
      [cur[a], cur[b]] = [cur[b], cur[a]];
    }
    let curCost = cost(cur);
    for (let it = 0; it < iterations; it++) {
      const temp = 6 * (1 - it / iterations) + 0.05;
      const next = cur.slice();
      const a = lo + Math.floor(rand() * (hi - lo + 1));
      const b = lo + Math.floor(rand() * (hi - lo + 1));
      if (a === b) continue;
      if (rand() < 0.5) [next[a], next[b]] = [next[b], next[a]];
      else next.splice(b, 0, next.splice(a, 1)[0]);
      const c = cost(next);
      if (c < curCost || rand() < Math.exp((curCost - c) / temp)) {
        cur = next; curCost = c;
        if (c < bestCost) { best = next; bestCost = c; }
      }
    }
  }
  return best;
}
