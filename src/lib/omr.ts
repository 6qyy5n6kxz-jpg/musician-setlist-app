// Light optical music recognition for reading help: find 5-line staves on a rendered PDF page and
// the notes that sit on ledger lines above/below them, then name those notes from their position.
//
// Deliberately narrow and robust: a note only counts if real ledger lines run between it and the
// staff, which filters out lyrics, tab numbers and chord symbols. 6-line tab staves are skipped.

export type Clef = "treble" | "bass";

export interface DarkImage {
  w: number;
  h: number;
  /** 1 = dark pixel */
  dark: Uint8Array;
}

export interface Staff {
  /** y of the 5 lines, top to bottom */
  lines: number[];
  space: number;
  x0: number;
  x1: number;
}

export interface FoundNote {
  x: number;
  y: number;
  /** Staff position: 0 = bottom line, 8 = top line, each step is a line or space. */
  step: number;
  staffIndex: number;
  /** How notehead-like the match is (higher = more confident). */
  score: number;
}

export function toDark(rgba: Uint8ClampedArray, w: number, h: number, threshold = 150): DarkImage {
  const dark = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < dark.length; i++, p += 4) {
    const lum = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
    dark[i] = rgba[p + 3] > 0 && lum < threshold ? 1 : 0;
  }
  return { w, h, dark };
}

/** Summed-area table for fast "how dark is this box" queries. */
export function integral(img: DarkImage): Uint32Array {
  const { w, h, dark } = img;
  const s = new Uint32Array((w + 1) * (h + 1));
  for (let y = 1; y <= h; y++) {
    let row = 0;
    for (let x = 1; x <= w; x++) {
      row += dark[(y - 1) * w + (x - 1)];
      s[y * (w + 1) + x] = s[(y - 1) * (w + 1) + x] + row;
    }
  }
  return s;
}

function boxRatio(ii: Uint32Array, img: DarkImage, x0: number, y0: number, x1: number, y1: number): number {
  const W = img.w + 1;
  const a = Math.max(0, Math.min(img.w, Math.round(x0))), b = Math.max(0, Math.min(img.w, Math.round(x1)));
  const c = Math.max(0, Math.min(img.h, Math.round(y0))), d = Math.max(0, Math.min(img.h, Math.round(y1)));
  const area = (b - a) * (d - c);
  if (area <= 0) return 0;
  return (ii[d * W + b] - ii[c * W + b] - ii[d * W + a] + ii[c * W + a]) / area;
}

/** Longest run of dark pixels in a row, with its extent. */
function longestRun(img: DarkImage, y: number): { len: number; x0: number; x1: number } {
  let best = { len: 0, x0: 0, x1: 0 };
  let start = -1;
  const row = y * img.w;
  for (let x = 0; x <= img.w; x++) {
    const d = x < img.w && img.dark[row + x];
    if (d && start < 0) start = x;
    if (!d && start >= 0) {
      if (x - start > best.len) best = { len: x - start, x0: start, x1: x - 1 };
      start = -1;
    }
  }
  return best;
}

/** Find 5-line staves (6-line tab staves are recognised and skipped). */
export function findStaves(img: DarkImage): Staff[] {
  const minRun = img.w * 0.2;
  const rows: { y: number; x0: number; x1: number }[] = [];
  for (let y = 0; y < img.h; y++) {
    const r = longestRun(img, y);
    if (r.len >= minRun) rows.push({ y, x0: r.x0, x1: r.x1 });
  }
  // Merge adjacent rows (thick lines) into single lines
  type Line = { y: number; x0: number; x1: number; n: number; lastRow: number };
  const lines: Line[] = [];
  for (const r of rows) {
    const last = lines[lines.length - 1];
    if (last && r.y - last.lastRow <= 1) {
      last.n++;
      last.y += (r.y - last.y) / last.n;
      last.lastRow = r.y;
      last.x0 = Math.min(last.x0, r.x0);
      last.x1 = Math.max(last.x1, r.x1);
    } else lines.push({ ...r, n: 1, lastRow: r.y });
  }
  const staves: Staff[] = [];
  const near = (a: number, b: number) => Math.abs(a - b) <= Math.max(1.5, b * 0.22);
  let i = 0;
  while (i + 4 < lines.length) {
    const d = lines[i + 1].y - lines[i].y;
    let count = 1;
    while (i + count < lines.length && near(lines[i + count].y - lines[i + count - 1].y, d)) count++;
    if (count === 5 && d >= 4) {
      const ls = lines.slice(i, i + 5);
      staves.push({
        lines: ls.map((l) => l.y),
        space: (ls[4].y - ls[0].y) / 4,
        x0: Math.max(...ls.map((l) => l.x0)),
        x1: Math.min(...ls.map((l) => l.x1)),
      });
      i += 5;
    } else if (count >= 5) {
      i += count; // tab staff (6 lines) or other ruled block: skip
    } else i += 1;
  }
  return staves;
}

/** Short horizontal dark runs (ledger lines) on a row, within [xa, xb]. */
function ledgerSegments(img: DarkImage, y: number, space: number, xa: number, xb: number): { x0: number; x1: number }[] {
  const out: { x0: number; x1: number }[] = [];
  const minLen = space * 1.15, maxLen = space * 4;
  const rows = [Math.round(y) - 1, Math.round(y), Math.round(y) + 1].filter((r) => r >= 0 && r < img.h);
  for (const r of rows) {
    let start = -1;
    for (let x = Math.max(0, Math.floor(xa)); x <= Math.min(img.w, Math.ceil(xb)); x++) {
      const d = x < img.w && img.dark[r * img.w + x];
      if (d && start < 0) start = x;
      if (!d && start >= 0) {
        const len = x - start;
        if (len >= minLen && len <= maxLen && !out.some((s) => Math.abs((s.x0 + s.x1) / 2 - (start + x - 1) / 2) < space)) {
          out.push({ x0: start, x1: x - 1 });
        }
        start = -1;
      }
    }
  }
  return out;
}

/** Is there a notehead (filled or hollow) centred near (cx, cy)? Returns refined x and a score. */
function notehead(ii: Uint32Array, img: DarkImage, cx: number, cy: number, s: number, onLine: boolean): { x: number; score: number } | null {
  const hw = s * 0.5, hh = s * 0.38;
  let best: { x: number; score: number } | null = null;
  for (let dx = -s * 0.5; dx <= s * 0.5; dx += Math.max(1, s / 8)) {
    const x = cx + dx;
    const full = boxRatio(ii, img, x - hw, cy - hh, x + hw, cy + hh);
    // Hollow heads (half/whole notes): dark ring, light middle (ignore the ledger row through it)
    const midTop = onLine ? cy - hh * 0.9 : cy - hh * 0.45;
    const inner = onLine
      ? (boxRatio(ii, img, x - hw * 0.4, midTop, x + hw * 0.4, cy - s * 0.12) + boxRatio(ii, img, x - hw * 0.4, cy + s * 0.12, x + hw * 0.4, cy + hh * 0.9)) / 2
      : boxRatio(ii, img, x - hw * 0.4, cy - hh * 0.45, x + hw * 0.4, cy + hh * 0.45);
    const above = boxRatio(ii, img, x - hw * 0.6, cy - hh * 1.05, x + hw * 0.6, cy - hh * 0.75);
    const below = boxRatio(ii, img, x - hw * 0.6, cy + hh * 0.75, x + hw * 0.6, cy + hh * 1.05);
    const filled = full >= 0.62;
    const hollow = full >= 0.3 && inner < 0.3 && above > 0.2 && below > 0.2;
    if (!filled && !hollow) continue;
    // Reject solid bars (beams, thick text): a notehead has light space just beyond its top or bottom
    const clearAbove = boxRatio(ii, img, x - hw * 0.5, cy - s * 0.95, x + hw * 0.5, cy - s * 0.6);
    const clearBelow = boxRatio(ii, img, x - hw * 0.5, cy + s * 0.6, x + hw * 0.5, cy + s * 0.95);
    if (clearAbove > 0.7 && clearBelow > 0.7) continue;
    const score = filled ? full : 0.5 + (0.3 - inner);
    if (!best || score > best.score) best = { x, score };
  }
  if (!best) return null;
  // Vertical fit: a head centred exactly here is darker in its middle band than half a step off
  const hw2 = s * 0.4;
  const centre = boxRatio(ii, img, best.x - hw2, cy - s * 0.18, best.x + hw2, cy + s * 0.18);
  const offUp = boxRatio(ii, img, best.x - hw2, cy - s * 0.68, best.x + hw2, cy - s * 0.32);
  const offDown = boxRatio(ii, img, best.x - hw2, cy + s * 0.32, best.x + hw2, cy + s * 0.68);
  return { x: best.x, score: best.score + centre - Math.max(offUp, offDown) * 0.5 };
}

/**
 * Notes on ledger lines (at least one ledger line beyond the staff), above and below each staff.
 * `maxLedgers` limits how far out to look.
 */
export function findLedgerNotes(img: DarkImage, staves: Staff[], maxLedgers = 5): FoundNote[] {
  const ii = integral(img);
  const notes: FoundNote[] = [];
  staves.forEach((st, si) => {
    const s = st.space;
    const bottom = st.lines[4];
    const yOf = (step: number) => bottom - (step * s) / 2;
    const upper = si > 0 ? (staves[si - 1].lines[4] + st.lines[0]) / 2 : 0;
    const lower = si < staves.length - 1 ? (st.lines[4] + staves[si + 1].lines[0]) / 2 : img.h;
    for (const dir of [1, -1] as const) {
      // Ledger steps: 10, 12, … above; -2, -4, … below
      let segsInner: { x0: number; x1: number }[] | null = null;
      for (let n = 1; n <= maxLedgers; n++) {
        const step = dir > 0 ? 8 + 2 * n : -2 * n;
        const y = yOf(step);
        if (y < upper + s * 0.5 || y > lower - s * 0.5) break;
        let segs = ledgerSegments(img, y, s, st.x0 - s, st.x1 + s);
        // Every ledger line beyond the first must sit over one closer to the staff
        if (segsInner) {
          const inner = segsInner;
          segs = segs.filter((g) => inner.some((h) => Math.abs((g.x0 + g.x1) / 2 - (h.x0 + h.x1) / 2) < s * 0.8));
        }
        if (!segs.length) break;
        for (const g of segs) {
          const cx = (g.x0 + g.x1) / 2;
          // The head is either ON this ledger or in the space just beyond it — never both.
          const onLine = notehead(ii, img, cx, yOf(step), s, true);
          const beyondOk = !ledgerSegments(img, yOf(step + 2 * dir), s, cx - s, cx + s).length;
          const beyond = beyondOk ? notehead(ii, img, cx, yOf(step + dir), s, false) : null;
          const pick = onLine && (!beyond || onLine.score >= beyond.score) ? { ...onLine, k: step } : beyond ? { ...beyond, k: step + dir } : null;
          if (pick) notes.push({ x: pick.x, y: yOf(pick.k), step: pick.k, staffIndex: si, score: pick.score });
        }
        segsInner = segs;
      }
    }
  });
  return dedupe(notes, staves);
}

/** One label per notehead: drop matches of the same head at the same or an adjacent step. */
function dedupe(notes: FoundNote[], staves: Staff[]): FoundNote[] {
  const sorted = [...notes].sort((a, b) => b.score - a.score);
  const kept: FoundNote[] = [];
  for (const n of sorted) {
    const s = staves[n.staffIndex].space;
    // Seconds in a chord are drawn side by side (about a head width apart), so they survive this
    const clash = kept.some((k) => k.staffIndex === n.staffIndex && Math.abs(k.step - n.step) <= 1 && Math.abs(k.x - n.x) < s * 0.6);
    if (!clash) kept.push(n);
  }
  return kept.sort((a, b) => a.staffIndex - b.staffIndex || a.x - b.x);
}

/**
 * Piano (grand staff) detection: two staves joined by a line down their left edge (the system
 * barline / brace side) form a treble + bass pair.
 */
export function pairedStaves(img: DarkImage, staves: Staff[]): boolean[] {
  const paired = staves.map(() => false);
  for (let i = 0; i + 1 < staves.length; i++) {
    if (paired[i]) continue;
    const a = staves[i], b = staves[i + 1];
    const y0 = Math.round(a.lines[0]), y1 = Math.round(b.lines[4]);
    const gap = b.lines[0] - a.lines[4];
    if (gap > a.space * 14) continue; // too far apart to be one system
    const x0 = Math.min(a.x0, b.x0);
    let joined = false;
    for (let x = Math.max(0, Math.round(x0 - a.space * 1.5)); x <= Math.min(img.w - 1, Math.round(x0 + a.space * 1.5)) && !joined; x++) {
      let dark = 0;
      for (let y = y0; y <= y1; y++) dark += img.dark[y * img.w + x];
      if (dark / (y1 - y0 + 1) > 0.92) joined = true;
    }
    if (joined) paired[i] = paired[i + 1] = true;
  }
  return paired;
}

const LETTERS = "CDEFGAB";

/** Note name for a staff step: treble bottom line = E4, bass bottom line = G2. */
export function noteName(step: number, clef: Clef, withOctave = false): string {
  const base = clef === "treble" ? 2 + 7 * 4 : 4 + 7 * 2; // diatonic index of the bottom line
  const idx = base + step;
  const letter = LETTERS[((idx % 7) + 7) % 7];
  return withOctave ? `${letter}${Math.floor(idx / 7)}` : letter;
}

export type ClefMode = "auto" | "treble" | "bass" | "grand";

/**
 * Clef for each staff. "auto": staves joined into a piano pair are treble (top) + bass (bottom),
 * single staves are treble. "grand": alternate treble/bass. Otherwise all the same.
 */
export function clefsFor(staffCount: number, mode: ClefMode, paired: boolean[] = []): Clef[] {
  if (mode === "auto") {
    const out: Clef[] = [];
    for (let i = 0; i < staffCount; i++) out.push(paired[i] && i > 0 && paired[i - 1] && out[i - 1] === "treble" ? "bass" : "treble");
    return out;
  }
  return Array.from({ length: staffCount }, (_, i) => (mode === "grand" ? (i % 2 === 0 ? "treble" : "bass") : mode));
}

export interface LabelBox {
  x0: number; y0: number; x1: number; y1: number;
}

/**
 * Find a spot for a note's label that covers as little ink as possible (notes, lyrics, markings)
 * and doesn't overlap labels already placed. Returns the text box in page pixels.
 */
export function placeLabel(
  ii: Uint32Array, img: DarkImage, note: FoundNote, space: number, text: string, placed: LabelBox[],
): LabelBox & { fontPx: number } {
  const f = space * 1.05; // font size
  const w = f * 0.62 * text.length + 2, h = f * 0.82;
  const { x: cx, y: cy } = note;
  const high = note.step > 4;
  const near = space * 0.62;
  const candidates: { box: LabelBox; bias: number }[] = [
    { box: { x0: cx - w / 2, y0: cy - near - h, x1: cx + w / 2, y1: cy - near }, bias: high ? 0 : 0.03 },
    { box: { x0: cx - w / 2, y0: cy + near, x1: cx + w / 2, y1: cy + near + h }, bias: high ? 0.03 : 0 },
    { box: { x0: cx - space * 0.8 - w, y0: cy - h / 2, x1: cx - space * 0.8, y1: cy + h / 2 }, bias: 0.04 },
    { box: { x0: cx + space * 0.9, y0: cy - h / 2, x1: cx + space * 0.9 + w, y1: cy + h / 2 }, bias: 0.05 },
    { box: { x0: cx - w / 2, y0: cy - near - h - space * 0.8, x1: cx + w / 2, y1: cy - near - space * 0.8 }, bias: 0.06 },
    { box: { x0: cx - w / 2, y0: cy + near + space * 0.8, x1: cx + w / 2, y1: cy + near + h + space * 0.8 }, bias: 0.06 },
  ];
  const overlap = (a: LabelBox, b: LabelBox) =>
    Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0)) * Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0)) / ((a.x1 - a.x0) * (a.y1 - a.y0));
  let best = candidates[0].box, bestScore = Infinity;
  for (const c of candidates) {
    const b = c.box;
    if (b.x0 < 0 || b.y0 < 0 || b.x1 > img.w || b.y1 > img.h) continue;
    const ink = boxRatio(ii, img, b.x0, b.y0, b.x1, b.y1);
    const clash = placed.reduce((m, p) => Math.max(m, overlap(b, p)), 0);
    const score = ink + clash * 3 + c.bias;
    if (score < bestScore) { bestScore = score; best = b; }
  }
  placed.push(best);
  return { ...best, fontPx: f };
}
