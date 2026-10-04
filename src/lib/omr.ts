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

/** Is there a notehead (filled or hollow) centred near (cx, cy)? Returns refined x or null. */
function notehead(ii: Uint32Array, img: DarkImage, cx: number, cy: number, s: number, onLine: boolean): number | null {
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
  return best ? best.x : null;
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
          for (const k of [step, step + dir]) {
            // A note on the next ledger out is found in the next round
            const isLine = k === step;
            const hx = notehead(ii, img, cx, yOf(k), s, isLine);
            if (hx === null) continue;
            if (!isLine && ledgerSegments(img, yOf(step + 2 * dir), s, cx - s, cx + s).length) continue;
            if (!notes.some((p) => p.staffIndex === si && p.step === k && Math.abs(p.x - hx) < s * 0.7)) {
              notes.push({ x: hx, y: yOf(k), step: k, staffIndex: si });
            }
          }
        }
        segsInner = segs;
      }
    }
  });
  return notes;
}

const LETTERS = "CDEFGAB";

/** Note name for a staff step: treble bottom line = E4, bass bottom line = G2. */
export function noteName(step: number, clef: Clef, withOctave = false): string {
  const base = clef === "treble" ? 2 + 7 * 4 : 4 + 7 * 2; // diatonic index of the bottom line
  const idx = base + step;
  const letter = LETTERS[((idx % 7) + 7) % 7];
  return withOctave ? `${letter}${Math.floor(idx / 7)}` : letter;
}

/** Clef for each staff: all treble/bass, or grand staff (alternating treble/bass in pairs). */
export function clefsFor(staffCount: number, mode: "treble" | "bass" | "grand"): Clef[] {
  return Array.from({ length: staffCount }, (_, i) => (mode === "grand" ? (i % 2 === 0 ? "treble" : "bass") : mode));
}
