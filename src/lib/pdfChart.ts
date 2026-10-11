// Turn a chart PDF into an editable ChordPro chart.
//   - Chord sheets (chords above lyrics): text is rebuilt line by line and run through the normal importer.
//   - Ultimate Guitar Pro tab scores: lyrics and section names are read from the text, chords are worked
//     out from the tab itself (the notes in each bar, using the printed tuning and capo).
//   - Notation-only scores: lyrics and sections only.
import { importChart, type ImportedChart } from "./music/convert";
import { noteIndex, noteName } from "./music/chords";
import { sectionTypeFromLabel } from "./music/chordpro";

export interface TextItem { page: number; x: number; y: number; w: number; h: number; str: string; font: string }
/** A vertical line on the page (bar lines), top-down coordinates. */
export interface VLine { page: number; x: number; y0: number; y1: number }
export interface PdfLayout { pages: { width: number; height: number }[]; items: TextItem[]; vlines: VLine[] }

export interface PdfChartResult extends ImportedChart {
  kind: "chords" | "tab" | "lyrics";
  /** Things the person should check before saving. */
  warnings: string[];
}

// ---------------------------------------------------------------- reading the PDF (pdf.js, browser or node)

type PdfDoc = {
  numPages: number;
  getPage(n: number): Promise<{
    getViewport(o: { scale: number }): { width: number; height: number; convertToViewportPoint(x: number, y: number): number[] };
    getTextContent(): Promise<{ items: unknown[] }>;
    getOperatorList(): Promise<{ fnArray: number[]; argsArray: unknown[] }>;
  }>;
};

type Mat = [number, number, number, number, number, number];
const mul = (m: Mat, n: Mat): Mat => [
  m[0] * n[0] + m[2] * n[1], m[1] * n[0] + m[3] * n[1],
  m[0] * n[2] + m[2] * n[3], m[1] * n[2] + m[3] * n[3],
  m[0] * n[4] + m[2] * n[5] + m[4], m[1] * n[4] + m[3] * n[5] + m[5],
];
const apply = (m: Mat, x: number, y: number) => [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];

/** Positioned text and vertical lines from every page. `OPS` is pdf.js's operator table. */
export async function readPdfLayout(doc: PdfDoc, OPS: Record<string, number>): Promise<PdfLayout> {
  const out: PdfLayout = { pages: [], items: [], vlines: [] };
  for (let p = 1; p <= doc.numPages; p++) {
    const page = await doc.getPage(p);
    const vp = page.getViewport({ scale: 1 });
    out.pages.push({ width: vp.width, height: vp.height });
    const tc = await page.getTextContent();
    for (const raw of tc.items as { str?: string; transform?: number[]; width?: number; height?: number; fontName?: string }[]) {
      if (!raw.str || !raw.transform) continue;
      const str = raw.str.replace(/[\uE000-\uF8FF]/g, "");
      if (!str.trim()) continue; // music-font glyphs (noteheads, clefs) aren't text
      const [x, y] = vp.convertToViewportPoint(raw.transform[4], raw.transform[5]);
      out.items.push({ page: p, x, y, w: raw.width ?? 0, h: raw.height || Math.abs(raw.transform[3]) || 10, str, font: raw.fontName ?? "" });
    }
    // Vertical strokes (bar lines) — they mark the top and bottom line of every staff
    const ol = await page.getOperatorList();
    let ctm: Mat = [1, 0, 0, 1, 0, 0];
    const stack: Mat[] = [];
    for (let i = 0; i < ol.fnArray.length; i++) {
      const fn = ol.fnArray[i];
      const args = ol.argsArray[i] as unknown[];
      if (fn === OPS.save) stack.push(ctm);
      else if (fn === OPS.restore) ctm = stack.pop() ?? [1, 0, 0, 1, 0, 0];
      else if (fn === OPS.transform) ctm = mul(ctm, args as unknown as Mat);
      else if (fn === OPS.constructPath) {
        const mm = args[2] as ArrayLike<number> | undefined;
        if (!mm || mm.length < 4) continue;
        const [ax, ay] = apply(ctm, mm[0], mm[1]);
        const [bx, by] = apply(ctm, mm[2], mm[3]);
        const [x0, y0] = vp.convertToViewportPoint(ax, ay);
        const [x1, y1] = vp.convertToViewportPoint(bx, by);
        if (Math.abs(x1 - x0) < 3 && Math.abs(y1 - y0) > 15) out.vlines.push({ page: p, x: (x0 + x1) / 2, y0: Math.min(y0, y1), y1: Math.max(y0, y1) });
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------- chord names from notes

const TEMPLATES: [string, number[]][] = [
  ["", [0, 4, 7]], ["m", [0, 3, 7]], ["7", [0, 4, 7, 10]], ["m7", [0, 3, 7, 10]], ["maj7", [0, 4, 7, 11]],
  ["sus4", [0, 5, 7]], ["sus2", [0, 2, 7]], ["add9", [0, 2, 4, 7]], ["madd9", [0, 2, 3, 7]], ["6", [0, 4, 7, 9]],
  ["m6", [0, 3, 7, 9]], ["7sus4", [0, 5, 7, 10]], ["9", [0, 2, 4, 7, 10]], ["dim", [0, 3, 6]], ["aug", [0, 4, 8]],
  ["dim7", [0, 3, 6, 9]], ["m7b5", [0, 3, 6, 10]], ["5", [0, 7]],
];
// Common pop spellings: Bb, Eb, Ab but C#, F#
const FLAT_PCS = new Set([3, 8, 10]);
const spell = (pc: number) => noteName(pc, FLAT_PCS.has(((pc % 12) + 12) % 12));

/**
 * Name the chord formed by these pitch classes (the bass note is the lowest one sounding).
 * Returns null for single notes or sets that aren't a recognisable chord.
 */
export function nameChord(pcsIn: number[], bass: number): string | null {
  const pcs = [...new Set(pcsIn.map((p) => ((p % 12) + 12) % 12))];
  if (pcs.length < 2) return null;
  let best: { name: string; score: number } | null = null;
  for (const root of pcs) {
    const rel = pcs.map((p) => (p - root + 12) % 12).sort((a, b) => a - b);
    for (const [q, t] of TEMPLATES) {
      const exact = rel.length === t.length && rel.every((v, i) => v === t[i]);
      // Allow the 5th to be missing (very common in tab voicings)
      const no5 = !exact && t.includes(7) && rel.length === t.length - 1 && rel.every((v) => t.includes(v)) && !rel.includes(7);
      if (!exact && !no5) continue;
      if (q === "5" && pcs.length !== 2) continue;
      let score = (exact ? 0 : 2) + TEMPLATES.findIndex((x) => x[0] === q) * 0.15 + (root === bass ? 0 : 1.5);
      if (pcs.length === 2 && q !== "5") score += 5; // two notes only make a power chord with confidence
      const name = spell(root) + q + (root === ((bass % 12) + 12) % 12 ? "" : `/${spell(bass)}`);
      if (!best || score < best.score) best = { name, score };
    }
  }
  return best && best.score < 6 ? best.name : null;
}

// ---------------------------------------------------------------- helpers

const median = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : 0; };
const isDigits = (s: string) => /^[\d\s()<>]+$/.test(s.trim());
// Marks printed around tab that aren't lyrics
const ANNOTATION = /^(H|P|S|B|T|h|p|s|b|x|PM|P\.M\.|let ring|harm\.?|A\.H\.|N\.H\.|vib\.?|~+|-+|\*|\||\/|\\|1\.|2\.|3\.|full|½|1\/2|Fine|D\.S\.|D\.C\.|Coda|rit\.?|accel\.?|tr|w\/bar|grad\.?|mf|f|p|mp|pp|ff|>)$/i;
const isLabel = (s: string) => !!sectionTypeFromLabel(s.replace(/[[\]:]/g, "").trim()) && s.trim().length < 24;

// Words that stand alone; any other fragment next to a lowercase fragment is a split syllable ("gon"+"na")
const COMMON = new Set(("a i the to of in on and but my me be we he she it is am are was you your our they them this that with for " +
  "at as so no not do don't can can't will won't all up out down over by if or oh yeah love baby way go got get like just one " +
  "now when what where who how her his him us its it's i'm i'll i've you're we're there here then than from into off too " +
  "night day time home back let more some come know see say make take want need feel gonna wanna said had has have").split(" "));
function joinSyllables(chunks: { x: number; str: string }[]): { x: number; str: string }[] {
  const out: { x: number; str: string }[] = [];
  for (const c of chunks) {
    const prev = out[out.length - 1];
    if (prev) {
      const last = prev.str.split(/\s+/).pop()!.toLowerCase();
      const first = c.str.trim().split(/\s+/)[0].toLowerCase().replace(/[^a-z']/g, "");
      const letters = /[a-z']$/i.test(prev.str) && /^[a-z]/.test(c.str.trim());
      // UG Pro prints one item per sung note, so words and syllables look alike — only join
      // fragments that clearly aren't words on their own ("gon" + "na", "wan" + "na")
      if (letters && !COMMON.has(last.replace(/[^a-z']/g, "")) && !COMMON.has(first) && first.length <= 3 && last.length <= 3) {
        prev.str += c.str.trim();
        continue;
      }
    }
    out.push({ x: c.x, str: c.str.trim() });
  }
  return out;
}

function headerMeta(layout: PdfLayout, firstContentY: number) {
  const meta: ImportedChart["meta"] = {};
  const top = layout.items.filter((i) => i.page === 1 && i.y < firstContentY && !isDigits(i.str));
  const bySize = [...top].sort((a, b) => b.h - a.h).filter((i) => !/tuning|capo|tempo|key|bpm|ultimate/i.test(i.str));
  if (bySize[0]) meta.title = bySize[0].str.trim();
  if (bySize[1] && bySize[1].h > median(layout.items.map((i) => i.h)) * 1.1) meta.artist = bySize[1].str.trim();
  for (const i of layout.items.filter((x) => x.page === 1)) {
    const capo = /capo\s*:?\s*(?:fret\s*)?(\d+)/i.exec(i.str);
    if (capo) meta.capo = Number(capo[1]);
    const tempo = /(?:^|\s|=)\s*(\d{2,3})\s*$/.exec(i.str);
    if (tempo && /=/.test(i.str)) meta.tempo = Number(tempo[1]);
  }
  return meta;
}

// ---------------------------------------------------------------- tab scores

interface Staff { page: number; top: number; bottom: number; bars: number[] }

/** Six-line tab staves: bar lines that share a top and bottom, with fret numbers between them. */
function findTabStaves(layout: PdfLayout, digits: TextItem[]): Staff[] {
  const groups = new Map<string, VLine[]>();
  for (const v of layout.vlines) {
    const k = `${v.page}:${Math.round(v.y0 / 4)}:${Math.round(v.y1 / 4)}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k)!.push(v);
  }
  const staves: Staff[] = [];
  for (const g of groups.values()) {
    const { page, y0, y1 } = g[0];
    const inside = digits.filter((d) => d.page === page && d.y - d.h * 0.35 >= y0 - 6 && d.y - d.h * 0.35 <= y1 + 6);
    if (inside.length < 2) continue;
    staves.push({ page, top: y0, bottom: y1, bars: [...new Set(g.map((v) => Math.round(v.x)))].sort((a, b) => a - b) });
  }
  // Drop staves nested inside another (e.g. a repeat bracket line)
  return staves
    .filter((s) => !staves.some((o) => o !== s && o.page === s.page && o.top <= s.top - 2 && o.bottom >= s.bottom + 2))
    .sort((a, b) => a.page - b.page || a.top - b.top);
}

function parseTuning(items: TextItem[]): number[] {
  const t = items.find((i) => /tuning/i.test(i.str));
  const notes = t?.str.split(/[:\s]+/).slice(1).filter((n) => /^[A-G][#b]?$/.test(n)) ?? [];
  if (notes.length === 6) {
    // Written low to high; give MIDI-ish pitches so the lowest sounding note is known
    const pcs = notes.map(noteIndex);
    const base = [40, 45, 50, 55, 59, 64];
    return pcs.map((pc, i) => { let m = base[i]; while (((m % 12) + 12) % 12 !== pc) m--; return m; });
  }
  return [40, 45, 50, 55, 59, 64]; // standard E A D G B E
}

function tabChart(layout: PdfLayout, staves: Staff[], digits: TextItem[], warnings: string[]): ImportedChart {
  const tuning = parseTuning(layout.items); // low E first
  const meta = headerMeta(layout, staves[0]?.top ?? Infinity);
  const capo = meta.capo ?? 0;
  const lines: string[] = [];
  let unnamed = 0;
  // The font used for clear section names ("Verse 1", "Chorus") on each page — other text that
  // happens to read like a label ("end", "intro" inside lyrics) is ignored
  const labelFontsByPage = new Map<number, Set<string>>();
  for (const i of layout.items) {
    if (!/^(intro|verse|chorus|pre-?chorus|bridge|outro|interlude|solo|break|instrumental|refrain|tag|ending)\b/i.test(i.str.trim())) continue;
    if (!labelFontsByPage.has(i.page)) labelFontsByPage.set(i.page, new Set());
    labelFontsByPage.get(i.page)!.add(i.font);
  }
  let prevChord: string | null = null;

  staves.forEach((st, si) => {
    const gap = (st.bottom - st.top) / 5;
    const prevBottom = staves[si - 1]?.page === st.page ? staves[si - 1].bottom : 0;
    // Fret numbers on this staff → sounding pitches at their x
    const notes: { x: number; pitch: number }[] = [];
    for (const d of digits.filter((x) => x.page === st.page)) {
      const cy = d.y - d.h * 0.35;
      if (cy < st.top - gap * 0.6 || cy > st.bottom + gap * 0.6) continue;
      const string = Math.round((cy - st.top) / gap); // 0 = high e
      if (string < 0 || string > 5) continue;
      const nums = d.str.trim().split(/\s+/).map((n) => n.replace(/[()<>]/g, "")).filter((n) => /^\d{1,2}$/.test(n));
      const cw = d.w / Math.max(1, d.str.length);
      nums.forEach((n, k) => notes.push({ x: d.x + k * cw * 2, pitch: tuning[5 - string] + Number(n) + capo }));
    }
    // A chord per bar; split a bar in two when its halves clearly hold different chords
    const bars = [...st.bars];
    if (!bars.length || bars[0] > (notes[0]?.x ?? 0)) bars.unshift(Math.min(...notes.map((n) => n.x), bars[0] ?? 0) - 1);
    const marks: { x: number; chord: string }[] = [];
    const chordIn = (from: number, to: number) => {
      const ns = notes.filter((n) => n.x >= from && n.x < to);
      if (ns.length < 2) return null;
      const bass = ns.reduce((a, b) => (b.pitch < a.pitch ? b : a)).pitch;
      return { chord: nameChord(ns.map((n) => n.pitch), bass), x: Math.min(...ns.map((n) => n.x)) };
    };
    for (let b = 0; b < bars.length; b++) {
      const from = bars[b], to = bars[b + 1] ?? Infinity;
      if (!notes.some((n) => n.x >= from && n.x < to)) continue;
      const mid = to === Infinity ? Infinity : (from + to) / 2;
      const left = chordIn(from, mid), right = chordIn(mid, to);
      const whole = chordIn(from, to);
      if (left?.chord && right?.chord && left.chord !== right.chord) marks.push(left as { x: number; chord: string }, right as { x: number; chord: string });
      else if (whole?.chord) marks.push(whole as { x: number; chord: string });
      else if (whole) unnamed++;
    }

    // Lyrics and section labels between the previous staff and this one
    const band = layout.items.filter((i) => i.page === st.page && i.y > prevBottom + gap && i.y < st.top + gap * 0.2 && !isDigits(i.str) && !ANNOTATION.test(i.str.trim()) && !/tuning|^=\s*\d/i.test(i.str));
    const labelFonts = labelFontsByPage.get(st.page) ?? new Set<string>();
    const labels = band.filter((i) => isLabel(i.str) && (labelFonts.size === 0 || labelFonts.has(i.font)));
    const lyricItems = band.filter((i) => !labels.includes(i) && i.y > st.top - gap * 4 && !/^[=♩]/.test(i.str.trim()));
    const lyricY = median(lyricItems.map((i) => i.y));
    const lyrics = joinSyllables(lyricItems.filter((i) => Math.abs(i.y - lyricY) < gap * 0.8).sort((a, b) => a.x - b.x).map((i) => ({ x: i.x, str: i.str })));

    // Merge everything left to right into ChordPro text, starting a new section at each label
    type Tok = { x: number; kind: "chord" | "lyric" | "label"; text: string };
    const toks: Tok[] = [
      ...marks.map((m) => ({ x: m.x, kind: "chord" as const, text: m.chord })),
      ...lyrics.map((l) => ({ x: l.x, kind: "lyric" as const, text: l.str })),
      ...labels.map((l) => ({ x: l.x - 1, kind: "label" as const, text: l.str.replace(/[[\]:]/g, "").trim() })),
    ].sort((a, b) => a.x - b.x || (a.kind === "label" ? -1 : a.kind === "chord" ? -1 : 1));
    let line = "";
    const flush = () => { if (line.trim()) lines.push(line.replace(/\s+$/, "")); line = ""; };
    for (const t of toks) {
      if (t.kind === "label") { flush(); lines.push("", `[${t.text}]`); continue; }
      if (t.kind === "chord") {
        if (t.text === prevChord) continue; // only show changes
        prevChord = t.text;
        line += `[${t.text}]`;
      } else line += (line && !line.endsWith("]") && !line.endsWith(" ") ? " " : "") + t.text + " ";
    }
    flush();
  });
  if (unnamed > 0) warnings.push(`${unnamed} bar${unnamed === 1 ? "" : "s"} had notes that didn't spell a clear chord (melody or riffs) — check the chords against a recording.`);
  warnings.push("Chords were worked out from the tab. They're usually right for strummed and picked parts; check riffs and passing chords.");
  // Section labels as ChordPro-friendly headers ("[Verse 1]" lines become sections in the importer)
  const text = [meta.title ? `{title: ${meta.title}}` : "", meta.artist ? `{artist: ${meta.artist}}` : "", meta.capo ? `{capo: ${meta.capo}}` : "", meta.tempo ? `{tempo: ${meta.tempo}}` : "", ...lines].filter((l, i) => l || i > 3).join("\n");
  return importChart(text);
}

// ---------------------------------------------------------------- chord sheets and lyrics-only

/** Rebuild the page as monospaced text lines (chords stay above the right syllables). */
function textLines(layout: PdfLayout, items: TextItem[]): string[] {
  const out: string[] = [];
  // Character width of body text (titles are bigger and would throw the columns off)
  const bodyH = median(items.map((i) => i.h));
  const charW = median(items.filter((i) => i.str.length > 3 && i.h <= bodyH * 1.15).map((i) => i.w / i.str.length)) || 5;
  for (let p = 1; p <= layout.pages.length; p++) {
    const pageItems = items.filter((i) => i.page === p).sort((a, b) => a.y - b.y || a.x - b.x);
    const rows: TextItem[][] = [];
    for (const it of pageItems) {
      const row = rows[rows.length - 1];
      if (row && Math.abs(row[0].y - it.y) < Math.max(2, row[0].h * 0.4)) row.push(it);
      else rows.push([it]);
    }
    let lastY = -Infinity;
    for (const row of rows) {
      row.sort((a, b) => a.x - b.x);
      if (row[0].y - lastY > row[0].h * 2.2 && out.length) out.push("");
      lastY = row[0].y;
      let s = "";
      for (const it of row) {
        const col = Math.max(0, Math.round(it.x / charW));
        if (s.length < col) s = s.padEnd(col, " ");
        else if (s.length && !s.endsWith(" ")) s += " ";
        s += it.str;
      }
      out.push(s.replace(/\s+$/, ""));
    }
  }
  // Trim the common left margin
  const indent = Math.min(...out.filter((l) => l.trim()).map((l) => l.length - l.trimStart().length));
  return out.map((l) => l.slice(Number.isFinite(indent) ? indent : 0));
}

const JUNK_LINE = /ultimate-guitar|ultimate guitar|^page \d+|^\d+\s*\/\s*\d+$|^\d+$|www\.|https?:|printed from|tabs\.ultimate/i;

// ---------------------------------------------------------------- entry point

export function pdfToChart(layout: PdfLayout): PdfChartResult {
  const warnings: string[] = [];
  const digits = layout.items.filter((i) => isDigits(i.str));
  const staves = findTabStaves(layout, digits);
  if (staves.length >= 2) {
    const chart = tabChart(layout, staves, digits, warnings);
    return { ...chart, kind: "tab", warnings };
  }
  // Chord sheet or notation: rebuild the text, dropping page furniture and notation numbers
  // (Tab markings like H/P/B aren't filtered here — on a chord sheet "B" and "F" are chords)
  const items = layout.items.filter((i) => !JUNK_LINE.test(i.str.trim()) && !(isDigits(i.str) && i.str.trim().length <= 3));
  const text = textLines(layout, items).filter((l) => !JUNK_LINE.test(l.trim())).join("\n").replace(/\n{3,}/g, "\n\n");
  const chart = importChart(text);
  const hasChords = /\[[A-G]/.test(chart.body);
  if (!hasChords) warnings.push("No chords found in this PDF (it looks like sheet music) — you'll get the lyrics and sections; add chords in the editor.");
  return { ...chart, kind: hasChords ? "chords" : "lyrics", warnings };
}
