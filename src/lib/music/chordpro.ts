// ChordPro parsing into sections/lines, plus helpers for flow (arrangement) and lyrics-only text.
import { parseChord, transposeChord } from "./chords";

export type SectionType =
  | "verse" | "chorus" | "prechorus" | "bridge" | "intro" | "outro" | "instrumental"
  | "solo" | "interlude" | "tag" | "ending" | "refrain" | "tab" | "none";

export interface Segment {
  chord: string | null;
  lyric: string;
}

export type Line =
  | { kind: "lyrics"; segments: Segment[] }
  | { kind: "comment"; text: string }
  | { kind: "tab"; text: string }
  | { kind: "empty" };

export interface Section {
  index: number;
  type: SectionType;
  label: string;
  lines: Line[];
}

export interface ParsedSong {
  meta: Record<string, string>;
  sections: Section[];
}

const META_ALIASES: Record<string, string> = {
  t: "title", title: "title",
  st: "artist", subtitle: "artist", artist: "artist",
  key: "key", tempo: "tempo", time: "time", capo: "capo", duration: "duration",
  album: "album", year: "year", ccli: "ccli", copyright: "copyright", composer: "composer",
  lyricist: "lyricist",
};

const SECTION_WORDS: [RegExp, SectionType][] = [
  [/^pre[- ]?chorus/i, "prechorus"],
  [/^chorus/i, "chorus"],
  [/^verse/i, "verse"],
  [/^bridge/i, "bridge"],
  [/^intro/i, "intro"],
  [/^outro/i, "outro"],
  [/^(instrumental|inst\b)/i, "instrumental"],
  [/^solo/i, "solo"],
  [/^interlude/i, "interlude"],
  [/^tag/i, "tag"],
  [/^(ending|end\b|coda)/i, "ending"],
  [/^refrain/i, "refrain"],
  [/^(turnaround|break|vamp|hook|post[- ]?chorus)/i, "interlude"],
];

export function sectionTypeFromLabel(label: string): SectionType | null {
  const l = label.trim();
  for (const [re, type] of SECTION_WORDS) if (re.test(l)) return type;
  return null;
}

const DEFAULT_LABELS: Partial<Record<SectionType, string>> = {
  verse: "Verse", chorus: "Chorus", prechorus: "Pre-Chorus", bridge: "Bridge", intro: "Intro",
  outro: "Outro", instrumental: "Instrumental", solo: "Solo", interlude: "Interlude", tag: "Tag",
  ending: "Ending", refrain: "Refrain", tab: "Tab",
};

function directiveSectionType(name: string): SectionType | null {
  const n = name.toLowerCase();
  if (n === "soc" || n === "start_of_chorus") return "chorus";
  if (n === "sov" || n === "start_of_verse") return "verse";
  if (n === "sob" || n === "start_of_bridge") return "bridge";
  if (n === "sot" || n === "start_of_tab" || n === "sog" || n === "start_of_grid") return "tab";
  if (n.startsWith("start_of_")) return sectionTypeFromLabel(n.slice(9).replace(/_/g, " ")) ?? "verse";
  return null;
}

function isEndDirective(name: string): boolean {
  const n = name.toLowerCase();
  return n.startsWith("end_of_") || ["eoc", "eov", "eob", "eot", "eog"].includes(n);
}

/**
 * A standalone section header line in loose formats:
 *   "Verse 1:"  "CHORUS"  "[Chorus]"  "{{title: Verse 1}}"  "(Bridge)"
 */
export function matchHeaderLine(line: string): string | null {
  let t = line.trim();
  if (!t || t.length > 40) return null;
  const legacy = /^\{\{\s*title\s*:\s*(.+?)\s*\}\}$/i.exec(t);
  if (legacy) return legacy[1];
  const bracket = /^[[(]([^\])]+)[\])]:?$/.exec(t);
  if (bracket) {
    if (parseChord(bracket[1])) return null; // "[G]" is a chord, not a header
    t = bracket[1];
    return sectionTypeFromLabel(t) ? t : null;
  }
  const colon = /^(.+?):$/.exec(t);
  if (colon && sectionTypeFromLabel(colon[1])) return colon[1];
  // bare "Chorus", "Verse 2", "Bridge x2"
  if (/^[A-Za-z][A-Za-z -]*\s*\d*\s*(\(?x\d+\)?)?$/.test(t) && sectionTypeFromLabel(t)) {
    const words = t.split(/\s+/);
    if (words.length <= 4) return t;
  }
  return null;
}

/** Split "Amaz[A]ing grace" into chord/lyric segments. */
export function parseLyricLine(line: string): Segment[] {
  const parts = line.split(/\[([^\]]*)\]/); // odd indices are chords
  const segments: Segment[] = [];
  if (parts[0]) segments.push({ chord: null, lyric: parts[0] });
  for (let i = 1; i < parts.length; i += 2) segments.push({ chord: parts[i].trim(), lyric: parts[i + 1] ?? "" });
  return segments.length ? segments : [{ chord: null, lyric: "" }];
}

export function parseChordPro(text: string): ParsedSong {
  const meta: Record<string, string> = {};
  const sections: Section[] = [];
  let current: Section | null = null;
  let explicit = false; // inside {start_of_x} ... {end_of_x}
  const counts: Partial<Record<SectionType, number>> = {};

  const open = (type: SectionType, label: string | null, isExplicit: boolean) => {
    closeImplicit();
    counts[type] = (counts[type] ?? 0) + 1;
    current = { index: sections.length, type, label: label ?? DEFAULT_LABELS[type] ?? "", lines: [] };
    sections.push(current);
    explicit = isExplicit;
  };
  const closeImplicit = () => {
    if (current) trimEmpty(current);
    current = null;
    explicit = false;
  };
  const ensure = () => {
    if (!current) {
      current = { index: sections.length, type: "none", label: "", lines: [] };
      sections.push(current);
    }
    return current;
  };

  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    const trimmed = line.trim();

    const dir = /^\{([^:}]+)(?::\s*(.*))?\}$/.exec(trimmed);
    if (dir && !trimmed.startsWith("{{")) {
      const name = dir[1].trim();
      const value = (dir[2] ?? "").trim();
      const lname = name.toLowerCase();
      const metaKey = META_ALIASES[lname];
      if (metaKey) {
        meta[metaKey] = value;
        continue;
      }
      if (lname === "meta") {
        const [k, ...v] = value.split(/\s+/);
        if (k) meta[k.toLowerCase()] = v.join(" ");
        continue;
      }
      const secType = directiveSectionType(lname);
      if (secType) {
        open(secType, value || null, true);
        continue;
      }
      if (isEndDirective(lname)) {
        closeImplicit();
        continue;
      }
      if (["c", "comment", "ci", "comment_italic", "cb", "comment_box", "highlight"].includes(lname)) {
        ensure().lines.push({ kind: "comment", text: value });
        continue;
      }
      if (lname === "chorus") {
        // "{chorus}" = repeat the chorus here
        closeImplicit();
        const label = value || "Chorus";
        const prev = [...sections].reverse().find((s) => s.type === "chorus");
        current = { index: sections.length, type: "chorus", label, lines: prev ? prev.lines.slice() : [] };
        sections.push(current);
        closeImplicit();
        continue;
      }
      // Unknown directives (new_page, column_break, textsize, ...) are ignored.
      continue;
    }

    if (current?.type === "tab" && explicit) {
      current.lines.push({ kind: "tab", text: line });
      continue;
    }

    const header = matchHeaderLine(trimmed);
    if (header && !explicit) {
      open(sectionTypeFromLabel(header) ?? "verse", header, false);
      continue;
    }

    if (!trimmed) {
      if (explicit) {
        current?.lines.push({ kind: "empty" });
      } else if (current && current.lines.length) {
        closeImplicit(); // blank line ends an implicit paragraph
      }
      continue;
    }

    if (trimmed.startsWith("#")) continue; // ChordPro comment line

    ensure().lines.push({ kind: "lyrics", segments: parseLyricLine(line) });
  }
  closeImplicit();

  // Drop empty implicit blocks; renumber
  const kept = sections.filter((s) => s.lines.length || s.type !== "none");
  kept.forEach((s, i) => (s.index = i));
  return { meta, sections: kept };
}

function trimEmpty(s: Section) {
  while (s.lines.length && s.lines[s.lines.length - 1].kind === "empty") s.lines.pop();
}

/** Short tag used in flows: "Verse 1" -> "V1", "Chorus" -> "C", "Pre-Chorus" -> "PC". */
export function sectionAbbrev(label: string, type: SectionType): string {
  const num = /(\d+)/.exec(label)?.[1] ?? "";
  const base: Partial<Record<SectionType, string>> = {
    verse: "V", chorus: "C", prechorus: "PC", bridge: "B", intro: "I", outro: "O",
    instrumental: "Inst", solo: "S", interlude: "It", tag: "T", ending: "E", refrain: "R", tab: "Tab",
  };
  return (base[type] ?? label.slice(0, 3)) + num;
}

/**
 * Arrange sections by a flow string like "I V1 C V2 C B C C".
 * Tokens match abbreviations or full labels (case-insensitive). Unmatched tokens are skipped.
 */
export function applyFlow(sections: Section[], flow: string | null | undefined): Section[] {
  if (!flow?.trim()) return sections;
  const tokens = flow.split(/[\s,]+/).filter(Boolean);
  const out: Section[] = [];
  for (const tok of tokens) {
    const t = tok.toLowerCase();
    const match =
      sections.find((s) => sectionAbbrev(s.label, s.type).toLowerCase() === t) ??
      sections.find((s) => s.label.toLowerCase() === t) ??
      // "C" should also match "Chorus 1" when there's no plain "Chorus"
      sections.find((s) => sectionAbbrev(s.label, s.type).toLowerCase().replace(/\d+$/, "") === t);
    if (match) out.push(match);
  }
  return out.length ? out : sections;
}

export function lineText(line: Line): string {
  if (line.kind === "lyrics") return line.segments.map((s) => s.lyric).join("").replace(/\s+/g, " ").trim();
  if (line.kind === "comment" || line.kind === "tab") return line.text;
  return "";
}

export interface Slide {
  /** Position of the section in the arranged list (what ChartView renders). */
  pos: number;
  label: string;
  lines: string[];
}

/** Sections reduced to lyric text, for the audience lyrics display. */
export function lyricSlides(sections: Section[]): Slide[] {
  return sections
    .map((s, pos) => ({
      pos,
      label: s.label,
      lines: s.type === "tab" ? [] : s.lines.filter((l) => l.kind === "lyrics").map(lineText).filter(Boolean),
    }))
    .filter((s) => s.lines.length);
}

/** The sections in performance order (flow applied when asked). */
export function arrangeSong(content: string, flow: string | null | undefined, useFlow: boolean): Section[] {
  const parsed = parseChordPro(content);
  return useFlow ? applyFlow(parsed.sections, flow) : parsed.sections;
}

export function allChords(song: ParsedSong): string[] {
  const out: string[] = [];
  for (const s of song.sections)
    for (const l of s.lines)
      if (l.kind === "lyrics") for (const seg of l.segments) if (seg.chord && parseChord(seg.chord)) out.push(seg.chord);
  return out;
}

/** Build ChordPro text with metadata directives from song fields + body. */
export function toChordProFile(fields: {
  title: string; artist?: string; song_key?: string | null; tempo?: number | null;
  time_signature?: string | null; capo?: number | null; duration_sec?: number | null; ccli?: string | null;
}, body: string): string {
  const d: string[] = [`{title: ${fields.title}}`];
  if (fields.artist) d.push(`{artist: ${fields.artist}}`);
  if (fields.song_key) d.push(`{key: ${fields.song_key}}`);
  if (fields.tempo) d.push(`{tempo: ${fields.tempo}}`);
  if (fields.time_signature) d.push(`{time: ${fields.time_signature}}`);
  if (fields.capo) d.push(`{capo: ${fields.capo}}`);
  if (fields.duration_sec) {
    const m = Math.floor(fields.duration_sec / 60), s = fields.duration_sec % 60;
    d.push(`{duration: ${m}:${String(s).padStart(2, "0")}}`);
  }
  if (fields.ccli) d.push(`{ccli: ${fields.ccli}}`);
  return d.join("\n") + "\n\n" + body.trim() + "\n";
}

/** Rewrite every [chord] in a chart by `semitones`, spelled for `targetKey`. */
export function transposeContent(content: string, semitones: number, preferFlats: boolean): string {
  if (!semitones) return content;
  return content.replace(/\[([^\]]+)\]/g, (whole, chord: string) => {
    const t = transposeChord(chord.trim(), semitones, preferFlats);
    return t === chord.trim() && !parseChord(chord.trim()) ? whole : `[${t}]`;
  });
}
