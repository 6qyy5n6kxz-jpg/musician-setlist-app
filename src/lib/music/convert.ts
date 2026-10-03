// Import any common chart format and normalize it to ChordPro:
//   - ChordPro (.cho/.chordpro/.pro), with metadata directives
//   - OnSong files (title/artist first lines, "Key: G" metadata, "Verse 1:" headers, inline [chords])
//   - Chords-over-lyrics text (Ultimate Guitar copy/paste, PDFs-to-text, emails)
//   - Ultimate Guitar markup ([ch]G[/ch], [tab]...[/tab])
import { isChordLineFiller, keyPrefersFlats, parseChord } from "./chords";
import { matchHeaderLine, sectionTypeFromLabel, transposeContent } from "./chordpro";

export interface ImportedChart {
  meta: {
    title?: string; artist?: string; key?: string; tempo?: number; time?: string; capo?: number;
    duration_sec?: number; ccli?: string; flow?: string; year?: number;
  };
  body: string;
}

const DIRECTIVE_META: Record<string, keyof ImportedChart["meta"]> = {
  t: "title", title: "title", st: "artist", subtitle: "artist", artist: "artist",
  key: "key", tempo: "tempo", time: "time", capo: "capo", duration: "duration_sec", ccli: "ccli", year: "year",
};

const ONSONG_META: Record<string, keyof ImportedChart["meta"]> = {
  title: "title", artist: "artist", author: "artist", key: "key", tempo: "tempo", bpm: "tempo",
  time: "time", "time signature": "time", capo: "capo", duration: "duration_sec", ccli: "ccli",
  flow: "flow", year: "year",
};

export function parseDuration(v: string): number | undefined {
  const m = /^(\d+):(\d{1,2})$/.exec(v.trim());
  if (m) return Number(m[1]) * 60 + Number(m[2]);
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? Math.round(n) : undefined;
}

function setMeta(meta: ImportedChart["meta"], key: keyof ImportedChart["meta"], value: string) {
  const v = value.trim();
  if (!v) return;
  switch (key) {
    case "tempo": case "capo": case "year": {
      const n = parseInt(v, 10);
      if (Number.isFinite(n)) meta[key] = n;
      break;
    }
    case "duration_sec": {
      const d = parseDuration(v);
      if (d) meta.duration_sec = d;
      break;
    }
    default:
      meta[key] = v;
  }
}

/** "[am]" -> "Am", "(G)" -> "G". Returns null if the token is not a chord. */
export function normalizeChordToken(token: string, allowLowercase: boolean): string | null {
  let t = token;
  const wrapped = /^[[(](.+)[\])]$/.exec(t);
  if (wrapped) t = wrapped[1];
  if (parseChord(t)) return t;
  if ((allowLowercase || wrapped) && /^[a-g]/.test(t)) {
    const cap = t[0].toUpperCase() + t.slice(1);
    if (parseChord(cap)) return cap;
  }
  return null;
}

/** A line made only of chords (and bar lines / repeat marks). Brackets around chords are allowed. */
export function isChordLine(line: string): boolean {
  const tokens = line.trim().split(/\s+/).filter(Boolean);
  if (!tokens.length) return false;
  let chords = 0;
  for (const tok of tokens) {
    if (normalizeChordToken(tok, false)) chords++;
    else if (!isChordLineFiller(tok)) return false;
  }
  return chords > 0;
}

const TAB_LINE = /^\s*[eBGDAEbgdaE]?\s*\|[-0-9hpbrsx/\\~|.()^ ]{4,}\|?\s*$/;

/** Place the chords from a chord line into the lyric line below it, by column. */
export function mergeChordLine(chordLine: string, lyric: string): string {
  const inserts: { col: number; chord: string }[] = [];
  const re = /\S+/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(chordLine))) {
    const chord = normalizeChordToken(m[0], false);
    if (chord) inserts.push({ col: m.index, chord });
  }
  let out = lyric;
  for (const { col, chord } of inserts.sort((a, b) => b.col - a.col)) {
    if (col > out.length) out = out.padEnd(col, " ");
    out = out.slice(0, col) + `[${chord}]` + out.slice(col);
  }
  // Runs of spaces were only there to line chords up; once merged they're noise.
  const lead = /^\s*/.exec(out)![0];
  return lead + out.slice(lead.length).replace(/ {2,}/g, " ").replace(/\s+$/, "");
}

function chordOnlyLine(line: string): string {
  return line
    .trim()
    .split(/\s+/)
    .map((tok) => {
      const c = normalizeChordToken(tok, false);
      return c ? `[${c}]` : tok;
    })
    .join(" ");
}

/** Fix chords already inline in brackets: "[am]" -> "[Am]". */
function normalizeInlineChords(line: string): string {
  return line.replace(/\[([^\]]+)\]/g, (whole, inner: string) => {
    const c = normalizeChordToken(inner.trim(), true);
    return c ? `[${c}]` : whole;
  });
}

function sectionDirectiveName(label: string): string {
  const type = sectionTypeFromLabel(label) ?? "verse";
  return type === "none" || type === "tab" ? "verse" : type;
}

/** Repair UTF-8 text that was decoded as Latin-1 ("Iâ€™ve" -> "I’ve"). */
export function fixMojibake(text: string): string {
  if (!/[\u00c2-\u00f4][\u0080-\u00bf]/.test(text)) return text;
  return text.replace(/[\u00c2-\u00f4][\u0080-\u00bf]{1,3}/g, (seq) => {
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(Uint8Array.from(seq, (c) => c.charCodeAt(0)));
    } catch {
      return seq;
    }
  });
}

export function importChart(input: string, filename = ""): ImportedChart {
  const meta: ImportedChart["meta"] = {};
  // Ultimate Guitar writes chords as played with the capo (shapes) but its key is the sounding key.
  const ultimateGuitar = /\[ch\]/i.test(input);
  let text = fixMojibake(input)
    .replace(/\r\n?/g, "\n")
    .replace(/\[ch\](.*?)\[\/ch\]/gi, "$1") // Ultimate Guitar chord tags
    .replace(/\[\/?tab\]/gi, "")
    .replace(/ /g, " ");

  // 1) ChordPro metadata directives anywhere
  text = text
    .split("\n")
    .filter((line) => {
      const d = /^\s*\{\s*([a-z_]+)\s*:\s*(.*?)\s*\}\s*$/i.exec(line);
      if (d && !line.trim().startsWith("{{")) {
        const key = DIRECTIVE_META[d[1].toLowerCase()];
        if (key) {
          setMeta(meta, key, d[2]);
          return false;
        }
      }
      return true;
    })
    .join("\n");

  let lines = text.split("\n").map((l) => l.replace(/\s+$/, ""));

  // 2) OnSong-style header: "Title", "Artist", then "Key: G" style lines, before the first section
  const isOnSong = /\.onsong$/i.test(filename) || lines.slice(0, 8).some((l) => /^\s*(key|tempo|capo|time|ccli|flow)\s*:/i.test(l));
  if (isOnSong) {
    let i = 0;
    const header: string[] = [];
    while (i < lines.length && i < 15) {
      const l = lines[i].trim();
      if (!l) { if (header.length) { i++; break; } i++; continue; }
      if (matchHeaderLine(l) || isChordLine(l) || /\[[^\]]+\]/.test(l)) break;
      const kv = /^([A-Za-z ]+?)\s*:\s*(.+)$/.exec(l);
      const key = kv ? ONSONG_META[kv[1].toLowerCase()] : undefined;
      if (key) setMeta(meta, key, kv![2]);
      else if (!meta.title) meta.title = l;
      else if (!meta.artist) meta.artist = l;
      else break;
      header.push(l);
      i++;
    }
    lines = lines.slice(i);
  }

  // 3) Body: headers -> section directives, chord lines merged into lyrics, tabs wrapped
  const out: string[] = [];
  let openSection: string | null = null;
  let inTab = false;
  const closeSection = () => {
    if (openSection) out.push(`{end_of_${openSection}}`);
    openSection = null;
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (TAB_LINE.test(line)) {
      if (!inTab) { out.push("{start_of_tab}"); inTab = true; }
      out.push(line);
      continue;
    }
    if (inTab) { out.push("{end_of_tab}"); inTab = false; }

    const header = matchHeaderLine(trimmed);
    if (header) {
      closeSection();
      if (out.length && out[out.length - 1] !== "") out.push("");
      const name = sectionDirectiveName(header);
      out.push(`{start_of_${name}: ${header.replace(/:$/, "")}}`);
      openSection = name;
      continue;
    }

    if (!trimmed) {
      out.push("");
      continue;
    }

    if (trimmed.startsWith("{")) {
      out.push(trimmed);
      continue;
    }

    if (isChordLine(line)) {
      const next = lines[i + 1];
      const nextIsLyric =
        next !== undefined && next.trim() !== "" && !isChordLine(next) && !matchHeaderLine(next.trim()) &&
        !next.trim().startsWith("{") && !TAB_LINE.test(next) && !/\[[^\]]+\]/.test(next);
      if (nextIsLyric) {
        out.push(mergeChordLine(line, next));
        i++;
      } else {
        out.push(chordOnlyLine(line));
      }
      continue;
    }

    out.push(normalizeInlineChords(line));
  }
  if (inTab) out.push("{end_of_tab}");
  closeSection();

  // Collapse 3+ blank lines; blank lines right after a section start are noise
  let body = out
    .join("\n")
    .replace(/(\{start_of_[a-z]+(?::[^}]*)?\})\n+/g, "$1\n")
    .replace(/\n+(\{end_of_[a-z]+\})/g, "\n$1")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  // This app stores sounding chords and derives capo shapes, so lift UG's shapes by the capo.
  if (ultimateGuitar && meta.capo) body = transposeContent(body, meta.capo, keyPrefersFlats(meta.key));

  return { meta, body };
}

/** Plain "chords over lyrics" text from ChordPro, for copy/paste and printing in monospace. */
export function chordProToChordsOverLyrics(body: string): string {
  const out: string[] = [];
  for (const line of body.split("\n")) {
    const t = line.trim();
    const dir = /^\{([^:}]+)(?::\s*(.*))?\}$/.exec(t);
    if (dir) {
      const name = dir[1].toLowerCase();
      if (name.startsWith("start_of_") || ["soc", "sov", "sob"].includes(name)) {
        out.push((dir[2] || name.replace("start_of_", "")).replace(/^\w/, (c) => c.toUpperCase()) + ":");
      } else if (["c", "comment", "ci", "cb"].includes(name)) {
        out.push(`(${dir[2] ?? ""})`);
      }
      continue;
    }
    if (!/\[[^\]]+\]/.test(line)) { out.push(line); continue; }
    let chords = "", lyrics = "";
    const parts = line.split(/\[([^\]]*)\]/);
    lyrics = parts[0];
    chords = " ".repeat(parts[0].length);
    for (let i = 1; i < parts.length; i += 2) {
      const chord = parts[i];
      const lyric = parts[i + 1] ?? "";
      if (chords.length > lyrics.length) lyrics = lyrics.padEnd(chords.length, " ");
      chords = chords.padEnd(lyrics.length, " ") + chord + " ";
      lyrics += lyric;
    }
    out.push(chords.replace(/\s+$/, ""));
    if (lyrics.trim()) out.push(lyrics.replace(/\s+$/, ""));
  }
  return out.join("\n");
}
