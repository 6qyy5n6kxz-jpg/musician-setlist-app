// Duo gear: Numa X Piano 73 (piano songs), Neural DSP Nano Cortex (electric guitar songs) and
// BeatBuddy (any song). The MIDI Captain does the switching; the app stores what each song needs,
// shows it on stage, and suggests settings from the performer's own preset lists.

export type Instrument = "piano" | "electric" | "acoustic";
export type LeadVocal = "kendra" | "devin" | "both";

export const INSTRUMENT_LABELS: Record<Instrument, string> = { piano: "Piano", electric: "Electric", acoustic: "Acoustic" };
export const VOCAL_LABELS: Record<LeadVocal, string> = { kendra: "Kendra", devin: "Devin", both: "Both" };

export type PianoCategory = "grand" | "bright" | "upright" | "rhodes" | "wurli" | "clav" | "organ" | "strings" | "synth";
export type GuitarCategory = "clean" | "edge" | "crunch" | "highgain" | "ambient";
export type BeatCategory = "pop" | "rock" | "country" | "train" | "ballad68" | "funk" | "blues" | "latin" | "swing" | "reggae";

export const PIANO_CATEGORIES: Record<PianoCategory, string> = {
  grand: "Grand piano", bright: "Bright / rock piano", upright: "Upright / honky-tonk", rhodes: "Rhodes / EP",
  wurli: "Wurlitzer", clav: "Clav", organ: "Organ", strings: "Strings / pad", synth: "Synth",
};
export const GUITAR_CATEGORIES: Record<GuitarCategory, string> = {
  clean: "Clean", edge: "Edge of breakup", crunch: "Crunch", highgain: "High gain", ambient: "Ambient / clean with effects",
};
export const BEAT_CATEGORIES: Record<BeatCategory, string> = {
  pop: "Pop", rock: "Rock", country: "Country", train: "Country train beat", ballad68: "6/8 ballad", funk: "Funk / R&B",
  blues: "Blues / shuffle", latin: "Latin", swing: "Swing / jazz", reggae: "Reggae",
};

/** One preset on a device, as the performer has it saved. */
export interface Preset {
  id: string;
  /** Program Change number the MIDI Captain sends (as shown on the device). */
  program: number | null;
  name: string;
  category: string;
  /** Which MIDI Captain switch/page calls it up, e.g. "Page 2 · B" (free text). */
  captain?: string;
}

export interface BeatPreset extends Preset {
  /** BeatBuddy folder number (bank) — songs are picked inside a folder. */
  folder: number | null;
}

export interface GearLibrary {
  numa?: Preset[];
  cortex?: Preset[];
  beatbuddy?: BeatPreset[];
}

/** What one song uses. Stored on songs.gear. */
export interface SongGear {
  numa?: string | null;      // Preset.id
  cortex?: string | null;    // Preset.id
  beatbuddy?: string | null; // BeatPreset.id
  /** BeatBuddy tempo for this song (defaults to the song's tempo). */
  bbTempo?: number | null;
}

interface SongFacts {
  title: string;
  genre: string | null;
  year: number | null;
  tempo: number | null;
  time_signature: string | null;
  tags: string[];
  instrument: Instrument | null;
  lead_vocal: LeadVocal | null;
}

const has = (s: string | null | undefined, re: RegExp) => !!s && re.test(s);
const isHoliday = (f: SongFacts) =>
  f.tags.includes("holiday") || has(f.genre, /holiday|christmas/i) ||
  /christmas|xmas|santa|jingle|sleigh|reindeer|mistletoe|snow|holiday|noel|silent night|o holy night|winter wonderland|lang syne|rudolph|frosty/i.test(f.title);

/** Piano sound that usually suits a song. */
export function suggestPianoCategory(f: SongFacts): { category: PianoCategory; why: string } {
  const g = f.genre ?? "";
  const t = f.tempo ?? 0;
  const y = f.year ?? 0;
  if (isHoliday(f)) return { category: "grand", why: "holiday standard" };
  if (has(g, /gospel|christian|worship/i)) return { category: "organ", why: "gospel" };
  if (has(g, /funk|soul|r&b|motown|disco/i)) return t >= 100 ? { category: "clav", why: "uptempo funk/soul" } : { category: "rhodes", why: "soul ballad" };
  if (has(g, /country|americana|bluegrass|folk/i)) return t >= 110 ? { category: "upright", why: "uptempo country" } : { category: "grand", why: "country ballad" };
  if (has(g, /blues/i)) return { category: "upright", why: "blues" };
  if (has(g, /jazz|vocal|standards|easy listening/i)) return { category: "grand", why: "jazz / standards" };
  if (y >= 1970 && y <= 1985 && has(g, /pop|rock|soft/i) && (!t || t < 110)) return { category: "rhodes", why: "70s/80s soft rock" };
  if (y >= 1980 && y <= 1989 && has(g, /pop|dance|new wave/i) && t >= 110) return { category: "synth", why: "80s pop" };
  if (has(g, /rock/i) && t >= 115) return { category: "bright", why: "uptempo rock" };
  if (t && t < 90) return { category: "grand", why: "ballad" };
  return { category: "bright", why: "pop / rock" };
}

/** Nano Cortex sound that usually suits a song. */
export function suggestGuitarCategory(f: SongFacts): { category: GuitarCategory; why: string } {
  const g = f.genre ?? "";
  const t = f.tempo ?? 0;
  const y = f.year ?? 0;
  if (isHoliday(f)) return { category: "clean", why: "holiday" };
  if (has(g, /metal|hard rock/i)) return { category: "highgain", why: "hard rock" };
  if (has(g, /country|americana|bluegrass/i)) return t && t < 80 ? { category: "clean", why: "country ballad" } : { category: "edge", why: "country twang" };
  if (has(g, /alternative|grunge|punk/i) || (y >= 1990 && y <= 2000 && has(g, /rock/i))) return { category: "crunch", why: "90s / alternative rock" };
  if (has(g, /rock/i)) return t && t < 85 ? { category: "edge", why: "rock ballad" } : { category: "crunch", why: "classic rock" };
  if (has(g, /blues/i)) return { category: "edge", why: "blues" };
  if (t && t < 80) return { category: "ambient", why: "slow song" };
  return { category: "clean", why: "pop / R&B" };
}

/** BeatBuddy style that usually suits a song. */
export function suggestBeatCategory(f: SongFacts): { category: BeatCategory; why: string } {
  const g = f.genre ?? "";
  const t = f.tempo ?? 0;
  if (/^(6|12)\/8$/.test(f.time_signature ?? "")) return { category: "ballad68", why: `${f.time_signature} time` };
  if (has(g, /reggae/i)) return { category: "reggae", why: "reggae" };
  if (has(g, /latin/i)) return { category: "latin", why: "latin" };
  if (has(g, /jazz|swing|standards|vocal/i) || (isHoliday(f) && (f.year ?? 2000) < 1970)) return { category: "swing", why: "swing / standards" };
  if (has(g, /funk|soul|r&b|disco|motown/i)) return { category: "funk", why: "funk / soul" };
  if (has(g, /blues/i)) return { category: "blues", why: "blues shuffle" };
  if (has(g, /country|americana|bluegrass/i)) return t >= 140 ? { category: "train", why: "fast country" } : { category: "country", why: "country" };
  if (has(g, /rock|alternative|grunge|punk/i)) return { category: "rock", why: "rock" };
  return { category: "pop", why: "pop" };
}

export interface GearSuggestion {
  numa?: { preset?: Preset; category: PianoCategory; why: string };
  cortex?: { preset?: Preset; category: GuitarCategory; why: string };
  beatbuddy?: { preset?: BeatPreset; category: BeatCategory; why: string };
}

/**
 * Suggest gear for a song from the performer's library. Prefers what they already chose for songs
 * by the same artist; otherwise the first preset in the suggested category.
 */
export function suggestGear(
  f: SongFacts & { artist: string },
  lib: GearLibrary,
  history: { artist: string; gear: SongGear }[] = [],
): GearSuggestion {
  const out: GearSuggestion = {};
  const sameArtist = history.filter((h) => h.artist && h.artist.toLowerCase() === f.artist.toLowerCase());
  const usual = (key: keyof SongGear) => {
    const counts = new Map<string, number>();
    for (const h of sameArtist) {
      const v = h.gear[key];
      if (typeof v === "string") counts.set(v, (counts.get(v) ?? 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
  };
  if (f.instrument === "piano") {
    const s = suggestPianoCategory(f);
    const prior = lib.numa?.find((p) => p.id === usual("numa"));
    out.numa = { ...s, preset: prior ?? lib.numa?.find((p) => p.category === s.category), why: prior ? `your usual for ${f.artist}` : s.why };
  }
  if (f.instrument === "electric") {
    const s = suggestGuitarCategory(f);
    const prior = lib.cortex?.find((p) => p.id === usual("cortex"));
    out.cortex = { ...s, preset: prior ?? lib.cortex?.find((p) => p.category === s.category), why: prior ? `your usual for ${f.artist}` : s.why };
  }
  const b = suggestBeatCategory(f);
  out.beatbuddy = { ...b, preset: lib.beatbuddy?.find((p) => p.category === b.category) };
  return out;
}

// ------------------------------------------------------------------ signature-show fit
export interface ShowRule {
  tag: string;
  label: string;
  test: (f: SongFacts & { artist: string }) => string | null;
}

/** Rules that suggest which signature shows a song fits (by the shows' tags). */
export const SHOW_RULES: ShowRule[] = [
  { tag: "holiday", label: "Home for the Holidays", test: (f) => (isHoliday(f) ? "holiday song" : null) },
  {
    tag: "women of country", label: "Women of Country",
    test: (f) => (has(f.genre, /country/i) && (f.lead_vocal === "kendra" || f.lead_vocal === "both") ? "country, Kendra sings lead" : null),
  },
  {
    tag: "90s acoustic", label: "90's Acoustic Rewind",
    test: (f) => (f.year && f.year >= 1990 && f.year <= 1999 && has(f.genre, /rock|alternative|pop|grunge/i) ? `${f.year} ${f.genre?.toLowerCase()}` : null),
  },
  { tag: "piano bar", label: "Piano Bar Classics", test: (f) => (f.instrument === "piano" && !isHoliday(f) ? "piano song" : null) },
  {
    tag: "americana", label: "Americana & Country Roads",
    test: (f) => (has(f.genre, /americana|folk|singer|bluegrass/i) || (has(f.genre, /country/i) && f.lead_vocal === "devin") ? "americana / country" : null),
  },
];

export function suggestShows(f: SongFacts & { artist: string }): { tag: string; label: string; why: string }[] {
  return SHOW_RULES.flatMap((r) => {
    if (f.tags.includes(r.tag)) return [];
    const why = r.test(f);
    return why ? [{ tag: r.tag, label: r.label, why }] : [];
  });
}

// ------------------------------------------------------------------ set balance
export interface BalanceStats {
  vocals: Record<LeadVocal | "unset", number>;
  instruments: Record<Instrument | "unset", number>;
  /** Back-to-back songs that need a different instrument. */
  switches: number;
}

export function setBalance(songs: { instrument: Instrument | null; lead_vocal: LeadVocal | null }[]): BalanceStats {
  const vocals = { kendra: 0, devin: 0, both: 0, unset: 0 };
  const instruments = { piano: 0, electric: 0, acoustic: 0, unset: 0 };
  let switches = 0;
  songs.forEach((s, i) => {
    vocals[s.lead_vocal ?? "unset"]++;
    instruments[s.instrument ?? "unset"]++;
    const prev = songs[i - 1];
    if (prev?.instrument && s.instrument && prev.instrument !== s.instrument) switches++;
  });
  return { vocals, instruments, switches };
}

export function presetLabel(p: Preset | undefined | null): string {
  if (!p) return "";
  return `${p.program !== null && p.program !== undefined ? `${p.program} · ` : ""}${p.name}`;
}
