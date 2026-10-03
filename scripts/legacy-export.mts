// Convert the old Flask app's musician.db into a Setlist Stage backup file (Import -> Restore backup).
// Usage: npx tsx scripts/legacy-export.mts path/to/musician.db out.json
import { DatabaseSync } from "node:sqlite";
import { randomUUID } from "node:crypto";
import { writeFileSync } from "node:fs";
import { importChart, fixMojibake } from "../src/lib/music/convert.ts";
import { guessKey, normalizeKey } from "../src/lib/music/chords.ts";
import { allChords, parseChordPro } from "../src/lib/music/chordpro.ts";

const [dbPath, outPath] = process.argv.slice(2);
if (!dbPath || !outPath) throw new Error("usage: legacy-export.mts musician.db out.json");
const sql = new DatabaseSync(dbPath, { readOnly: true });
const now = new Date().toISOString();
const base = () => ({ id: randomUUID(), created_at: now, updated_at: now, deleted_at: null, dirty: 1 });

type Row = Record<string, string | number | null>;
const songIds = new Map<number, string>();
const songs = (sql.prepare("select * from song where deleted_at is null").all() as Row[]).map((r) => {
  const chart = r.chord_chart ? importChart(String(r.chord_chart)) : null;
  const s = {
    ...base(),
    title: fixMojibake(String(r.title ?? "")).trim(),
    artist: fixMojibake(String(r.artist ?? "")).trim(),
    // song_key must be the key the chart is written in, so a chart's own key (or its chords) wins
    // over the old AI-guessed musical_key.
    song_key: chart?.body
      ? chart.meta.key ?? guessKey(allChords(parseChordPro(chart.body)))
      : r.musical_key ? normalizeKey(String(r.musical_key).replace(/\s*major$/i, "")) : null,
    tempo: (r.tempo_bpm as number) || null,
    time_signature: null,
    duration_sec: (r.duration_override_sec as number) || null,
    capo: 0,
    // The old app's AI tagger added "auto"/"ai"/"general" and decade tags that were often wrong.
    tags: String(r.tags ?? "").split(",").map((t) => t.trim().toLowerCase())
      .filter((t) => t && !["auto", "ai", "general"].includes(t) && !/^(\d0|\d{3}0)s$/.test(t)),
    genre: r.genre ? String(r.genre) : null,
    year: (r.release_year as number) || null,
    ccli: null,
    content: chart?.body ?? "",
    notes: null,
    flow: null,
    requestable: true,
  };
  return { legacyId: Number(r.id), song: s };
}).reduce<ReturnType<typeof base>[]>((kept, { legacyId, song }) => {
  // Merge duplicates (same title + artist): keep the copy that has a chart.
  const key = (x: { title: string; artist: string }) => `${x.title.toLowerCase()}|${x.artist.toLowerCase()}`;
  const dup = (kept as unknown as typeof song[]).find((k) => key(k) === key(song));
  if (dup) {
    if (!dup.content && song.content) Object.assign(dup, { ...song, id: dup.id });
    songIds.set(legacyId, dup.id);
    return kept;
  }
  songIds.set(legacyId, song.id);
  return [...kept, song];
}, []);

const setlistIds = new Map<number, string>();
const setlists = (sql.prepare("select * from setlist").all() as Row[]).map((r) => {
  const s = { ...base(), name: String(r.name ?? "Setlist"), event_date: null, venue: r.venue_type ? String(r.venue_type) : null, notes: r.notes ? String(r.notes) : null };
  setlistIds.set(Number(r.id), s.id);
  return s;
});

// Old setlists marked sections per row ("Set 1", "Break", "Encore"); turn changes into break rows.
const setlist_items: object[] = [];
for (const [oldId, newId] of setlistIds) {
  const rows = sql.prepare("select * from setlist_song where setlist_id = ? order by position").all(oldId) as Row[];
  let pos = 0;
  let lastSection: string | null = null;
  for (const r of rows) {
    const songId = songIds.get(Number(r.song_id));
    if (!songId) continue;
    const section = r.section_name ? String(r.section_name).trim() : null;
    if (section && section !== lastSection && (pos > 0 || !/^set 1$/i.test(section))) {
      setlist_items.push({ ...base(), setlist_id: newId, song_id: null, kind: "break", label: section, position: (pos += 1000), key_override: null, capo_override: null, notes: null });
    }
    if (section) lastSection = section;
    setlist_items.push({ ...base(), setlist_id: newId, song_id: songId, kind: "song", label: null, position: (pos += 1000), key_override: null, capo_override: null, notes: r.notes ? String(r.notes) : null });
  }
}

writeFileSync(outPath, JSON.stringify({ app: "setlist-stage", version: 1, exported_at: now, songs, setlists, setlist_items }, null, 1));
console.log(`${songs.length} songs, ${setlists.length} setlists, ${setlist_items.length} set items -> ${outPath}`);
