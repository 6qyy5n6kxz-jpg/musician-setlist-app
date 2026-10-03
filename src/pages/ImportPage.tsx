import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { IconBack } from "../components/Icons";
import { addSongFile, blankSong, db, live, saveRow, type Setlist, type SetlistItem, type Song } from "../lib/db";
import { importChart, parseDuration } from "../lib/music/convert";

interface Candidate {
  key: string;
  song: Song;
  source: string;
  file?: File; // PDF to attach
  duplicateOf?: string;
  include: boolean;
}

interface Backup {
  app: "setlist-stage";
  version: 1;
  songs: Song[];
  setlists?: Setlist[];
  setlist_items?: SetlistItem[];
}

const norm = (s: string) => s.trim().toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

function songFromText(text: string, filename: string): Song {
  const { meta, body } = importChart(text, filename);
  const fallbackTitle = filename.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim();
  return blankSong({
    title: meta.title || fallbackTitle || "Untitled",
    artist: meta.artist || "",
    song_key: meta.key || null,
    tempo: meta.tempo ?? null,
    time_signature: meta.time || null,
    capo: meta.capo ?? 0,
    duration_sec: meta.duration_sec ?? null,
    ccli: meta.ccli || null,
    flow: meta.flow || null,
    year: meta.year ?? null,
    content: body,
  });
}

/** CSV with a header row: Title, Artist, Key, Tempo/BPM, Genre, Tags, Year, Duration. */
function songsFromCsv(text: string): Song[] {
  const rows = parseCsv(text);
  if (rows.length < 2) return [];
  const header = rows[0].map((h) => h.trim().toLowerCase());
  const col = (...names: string[]) => header.findIndex((h) => names.some((n) => h.startsWith(n)));
  const c = {
    title: col("title", "song"), artist: col("artist"), key: col("key", "musical"), tempo: col("tempo", "bpm"),
    genre: col("genre"), tags: col("tags"), year: col("release", "year"), duration: col("duration", "length"),
  };
  if (c.title < 0) return [];
  return rows.slice(1).filter((r) => r[c.title]?.trim()).map((r) => blankSong({
    title: r[c.title].trim(),
    artist: c.artist >= 0 ? r[c.artist]?.trim() ?? "" : "",
    song_key: c.key >= 0 ? r[c.key]?.trim() || null : null,
    tempo: c.tempo >= 0 ? parseInt(r[c.tempo], 10) || null : null,
    genre: c.genre >= 0 ? r[c.genre]?.trim() || null : null,
    tags: c.tags >= 0 ? (r[c.tags] ?? "").split(/[,;]/).map((t) => t.trim().toLowerCase()).filter(Boolean) : [],
    year: c.year >= 0 ? parseInt(r[c.year], 10) || null : null,
    duration_sec: c.duration >= 0 && r[c.duration] ? parseDuration(r[c.duration]) ?? null : null,
  }));
}

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field); rows.push(row); row = []; field = "";
    } else field += ch;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows;
}

export function ImportPage() {
  const navigate = useNavigate();
  const [paste, setPaste] = useState("");
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [backup, setBackup] = useState<Backup | null>(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<string | null>(null);

  const markDuplicates = async (list: Candidate[]) => {
    const existing = live(await db.songs.toArray());
    const byKey = new Map(existing.map((s) => [norm(s.title) + "|" + norm(s.artist), s.id]));
    return list.map((c) => {
      const dup = byKey.get(norm(c.song.title) + "|" + norm(c.song.artist));
      return { ...c, duplicateOf: dup, include: !dup };
    });
  };

  const addFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setDone(null);
    const list: Candidate[] = [];
    for (const f of Array.from(files)) {
      if (f.name.startsWith("._")) continue;
      if (/\.json$/i.test(f.name)) {
        try {
          const data = JSON.parse(await f.text()) as Backup;
          if (data.app === "setlist-stage" && Array.isArray(data.songs)) { setBackup(data); continue; }
        } catch { /* fall through */ }
      }
      if (/\.pdf$/i.test(f.name) || f.type === "application/pdf") {
        const title = f.name.replace(/\.pdf$/i, "").replace(/[_]+/g, " ");
        const [t, a] = title.split(/\s+-\s+/);
        list.push({ key: crypto.randomUUID(), song: blankSong({ title: t.trim(), artist: (a ?? "").trim() }), source: f.name, file: f, include: true });
        continue;
      }
      const text = await f.text();
      if (/\.csv$/i.test(f.name)) {
        songsFromCsv(text).forEach((s) => list.push({ key: s.id, song: s, source: f.name, include: true }));
        continue;
      }
      const s = songFromText(text, f.name);
      list.push({ key: s.id, song: s, source: f.name, include: true });
    }
    setCandidates(await markDuplicates([...candidates, ...list]));
  };

  const addPaste = async () => {
    if (!paste.trim()) return;
    const s = songFromText(paste, "");
    // pasted text without metadata: first line is usually the title
    if (s.title === "Untitled") {
      const first = paste.trim().split("\n")[0].trim();
      if (first.length < 80) s.title = first;
    }
    setCandidates(await markDuplicates([...candidates, { key: s.id, song: s, source: "Pasted text", include: true }]));
    setPaste("");
  };

  const runImport = async () => {
    setBusy(true);
    let n = 0;
    for (const c of candidates.filter((c) => c.include)) {
      const song = c.duplicateOf ? { ...c.song, id: c.duplicateOf } : c.song;
      if (c.duplicateOf) {
        const old = await db.songs.get(c.duplicateOf);
        if (old) Object.assign(song, { created_at: old.created_at, tags: song.tags.length ? song.tags : old.tags });
      }
      await saveRow(db.songs, song);
      if (c.file) await addSongFile(song.id, c.file);
      n++;
    }
    setCandidates([]);
    setBusy(false);
    setDone(`Imported ${n} song${n === 1 ? "" : "s"}.`);
  };

  const restoreBackup = async () => {
    if (!backup) return;
    setBusy(true);
    const existing = new Set((await db.songs.toArray()).map((s) => s.id));
    let songs = 0;
    for (const s of backup.songs) {
      if (existing.has(s.id)) continue;
      await saveRow(db.songs, { ...blankSong(), ...s, deleted_at: null });
      songs++;
    }
    const sets = new Set((await db.setlists.toArray()).map((s) => s.id));
    for (const sl of backup.setlists ?? []) if (!sets.has(sl.id)) await saveRow(db.setlists, { ...sl, deleted_at: null, dirty: 1 });
    const items = new Set((await db.setlist_items.toArray()).map((s) => s.id));
    for (const it of backup.setlist_items ?? []) if (!items.has(it.id)) await saveRow(db.setlist_items, { ...it, deleted_at: null, dirty: 1 });
    setBusy(false);
    setBackup(null);
    setDone(`Restored ${songs} songs and ${backup.setlists?.length ?? 0} setlists (existing items were left alone).`);
  };

  return (
    <div className="page">
      <div className="row" style={{ marginBottom: 14 }}>
        <button className="btn ghost icon" onClick={() => navigate(-1)} aria-label="Back"><IconBack /></button>
        <h1 className="grow">Import</h1>
      </div>

      {done && <div className="card" style={{ marginBottom: 12, borderColor: "var(--ok)" }}>{done} <a href="#/">Go to songs</a></div>}

      <div className="editor-grid">
        <div className="card stack">
          <h2 style={{ fontSize: "1.1rem" }}>Paste a chart</h2>
          <p className="small dim">From Ultimate Guitar, a website, an email or a PDF — chords above lyrics, ChordPro or OnSong text all work.</p>
          <textarea className="code-area" style={{ minHeight: 260 }} value={paste} onChange={(e) => setPaste(e.target.value)}
            placeholder={"Wonderwall\nOasis\n\n[Verse 1]\nEm7        G\nToday is gonna be the day"} spellCheck={false} />
          <button className="btn primary" onClick={addPaste} disabled={!paste.trim()}>Add to import list</button>
        </div>
        <div className="card stack">
          <h2 style={{ fontSize: "1.1rem" }}>Import files</h2>
          <p className="small dim">
            ChordPro (.cho, .chopro, .pro), OnSong (.onsong), text charts (.txt), PDFs (title from the filename,
            “Title - Artist.pdf”), a CSV song list, or a Setlist Stage backup (.json). Select many at once.
          </p>
          <label className="btn">
            Choose files…
            <input type="file" multiple hidden accept=".cho,.chopro,.chordpro,.pro,.crd,.onsong,.txt,.pdf,.csv,.json,text/plain,application/pdf"
              onChange={(e) => { void addFiles(e.target.files); e.target.value = ""; }} />
          </label>
          {backup && (
            <div className="card" style={{ background: "var(--bg-sunken)" }}>
              <div style={{ fontWeight: 700 }}>Backup file: {backup.songs.length} songs, {backup.setlists?.length ?? 0} setlists</div>
              <p className="small dim">Adds anything not already on this device. Nothing is overwritten.</p>
              <button className="btn primary" onClick={restoreBackup} disabled={busy}>Restore backup</button>
            </div>
          )}
        </div>
      </div>

      {candidates.length > 0 && (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="row" style={{ marginBottom: 10 }}>
            <h2 className="grow" style={{ fontSize: "1.1rem" }}>Ready to import ({candidates.filter((c) => c.include).length} of {candidates.length})</h2>
            <button className="btn" onClick={() => setCandidates([])}>Clear</button>
            <button className="btn primary" onClick={runImport} disabled={busy || !candidates.some((c) => c.include)}>Import</button>
          </div>
          <ul className="list">
            {candidates.map((c) => (
              <li key={c.key} className="list-item">
                <input type="checkbox" style={{ width: 22, height: 22 }} checked={c.include}
                  onChange={(e) => setCandidates((all) => all.map((x) => (x.key === c.key ? { ...x, include: e.target.checked } : x)))} />
                <div className="grow">
                  <input className="input" style={{ minHeight: 34, marginBottom: 4 }} value={c.song.title}
                    onChange={(e) => setCandidates((all) => all.map((x) => (x.key === c.key ? { ...x, song: { ...x.song, title: e.target.value } } : x)))} />
                  <div className="small dim">
                    {c.song.artist || "Unknown artist"}{c.song.song_key ? ` · ${c.song.song_key}` : ""} · {c.source}
                    {c.file ? " · PDF attached" : c.song.content ? ` · ${c.song.content.split("\n").length} lines` : " · no chart"}
                    {c.duplicateOf && <span className="chip accent" style={{ marginLeft: 6 }}>already in library — check to replace</span>}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
