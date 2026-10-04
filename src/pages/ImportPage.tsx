import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ChartView } from "../components/ChartView";
import { IconBack } from "../components/Icons";
import { useSyncStatus } from "../lib/sync";
import { addSongFile, blankSong, db, live, saveRow, type Setlist, type SetlistItem, type Song } from "../lib/db";
import { importChart, parseDuration } from "../lib/music/convert";
import { expandZips } from "../lib/zipImport";
import { guessKey } from "../lib/music/chords";
import { allChords, parseChordPro } from "../lib/music/chordpro";

interface Candidate {
  key: string;
  song: Song;
  source: string;
  file?: File; // PDF to attach
  duplicateOf?: string;
  /** The existing song has no chart, so this import fills it in (keeps its setlists, tags, etc.). */
  fillsChart?: boolean;
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
    // The chart decides the written key: its {key}, else the chords themselves.
    song_key: meta.key || guessKey(allChords(parseChordPro(body))),
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
  const { userEmail } = useSyncStatus();
  const standalone = window.matchMedia("(display-mode: standalone)").matches || !!(navigator as { standalone?: boolean }).standalone;
  const [paste, setPaste] = useState("");
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [backup, setBackup] = useState<Backup | null>(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<string | null>(null);

  // Arriving from the "Send to Stage" shortcut: #/import?text=… (the chart travels in the URL fragment,
  // which never leaves the device).
  const [params, setParams] = useSearchParams();
  const [incomingError, setIncomingError] = useState<string | null>(null);
  const handled = useRef(false);
  useEffect(() => {
    if (handled.current) return;
    const text = params.get("text");
    const err = params.get("error");
    if (!text && !err) return;
    handled.current = true;
    if (err) setIncomingError(err);
    if (text) void addText(text, "Ultimate Guitar (shared)");
    setParams({}, { replace: true });
  }, [params]); // eslint-disable-line react-hooks/exhaustive-deps

  const addText = async (text: string, source: string) => {
    const s = songFromText(text, "");
    if (s.title === "Untitled") {
      const first = text.trim().split("\n")[0].trim();
      if (first.length < 80) s.title = first;
    }
    const marked = await markDuplicates([{ key: s.id, song: s, source, include: true }]);
    setCandidates((prev) => [...prev, ...marked]);
    setDone(null);
  };

  const pasteClipboard = async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text.trim()) await addText(text, "Clipboard");
      else setIncomingError("The clipboard is empty — copy the chart first.");
    } catch {
      setIncomingError("This browser blocked clipboard access — paste into the box below instead.");
    }
  };

  const markDuplicates = async (list: Candidate[]) => {
    const existing = live(await db.songs.toArray());
    const byKey = new Map(existing.map((s) => [norm(s.title) + "|" + norm(s.artist), s]));
    const byTitle = new Map<string, Song[]>();
    for (const s of existing) byTitle.set(norm(s.title), [...(byTitle.get(norm(s.title)) ?? []), s]);
    return list.map((c) => {
      // Same title + artist, or the only song with that title (artist names vary: "ft.", "&", "The")
      const sameTitle = byTitle.get(norm(c.song.title));
      const dup = byKey.get(norm(c.song.title) + "|" + norm(c.song.artist)) ?? (sameTitle?.length === 1 ? sameTitle[0] : undefined);
      const fillsChart = !!dup && !dup.content.trim() && !!c.song.content.trim();
      return { ...c, duplicateOf: dup?.id, fillsChart, include: !dup || fillsChart };
    });
  };

  const addFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setDone(null);
    const list: Candidate[] = [];
    if (Array.from(files).some((f) => /\.backup$/i.test(f.name))) {
      setIncomingError("That's a full OnSong backup (it holds OnSong's own database). Send it to your Mac and it can be converted there — or export your library from OnSong as ChordPro and import that zip here.");
    }
    let expanded: File[];
    try {
      expanded = await expandZips(Array.from(files).filter((f) => !/\.backup$/i.test(f.name)));
    } catch {
      setIncomingError("Couldn't open that zip file.");
      return;
    }
    for (const f of expanded) {
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
    await addText(paste, "Pasted text");
    setPaste("");
  };

  const runImport = async () => {
    setBusy(true);
    let n = 0;
    for (const c of candidates.filter((c) => c.include)) {
      let song = c.song;
      const old = c.duplicateOf ? await db.songs.get(c.duplicateOf) : undefined;
      if (old) {
        // Keep the existing song (setlists, tags, karaoke, notes, length…) and bring in the chart.
        song = {
          ...old,
          content: c.song.content || old.content,
          song_key: c.song.content ? c.song.song_key ?? old.song_key : old.song_key,
          capo: c.song.content ? c.song.capo : old.capo,
          tempo: old.tempo ?? c.song.tempo,
          time_signature: old.time_signature ?? c.song.time_signature,
          duration_sec: old.duration_sec ?? c.song.duration_sec,
          flow: old.flow ?? c.song.flow,
          timings: c.song.content && c.song.content !== old.content ? null : old.timings,
        };
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
        <button className="btn" onClick={pasteClipboard}>Paste from clipboard</button>
      </div>

      {done && <div className="card" style={{ marginBottom: 12, borderColor: "var(--ok)" }}>{done} <a href="#/">Go to songs</a></div>}
      {incomingError && <div className="card" style={{ marginBottom: 12, borderColor: "var(--danger)" }}>{incomingError}</div>}
      {candidates.length > 0 && !userEmail && !standalone && (
        <div className="sticky-note">
          This opened in Safari, which keeps its own copy of the app. <a href="#/settings">Sign in here once</a> and anything you
          import syncs to the home-screen app automatically.
        </div>
      )}
      {candidates.length > 0 && (
        <div className="card" style={{ marginBottom: 16, borderColor: "var(--accent)" }}>
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
                    {c.fillsChart && <span className="chip accent" style={{ marginLeft: 6 }}>adds the chart to your existing song</span>}
                    {c.duplicateOf && !c.fillsChart && <span className="chip accent" style={{ marginLeft: 6 }}>already in library — check to replace its chart</span>}
                  </div>
                </div>
              </li>
            ))}
          </ul>
          {candidates.length === 1 && candidates[0].song.content && (
            <div className="preview-pane" style={{ marginTop: 12, maxHeight: "45vh" }}>
              <ChartView sections={parseChordPro(candidates[0].song.content).sections} songKey={candidates[0].song.song_key}
                transpose={0} capo={0} showChords nashville={false} columns={1} fontScale={0.8} />
            </div>
          )}
        </div>
      )}

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
            “Title - Artist.pdf”), a CSV song list, a Setlist Stage backup (.json), or a <strong>.zip</strong> of any of these
            (e.g. your whole OnSong library exported as ChordPro). Select many at once.
          </p>
          <label className="btn">
            Choose files…
            <input type="file" multiple hidden accept=".cho,.chopro,.chordpro,.pro,.crd,.onsong,.txt,.pdf,.csv,.json,.zip,.backup,text/plain,application/pdf,application/zip"
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

    </div>
  );
}
