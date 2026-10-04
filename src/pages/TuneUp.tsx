import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { DuoPickers } from "../components/GearUI";
import { IconBack } from "../components/Icons";
import { db, patchRow, saveRow, type Song } from "../lib/db";
import { useSongs } from "../lib/hooks";
import { ALL_KEYS, guessKey } from "../lib/music/chords";
import { allChords, parseChordPro } from "../lib/music/chordpro";
import { importChart } from "../lib/music/convert";
import { formatDuration, parseDurationInput } from "../lib/stage";
import { supabase } from "../lib/supabase";
import { useSyncStatus } from "../lib/sync";

interface Suggestion {
  duration_sec?: number | null;
  year?: number | null;
  genre?: string | null;
  tempo?: number | null;
  url?: string | null;
  error?: string;
}

type Filter = "all" | "chart" | "info" | "duo";

const chartSearchUrl = (s: Song) =>
  `https://www.ultimate-guitar.com/search.php?search_type=title&value=${encodeURIComponent(`${s.title} ${s.artist}`)}`;

/** Fill in charts and song details fast: one row per song, everything editable in place. */
export function TuneUp() {
  const songs = useSongs();
  const navigate = useNavigate();
  const { userEmail } = useSyncStatus();
  const [filter, setFilter] = useState<Filter>("chart");
  const [sugg, setSugg] = useState<Record<string, Suggestion>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const list = useMemo(() => (songs ?? []).filter((s) =>
    filter === "all" ? true : filter === "chart" ? !s.content.trim() : filter === "duo" ? !s.instrument || !s.lead_vocal : !s.duration_sec || !s.tempo || !s.year,
  ), [songs, filter]);
  const counts = useMemo(() => ({
    chart: (songs ?? []).filter((s) => !s.content.trim()).length,
    info: (songs ?? []).filter((s) => !s.duration_sec || !s.tempo || !s.year).length,
    duo: (songs ?? []).filter((s) => !s.instrument || !s.lead_vocal).length,
  }), [songs]);

  const flash = (msg: string) => { setToast(msg); setTimeout(() => setToast(null), 3500); };

  const lookUp = async (s: Song): Promise<Suggestion> => {
    const { data, error } = await supabase.functions.invoke("song-lookup", { body: { title: s.title, artist: s.artist } });
    const res: Suggestion = error || !data
      ? { error: "Lookup failed" }
      : data.catalog || data.bpm
        ? { duration_sec: data.catalog?.duration_sec, year: data.catalog?.year, genre: data.catalog?.genre, url: data.catalog?.url, tempo: data.bpm }
        : { error: "Not found" };
    setSugg((m) => ({ ...m, [s.id]: res }));
    return res;
  };

  const lookUpAll = async () => {
    const targets = list.filter((s) => !s.duration_sec || !s.tempo || !s.year);
    for (let i = 0; i < targets.length; i++) {
      setProgress(`Looking up ${i + 1} of ${targets.length}…`);
      await lookUp(targets[i]);
    }
    setProgress(null);
  };

  /** Only fills what's empty (tempo can be replaced: the old values were guesses). */
  const accept = async (s: Song, which: "duration" | "year" | "tempo" | "all") => {
    const g = sugg[s.id];
    if (!g) return;
    const patch: Partial<Song> = {};
    if ((which === "duration" || which === "all") && g.duration_sec) patch.duration_sec = g.duration_sec;
    if ((which === "year" || which === "all") && g.year) patch.year = g.year;
    if ((which === "tempo" || which === "all") && g.tempo) patch.tempo = g.tempo;
    if (which === "all" && g.genre && !s.genre) patch.genre = g.genre;
    await patchRow(db.songs, s.id, patch);
  };

  const acceptAll = async () => {
    for (const s of list) if (sugg[s.id] && !sugg[s.id].error) await accept(s, "all");
    flash("Suggestions applied.");
  };

  const pasteChart = async (s: Song) => {
    setBusy(s.id);
    try {
      const text = await navigator.clipboard.readText();
      if (!text.trim()) throw new Error("empty");
      const { meta, body } = importChart(text);
      const detected = meta.key || guessKey(allChords(parseChordPro(body)));
      await saveRow(db.songs, {
        ...s,
        content: body,
        // The chart decides the written key; the old value was a guess.
        song_key: detected ?? s.song_key,
        capo: meta.capo ?? s.capo,
        time_signature: s.time_signature ?? meta.time ?? null,
        tempo: s.tempo ?? meta.tempo ?? null,
      });
      flash(`Chart added to “${s.title}”${detected ? ` — key ${detected}` : ""}. Tap the title to check it.`);
    } catch {
      // Clipboard blocked or empty: fall back to the editor's paste box
      navigate(`/song/${s.id}/edit`);
    } finally {
      setBusy(null);
    }
  };

  if (!songs) return null;

  return (
    <div className="page" style={{ maxWidth: 1200 }}>
      <div className="row wrap" style={{ marginBottom: 10 }}>
        <button className="btn ghost icon" onClick={() => navigate(-1)} aria-label="Back"><IconBack /></button>
        <h1 className="grow">Tune-up</h1>
        <div className="seg">
          <button className={filter === "chart" ? "on" : ""} onClick={() => setFilter("chart")}>Needs chart ({counts.chart})</button>
          <button className={filter === "info" ? "on" : ""} onClick={() => setFilter("info")}>Needs details ({counts.info})</button>
          <button className={filter === "duo" ? "on" : ""} onClick={() => setFilter("duo")}>Needs vocal/instrument ({counts.duo})</button>
          <button className={filter === "all" ? "on" : ""} onClick={() => setFilter("all")}>All</button>
        </div>
      </div>
      <div className="card small dim" style={{ marginBottom: 12 }}>
        <strong style={{ color: "var(--text)" }}>Adding charts:</strong> tap <em>Find chart</em>, copy the chart you trust (select the text, Copy), come back and tap <em>Paste chart</em>.
        It's converted and the song's key is set from the chart. <strong style={{ color: "var(--text)" }}>Details:</strong> <em>Look up</em> pulls
        length and year from Apple's catalog and an approximate tempo from Deezer — accept what's right.
      </div>
      <div className="row wrap" style={{ marginBottom: 10 }}>
        <button className="btn" onClick={lookUpAll} disabled={!userEmail || !!progress}>{progress ?? "Look up details for this list"}</button>
        {Object.values(sugg).some((g) => !g.error) && <button className="btn primary" onClick={acceptAll}>Accept all suggestions</button>}
        {!userEmail && <span className="small dim">Sign in to use lookups.</span>}
      </div>

      {list.length === 0 ? (
        <div className="empty-state card">Nothing here — nice work.</div>
      ) : (
        <ul className="list">
          {list.map((s) => {
            const g = sugg[s.id];
            const hasChart = !!s.content.trim();
            const chartKey = hasChart ? guessKey(allChords(parseChordPro(s.content))) : null;
            return (
              <li key={s.id} style={{ padding: "10px 14px" }}>
                <div className="row wrap" style={{ gap: 10 }}>
                  <Link to={`/song/${s.id}`} className="grow" style={{ color: "inherit", textDecoration: "none", minWidth: 200 }}>
                    <div style={{ fontWeight: 700 }}>{s.title}</div>
                    <div className="small dim">{s.artist}</div>
                  </Link>
                  {hasChart ? <span className="chip accent">chart ✓</span> : <span className="chip">no chart</span>}
                  <a className="btn small" href={chartSearchUrl(s)} target="_blank" rel="noreferrer">Find chart</a>
                  <button className="btn small primary" disabled={busy === s.id} onClick={() => pasteChart(s)}>{hasChart ? "Replace chart" : "Paste chart"}</button>
                  <button className="btn small" disabled={!userEmail} onClick={() => lookUp(s)}>Look up</button>
                </div>
                <div className="row wrap" style={{ gap: 10, marginTop: 8 }}>
                  <label className="row small" style={{ gap: 4 }}>Key
                    <select className="select" style={{ width: 80, minHeight: 34, padding: "0 6px" }} value={s.song_key ?? ""}
                      onChange={(e) => patchRow(db.songs, s.id, { song_key: e.target.value || null })}>
                      <option value="">—</option>
                      {ALL_KEYS.map((k) => <option key={k}>{k}</option>)}
                    </select>
                  </label>
                  {hasChart && chartKey && chartKey !== s.song_key && (
                    <button className="btn small" onClick={() => patchRow(db.songs, s.id, { song_key: chartKey })} title="The chart's chords suggest this key">
                      Chart looks like {chartKey}
                    </button>
                  )}
                  <label className="row small" style={{ gap: 4 }}>BPM
                    <input className="input" style={{ width: 70, minHeight: 34 }} inputMode="numeric" defaultValue={s.tempo ?? ""} key={`t${s.tempo}`}
                      onBlur={(e) => patchRow(db.songs, s.id, { tempo: Number(e.target.value) || null })} />
                  </label>
                  {g?.tempo && g.tempo !== s.tempo && <button className="chip accent" onClick={() => accept(s, "tempo")}>~{g.tempo} ✓</button>}
                  <label className="row small" style={{ gap: 4 }}>Time
                    <select className="select" style={{ width: 76, minHeight: 34, padding: "0 6px" }} value={s.time_signature ?? ""}
                      onChange={(e) => patchRow(db.songs, s.id, { time_signature: e.target.value || null })}>
                      <option value="">—</option>
                      {["4/4", "3/4", "6/8", "2/4", "12/8"].map((t) => <option key={t}>{t}</option>)}
                    </select>
                  </label>
                  <label className="row small" style={{ gap: 4 }}>Length
                    <input className="input" style={{ width: 70, minHeight: 34 }} placeholder="m:ss" defaultValue={formatDuration(s.duration_sec)} key={`d${s.duration_sec}`}
                      onBlur={(e) => patchRow(db.songs, s.id, { duration_sec: parseDurationInput(e.target.value) })} />
                  </label>
                  {g?.duration_sec && g.duration_sec !== s.duration_sec && <button className="chip accent" onClick={() => accept(s, "duration")}>{formatDuration(g.duration_sec)} ✓</button>}
                  <label className="row small" style={{ gap: 4 }}>Year
                    <input className="input" style={{ width: 70, minHeight: 34 }} inputMode="numeric" defaultValue={s.year ?? ""} key={`y${s.year}`}
                      onBlur={(e) => patchRow(db.songs, s.id, { year: Number(e.target.value) || null })} />
                  </label>
                  {g?.year && g.year !== s.year && <button className="chip accent" onClick={() => accept(s, "year")}>{g.year} ✓</button>}
                  <DuoPickers compact song={s} onChange={(patch) => patchRow<Song>(db.songs, s.id, patch)} />
                  <label className="check small" style={{ minHeight: 34 }}>
                    <input type="checkbox" checked={s.karaoke ?? false} onChange={(e) => patchRow(db.songs, s.id, { karaoke: e.target.checked })} /> Karaoke
                  </label>
                  {g?.error && <span className="small dim">{g.error}</span>}
                  {g?.url && <a className="small" href={g.url} target="_blank" rel="noreferrer">listen</a>}
                </div>
              </li>
            );
          })}
        </ul>
      )}
      {toast && <div className="toast-stack"><div className="toast">{toast}</div></div>}
    </div>
  );
}
