import { useLiveQuery } from "dexie-react-hooks";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { IconImport, IconPlus, IconSearch } from "../components/Icons";
import { SyncBadge } from "../components/SyncBadge";
import { DuoBadges } from "../components/GearUI";
import { blankSong, db, live, saveRow, type Song } from "../lib/db";
import { sortTitle, useSongs } from "../lib/hooks";
import { formatDuration } from "../lib/stage";
import { rememberSongOrder } from "../lib/swipe";

type SortMode = "title" | "artist" | "recent" | "key";

export function Library() {
  const songs = useSongs();
  const navigate = useNavigate();
  const [q, setQ] = useState(() => sessionStorage.getItem("lib-q") ?? "");
  const [tag, setTag] = useState("");
  const [sort, setSort] = useState<SortMode>("title");
  const [inLyrics, setInLyrics] = useState(false);
  const [instrument, setInstrument] = useState("");
  const [vocal, setVocal] = useState("");

  const fileKinds = useLiveQuery(async () => {
    const map = new Map<string, Set<string>>();
    for (const f of live(await db.song_files.toArray())) {
      if (!map.has(f.song_id)) map.set(f.song_id, new Set());
      map.get(f.song_id)!.add(f.kind);
    }
    return map;
  }, []);

  const tags = useMemo(() => {
    const set = new Set<string>();
    songs?.forEach((s) => s.tags.forEach((t) => set.add(t)));
    return [...set].sort((a, b) => a.localeCompare(b));
  }, [songs]);

  const filtered = useMemo(() => {
    if (!songs) return [];
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    let list = songs.filter((s) => {
      if (tag && !s.tags.includes(tag)) return false;
      if (instrument && (s.instrument ?? "none") !== instrument) return false;
      if (vocal && (s.lead_vocal ?? "none") !== vocal) return false;
      if (!terms.length) return true;
      const hay = [s.title, s.artist, s.song_key ?? "", s.genre ?? "", s.tags.join(" "), inLyrics ? s.content : ""]
        .join(" ")
        .toLowerCase();
      return terms.every((t) => hay.includes(t));
    });
    if (sort === "artist") list = [...list].sort((a, b) => sortTitle(a.artist).localeCompare(sortTitle(b.artist)) || sortTitle(a.title).localeCompare(sortTitle(b.title)));
    if (sort === "recent") list = [...list].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
    if (sort === "key") list = [...list].sort((a, b) => (a.song_key ?? "~").localeCompare(b.song_key ?? "~"));
    return list;
  }, [songs, q, tag, sort, inLyrics, instrument, vocal]);

  // The song page swipes through this exact list (search, filters and sort included)
  useEffect(() => { if (songs) rememberSongOrder(filtered.map((s) => s.id)); }, [songs, filtered]);

  const addSong = async () => {
    const s = await saveRow(db.songs, blankSong());
    navigate(`/song/${s.id}/edit`);
  };

  const groupKey = (s: Song) => {
    if (sort === "title") return (sortTitle(s.title)[0] ?? "#").toUpperCase().replace(/[^A-Z]/, "#");
    if (sort === "artist") return (sortTitle(s.artist)[0] ?? "#").toUpperCase().replace(/[^A-Z]/, "#");
    if (sort === "key") return s.song_key ?? "No key";
    return null;
  };

  return (
    <div className="page">
      <div className="row" style={{ marginBottom: 14 }}>
        <h1 className="grow">Songs <span className="dim small">{songs ? songs.length : ""}</span></h1>
        <SyncBadge />
        {songs && songs.some((x) => !x.content.trim()) && (
          <Link className="btn" to="/tuneup" title="Add charts and fix song details fast">
            Tune-up <span className="badge" style={{ background: "var(--accent)", color: "var(--accent-ink)" }}>{songs.filter((x) => !x.content.trim()).length}</span>
          </Link>
        )}
        <Link className="btn" to="/import"><IconImport /> Import</Link>
        <button className="btn primary" onClick={addSong}><IconPlus /> New song</button>
      </div>

      <div className="toolbar">
        <label className="search row" style={{ position: "relative" }}>
          <span style={{ position: "absolute", left: 12, display: "flex" }} className="dim"><IconSearch size={18} /></span>
          <input
            className="input"
            style={{ paddingLeft: 38 }}
            type="search"
            placeholder="Search title, artist, key, tag…"
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              sessionStorage.setItem("lib-q", e.target.value);
            }}
          />
        </label>
        <select className="select" style={{ width: "auto" }} value={tag} onChange={(e) => setTag(e.target.value)} aria-label="Filter by tag">
          <option value="">All tags</option>
          {tags.map((t) => <option key={t}>{t}</option>)}
        </select>
        <select className="select" style={{ width: "auto" }} value={instrument} onChange={(e) => setInstrument(e.target.value)} aria-label="Filter by instrument">
          <option value="">Any instrument</option>
          <option value="piano">🎹 Piano</option><option value="electric">⚡ Electric</option><option value="acoustic">🎸 Acoustic</option>
          <option value="none">Not set</option>
        </select>
        <select className="select" style={{ width: "auto" }} value={vocal} onChange={(e) => setVocal(e.target.value)} aria-label="Filter by lead vocal">
          <option value="">Any vocal</option>
          <option value="kendra">Kendra</option><option value="devin">Devin</option><option value="both">Both</option>
          <option value="none">Not set</option>
        </select>
        <div className="seg" role="group" aria-label="Sort">
          {(["title", "artist", "key", "recent"] as SortMode[]).map((m) => (
            <button key={m} className={sort === m ? "on" : ""} onClick={() => setSort(m)}>
              {m[0].toUpperCase() + m.slice(1)}
            </button>
          ))}
        </div>
        <label className="check small"><input type="checkbox" checked={inLyrics} onChange={(e) => setInLyrics(e.target.checked)} /> Search lyrics</label>
      </div>

      {songs && songs.length === 0 ? (
        <div className="empty-state card">
          <h2>Your song library is empty</h2>
          <p>Add a song, paste a chart from Ultimate Guitar, or import ChordPro / OnSong files.</p>
          <div className="row" style={{ justifyContent: "center", marginTop: 12 }}>
            <Link className="btn" to="/import">Import charts</Link>
            <button className="btn primary" onClick={addSong}>New song</button>
          </div>
        </div>
      ) : filtered.length === 0 && songs ? (
        <div className="empty-state">No songs match “{q}”.</div>
      ) : (
        <ul className="list">
          {filtered.map((s, i) => {
            const g = groupKey(s);
            const showHead = g !== null && (i === 0 || groupKey(filtered[i - 1]) !== g);
            const kinds = fileKinds?.get(s.id);
            return (
              <li key={s.id}>
                {showHead && <div className="alpha-head">{g}</div>}
                <Link className="list-item" to={`/song/${s.id}`}>
                  <span className="key-pill">{s.song_key || "–"}</span>
                  <div className="grow">
                    <div className="title truncate">{s.title || "Untitled"}</div>
                    <div className="meta small dim">
                      <span className="truncate">{s.artist}</span>
                      <DuoBadges song={s} />
                      {s.tempo ? <span className="chip">{s.tempo} bpm</span> : null}
                      {s.duration_sec ? <span className="chip">{formatDuration(s.duration_sec)}</span> : null}
                      {!s.content.trim() && !kinds?.has("pdf") ? <span className="chip">no chart</span> : null}
                      {kinds?.has("pdf") ? <span className="chip">PDF</span> : null}
                      {kinds?.has("audio") ? <span className="chip accent">track</span> : null}
                      {s.tags.slice(0, 3).map((t) => <span key={t} className="chip">{t}</span>)}
                    </div>
                  </div>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
