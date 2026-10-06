import { useLiveQuery } from "dexie-react-hooks";
import { useMemo, useState } from "react";
import { singerKey } from "../lib/gear";
import { playStats } from "../lib/gigs";
import { Link, useNavigate, useParams } from "react-router-dom";
import { IconBack, IconEdit, IconPlay, IconSets } from "../components/Icons";
import { SongStage } from "../components/SongStage";
import { blankItem, db, live, positionBetween, saveRow } from "../lib/db";
import { useSetlists, useSong, useSongs } from "../lib/hooks";
import { neighbours, songOrder, useSwipe, type SwipeDir } from "../lib/swipe";
import { keyPrefersFlats } from "../lib/music/chords";
import { toChordProFile, transposeContent } from "../lib/music/chordpro";
import { keyDistance } from "../lib/music/chords";

export function SongPage() {
  const { id } = useParams();
  const song = useSong(id);
  const setlists = useSetlists();
  const navigate = useNavigate();
  const [pendingKey, setPendingKey] = useState<string | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const stats = useLiveQuery(async () => (id ? (await playStats()).get(id) : undefined), [id]);
  const allSongs = useSongs();
  // Swipe through the Songs list as last shown (search/filter/sort); otherwise A–Z
  const order = useMemo(() => {
    const ids = allSongs?.map((s) => s.id) ?? [];
    const saved = songOrder()?.filter((x) => ids.includes(x));
    return saved && id && saved.includes(id) ? saved : ids;
  }, [allSongs, id]);
  const nb = id ? neighbours(order, id) : { prev: null, next: null, index: -1 };
  const [entered, setEntered] = useState<SwipeDir | null>(null);
  const goTo = (dir: SwipeDir) => {
    const target = dir === "left" ? nb.next : nb.prev;
    if (!target) return;
    setPendingKey(null);
    setAddOpen(false);
    setEntered(dir);
    // replace: Back still returns to the Songs list rather than through every song
    navigate(`/song/${target}`, { replace: true });
  };
  const swipe = useSwipe(goTo);

  if (song === undefined) return null;
  if (song === null || song.deleted_at) return <div className="page empty-state">Song not found. <Link to="/">Back to songs</Link></div>;

  const saveKey = async () => {
    if (!pendingKey || !song.song_key) return;
    const semis = keyDistance(song.song_key, pendingKey);
    await saveRow(db.songs, { ...song, content: transposeContent(song.content, semis, keyPrefersFlats(pendingKey)), song_key: pendingKey });
    setPendingKey(null);
  };

  const addToSet = async (setlistId: string) => {
    const items = live(await db.setlist_items.where("setlist_id").equals(setlistId).toArray()).sort((a, b) => a.position - b.position);
    await saveRow(db.setlist_items, blankItem({
      setlist_id: setlistId, song_id: song.id, position: positionBetween(items[items.length - 1]?.position, undefined),
    }));
    setAddOpen(false);
  };

  const exportCho = () => {
    const text = toChordProFile(song, song.content);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
    a.download = `${song.title || "song"}.cho`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="song-view">
      <div className="topbar no-print">
        <button className="btn ghost icon" onClick={() => navigate(-1)} aria-label="Back"><IconBack /></button>
        <button className="btn ghost icon" onClick={() => goTo("right")} disabled={!nb.prev} aria-label="Previous song">‹</button>
        <div className="grow truncate" style={{ fontWeight: 700 }}>
          {song.title}
          {nb.index >= 0 && order.length > 1 && <span className="dim small" style={{ fontWeight: 400 }}> · {nb.index + 1}/{order.length}</span>}
        </div>
        <button className="btn ghost icon" onClick={() => goTo("left")} disabled={!nb.next} aria-label="Next song">›</button>
        {pendingKey && song.song_key && (
          <button className="btn small primary" onClick={saveKey} title="Rewrite the chart's chords in this key">Save in {pendingKey}</button>
        )}
        <div style={{ position: "relative" }}>
          <button className="btn small" onClick={() => setAddOpen(!addOpen)}><IconSets size={18} /> Add to set</button>
          {addOpen && (
            <div className="card" style={{ position: "absolute", right: 0, top: 44, zIndex: 30, width: 260, padding: 6, maxHeight: 320, overflowY: "auto" }}>
              {setlists?.length ? setlists.map((s) => (
                <button key={s.id} className="btn ghost" style={{ width: "100%", justifyContent: "flex-start" }} onClick={() => addToSet(s.id)}>
                  {s.name || "Untitled set"}
                </button>
              )) : <div className="dim small" style={{ padding: 8 }}>No setlists yet.</div>}
            </div>
          )}
        </div>
        <button className="btn small" onClick={exportCho} title="Download ChordPro file">.cho</button>
        <button className="btn small" onClick={() => window.print()}>Print</button>
        <Link className="btn small" to={`/perform/song/${song.id}`}><IconPlay size={18} /> Perform</Link>
        <Link className="btn small primary" to={`/song/${song.id}/edit`}><IconEdit size={18} /> Edit</Link>
      </div>
      {stats && stats.count > 0 && (
        <div className="small dim no-print" style={{ padding: "6px 20px 0" }}>
          Played {stats.count} time{stats.count === 1 ? "" : "s"}{stats.requests ? ` (${stats.requests} by request)` : ""}
          {stats.last ? ` · last on ${new Date(stats.last.date + "T12:00").toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}${stats.last.venue ? ` at ${stats.last.venue}` : ""}` : ""}
        </div>
      )}
      <div key={song.id} className={`swipe-area ${entered ? `swipe-in-${entered}` : ""}`} {...swipe}>
        <SongStage song={song} performKey={singerKey(song)} onPerformKeyChange={setPendingKey} />
      </div>
    </div>
  );
}
