import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { IconBack, IconEdit, IconPlay, IconSets } from "../components/Icons";
import { SongStage } from "../components/SongStage";
import { blankItem, db, live, positionBetween, saveRow } from "../lib/db";
import { useSetlists, useSong } from "../lib/hooks";
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
        <div className="grow truncate" style={{ fontWeight: 700 }}>{song.title}</div>
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
      <SongStage song={song} onPerformKeyChange={setPendingKey} />
    </div>
  );
}
