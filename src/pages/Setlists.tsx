import { useLiveQuery } from "dexie-react-hooks";
import { Link, useNavigate } from "react-router-dom";
import { IconPlay, IconPlus } from "../components/Icons";
import { SyncBadge } from "../components/SyncBadge";
import { blankItem, blankSetlist, db, live, newId, nowIso, saveRow, softDelete, type Setlist } from "../lib/db";
import { useSetlists } from "../lib/hooks";
import { estimateDuration, formatDuration } from "../lib/stage";

export function Setlists() {
  const setlists = useSetlists();
  const navigate = useNavigate();
  const stats = useLiveQuery(async () => {
    const items = live(await db.setlist_items.toArray());
    const songs = new Map(live(await db.songs.toArray()).map((s) => [s.id, s]));
    const out = new Map<string, { count: number; seconds: number }>();
    for (const it of items) {
      if (it.kind !== "song" || !it.song_id) continue;
      const s = songs.get(it.song_id);
      if (!s) continue;
      const o = out.get(it.setlist_id) ?? { count: 0, seconds: 0 };
      o.count++;
      o.seconds += s.duration_sec || estimateDuration(s.tempo);
      out.set(it.setlist_id, o);
    }
    return out;
  }, []);

  const create = async () => {
    const s = await saveRow(db.setlists, blankSetlist({ name: "New setlist", event_date: new Date().toISOString().slice(0, 10) }));
    navigate(`/sets/${s.id}`);
  };

  const duplicate = async (sl: Setlist, forGig = false) => {
    const today = new Date().toISOString().slice(0, 10);
    const copy = await saveRow(db.setlists, {
      ...sl, id: newId(), created_at: nowIso(), dirty: 1,
      // A gig copy of a signature show is an ordinary, dated setlist you can change freely
      signature: false,
      name: forGig ? `${sl.name} — ${new Date(today + "T12:00").toLocaleDateString(undefined, { month: "short", day: "numeric" })}` : `${sl.name} (copy)`,
      event_date: forGig ? today : sl.event_date,
      notes: forGig ? null : sl.notes,
    });
    const items = live(await db.setlist_items.where("setlist_id").equals(sl.id).toArray());
    for (const it of items) await saveRow(db.setlist_items, blankItem({ ...it, id: newId(), setlist_id: copy.id }));
    navigate(`/sets/${copy.id}`);
  };

  const remove = async (sl: Setlist) => {
    if (sl.signature) {
      const typed = prompt(`“${sl.name}” is a signature show. Type DELETE to remove it permanently.`);
      if (typed?.trim().toUpperCase() !== "DELETE") return;
    } else if (!confirm(`Delete setlist “${sl.name}”?`)) return;
    await softDelete(db.setlists, sl.id);
  };

  const signature = (setlists ?? []).filter((s) => s.signature).sort((a, b) => a.name.localeCompare(b.name));
  const gigs = (setlists ?? []).filter((s) => !s.signature);

  const row = (sl: Setlist) => {
    const st = stats?.get(sl.id);
    return (
      <li key={sl.id} className="list-item">
        <Link to={`/sets/${sl.id}`} className="grow" style={{ color: "inherit", textDecoration: "none" }}>
          <div className="title">{sl.signature && <span style={{ color: "var(--accent)" }}>★ </span>}{sl.name || "Untitled set"}</div>
          <div className="small dim">
            {[!sl.signature && sl.event_date && new Date(sl.event_date + "T12:00").toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric", year: "numeric" }), sl.venue, st && `${st.count} songs · ${formatDuration(st.seconds)}`]
              .filter(Boolean).join(" · ")}
          </div>
        </Link>
        {sl.signature
          ? <button className="btn small primary" onClick={() => duplicate(sl, true)}>Duplicate for a gig</button>
          : <button className="btn small" onClick={() => duplicate(sl)}>Duplicate</button>}
        <button className="btn small danger" onClick={() => remove(sl)}>Delete</button>
        <Link className="btn small" to={`/perform/${sl.id}`}><IconPlay size={18} /> Perform</Link>
      </li>
    );
  };

  return (
    <div className="page">
      <div className="row" style={{ marginBottom: 14 }}>
        <h1 className="grow">Setlists</h1>
        <SyncBadge />
        <button className="btn primary" onClick={create}><IconPlus /> New setlist</button>
      </div>
      {setlists && setlists.length === 0 ? (
        <div className="empty-state card">
          <h2>No setlists yet</h2>
          <p>Build a set for your next gig — add songs, set breaks and keys, then hit Perform.</p>
          <button className="btn primary" onClick={create}>New setlist</button>
        </div>
      ) : (
        <>
          {signature.length > 0 && (
            <>
              <h2 className="dim" style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.06em", margin: "4px 0 8px" }}>Signature shows</h2>
              <ul className="list" style={{ marginBottom: 20 }}>{signature.map(row)}</ul>
            </>
          )}
          {gigs.length > 0 && (
            <>
              {signature.length > 0 && <h2 className="dim" style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.06em", margin: "4px 0 8px" }}>Gigs & other sets</h2>}
              <ul className="list">{gigs.map(row)}</ul>
            </>
          )}
        </>
      )}
    </div>
  );
}
