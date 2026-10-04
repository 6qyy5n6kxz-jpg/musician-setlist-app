import { useLiveQuery } from "dexie-react-hooks";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { IconBack } from "../components/Icons";
import { db, live, patchRow, softDelete, type Gig } from "../lib/db";
import { playStats } from "../lib/gigs";
import { useProfile } from "../lib/hooks";
import { formatDuration } from "../lib/stage";

const fmtDate = (d: string) => new Date(d + "T12:00").toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric", year: "numeric" });

/** Every gig Perform mode logged, newest first, plus most-played and most-requested songs. */
export function GigLog() {
  const navigate = useNavigate();
  const profile = useProfile();
  const data = useLiveQuery(async () => {
    const gigs = live(await db.gigs.toArray()).sort((a, b) => b.gig_date.localeCompare(a.gig_date) || b.created_at.localeCompare(a.created_at));
    const played = live(await db.gig_songs.toArray());
    const songs = new Map(live(await db.songs.toArray()).map((s) => [s.id, s]));
    const perGig = new Map<string, { count: number; seconds: number }>();
    for (const p of played) {
      const s = p.song_id ? songs.get(p.song_id) : undefined;
      const o = perGig.get(p.gig_id) ?? { count: 0, seconds: 0 };
      o.count++;
      o.seconds += s?.duration_sec ?? 0;
      perGig.set(p.gig_id, o);
    }
    const stats = await playStats();
    const top = [...stats.entries()].map(([id, st]) => ({ song: songs.get(id), ...st })).filter((x) => x.song);
    return {
      gigs, perGig,
      mostPlayed: [...top].sort((a, b) => b.count - a.count).slice(0, 10),
      mostRequested: top.filter((x) => x.requests).sort((a, b) => b.requests - a.requests).slice(0, 10),
    };
  }, []);
  const actName = (id: string | null) => profile?.acts?.find((a) => a.id === id)?.name;

  return (
    <div className="page">
      <div className="row" style={{ marginBottom: 14 }}>
        <button className="btn ghost icon" onClick={() => navigate(-1)} aria-label="Back"><IconBack /></button>
        <h1 className="grow">Gig log</h1>
      </div>
      <p className="small dim" style={{ marginTop: 0 }}>
        Perform mode logs a song once it's been on screen for 45 seconds. Set a venue on the setlist so the app can warn you about
        repeats next time you play there.
      </p>
      {data && data.gigs.length === 0 ? (
        <div className="empty-state card">No gigs logged yet — start Perform mode with a setlist and play.</div>
      ) : (
        <div className="editor-grid" style={{ gridTemplateColumns: "3fr 2fr" }}>
          <ul className="list" style={{ alignSelf: "start" }}>
            {data?.gigs.map((g) => {
              const st = data.perGig.get(g.id);
              return (
                <li key={g.id}>
                  <Link className="list-item" to={`/gigs/${g.id}`}>
                    <div className="grow">
                      <div className="title">{fmtDate(g.gig_date)}{g.venue ? ` · ${g.venue}` : ""}</div>
                      <div className="small dim">{[actName(g.act_id), g.name, st ? `${st.count} song${st.count === 1 ? "" : "s"}${st.seconds ? ` · ${formatDuration(st.seconds)}` : ""}` : "nothing logged"].filter(Boolean).join(" · ")}</div>
                    </div>
                  </Link>
                </li>
              );
            })}
          </ul>
          <div className="stack">
            {!!data?.mostRequested.length && (
              <div className="card">
                <strong>Most requested</strong>
                <ol className="small" style={{ margin: "8px 0 0", paddingLeft: 20 }}>
                  {data.mostRequested.map((x) => <li key={x.song!.id}><Link to={`/song/${x.song!.id}`}>{x.song!.title}</Link> <span className="dim">· {x.requests}</span></li>)}
                </ol>
              </div>
            )}
            {!!data?.mostPlayed.length && (
              <div className="card">
                <strong>Most played</strong>
                <ol className="small" style={{ margin: "8px 0 0", paddingLeft: 20 }}>
                  {data.mostPlayed.map((x) => <li key={x.song!.id}><Link to={`/song/${x.song!.id}`}>{x.song!.title}</Link> <span className="dim">· {x.count}</span></li>)}
                </ol>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function GigDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const gig = useLiveQuery(async () => (id ? (await db.gigs.get(id)) ?? null : null), [id]);
  const played = useLiveQuery(async () => {
    if (!id) return [];
    const rows = live(await db.gig_songs.where("gig_id").equals(id).toArray()).sort((a, b) => a.played_at.localeCompare(b.played_at));
    const songs = new Map(live(await db.songs.toArray()).map((s) => [s.id, s]));
    return rows.map((r) => ({ row: r, song: r.song_id ? songs.get(r.song_id) : undefined }));
  }, [id]);
  const [form, setForm] = useState<Gig | null>(null);
  useEffect(() => { if (gig && form?.id !== gig.id) setForm(gig); }, [gig, form]);

  if (gig === undefined) return null;
  if (!gig || gig.deleted_at || !form) return <div className="page empty-state">Gig not found. <Link to="/gigs">Back</Link></div>;
  const save = (patch: Partial<Gig>) => { setForm({ ...form, ...patch }); void patchRow(db.gigs, gig.id, patch); };

  return (
    <div className="page" style={{ maxWidth: 900 }}>
      <div className="row" style={{ marginBottom: 12 }}>
        <button className="btn ghost icon" onClick={() => navigate(-1)} aria-label="Back"><IconBack /></button>
        <h1 className="grow">{fmtDate(gig.gig_date)}</h1>
        <button className="btn danger" onClick={async () => { if (confirm("Delete this gig from the log?")) { await softDelete(db.gigs, gig.id); navigate("/gigs"); } }}>Delete</button>
      </div>
      <div className="meta-grid" style={{ marginBottom: 14 }}>
        <label className="field"><span>Date</span><input className="input" type="date" value={form.gig_date} onChange={(e) => e.target.value && save({ gig_date: e.target.value })} /></label>
        <label className="field"><span>Venue</span><input className="input" value={form.venue ?? ""} onChange={(e) => setForm({ ...form, venue: e.target.value })} onBlur={() => save({ venue: form.venue || null })} /></label>
        <label className="field" style={{ gridColumn: "span 2" }}><span>Notes</span><input className="input" value={form.notes ?? ""} placeholder="Crowd, pay, sound, what worked…" onChange={(e) => setForm({ ...form, notes: e.target.value })} onBlur={() => save({ notes: form.notes || null })} /></label>
      </div>
      <ul className="list">
        {played?.map(({ row, song }, i) => (
          <li key={row.id} className="list-item">
            <span className="dim" style={{ width: 24, textAlign: "right" }}>{i + 1}</span>
            <div className="grow">
              <div className="title">{song ? <Link to={`/song/${song.id}`} style={{ color: "inherit" }}>{song.title}</Link> : "Deleted song"}</div>
              <div className="small dim">{song?.artist} · {new Date(row.played_at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}{row.from_request ? " · by request" : ""}</div>
            </div>
            <button className="btn small ghost" onClick={() => softDelete(db.gig_songs, row.id)}>Remove</button>
          </li>
        ))}
        {played?.length === 0 && <li className="list-item dim">Nothing logged for this gig.</li>}
      </ul>
    </div>
  );
}
