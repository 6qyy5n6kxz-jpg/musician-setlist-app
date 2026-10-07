import { closestCenter, DndContext, KeyboardSensor, PointerSensor, TouchSensor, useSensor, useSensors, type DragEndEvent } from "@dnd-kit/core";
import { SortableContext, sortableKeyboardCoordinates, useSortable, verticalListSortingStrategy } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { IconBack, IconDrag, IconPlay, IconPlus, IconSearch, IconTrash } from "../components/Icons";
import { blankItem, db, patchRow, positionBetween, saveRow, softDelete, type Setlist, type SetlistItem, type Song } from "../lib/db";
import { useProfile, useSetlistItems, useSetlists, useSongs } from "../lib/hooks";
import { ALL_KEYS } from "../lib/music/chords";
import { estimateDuration, formatDuration } from "../lib/stage";
import { setBalance, singerKey } from "../lib/gear";
import { daysAgo, playedAtVenue } from "../lib/gigs";
import { useLiveQuery } from "dexie-react-hooks";
import { isSlow, optimizeSet, scoreSet, songEnergy, type FlowReport, type OptOptions, type OptSong } from "../lib/setOptimizer";

export function SetlistEditor() {
  const { id } = useParams();
  const navigate = useNavigate();
  const setlists = useSetlists();
  const setlist = setlists?.find((s) => s.id === id);
  const items = useSetlistItems(id);
  const songs = useSongs();
  const profile = useProfile();
  const venueHistory = useLiveQuery(async () => (setlist?.venue && !setlist.signature ? playedAtVenue(setlist.venue, setlist.event_date ?? undefined) : undefined), [setlist?.venue, setlist?.event_date, setlist?.signature]);
  const songMap = useMemo(() => new Map((songs ?? []).map((s) => [s.id, s])), [songs]);
  const [meta, setMeta] = useState<Setlist | null>(null);
  const [q, setQ] = useState("");
  const [showBuild, setShowBuild] = useState(false);
  const [showOptimize, setShowOptimize] = useState(false);

  useEffect(() => { if (setlist && meta?.id !== setlist.id) setMeta(setlist); }, [setlist, meta]);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(TouchSensor, { activationConstraint: { delay: 120, tolerance: 8 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  if (setlists === undefined || items === undefined) return null;
  if (!setlist || !meta) return <div className="page empty-state">Setlist not found. <Link to="/sets">Back</Link></div>;

  const saveMeta = (patch: Partial<Setlist>) => {
    const next = { ...meta, ...patch };
    setMeta(next);
    void saveRow(db.setlists, next);
  };

  const inSet = new Set(items.filter((i) => i.song_id).map((i) => i.song_id));
  const lastPos = items[items.length - 1]?.position;

  const addSong = (song: Song) => saveRow(db.setlist_items, blankItem({ setlist_id: setlist.id, song_id: song.id, position: positionBetween(lastPos, undefined) }));
  const addBreak = () => {
    const sets = items.filter((i) => i.kind === "break").length;
    return saveRow(db.setlist_items, blankItem({ setlist_id: setlist.id, kind: "break", label: `Set ${sets + 2}`, position: positionBetween(lastPos, undefined) }));
  };

  const onDragEnd = ({ active, over }: DragEndEvent) => {
    if (!over || active.id === over.id) return;
    // Index of the drop target in the original list == insertion index in the list without the dragged row
    const to = items.findIndex((i) => i.id === over.id);
    const without = items.filter((i) => i.id !== active.id);
    const before = without[to - 1];
    const after = without[to];
    void patchRow(db.setlist_items, String(active.id), { position: positionBetween(before?.position, after?.position) });
  };

  // Totals per set (split at breaks)
  const blocks: { label: string; count: number; seconds: number; songs: Song[] }[] = [{ label: "Set 1", count: 0, seconds: 0, songs: [] }];
  for (const it of items) {
    if (it.kind === "break") blocks.push({ label: it.label || "Set", count: 0, seconds: 0, songs: [] });
    else if (it.song_id && songMap.get(it.song_id)) {
      const s = songMap.get(it.song_id)!;
      blocks[blocks.length - 1].count++;
      blocks[blocks.length - 1].seconds += s.duration_sec || estimateDuration(s.tempo);
      blocks[blocks.length - 1].songs.push(s);
    }
  }
  const balanceText = (list: Song[]) => {
    const b = setBalance(list.map((s) => ({ instrument: s.instrument ?? null, lead_vocal: s.lead_vocal ?? null })));
    if (b.vocals.unset === list.length && b.instruments.unset === list.length) return "";
    return ` · K ${b.vocals.kendra} / D ${b.vocals.devin} / Both ${b.vocals.both} · 🎹${b.instruments.piano} ⚡${b.instruments.electric} 🎸${b.instruments.acoustic} · ${b.switches} instrument switch${b.switches === 1 ? "" : "es"}`;
  };
  const allSetSongs = blocks.flatMap((b) => b.songs);
  const total = blocks.reduce((a, b) => a + b.seconds, 0);
  const totalCount = blocks.reduce((a, b) => a + b.count, 0);

  let num = 0;
  const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
  const pool = (songs ?? []).filter((s) => terms.every((t) => `${s.title} ${s.artist} ${s.tags.join(" ")} ${s.song_key ?? ""}`.toLowerCase().includes(t)));

  return (
    <div className="page" style={{ maxWidth: 1300 }}>
      <div className="no-print">
      <div className="row wrap" style={{ marginBottom: 12 }}>
        <button className="btn ghost icon" onClick={() => navigate("/sets")} aria-label="Back"><IconBack /></button>
        <input className="input grow" style={{ fontSize: "1.3rem", fontWeight: 700, maxWidth: 520 }} value={meta.name}
          onChange={(e) => saveMeta({ name: e.target.value })} aria-label="Setlist name" />
        <span className="spacer" />
        <label className="check small" title="Signature shows stay pinned at the top and are protected from deletion">
          <input type="checkbox" checked={meta.signature ?? false} onChange={(e) => saveMeta({ signature: e.target.checked })} /> ★ Signature show
        </label>
        <button className="btn" onClick={() => window.print()}>Print</button>
        <Link className="btn primary" to={`/perform/${setlist.id}`}><IconPlay size={18} /> Perform</Link>
      </div>
      <div className="meta-grid" style={{ marginBottom: 14 }}>
        <label className="field"><span>Date</span>
          <input className="input" type="date" value={meta.event_date ?? ""} onChange={(e) => saveMeta({ event_date: e.target.value || null })} />
        </label>
        <label className="field"><span>Act</span>
          <select className="select" value={meta.act_id ?? ""} onChange={(e) => saveMeta({ act_id: e.target.value || null })}>
            <option value="">—</option>
            {(profile?.acts ?? []).map((a) => <option key={a.id} value={a.id}>{a.name || "Untitled act"}</option>)}
          </select>
        </label>
        <label className="field"><span>Venue</span>
          <input className="input" value={meta.venue ?? ""} onChange={(e) => saveMeta({ venue: e.target.value || null })} />
        </label>
        <label className="field" style={{ gridColumn: "span 2" }}><span>Notes</span>
          <input className="input" value={meta.notes ?? ""} onChange={(e) => saveMeta({ notes: e.target.value || null })} placeholder="Load-in 6pm · first set 7:30" />
        </label>
      </div>

      <div className="editor-grid" style={{ gridTemplateColumns: "3fr 2fr" }}>
        <div>
          <div className="set-totals" style={{ marginBottom: 8 }}>
            <strong style={{ color: "var(--text)" }}>{totalCount} songs · {formatDuration(total)}{balanceText(allSetSongs)}</strong>
            {blocks.length > 1 && blocks.map((b, i) => <span key={i}>{b.label}: {b.count} · {formatDuration(b.seconds)}{balanceText(b.songs)}</span>)}
          </div>
          {items.length === 0 ? (
            <div className="empty-state card">Add songs from your library on the right, or auto-build a set.</div>
          ) : (
            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
              <SortableContext items={items.map((i) => i.id)} strategy={verticalListSortingStrategy}>
                <ul className="list">
                  {items.map((it, i) => {
                    if (it.kind === "song") num++;
                    else num = 0;
                    const song = it.song_id ? songMap.get(it.song_id) : undefined;
                    const prevItem = items[i - 1];
                    const prev = prevItem?.kind === "song" && prevItem.song_id ? songMap.get(prevItem.song_id) : undefined;
                    const switching = !!(prev?.instrument && song?.instrument && prev.instrument !== song.instrument);
                    return <SetRow key={it.id} item={it} song={song} number={num} switching={switching} playedHere={song ? venueHistory?.get(song.id) : undefined} />;
                  })}
                </ul>
              </SortableContext>
            </DndContext>
          )}
          <div className="row" style={{ marginTop: 10 }}>
            <button className="btn" onClick={addBreak}><IconPlus size={18} /> Set break</button>
            <button className="btn" onClick={() => { setShowBuild(!showBuild); setShowOptimize(false); }}>Auto-build…</button>
            <button className="btn" disabled={totalCount < 3} onClick={() => { setShowOptimize(!showOptimize); setShowBuild(false); }}>Optimize order…</button>
          </div>
          {showBuild && <AutoBuild setlist={setlist} items={items} songs={songs ?? []} onDone={() => setShowBuild(false)} />}
          {showOptimize && <OptimizePanel items={items} songMap={songMap} onClose={() => setShowOptimize(false)} />}
        </div>

        <div className="card" style={{ padding: 10, alignSelf: "start", position: "sticky", top: 8 }}>
          <label className="row" style={{ position: "relative", marginBottom: 8 }}>
            <span style={{ position: "absolute", left: 12, display: "flex" }} className="dim"><IconSearch size={18} /></span>
            <input className="input" style={{ paddingLeft: 38 }} type="search" placeholder="Add songs…" value={q} onChange={(e) => setQ(e.target.value)} />
          </label>
          <div style={{ maxHeight: "62vh", overflowY: "auto" }}>
            {pool.map((s) => (
              <button key={s.id} className="song-pick" onClick={() => addSong(s)} style={{ opacity: inSet.has(s.id) ? 0.55 : 1 }}>
                <span className="key-pill" style={{ minWidth: 38, height: 28 }}>{s.song_key || "–"}</span>
                <span className="grow">
                  <span className="truncate" style={{ display: "block", fontWeight: 600 }}>{s.title}</span>
                  <span className="small dim truncate" style={{ display: "block" }}>{s.artist}</span>
                </span>
                {inSet.has(s.id) ? <span className="small dim">in set</span> : <IconPlus size={18} />}
              </button>
            ))}
          </div>
        </div>
      </div>

      </div>
      {/* Print view: big, simple stage list */}
      <div className="print-only">
        <h1>{meta.name}</h1>
        <ol>{items.map((it) => it.kind === "break" ? <h2 key={it.id}>{it.label}</h2> : <li key={it.id}>{songMap.get(it.song_id ?? "")?.title} {it.key_override || songMap.get(it.song_id ?? "")?.song_key ? `(${it.key_override || songMap.get(it.song_id ?? "")?.song_key})` : ""}</li>)}</ol>
      </div>
    </div>
  );
}

function SetRow({ item, song, number, switching, playedHere }: { item: SetlistItem; song?: Song; number: number; switching?: boolean; playedHere?: string }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id: item.id });
  const style = { transform: CSS.Transform.toString(transform), transition, opacity: isDragging ? 0.6 : 1, zIndex: isDragging ? 5 : undefined, position: "relative" as const };

  if (item.kind === "break") {
    return (
      <li ref={setNodeRef} style={style} className="set-row break">
        <span className="drag-handle" {...attributes} {...listeners}><IconDrag /></span>
        <input className="input grow" style={{ minHeight: 36, fontWeight: 700 }} value={item.label ?? ""}
          onChange={(e) => patchRow(db.setlist_items, item.id, { label: e.target.value })} aria-label="Break label" />
        <button className="btn small ghost" onClick={() => softDelete(db.setlist_items, item.id)} aria-label="Remove break"><IconTrash size={18} /></button>
      </li>
    );
  }
  return (
    <li ref={setNodeRef} style={style} className="set-row">
      <span className="drag-handle" {...attributes} {...listeners}><IconDrag /></span>
      <span className="num">{number}</span>
      <div className="grow">
        {song ? (
          <Link to={`/song/${song.id}`} style={{ color: "inherit", textDecoration: "none" }}>
            <div className="truncate" style={{ fontWeight: 600 }}>{song.title}</div>
            <div className="small dim truncate">
              {song.artist} · {formatDuration(song.duration_sec || estimateDuration(song.tempo))}
              {song.lead_vocal ? ` · 🎤 ${song.lead_vocal === "both" ? "K+D" : song.lead_vocal === "kendra" ? "Kendra" : "Devin"}` : ""}
              {song.instrument ? ` · ${song.instrument === "piano" ? "🎹" : song.instrument === "electric" ? "⚡" : "🎸"}` : ""}
              {switching ? " · ⇄ switch" : ""}
            </div>
            {playedHere && <div className="small" style={{ color: "var(--accent)" }}>Played here {daysAgo(playedHere)} days ago</div>}
          </Link>
        ) : <div className="dim">Song was deleted</div>}
      </div>
      <select className="select" style={{ width: 82, minHeight: 36, padding: "0 6px" }} value={item.key_override ?? ""}
        onChange={(e) => patchRow(db.setlist_items, item.id, { key_override: e.target.value || null })} aria-label="Key for this set">
        <option value="">{song ? singerKey(song) ?? song.song_key ?? "Key" : "Key"}</option>
        {ALL_KEYS.map((k) => <option key={k} value={k}>{k}</option>)}
      </select>
      <button className="btn small ghost" onClick={() => softDelete(db.setlist_items, item.id)} aria-label="Remove from set"><IconTrash size={18} /></button>
    </li>
  );
}

const toOpt = (item: SetlistItem, s: Song): OptSong => ({
  id: item.id, title: s.title, artist: s.artist, key: item.key_override || singerKey(s) || s.song_key,
  tempo: s.tempo, tags: s.tags, time_signature: s.time_signature, instrument: s.instrument ?? null, lead_vocal: s.lead_vocal ?? null,
});

/** Split the set at breaks; each set is optimized on its own and breaks never move. */
function setsOf(items: SetlistItem[], songMap: Map<string, Song>) {
  const sets: { songs: OptSong[]; items: SetlistItem[] }[] = [{ songs: [], items: [] }];
  for (const it of items) {
    if (it.kind === "break") { sets.push({ songs: [], items: [] }); continue; }
    const s = it.song_id ? songMap.get(it.song_id) : undefined;
    if (!s) continue;
    sets[sets.length - 1].songs.push(toOpt(it, s));
    sets[sets.length - 1].items.push(it);
  }
  return sets;
}

const sumReports = (rs: FlowReport[]): FlowReport => rs.reduce((a, r) => ({
  sameKey: a.sameKey + r.sameKey, relativeKey: a.relativeKey + r.relativeKey, slowStacked: a.slowStacked + r.slowStacked,
  weakOpenOrClose: a.weakOpenOrClose + r.weakOpenOrClose, switches: a.switches + r.switches, vocalRuns: a.vocalRuns + r.vocalRuns,
  sameArtist: a.sameArtist + r.sameArtist, cost: a.cost + r.cost,
}), { sameKey: 0, relativeKey: 0, slowStacked: 0, weakOpenOrClose: 0, switches: 0, vocalRuns: 0, sameArtist: 0, cost: 0 });

/**
 * Suggest a better running order inside each set: keys, slow songs, energy arc, instrument
 * switches, singers. Shows before/after and the new order; nothing changes until Apply.
 */
function OptimizePanel({ items, songMap, onClose }: { items: SetlistItem[]; songMap: Map<string, Song>; onClose: () => void }) {
  const [opts, setOpts] = useState<OptOptions>({ keepOpener: false, keepCloser: false, fewerSwitches: true });
  const [undo, setUndo] = useState<{ id: string; position: number }[] | null>(null);
  const [busy, setBusy] = useState(false);
  // Work from a snapshot so applying (which moves rows one by one) doesn't re-run the optimizer each time
  const [snap, setSnap] = useState(items);
  useEffect(() => { if (!busy && !undo) setSnap(items); }, [items, busy, undo]);
  const sets = useMemo(() => setsOf(snap, songMap), [snap, songMap]);
  const result = useMemo(() => sets.map((s) => {
    const order = optimizeSet(s.songs, opts);
    return { before: scoreSet(s.songs, opts), after: scoreSet(order, opts), order };
  }), [sets, opts]);
  const before = sumReports(result.map((r) => r.before));
  const after = sumReports(result.map((r) => r.after));
  const changed = result.some((r, i) => r.order.some((s, j) => s.id !== sets[i].songs[j]?.id));
  const noTempo = sets.flatMap((s) => s.songs).filter((s) => !s.tempo).length;
  const noKey = sets.flatMap((s) => s.songs).filter((s) => !s.key).length;
  // With a strong opener and closer, a set of n songs can space out at most floor((n - 1) / 2) slow ones
  const tooManySlow = sets.map((s, i) => ({
    label: sets.length > 1 ? (i === 0 ? "Set 1" : snap.filter((x) => x.kind === "break")[i - 1]?.label || `Set ${i + 1}`) : "This set",
    slow: s.songs.filter((x) => isSlow(songEnergy(x))).length,
    n: s.songs.length,
  })).filter((x) => x.n >= 3 && x.slow > Math.floor((x.n - 1) / 2));

  const apply = async () => {
    // Reuse each set's existing positions in the new order, so breaks stay exactly where they are
    setBusy(true);
    const saved = snap.map((i) => ({ id: i.id, position: i.position }));
    for (let i = 0; i < sets.length; i++) {
      const slots = sets[i].items.map((it) => it.position).sort((a, b) => a - b);
      for (let j = 0; j < result[i].order.length; j++) {
        const id = result[i].order[j].id;
        const it = sets[i].items.find((x) => x.id === id)!;
        if (it.position !== slots[j]) await patchRow(db.setlist_items, id, { position: slots[j] });
      }
    }
    setUndo(saved);
    setBusy(false);
  };
  const revert = async () => {
    if (!undo) return;
    setBusy(true);
    for (const u of undo) {
      const it = items.find((i) => i.id === u.id);
      if (it && it.position !== u.position) await patchRow(db.setlist_items, u.id, { position: u.position });
    }
    setUndo(null);
    setBusy(false);
  };

  const rows: [string, keyof FlowReport][] = [
    ["Same key back to back", "sameKey"],
    ["Slow songs back to back", "slowStacked"],
    ["Slow opener or closer", "weakOpenOrClose"],
    ["Instrument switches", "switches"],
    ["Same artist back to back", "sameArtist"],
    ["4+ songs in a row, same singer", "vocalRuns"],
  ];

  if (undo) {
    return (
      <div className="card stack" style={{ marginTop: 10 }}>
        <strong>New order applied.</strong>
        <div className="row">
          <button className="btn" onClick={revert}>Undo</button>
          <button className="btn primary" onClick={onClose}>Done</button>
        </div>
      </div>
    );
  }

  return (
    <div className="card stack" style={{ marginTop: 10 }}>
      <div className="row">
        <strong className="grow">Optimize running order</strong>
        <button className="btn small ghost" onClick={onClose}>Close</button>
      </div>
      <div className="small dim">
        Keeps your songs and set breaks; only changes the order inside each set. Aims for a strong opener, a breather in
        the middle, a big finish, no two songs in a row in the same key, and slow songs spread out.
      </div>
      <div className="row wrap" style={{ gap: 14 }}>
        <label className="check small"><input type="checkbox" checked={opts.keepOpener} onChange={(e) => setOpts({ ...opts, keepOpener: e.target.checked })} /> Keep my opener</label>
        <label className="check small"><input type="checkbox" checked={opts.keepCloser} onChange={(e) => setOpts({ ...opts, keepCloser: e.target.checked })} /> Keep my closer</label>
        <label className="check small"><input type="checkbox" checked={opts.fewerSwitches} onChange={(e) => setOpts({ ...opts, fewerSwitches: e.target.checked })} /> Fewer instrument switches</label>
      </div>
      <table className="small" style={{ borderCollapse: "collapse", width: "100%", maxWidth: 420 }}>
        <thead><tr><th style={{ textAlign: "left" }}></th><th>Now</th><th>Optimized</th></tr></thead>
        <tbody>
          {rows.map(([label, k]) => (
            <tr key={k}>
              <td style={{ padding: "3px 0" }}>{label}</td>
              <td style={{ textAlign: "center" }}>{before[k]}</td>
              <td style={{ textAlign: "center", fontWeight: 700, color: after[k] < before[k] ? "var(--ok)" : undefined }}>{after[k]}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {tooManySlow.map((x) => (
        <div key={x.label} className="small" style={{ color: "var(--accent)" }}>
          {x.label} has {x.slow} slow songs out of {x.n} — too many to keep apart without a slow opener or closer. Swap one for an upbeat song to clear it.
        </div>
      ))}
      {(noTempo > 0 || noKey > 0) && (
        <div className="small" style={{ color: "var(--accent)" }}>
          {noTempo > 0 && `${noTempo} song${noTempo === 1 ? " has" : "s have"} no tempo (treated as medium) — add BPM or tag "mellow"/"upbeat" for a better arc. `}
          {noKey > 0 && `${noKey} song${noKey === 1 ? " has" : "s have"} no key, so key clashes can't be checked for ${noKey === 1 ? "it" : "them"}.`}
        </div>
      )}
      {result.map((r, i) => r.order.length > 0 && (
        <div key={i}>
          {sets.length > 1 && <div className="small dim" style={{ fontWeight: 700, margin: "6px 0 2px" }}>{i === 0 ? "Set 1" : items.filter((x) => x.kind === "break")[i - 1]?.label || `Set ${i + 1}`}</div>}
          <ol className="small" style={{ margin: 0, paddingLeft: 22 }}>
            {r.order.map((s, j) => {
              const e = songEnergy(s);
              const prev = r.order[j - 1];
              return (
                <li key={s.id} style={{ padding: "2px 0" }}>
                  <span className="energy-bar" style={{ width: `${8 + e * 40}px`, background: isSlow(e) ? "var(--text-dim)" : "var(--accent)" }} title={`Energy ${Math.round(e * 100)}`} />
                  {" "}{s.title} <span className="dim">· {s.key ?? "no key"}{s.tempo ? ` · ${s.tempo}` : ""}{prev?.instrument && s.instrument && prev.instrument !== s.instrument ? " · ⇄" : ""}</span>
                </li>
              );
            })}
          </ol>
        </div>
      ))}
      <div className="row">
        <button className="btn primary" disabled={!changed || busy} onClick={apply}>{changed ? "Apply new order" : "Already in a good order"}</button>
        <span className="small dim">Bar = energy (grey = slow song). Drag to fine-tune afterwards.</span>
      </div>
    </div>
  );
}

/** Fill the set to a target length from the library, optionally limited to tags. */
function AutoBuild({ setlist, items, songs, onDone }: { setlist: Setlist; items: SetlistItem[]; songs: Song[]; onDone: () => void }) {
  const [minutes, setMinutes] = useState(90);
  const [tags, setTags] = useState("");
  const [noRepeatArtist, setNoRepeatArtist] = useState(true);
  const [breaks, setBreaks] = useState(0);

  const build = async () => {
    const wanted = tags.split(",").map((t) => t.trim().toLowerCase()).filter(Boolean);
    const already = new Set(items.map((i) => i.song_id));
    const artists = new Set(items.map((i) => songs.find((s) => s.id === i.song_id)?.artist.toLowerCase()).filter(Boolean));
    let pool = songs.filter((s) => !already.has(s.id) && (!wanted.length || s.tags.some((t) => wanted.includes(t))));
    pool = pool.sort(() => Math.random() - 0.5);
    let seconds = 0;
    const picked: Song[] = [];
    for (const s of pool) {
      if (seconds >= minutes * 60) break;
      if (noRepeatArtist && s.artist && artists.has(s.artist.toLowerCase())) continue;
      picked.push(s);
      artists.add(s.artist.toLowerCase());
      seconds += s.duration_sec || estimateDuration(s.tempo);
    }
    // Spread tempos: alternate faster/slower so it doesn't sag
    picked.sort((a, b) => (b.tempo ?? 100) - (a.tempo ?? 100));
    const order: Song[] = [];
    while (picked.length) order.push(picked.shift()!, ...(picked.length ? [picked.pop()!] : []));
    let pos = items[items.length - 1]?.position;
    const perSet = breaks > 0 ? Math.ceil(order.length / (breaks + 1)) : Infinity;
    for (let i = 0; i < order.length; i++) {
      if (i > 0 && i % perSet === 0) {
        pos = positionBetween(pos, undefined);
        await saveRow(db.setlist_items, blankItem({ setlist_id: setlist.id, kind: "break", label: `Set ${i / perSet + 1}`, position: pos }));
      }
      pos = positionBetween(pos, undefined);
      await saveRow(db.setlist_items, blankItem({ setlist_id: setlist.id, song_id: order[i].id, position: pos }));
    }
    onDone();
  };

  return (
    <div className="card stack" style={{ marginTop: 10 }}>
      <div className="meta-grid">
        <label className="field"><span>Target minutes</span>
          <input className="input" inputMode="numeric" value={minutes} onChange={(e) => setMinutes(Number(e.target.value) || 0)} />
        </label>
        <label className="field"><span>Set breaks</span>
          <select className="select" value={breaks} onChange={(e) => setBreaks(Number(e.target.value))}>
            {[0, 1, 2, 3].map((n) => <option key={n} value={n}>{n === 0 ? "None" : `${n} (${n + 1} sets)`}</option>)}
          </select>
        </label>
        <label className="field" style={{ gridColumn: "span 2" }}><span>Only tags (optional)</span>
          <input className="input" placeholder="country, upbeat" value={tags} onChange={(e) => setTags(e.target.value)} />
        </label>
      </div>
      <label className="check"><input type="checkbox" checked={noRepeatArtist} onChange={(e) => setNoRepeatArtist(e.target.checked)} /> One song per artist</label>
      <div className="row">
        <button className="btn primary" onClick={build}>Build</button>
        <span className="small dim">Adds songs after what's already in the set. Drag to fine-tune.</span>
      </div>
    </div>
  );
}

