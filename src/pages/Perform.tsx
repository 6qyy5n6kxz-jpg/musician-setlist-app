import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { IconClose, IconInbox, IconList, IconMic, IconTv } from "../components/Icons";
import { SongStage, type StageHandle } from "../components/SongStage";
import { blankItem, db, patchRow, positionBetween, saveRow, type Song } from "../lib/db";
import { useProfile, useSetlistItems, useSetlists, useSong, useSongs } from "../lib/hooks";
import { useLivePublisher } from "../lib/live";
import { guessKey, keyDistance } from "../lib/music/chords";
import { allChords, lyricSlides, parseChordPro, type Section } from "../lib/music/chordpro";
import { describeRequest, karaokeLineup, openQueue, useRequests, type SongRequest } from "../lib/requests";
import { KaraokePanel } from "../components/KaraokePanel";
import { useSettings } from "../lib/settings";
import { useSyncStatus } from "../lib/sync";
import { useWakeLock } from "../lib/stage";

export function Perform() {
  const { setlistId, songId } = useParams();
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const settings = useSettings();
  const setlists = useSetlists();
  const setlist = setlistId ? setlists?.find((s) => s.id === setlistId) : null;
  const items = useSetlistItems(setlistId);
  const songs = useSongs();
  const singleSong = useSong(songId);
  const profile = useProfile();
  const { userEmail } = useSyncStatus();
  const { requests, newCount, toasts, dismissToast, setStatus } = useRequests();

  const songMap = useMemo(() => new Map((songs ?? []).map((s) => [s.id, s])), [songs]);
  const playable = useMemo(
    () => (items ?? []).filter((i) => i.kind === "song" && i.song_id && songMap.has(i.song_id)),
    [items, songMap],
  );
  const index = Math.min(Math.max(0, Number(params.get("i") ?? 0)), Math.max(0, playable.length - 1));
  const [extra, setExtra] = useState<Song | null>(null); // a request played outside the set
  const item = setlistId ? playable[index] : undefined;
  const song: Song | null | undefined = extra ?? (setlistId ? (item ? songMap.get(item.song_id!) : null) : singleSong);

  const [sideOpen, setSideOpen] = useState(() => window.innerWidth > 1000);
  const [drawer, setDrawer] = useState<"requests" | "live" | "karaoke" | null>(null);
  const [singerId, setSingerId] = useState<string | null>(null);
  const [slidePos, setSlidePos] = useState(0);
  const [blank, setBlank] = useState(false);
  const [message, setMessage] = useState("");
  const sectionsRef = useRef<{ sections: Section[]; flowUsed: boolean }>({ sections: [], flowUsed: false });
  const stage = useRef<StageHandle>(null);

  useWakeLock(true);

  const go = useCallback((i: number) => {
    setExtra(null);
    setSlidePos(0);
    setParams({ i: String(Math.max(0, Math.min(playable.length - 1, i))) }, { replace: true });
  }, [playable.length, setParams]);
  const next = useCallback(() => { if (extra) setExtra(null); else if (index < playable.length - 1) go(index + 1); }, [extra, index, playable.length, go]);
  const prev = useCallback(() => { if (extra) setExtra(null); else if (index > 0) go(index - 1); }, [extra, index, go]);

  // ------------------------------------------------------------ live broadcast
  const liveEnabled = !!userEmail && !!profile?.live_token;
  const publish = useLivePublisher(profile?.live_token, liveEnabled);
  const performKey = extra ? null : item?.key_override ?? null;

  const publishSong = useCallback(() => {
    if (!song) return;
    const { sections, flowUsed } = sectionsRef.current;
    const written = song.song_key || guessKey(allChords(parseChordPro(song.content)));
    const upNextItem = !extra && setlistId ? playable[index + 1] : undefined;
    const upNext = upNextItem ? songMap.get(upNextItem.song_id!) : undefined;
    publish({
      setlist: setlist?.name ?? null,
      song: {
        id: song.id, title: song.title, artist: song.artist, key: written,
        performKey: performKey ?? written, transpose: performKey && written ? keyDistance(written, performKey) : 0,
        capo: item?.capo_override ?? song.capo, tempo: song.tempo, timeSignature: song.time_signature,
        flow: flowUsed ? song.flow : null, content: song.content,
      },
      slides: lyricSlides(sections),
      upNext: upNext ? { title: upNext.title, artist: upNext.artist } : null,
    });
  }, [song, extra, setlistId, playable, index, songMap, setlist?.name, performKey, item?.capo_override, publish]);

  const onSectionsChange = useCallback((sections: Section[], flowUsed: boolean) => {
    sectionsRef.current = { sections, flowUsed };
    publishSong();
  }, [publishSong]);

  useEffect(() => { publish({ slide: slidePos }); }, [slidePos]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { publish({ blank }); }, [blank]); // eslint-disable-line react-hooks/exhaustive-deps

  const onActiveSection = useCallback((pos: number) => {
    if (settings.displayFollowsScroll) setSlidePos(pos);
  }, [settings.displayFollowsScroll]);

  const stepSlide = useCallback((dir: 1 | -1) => {
    const slides = lyricSlides(sectionsRef.current.sections);
    if (!slides.length) return;
    const idx = slides.findIndex((s) => s.pos === slidePos);
    const target = idx >= 0
      ? slides[Math.max(0, Math.min(slides.length - 1, idx + dir))]
      : dir === 1
        ? slides.find((s) => s.pos > slidePos) ?? slides[slides.length - 1]
        : [...slides].reverse().find((s) => s.pos < slidePos) ?? slides[0];
    setSlidePos(target.pos);
    stage.current?.scrollToPos(target.pos);
  }, [slidePos]);

  // ------------------------------------------------------------ requests
  const queue = openQueue(requests).filter((r) => r.status === "new" || r.status === "queued");
  const playRequest = async (r: SongRequest) => {
    const s = r.song_id ? songMap.get(r.song_id) : undefined;
    void setStatus(r.id, "played");
    setDrawer(null);
    if (!s) return;
    const at = playable.findIndex((p) => p.song_id === s.id);
    if (at >= 0) go(at);
    else setExtra(s);
  };
  const addRequestNext = async (r: SongRequest) => {
    if (!r.song_id || !setlistId || !items) return;
    const curItem = playable[index];
    const all = items;
    const curIdx = all.findIndex((i) => i.id === curItem?.id);
    await saveRow(db.setlist_items, blankItem({
      setlist_id: setlistId, song_id: r.song_id,
      position: positionBetween(all[curIdx]?.position, all[curIdx + 1]?.position),
    }));
    void setStatus(r.id, "queued");
  };

  // ------------------------------------------------------------ karaoke
  const lineup = karaokeLineup(requests).filter((r) => r.id !== singerId);
  const singer = singerId ? requests.find((r) => r.id === singerId) : undefined;
  const startSinger = (r: SongRequest) => {
    if (singerId && singerId !== r.id) void setStatus(singerId, "played");
    setSingerId(r.id);
    if (r.status === "new") void setStatus(r.id, "queued");
    setDrawer(null);
    const s = r.song_id ? songMap.get(r.song_id) : undefined;
    if (s) {
      const at = playable.findIndex((p) => p.song_id === s.id);
      if (at >= 0) go(at);
      else setExtra(s);
    }
  };
  const finishSinger = () => {
    if (singerId) void setStatus(singerId, "played");
    setSingerId(null);
    setExtra(null);
  };
  const lineupKey = lineup.map((r) => r.id).join(",");
  useEffect(() => {
    publish({
      singer: singer?.patron_name ?? null,
      nextSingers: lineup.slice(0, 3).map((r) => ({ name: r.patron_name ?? "", title: r.title })),
    });
  }, [singer?.patron_name, lineupKey]); // eslint-disable-line react-hooks/exhaustive-deps

  if (setlistId && (setlists === undefined || items === undefined || songs === undefined)) return null;
  if (songId && singleSong === undefined) return null;

  const exit = () => navigate(setlistId ? `/sets/${setlistId}` : songId ? `/song/${songId}` : "/");
  const popups = settings.requestPopups ? toasts : [];
  const currentSlide = lyricSlides(sectionsRef.current.sections).find((s) => s.pos === slidePos);

  let n = 0;
  return (
    <div className="perform">
      <div className="perform-top">
        <button className="btn icon ghost" onClick={exit} aria-label="Exit perform mode"><IconClose /></button>
        {setlistId && <button className={`btn icon ${sideOpen ? "on" : ""}`} onClick={() => setSideOpen(!sideOpen)} aria-label="Show set"><IconList /></button>}
        <div className="grow" style={{ minWidth: 0 }}>
          <div className="title truncate">
            {setlistId && !extra && <span className="dim">{index + 1}/{playable.length} · </span>}
            {extra && <span className="chip accent" style={{ marginRight: 6 }}>Request</span>}
            {song?.title ?? "Empty set"}
          </div>
          {setlistId && !extra && playable[index + 1] && (
            <div className="up-next truncate">Next: {songMap.get(playable[index + 1].song_id!)?.title}</div>
          )}
        </div>
        {liveEnabled && (
          <div className="row" style={{ gap: 4 }}>
            <button className="btn small icon" onClick={() => stepSlide(-1)} aria-label="Previous lyric section">‹</button>
            <button className={`btn small ${blank ? "" : "on"}`} onClick={() => setDrawer(drawer === "live" ? null : "live")} title="Lyrics display & band">
              <IconTv size={18} /> <span className="truncate" style={{ maxWidth: 110 }}>{blank ? "Blank" : currentSlide?.label || "Live"}</span>
            </button>
            <button className="btn small icon" onClick={() => stepSlide(1)} aria-label="Next lyric section">›</button>
          </div>
        )}
        {(profile?.karaoke_open || lineup.length > 0 || singer) && (
          <button className={`btn small ${singer ? "on" : ""}`} onClick={() => setDrawer(drawer === "karaoke" ? null : "karaoke")} title="Karaoke lineup">
            <IconMic size={18} /> {singer ? <span className="truncate" style={{ maxWidth: 90 }}>{singer.patron_name}</span> : lineup.length || ""}
          </button>
        )}
        <button className="btn small" onClick={() => setDrawer(drawer === "requests" ? null : "requests")} style={{ position: "relative" }}>
          <IconInbox size={18} /> {queue.length || ""}
          {newCount > 0 && <span className="badge" style={{ position: "absolute", top: -6, right: -6 }}>{newCount}</span>}
        </button>
        <button className="btn small" onClick={prev} disabled={!extra && index === 0}>‹ Prev</button>
        <button className="btn small primary" onClick={next} disabled={!extra && index >= playable.length - 1}>Next ›</button>
      </div>

      <div className="perform-body">
        {setlistId && (
          <aside className={`perform-side ${sideOpen ? "" : "hidden"}`}>
            {(items ?? []).map((it) => {
              if (it.kind === "break") { n = 0; return <div key={it.id} className="brk">{it.label}</div>; }
              const s = it.song_id ? songMap.get(it.song_id) : undefined;
              if (!s) return null;
              n++;
              const pIdx = playable.findIndex((p) => p.id === it.id);
              return (
                <button key={it.id} className={pIdx === index && !extra ? "current" : ""} onClick={() => go(pIdx)}>
                  <span className="dim small" style={{ width: 20, textAlign: "right" }}>{n}</span>
                  <span className="grow">
                    <span style={{ display: "block", fontWeight: 600 }}>{s.title}</span>
                    <span className="small dim">{it.key_override || s.song_key || ""}</span>
                  </span>
                </button>
              );
            })}
          </aside>
        )}
        <div className="grow" style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
          {song ? (
            <SongStage
              key={song.id}
              song={song}
              handleRef={stage}
              performKey={performKey}
              capoOverride={extra ? null : item?.capo_override ?? null}
              onPerformKeyChange={item && !extra ? (k) => patchRow(db.setlist_items, item.id, { key_override: k }) : undefined}
              onCapoChange={item && !extra ? (c) => patchRow(db.setlist_items, item.id, { capo_override: c === song.capo ? null : c }) : undefined}
              onSectionsChange={onSectionsChange}
              onActiveSection={onActiveSection}
              onTimedSection={setSlidePos}
              onReachEnd={setlistId ? next : undefined}
              onReachStart={setlistId ? prev : undefined}
              activePos={liveEnabled ? slidePos : null}
              pedalsEnabled={!drawer}
              pedalHandlers={{
                nextSong: next, prevSong: prev,
                nextSlide: () => stepSlide(1), prevSlide: () => stepSlide(-1),
                blankDisplay: () => setBlank((b) => !b),
              }}
              kicker={item?.notes ? <div className="sticky-note">{item.notes}</div> : null}
            />
          ) : (
            <div className="empty-state">This set has no songs yet.</div>
          )}
        </div>
      </div>

      {popups.length > 0 && !drawer && (
        <div className="toast-stack">
          {popups.map((t) => (
            <div key={t.id} className="toast">
              <span className="badge">!</span>
              <div className="grow" onClick={() => { dismissToast(t.id); setDrawer(t.kind === "karaoke" ? "karaoke" : "requests"); }}>
                <div style={{ fontWeight: 700 }}>{describeRequest(t).title}</div>
                <div className="small dim">{describeRequest(t).detail}</div>
              </div>
              <button className="btn small" onClick={() => { void setStatus(t.id, "queued"); dismissToast(t.id); }}>{t.kind === "karaoke" ? "Approve" : "Queue"}</button>
              <button className="btn small ghost icon" onClick={() => dismissToast(t.id)} aria-label="Dismiss"><IconClose size={18} /></button>
            </div>
          ))}
        </div>
      )}

      {drawer && <div className="scrim" onClick={() => setDrawer(null)} />}
      {drawer === "requests" && (
        <div className="drawer">
          <div className="drawer-head">
            <h2 className="grow" style={{ fontSize: "1.15rem" }}>Request queue</h2>
            <button className="btn icon ghost" onClick={() => setDrawer(null)} aria-label="Close"><IconClose /></button>
          </div>
          <div className="drawer-body">
            {!userEmail && <p className="dim">Sign in (Settings) to receive requests.</p>}
            {userEmail && queue.length === 0 && <p className="dim">No requests yet. Turn requests on and put the QR code out from the Requests tab.</p>}
            {queue.map((r) => {
              const inLibrary = r.song_id && songMap.has(r.song_id);
              return (
                <div key={r.id} className={`req ${r.status}`}>
                  <div className="row">
                    <div className="grow">
                      <div className="req-title">{r.title}</div>
                      <div className="small dim">{r.artist}{r.patron_name ? ` · for ${r.patron_name}` : ""} · {new Date(r.created_at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}</div>
                    </div>
                    {r.status === "queued" && <span className="chip accent">queued</span>}
                  </div>
                  {r.message && <div className="req-msg">“{r.message}”</div>}
                  <div className="req-actions">
                    {inLibrary && <button className="btn small primary" onClick={() => playRequest(r)}>Play now</button>}
                    {inLibrary && setlistId && <button className="btn small" onClick={() => addRequestNext(r)}>Play next</button>}
                    {r.status === "new" && <button className="btn small" onClick={() => setStatus(r.id, "queued")}>Queue</button>}
                    {!inLibrary && <button className="btn small" onClick={() => setStatus(r.id, "played")}>Played</button>}
                    <button className="btn small ghost danger" onClick={() => setStatus(r.id, "declined")}>Decline</button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
      {drawer === "karaoke" && (
        <div className="drawer">
          <div className="drawer-head">
            <h2 className="grow" style={{ fontSize: "1.15rem" }}>Karaoke</h2>
            {singer && <button className="btn small primary" onClick={finishSinger}>Done singing</button>}
            <button className="btn icon ghost" onClick={() => setDrawer(null)} aria-label="Close"><IconClose /></button>
          </div>
          <div className="drawer-body">
            {userEmail ? <KaraokePanel onStart={startSinger} currentSingerId={singerId} /> : <p className="dim">Sign in (Settings) to run karaoke sign-up.</p>}
          </div>
        </div>
      )}
      {drawer === "live" && (
        <div className="drawer">
          <div className="drawer-head">
            <h2 className="grow" style={{ fontSize: "1.15rem" }}>Lyrics display & band</h2>
            <button className="btn icon ghost" onClick={() => setDrawer(null)} aria-label="Close"><IconClose /></button>
          </div>
          <div className="drawer-body stack">
            <div className="row">
              <button className={`btn grow ${blank ? "on" : ""}`} onClick={() => setBlank(!blank)}>{blank ? "Display is blank — tap to show" : "Blank the display"}</button>
            </div>
            <div className="stack" style={{ gap: 6 }}>
              {lyricSlides(sectionsRef.current.sections).map((s) => (
                <button key={s.pos} className={`btn ${s.pos === slidePos ? "on" : ""}`} style={{ justifyContent: "flex-start", height: "auto", padding: "8px 12px", textAlign: "left" }}
                  onClick={() => { setSlidePos(s.pos); stage.current?.scrollToPos(s.pos); }}>
                  <span>
                    <span style={{ display: "block", fontSize: "0.75rem", textTransform: "uppercase", opacity: 0.7 }}>{s.label || "—"}</span>
                    <span style={{ fontWeight: 400 }}>{s.lines[0]}</span>
                  </span>
                </button>
              ))}
            </div>
            <label className="field"><span>Message to band screens</span>
              <div className="row">
                <input className="input" value={message} onChange={(e) => setMessage(e.target.value)} placeholder="Skip the bridge · Key change!" />
                <button className="btn" onClick={() => publish({ message: message.trim() || null })}>Send</button>
                <button className="btn ghost" onClick={() => { setMessage(""); publish({ message: null }); }}>Clear</button>
              </div>
            </label>
            <p className="small dim">
              {settings.displayFollowsScroll ? "The display follows the section at the top of your screen. " : ""}
              Pedal “next section” moves both. Links for the TV and band are in Settings → Live screens.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
