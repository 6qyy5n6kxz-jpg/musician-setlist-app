import { useLiveQuery } from "dexie-react-hooks";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { IconBack } from "../components/Icons";
import { db, live } from "../lib/db";
import { useProfile } from "../lib/hooks";
import { appUrl } from "../lib/links";
import { useRequests } from "../lib/requests";
import { PEDAL_ACTION_LABELS, useSettings } from "../lib/settings";
import { Metronome } from "../lib/stage";
import { supabase } from "../lib/supabase";
import { syncNow, useSyncStatus } from "../lib/sync";

type Status = "ok" | "warn" | "fail" | "pending";

function Row({ status, title, detail, children }: { status: Status; title: string; detail?: ReactNode; children?: ReactNode }) {
  const icon = { ok: "✓", warn: "!", fail: "✕", pending: "…" }[status];
  const color = { ok: "var(--ok)", warn: "var(--accent)", fail: "var(--danger)", pending: "var(--text-dim)" }[status];
  return (
    <li className="list-item" style={{ alignItems: "flex-start", padding: "14px" }}>
      <span style={{ width: 28, height: 28, borderRadius: 14, display: "inline-flex", alignItems: "center", justifyContent: "center", background: color, color: "#111", fontWeight: 800, flexShrink: 0 }}>{icon}</span>
      <div className="grow">
        <div style={{ fontWeight: 700 }}>{title}</div>
        {detail && <div className="small dim" style={{ marginTop: 2 }}>{detail}</div>}
        {children && <div className="row wrap" style={{ marginTop: 8 }}>{children}</div>}
      </div>
    </li>
  );
}

/** Pre-gig self test: run it on the iPad at soundcheck. */
export function GigCheck() {
  const navigate = useNavigate();
  const settings = useSettings();
  const profile = useProfile();
  const sync = useSyncStatus();
  const { requests, remove } = useRequests();

  const standalone = window.matchMedia("(display-mode: standalone)").matches || !!(navigator as { standalone?: boolean }).standalone;
  const [persisted, setPersisted] = useState<boolean | null>(null);
  const [swReady, setSwReady] = useState<boolean | null>(null);
  const [rt, setRt] = useState<{ status: Status; ms?: number }>({ status: "pending" });
  const [wake, setWake] = useState<Status>("pending");
  const [audio, setAudio] = useState<Status>("warn");
  const [lastKey, setLastKey] = useState<string | null>(null);
  const [reqTest, setReqTest] = useState<{ status: Status; msg: string; sentAt?: number }>({ status: "warn", msg: "Not run yet" });
  const metro = useRef(new Metronome());

  const counts = useLiveQuery(async () => {
    const files = live(await db.song_files.toArray());
    const blobs = new Set((await db.blobs.toArray()).map((b) => b.id));
    const songs = live(await db.songs.toArray());
    return { songs: songs.length, charted: songs.filter((s) => s.content.trim()).length, files: files.length, offline: files.filter((f) => blobs.has(f.id)).length };
  }, []);

  useEffect(() => {
    void navigator.storage?.persisted?.().then(setPersisted);
    setSwReady(!!navigator.serviceWorker?.controller);
    void syncNow();
  }, []);

  // Live connection: broadcast to ourselves and time the round trip.
  const testRealtime = () => {
    setRt({ status: "pending" });
    const ch = supabase.channel(`gigcheck-${crypto.randomUUID()}`, { config: { broadcast: { self: true } } });
    const started = performance.now();
    const timeout = setTimeout(() => { setRt({ status: "fail" }); void ch.unsubscribe(); }, 8000);
    ch.on("broadcast", { event: "ping" }, () => {
      clearTimeout(timeout);
      setRt({ status: "ok", ms: Math.round(performance.now() - started) });
      void ch.unsubscribe();
    });
    ch.subscribe((s) => { if (s === "SUBSCRIBED") void ch.send({ type: "broadcast", event: "ping", payload: {} }); });
  };
  useEffect(() => { if (navigator.onLine) testRealtime(); else setRt({ status: "fail" }); }, []);

  const testWake = async () => {
    if (!("wakeLock" in navigator)) return setWake("fail");
    try {
      const l = await navigator.wakeLock.request("screen");
      setWake("ok");
      setTimeout(() => void l.release(), 1000);
    } catch {
      setWake("fail");
    }
  };
  useEffect(() => { void testWake(); }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT") return;
      setLastKey(e.key);
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Request pipeline: send a request through the public endpoint and wait for it to arrive here.
  const sendTestRequest = async () => {
    if (!profile) return;
    setReqTest({ status: "pending", msg: "Sending…" });
    const { data, error } = await supabase.rpc("submit_request", {
      p_token: profile.request_token, p_song_id: null, p_title: "Gig check test", p_artist: null, p_name: "Gig check", p_message: null,
    });
    const res = data as { ok: boolean; error?: string } | null;
    if (error || !res?.ok) return setReqTest({ status: "fail", msg: res?.error ?? error?.message ?? "Failed" });
    setReqTest({ status: "pending", msg: "Sent — waiting for it to arrive…", sentAt: Date.now() });
  };
  useEffect(() => {
    if (reqTest.status !== "pending" || !reqTest.sentAt) return;
    const hit = requests.find((r) => r.title === "Gig check test" && r.patron_name === "Gig check");
    if (hit) {
      setReqTest({ status: "ok", msg: `Arrived in ${((Date.now() - reqTest.sentAt) / 1000).toFixed(1)}s` });
      void remove(hit.id);
      return;
    }
    const t = setTimeout(() => setReqTest((r) => (r.status === "pending" ? { status: "fail", msg: "Sent, but it didn't arrive within 10 seconds." } : r)), 10_000);
    return () => clearTimeout(t);
  }, [requests, reqTest, remove]);

  const playClick = () => {
    const m = metro.current;
    m.bpm = 120; m.beatsPerBar = 4; m.volume = settings.metronomeVolume;
    m.onDone = () => setAudio("ok");
    m.start(1);
  };

  const mapped = lastKey ? settings.pedalMap[lastKey] : undefined;
  const signedIn = !!sync.userEmail;

  return (
    <div className="page" style={{ maxWidth: 820 }}>
      <div className="row" style={{ marginBottom: 6 }}>
        <button className="btn ghost icon" onClick={() => navigate(-1)} aria-label="Back"><IconBack /></button>
        <h1 className="grow">Gig check</h1>
        <button className="btn" onClick={() => location.reload()}>Run again</button>
      </div>
      <p className="dim" style={{ marginTop: 0 }}>Run this on the iPad at soundcheck, on the same connection you'll use for the show (usually your phone's hotspot).</p>

      <h2 style={{ fontSize: "1rem", margin: "18px 0 8px" }} className="dim">Works with no signal</h2>
      <ul className="list">
        <Row status={standalone ? "ok" : "fail"} title={standalone ? "Installed on the home screen" : "Not opened from the home screen"}
          detail={standalone ? "Safari won't clear this app's storage." : "In Safari: Share → Add to Home Screen, then open the app from that icon. Browser tabs can lose offline data."} />
        <Row status={swReady ? "ok" : "warn"} title={swReady ? "App is saved for offline use" : "Offline copy not active yet"}
          detail={swReady ? "The app opens without internet." : "Close and reopen the app once while online."} />
        <Row status={persisted ? "ok" : "warn"} title={persisted ? "Storage protected" : "Storage not protected"}
          detail={persisted ? "The browser won't evict your songs." : "Tap Protect so iPadOS keeps your library under storage pressure."}>
          {!persisted && <button className="btn small" onClick={async () => setPersisted((await navigator.storage?.persist?.()) ?? false)}>Protect</button>}
        </Row>
        <Row status={counts && counts.offline === counts.files ? "ok" : "warn"} title={counts ? `${counts.songs} songs on this iPad (${counts.charted} with charts)` : "Counting…"}
          detail={counts ? (counts.files ? `${counts.offline} of ${counts.files} PDFs / backing tracks downloaded.` : "No PDFs or backing tracks attached yet.") + (counts.offline < counts.files ? " Stay online until the rest finish downloading." : "") : undefined} />
        <Row status={wake} title={wake === "ok" ? "Screen can stay awake" : "Screen may dim and lock"}
          detail={wake === "ok" ? "Perform mode keeps the display on." : "This iPadOS version can't hold the screen on from a home-screen app. For gigs set Settings → Display & Brightness → Auto-Lock → Never."} />
        <Row status={audio} title={audio === "ok" ? "Sound works" : "Test the click"} detail="Plays one bar of metronome. Check the iPad isn't muted and the volume is up.">
          <button className="btn small" onClick={playClick}>Play click</button>
        </Row>
        <Row status={lastKey ? (mapped ? "ok" : "warn") : "warn"} title="Foot pedal"
          detail={lastKey ? (mapped ? `Got “${lastKey === " " ? "Space" : lastKey}” → ${PEDAL_ACTION_LABELS[mapped]}` : `Got “${lastKey}”, which isn't mapped to anything. Map it in Settings → Foot pedals.`) : "Press each pedal now. You should see what it does here."} />
      </ul>

      <h2 style={{ fontSize: "1rem", margin: "22px 0 8px" }} className="dim">Needs a connection (hotspot)</h2>
      <ul className="list">
        <Row status={!navigator.onLine ? "fail" : signedIn ? (sync.phase === "error" ? "fail" : "ok") : "fail"}
          title={!navigator.onLine ? "Offline" : signedIn ? `Signed in — ${sync.phase === "error" ? "sync error" : sync.pending ? `${sync.pending} changes waiting` : "everything synced"}` : "Not signed in"}
          detail={sync.error ?? (signedIn ? sync.userEmail : "Sign in under Settings to sync and take requests.")} />
        <Row status={rt.status} title={rt.status === "ok" ? `Live connection works (${rt.ms} ms)` : rt.status === "pending" ? "Testing live connection…" : "Live connection failed"}
          detail="Used by requests, the lyrics display and band screens. Under ~500 ms is great; over 2 s will feel laggy.">
          <button className="btn small" onClick={testRealtime}>Test again</button>
        </Row>
        <Row status={!profile ? "fail" : reqTest.status} title="Audience requests reach you"
          detail={!profile ? "Sign in first." : !profile.requests_open ? "Requests are closed — open them on the Requests tab, then run this." : reqTest.msg}>
          {profile?.requests_open && <button className="btn small" onClick={sendTestRequest}>Send test request</button>}
        </Row>
        <Row status={profile ? "ok" : "warn"} title="Lyrics display" detail="Open it on the TV device and check it says “Lyrics will appear here”. Then start Perform mode and scroll.">
          {profile && <a className="btn small" href={appUrl(`/display/${profile.live_token}`)} target="_blank" rel="noreferrer">Open display</a>}
        </Row>
      </ul>
    </div>
  );
}
