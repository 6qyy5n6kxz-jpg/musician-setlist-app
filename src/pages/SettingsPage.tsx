import { useLiveQuery } from "dexie-react-hooks";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { GearLibraryEditor } from "../components/GearLibraryEditor";
import { MidiOutPanel } from "../components/MidiOutPanel";
import { QrCode } from "../components/QrCode";
import { SyncBadge } from "../components/SyncBadge";
import { db, live } from "../lib/db";
import { useProfile } from "../lib/hooks";
import { appUrl } from "../lib/links";
import { DEFAULT_SETTINGS, PEDAL_ACTION_LABELS, updateSettings, useSettings, type PedalAction, type Theme } from "../lib/settings";
import { supabase } from "../lib/supabase";
import { signOut, syncNow, useSyncStatus } from "../lib/sync";
import SEND_TO_STAGE_JS from "../../shortcut/send-to-stage.js?raw";

export function SettingsPage() {
  return (
    <div className="page stack" style={{ gap: 18, maxWidth: 900 }}>
      <div className="row"><h1 className="grow">Settings</h1><SyncBadge /></div>
      <div className="card row wrap" style={{ borderColor: "var(--accent)" }}>
        <div className="grow">
          <strong>Gig check</strong>
          <div className="small dim">Run at soundcheck: install, offline library, screen, sound, pedal, requests and live screens.</div>
        </div>
        <Link className="btn primary" to="/gigcheck">Run gig check</Link>
      </div>
      <Account />
      <SendToStage />
      <GearSection />
      <LiveScreens />
      <StageDisplay />
      <Pedals />
      <Performance />
      <Device />
      <Backup />
    </div>
  );
}

function Section({ title, children, hint }: { title: string; children: React.ReactNode; hint?: string }) {
  return (
    <section className="card stack">
      <div>
        <h2 style={{ fontSize: "1.15rem" }}>{title}</h2>
        {hint && <p className="small dim" style={{ margin: "4px 0 0" }}>{hint}</p>}
      </div>
      {children}
    </section>
  );
}

function Account() {
  const { userEmail, phase, lastSynced, error, pending, lostSession } = useSyncStatus();
  const [email, setEmail] = useState(lostSession ?? "");
  const [password, setPassword] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const signIn = async () => {
    setBusy(true); setMsg(null);
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    setBusy(false);
    if (error) setMsg(error.message === "Email not confirmed" ? "Check your email and tap the confirmation link first." : error.message);
  };
  const signUp = async () => {
    if (password.length < 8) return setMsg("Use at least 8 characters for your password.");
    setBusy(true); setMsg(null);
    const { data, error } = await supabase.auth.signUp({ email, password, options: { emailRedirectTo: appUrl("/settings") } });
    setBusy(false);
    if (error) setMsg(error.message.includes("Registration is closed") || error.message.includes("Database error") ? "This app already has its owner account. Sign in instead." : error.message);
    else if (!data.session) setMsg("Account created. Check your email for a confirmation link, then sign in here.");
  };
  const reset = async () => {
    if (!email) return setMsg("Enter your email first.");
    const { error } = await supabase.auth.resetPasswordForEmail(email, { redirectTo: appUrl("/settings") });
    setMsg(error ? error.message : "Password reset email sent.");
  };

  if (userEmail) {
    return (
      <Section title="Account & sync" hint="Everything is stored on this device and synced to the cloud when you have a connection.">
        <div className="row wrap">
          <div className="grow">
            <div>Signed in as <strong>{userEmail}</strong></div>
            <div className="small dim">
              {phase === "offline" ? "Offline — changes are saved on this device and will sync later." : lastSynced ? `Last synced ${new Date(lastSynced).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}` : "Not synced yet"}
              {pending ? ` · ${pending} change${pending === 1 ? "" : "s"} waiting` : ""}
            </div>
            {error && <div className="small" style={{ color: "var(--danger)" }}>{error}</div>}
          </div>
          <button className="btn" onClick={() => void syncNow()}>Sync now</button>
          <button className="btn ghost" onClick={() => void signOut()}>Sign out</button>
        </div>
        <ChangePassword />
      </Section>
    );
  }
  return (
    <Section title="Account & sync" hint="Sign in to sync between devices, take audience requests and drive the lyrics display. The app works offline either way.">
      <div className="meta-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <label className="field"><span>Email</span><input className="input" type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} /></label>
        <label className="field"><span>Password</span><input className="input" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} onKeyDown={(e) => e.key === "Enter" && signIn()} /></label>
      </div>
      <div className="row wrap">
        <button className="btn primary" onClick={signIn} disabled={busy || !email || !password}>Sign in</button>
        <button className="btn" onClick={signUp} disabled={busy || !email || !password}>Create owner account</button>
        <button className="btn ghost" onClick={reset}>Forgot password</button>
      </div>
      {msg && <div className="small">{msg}</div>}
    </Section>
  );
}

function ChangePassword() {
  const [open, setOpen] = useState(false);
  const [pw, setPw] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  // Arriving from a password-reset email opens this automatically
  useEffect(() => supabase.auth.onAuthStateChange((e) => { if (e === "PASSWORD_RECOVERY") setOpen(true); }).data.subscription.unsubscribe, []);
  if (!open) return <button className="btn small ghost" style={{ alignSelf: "flex-start" }} onClick={() => setOpen(true)}>Change password</button>;
  return (
    <div className="row wrap">
      <input className="input" style={{ maxWidth: 280 }} type="password" autoComplete="new-password" placeholder="New password (8+ characters)" value={pw} onChange={(e) => setPw(e.target.value)} />
      <button className="btn" disabled={pw.length < 8} onClick={async () => {
        const { error } = await supabase.auth.updateUser({ password: pw });
        setMsg(error ? error.message : "Password updated.");
        if (!error) { setPw(""); setOpen(false); }
      }}>Save</button>
      {msg && <span className="small">{msg}</span>}
    </div>
  );
}

function GearSection() {
  const profile = useProfile();
  return (
    <Section title="Gear & MIDI Captain" hint="Numa X, Nano Cortex and BeatBuddy settings per song, switched from your MIDI Captain.">
      <div className="sticky-note" style={{ margin: 0 }}>
        <strong>MIDI Captain → this app:</strong> set a switch to send a keyboard key over USB (HID mode in the Captain's
        settings — e.g. Page Down), plug the Captain into the iPad, then map that key under <em>Foot pedals & keys</em> below with
        <em> Learn</em>. One switch can send a key to the app and MIDI to your gear at the same time.
      </div>
      {profile ? <GearLibraryEditor profile={profile} /> : <div className="small dim">Sign in to set up your gear.</div>}
      {profile && <MidiOutPanel profile={profile} />}
    </Section>
  );
}

function SendToStage() {
  const [copied, setCopied] = useState(false);
  return (
    <Section title="Send charts from Ultimate Guitar" hint="Share a chart straight into the app, like sending it to OnSong.">
      <ol className="small" style={{ margin: 0, paddingLeft: 20, lineHeight: 1.7 }}>
        <li>Install the <strong>Send to Stage</strong> shortcut on the iPad (AirDrop the file from your Mac, or build it with the steps below).</li>
        <li>In Safari, open a chart on ultimate-guitar.com, tap <strong>Share</strong> → <strong>Send to Stage</strong>.</li>
        <li>The import screen opens with the chart converted — tap <strong>Import</strong>. If the song is already in your library without a chart, the chart is added to it.</li>
      </ol>
      <details>
        <summary style={{ cursor: "pointer", fontWeight: 600 }}>Build the shortcut by hand</summary>
        <ol className="small" style={{ paddingLeft: 20, lineHeight: 1.7 }}>
          <li>Shortcuts app → <strong>+</strong> → name it “Send to Stage”.</li>
          <li>Tap the <strong>ⓘ</strong> (details) → turn on <strong>Show in Share Sheet</strong>, and set it to receive <strong>Safari web pages</strong>.</li>
          <li>Add action <strong>Run JavaScript on Web Page</strong> (input: Shortcut Input). Replace its script with the one you copy here.</li>
          <li>Add action <strong>Open URLs</strong> (it uses the result of the JavaScript).</li>
        </ol>
        <button className="btn small" onClick={async () => { await navigator.clipboard.writeText(SEND_TO_STAGE_JS); setCopied(true); setTimeout(() => setCopied(false), 1500); }}>
          {copied ? "Copied" : "Copy script"}
        </button>
      </details>
      <p className="small dim" style={{ margin: 0 }}>
        The shortcut opens in Safari, which keeps its own copy of the app. Sign in there once and shared charts sync to the home-screen app within seconds.
        No signal? Copy the chart text instead and use Import → Paste from clipboard.
      </p>
    </Section>
  );
}

function LiveScreens() {
  const profile = useProfile();
  const { userEmail } = useSyncStatus();
  const [show, setShow] = useState<"display" | "band" | null>(null);
  if (!userEmail || !profile) {
    return <Section title="Live screens" hint="Sign in to get your lyrics-display and band links." >{null}</Section>;
  }
  const display = appUrl(`/display/${profile.live_token}`);
  const band = appUrl(`/band/${profile.live_token}`);
  const rotate = async () => {
    if (!confirm("Make new links? Screens using the old links will stop updating.")) return;
    const token = Array.from(crypto.getRandomValues(new Uint8Array(15)), (b) => "abcdefghjkmnpqrstuvwxyz23456789"[b % 31]).join("");
    await supabase.from("profiles").update({ live_token: token, updated_at: new Date().toISOString() }).eq("id", profile.id);
    await syncNow();
  };
  return (
    <Section title="Live screens" hint="Open these links on any device with a browser. They follow whatever you're playing in Perform mode.">
      <div className="stack">
        <div className="row wrap">
          <div className="grow">
            <strong>Lyrics display</strong> <span className="dim small">— karaoke / audience screen</span>
            <div className="small dim">Open on whatever drives the TV: a laptop, Fire TV or Chromecast browser, a smart TV browser, or a second iPad over HDMI. Big lyrics, section by section.</div>
          </div>
          <button className="btn small" onClick={() => setShow(show === "display" ? null : "display")}>QR</button>
          <a className="btn small" href={display} target="_blank" rel="noreferrer">Open</a>
          <button className="btn small" onClick={() => navigator.clipboard.writeText(display)}>Copy</button>
        </div>
        {show === "display" && <QrCode value={display} />}
        <div className="row wrap">
          <div className="grow">
            <strong>Band follow</strong> <span className="dim small">— bandmates' phones/tablets</span>
            <div className="small dim">Shows your current song with chords. Each person can pick their own key, capo or lyrics-only view.</div>
          </div>
          <button className="btn small" onClick={() => setShow(show === "band" ? null : "band")}>QR</button>
          <a className="btn small" href={band} target="_blank" rel="noreferrer">Open</a>
          <button className="btn small" onClick={() => navigator.clipboard.writeText(band)}>Copy</button>
        </div>
        {show === "band" && <QrCode value={band} />}
        <div><button className="btn small ghost" onClick={rotate}>Make new links</button></div>
      </div>
    </Section>
  );
}

function StageDisplay() {
  const s = useSettings();
  return (
    <Section title="Stage display">
      <div className="row wrap">
        <span className="grow">Theme</span>
        <div className="seg">
          {(["dark", "lowlight", "light"] as Theme[]).map((t) => (
            <button key={t} className={s.theme === t ? "on" : ""} onClick={() => updateSettings({ theme: t })}>
              {t === "lowlight" ? "Low light" : t[0].toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
      </div>
      <div className="row wrap">
        <span className="grow">Text size ({Math.round(s.fontScale * 100)}%)</span>
        <input type="range" min={0.6} max={2.6} step={0.05} value={s.fontScale} onChange={(e) => updateSettings({ fontScale: Number(e.target.value) })} style={{ width: 240, accentColor: "var(--accent)" }} />
      </div>
      <label className="check"><input type="checkbox" checked={s.showChords} onChange={(e) => updateSettings({ showChords: e.target.checked })} /> Show chords (off = lyrics only)</label>
      <label className="check"><input type="checkbox" checked={s.nashville} onChange={(e) => updateSettings({ nashville: e.target.checked })} /> Nashville numbers instead of chord names</label>
      <label className="check"><input type="checkbox" checked={s.columns === 2} onChange={(e) => updateSettings({ columns: e.target.checked ? 2 : 1 })} /> Two columns (great in landscape)</label>
    </Section>
  );
}

function Pedals() {
  const s = useSettings();
  const [learning, setLearning] = useState<PedalAction | null>(null);
  useEffect(() => {
    if (!learning) return;
    const onKey = (e: KeyboardEvent) => {
      e.preventDefault();
      const map = { ...s.pedalMap, [e.key]: learning }; // a key maps to one action; an action can have several keys
      updateSettings({ pedalMap: map });
      setLearning(null);
    };
    window.addEventListener("keydown", onKey, { once: true });
    return () => window.removeEventListener("keydown", onKey);
  }, [learning, s.pedalMap]);

  const keysFor = (a: PedalAction) => Object.entries(s.pedalMap).filter(([, v]) => v === a).map(([k]) => (k === " " ? "Space" : k));
  return (
    <Section title="Foot pedals & keys" hint="Bluetooth page turners (AirTurn, PageFlip, iRig BlueTurn, Donner…) act like a keyboard. Pair the pedal in iPad Settings → Bluetooth, set it to a keyboard / arrow-key mode, then tap “Learn” and press the pedal.">
      <ul className="list">
        {(Object.keys(PEDAL_ACTION_LABELS) as PedalAction[]).map((a) => (
          <li key={a} className="list-item" style={{ minHeight: 52 }}>
            <span className="grow">{PEDAL_ACTION_LABELS[a]}</span>
            {keysFor(a).map((k) => (
              <button key={k} className="chip" title="Remove" onClick={() => {
                const map = { ...s.pedalMap };
                delete map[k === "Space" ? " " : k];
                updateSettings({ pedalMap: map });
              }}>{k} ✕</button>
            ))}
            <button className={`btn small ${learning === a ? "on" : ""}`} onClick={() => setLearning(learning === a ? null : a)}>
              {learning === a ? "Press a key…" : "Learn"}
            </button>
          </li>
        ))}
      </ul>
      <div className="row wrap">
        <label className="check grow"><input type="checkbox" checked={s.pageDownAdvances} onChange={(e) => updateSettings({ pageDownAdvances: e.target.checked })} /> “Scroll down” at the end of a song goes to the next song</label>
        <button className="btn small ghost" onClick={() => updateSettings({ pedalMap: DEFAULT_SETTINGS.pedalMap })}>Reset keys</button>
      </div>
    </Section>
  );
}

function Performance() {
  const s = useSettings();
  const num = (v: string, d: number) => (Number.isFinite(Number(v)) ? Number(v) : d);
  return (
    <Section title="Autoscroll, metronome & live">
      <div className="meta-grid">
        <label className="field"><span>Autoscroll delay (sec)</span><input className="input" inputMode="numeric" value={s.scrollDelay} onChange={(e) => updateSettings({ scrollDelay: num(e.target.value, 8) })} /></label>
        <label className="field"><span>Default song length (sec)</span><input className="input" inputMode="numeric" value={s.defaultDurationSec} onChange={(e) => updateSettings({ defaultDurationSec: num(e.target.value, 210) })} /></label>
        <label className="field"><span>Count-in before tracks</span>
          <select className="select" value={s.countInBars} onChange={(e) => updateSettings({ countInBars: Number(e.target.value) })}>
            {[0, 1, 2].map((n) => <option key={n} value={n}>{n === 0 ? "Off" : `${n} bar${n > 1 ? "s" : ""}`}</option>)}
          </select>
        </label>
        <label className="field"><span>Click volume</span><input type="range" min={0.05} max={1} step={0.05} value={s.metronomeVolume} onChange={(e) => updateSettings({ metronomeVolume: Number(e.target.value) })} style={{ accentColor: "var(--accent)", minHeight: 44 }} /></label>
      </div>
      <label className="check"><input type="checkbox" checked={s.clickAccent} onChange={(e) => updateSettings({ clickAccent: e.target.checked })} /> Accent beat one</label>
      <label className="check"><input type="checkbox" checked={s.flashOnBeat} onChange={(e) => updateSettings({ flashOnBeat: e.target.checked })} /> Flash the screen edge on each beat (silent visual click)</label>
      <label className="check"><input type="checkbox" checked={s.displayFollowsScroll} onChange={(e) => updateSettings({ displayFollowsScroll: e.target.checked })} /> Lyrics display follows the section at the top of my screen</label>
      <label className="check"><input type="checkbox" checked={s.requestPopups} onChange={(e) => updateSettings({ requestPopups: e.target.checked })} /> Pop up new requests while performing</label>
    </Section>
  );
}

function Device() {
  const [persisted, setPersisted] = useState<boolean | null>(null);
  const [usage, setUsage] = useState<string>("");
  const counts = useLiveQuery(async () => {
    const files = live(await db.song_files.toArray());
    const blobs = new Set((await db.blobs.toArray()).map((b) => b.id));
    return { songs: live(await db.songs.toArray()).length, files: files.length, offline: files.filter((f) => blobs.has(f.id)).length };
  }, []);
  useEffect(() => {
    void navigator.storage?.persisted?.().then(setPersisted);
    void navigator.storage?.estimate?.().then((e) => e.usage !== undefined && setUsage(`${(e.usage / 1048576).toFixed(1)} MB used`));
  }, []);
  const standalone = window.matchMedia("(display-mode: standalone)").matches || (navigator as { standalone?: boolean }).standalone;
  return (
    <Section title="This device">
      {!standalone && (
        <div className="sticky-note" style={{ margin: 0 }}>
          <strong>Install on your iPad for offline gigs:</strong> open this page in Safari, tap the Share button, then “Add to Home Screen”.
          Launch it from the home-screen icon — installed apps keep their songs and files offline, and don't get cleared by Safari.
        </div>
      )}
      <div className="small dim">
        {counts ? `${counts.songs} songs on this device · ${counts.offline}/${counts.files} attachments saved offline` : ""}{usage ? ` · ${usage}` : ""}
      </div>
      <div className="row wrap">
        <span className="grow small">{persisted ? "Storage is protected from automatic clearing." : "Ask the browser to protect this app's storage from being cleared."}</span>
        {!persisted && <button className="btn small" onClick={async () => setPersisted((await navigator.storage?.persist?.()) ?? false)}>Protect storage</button>}
      </div>
    </Section>
  );
}

function Backup() {
  const exportJson = async () => {
    const data = {
      app: "setlist-stage", version: 1, exported_at: new Date().toISOString(),
      songs: live(await db.songs.toArray()),
      setlists: live(await db.setlists.toArray()),
      setlist_items: live(await db.setlist_items.toArray()),
    };
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], { type: "application/json" }));
    a.download = `setlist-stage-backup-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  return (
    <Section title="Backup" hint="Songs and setlists as one file (attachments stay in the cloud). Restore it from Import.">
      <div><button className="btn" onClick={exportJson}>Download backup</button></div>
    </Section>
  );
}
