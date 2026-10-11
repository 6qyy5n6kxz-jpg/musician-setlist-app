import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { availableMethods, formatMoney, PAY_LABELS, payLink, type PayHandles, type PayMethod } from "../../lib/pay";
import { supabase } from "../../lib/supabase";

interface CatalogSong {
  id: string;
  title: string;
  artist: string;
  karaoke?: boolean;
  requestable?: boolean;
}

interface Catalog {
  open: boolean;
  karaoke_open?: boolean;
  performer: string;
  message: string | null;
  tip_url: string | null;
  pay?: PayHandles;
  amounts?: number[];
  priority?: boolean;
  shoutouts?: boolean;
  email_list?: boolean;
  booking?: boolean;
  songs: CatalogSong[];
}

type Mode = "request" | "karaoke" | "shoutout" | "booking";

/**
 * Audience page reached from the QR code: song requests (optionally with a tip), shoutouts,
 * karaoke sign-up, booking inquiries and an email list. Any phone, no login.
 */
export function PublicRequest() {
  const { token } = useParams();
  const [catalog, setCatalog] = useState<Catalog | null | "invalid">(null);
  const [mode, setMode] = useState<Mode>("request");
  const [q, setQ] = useState("");
  const [picked, setPicked] = useState<string | null>(null);
  const [custom, setCustom] = useState("");
  const [name, setName] = useState(() => localStorage.getItem("req-name") ?? "");
  const [message, setMessage] = useState("");
  const [tip, setTip] = useState<number | null>(null);
  const [otherTip, setOtherTip] = useState("");
  const [method, setMethod] = useState<PayMethod | null>(null);
  const [contact, setContact] = useState("");
  const [eventDate, setEventDate] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [lineup, setLineup] = useState<{ name: string; title: string }[]>([]);
  const [sentTitle, setSentTitle] = useState("");
  const [sentTip, setSentTip] = useState<{ amount: number; method: PayMethod } | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = "dark";
    supabase.rpc("request_catalog", { p_token: token }).then(({ data, error }) => {
      if (error || !data) return setCatalog("invalid");
      const c = data as Catalog;
      setCatalog(c);
      if (!c.open && c.karaoke_open) setMode("karaoke");
    });
  }, [token]);

  const loadLineup = useCallback(async () => {
    const { data } = await supabase.rpc("karaoke_lineup", { p_token: token });
    if (Array.isArray(data)) setLineup(data);
  }, [token]);
  useEffect(() => {
    if (mode !== "karaoke") return;
    void loadLineup();
    const t = setInterval(loadLineup, 20_000);
    return () => clearInterval(t);
  }, [mode, loadLineup]);

  const methods = useMemo(() => (catalog && catalog !== "invalid" ? availableMethods(catalog.pay ?? {}) : []), [catalog]);
  useEffect(() => { if (methods.length && !method) setMethod(methods[0]); }, [methods, method]);

  const list = useMemo(() => {
    if (!catalog || catalog === "invalid") return [];
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    return catalog.songs
      .filter((s) => (mode === "karaoke" ? s.karaoke : s.requestable !== false))
      .filter((s) => terms.every((t) => `${s.title} ${s.artist}`.toLowerCase().includes(t)));
  }, [catalog, q, mode]);

  const switchMode = (m: Mode) => {
    setMode(m);
    setPicked(null);
    setCustom("");
    setMessage("");
    setTip(null);
    setOtherTip("");
    setState("idle");
    setError(null);
  };

  const tipAmount = tip === -1 ? Math.max(0, Math.round(Number(otherTip.replace(/[^0-9.]/g, "")) * 100) / 100) || null : tip;

  const send = async () => {
    if (!catalog || catalog === "invalid") return;
    setState("sending");
    setError(null);
    localStorage.setItem("req-name", name);
    const tipping = (mode === "request" || mode === "shoutout") && tipAmount && method ? { amount: tipAmount, method } : null;
    const { data, error } = await supabase.rpc("submit_request", {
      p_token: token, p_song_id: mode === "request" || mode === "karaoke" ? picked : null,
      p_title: mode === "request" && !picked ? custom : null, p_artist: null,
      p_name: name || null, p_message: message || null, p_kind: mode,
      p_tip: tipping?.amount ?? null, p_tip_method: tipping?.method ?? null,
      p_contact: mode === "booking" ? contact : null, p_event_date: mode === "booking" && eventDate ? eventDate : null,
    });
    const res = data as { ok: boolean; error?: string } | null;
    if (error || !res?.ok) {
      setState("error");
      setError(res?.error ?? "Couldn't send — check your connection and try again.");
      return;
    }
    const pickedSong = catalog.songs.find((s) => s.id === picked);
    setSentTitle(mode === "shoutout" ? "your shoutout" : pickedSong ? `“${pickedSong.title}”` : `“${custom}”`);
    setSentTip(tipping);
    setState("sent");
    // The request is in; now hand off to the payment app with everything filled in
    if (tipping) window.location.href = payLink(tipping.method, catalog.pay?.[tipping.method] ?? "", tipping.amount, `${mode === "shoutout" ? "Shoutout" : `Request: ${pickedSong?.title ?? custom}`} 🎶`);
    if (mode === "karaoke") void loadLineup();
  };

  if (catalog === null) return <div className="public"><div className="public-page dim">Loading…</div></div>;
  if (catalog === "invalid") return <div className="public"><div className="public-page empty-state"><h2>This request link isn't active</h2><p>Ask the performer for the current QR code.</p></div></div>;

  const pickedSong = catalog.songs.find((s) => s.id === picked);
  const isOpen = mode === "karaoke" ? !!catalog.karaoke_open : mode === "booking" ? true : catalog.open;
  const myPlace = mode === "karaoke" && name ? lineup.findIndex((l) => l.name?.toLowerCase() === name.trim().toLowerCase()) : -1;
  const amounts = (catalog.amounts?.length ? catalog.amounts : [5, 10, 20]).slice(0, 4);
  const canTip = methods.length > 0 && (mode === "request" || mode === "shoutout");
  const tabs: [Mode, string][] = [
    ...(catalog.open ? [["request", "Request"] as [Mode, string]] : []),
    ...(catalog.open && catalog.shoutouts !== false ? [["shoutout", "Shoutout 📣"] as [Mode, string]] : []),
    ...(catalog.karaoke_open ? [["karaoke", "Sing 🎤"] as [Mode, string]] : []),
  ];

  const tipPicker = canTip && (
    <div className="card stack" style={{ padding: 12 }}>
      <div style={{ fontWeight: 600 }}>
        Add a tip <span className="dim small">(optional{catalog.priority !== false && mode === "request" ? " — tipped requests move up the list" : ""})</span>
      </div>
      <div className="row wrap" style={{ gap: 6 }}>
        <button className={`btn ${tip === null ? "on" : ""}`} onClick={() => setTip(null)}>No tip</button>
        {amounts.map((a) => <button key={a} className={`btn ${tip === a ? "on" : ""}`} onClick={() => setTip(a)}>{formatMoney(a)}</button>)}
        <button className={`btn ${tip === -1 ? "on" : ""}`} onClick={() => setTip(-1)}>Other</button>
        {tip === -1 && <input className="input" style={{ width: 90 }} inputMode="decimal" placeholder="$" value={otherTip} onChange={(e) => setOtherTip(e.target.value)} autoFocus />}
      </div>
      {tipAmount && methods.length > 1 && (
        <div className="row wrap" style={{ gap: 6 }}>
          <span className="small dim">Pay with</span>
          {methods.map((m) => <button key={m} className={`btn small ${method === m ? "on" : ""}`} onClick={() => setMethod(m)}>{PAY_LABELS[m]}</button>)}
        </div>
      )}
      {tipAmount && <div className="small dim">After you send, {method ? PAY_LABELS[method] : "your payment app"} opens with {formatMoney(tipAmount)} filled in.</div>}
    </div>
  );

  const sendLabel = state === "sending" ? "Sending…"
    : mode === "karaoke" ? (pickedSong ? `Sign up to sing “${pickedSong.title}”` : "Pick a song to sing")
    : mode === "shoutout" ? `Send shoutout${tipAmount ? ` + ${formatMoney(tipAmount)} tip` : ""}`
    : mode === "booking" ? "Send booking request"
    : `${pickedSong ? `Request “${pickedSong.title}”` : "Send request"}${tipAmount ? ` + ${formatMoney(tipAmount)} tip` : ""}`;
  const disabled = state === "sending" || (
    mode === "karaoke" ? !picked || !name.trim()
    : mode === "shoutout" ? !message.trim()
    : mode === "booking" ? !name.trim() || !contact.trim()
    : !picked && !custom.trim()) || (tip === -1 && !tipAmount);

  return (
    <div className="public">
      <div className="public-page">
        <div className="public-hero">
          <div className="dim small" style={{ textTransform: "uppercase", letterSpacing: "0.1em" }}>
            {mode === "karaoke" ? "Live band karaoke" : mode === "booking" ? "Book us" : mode === "shoutout" ? "Shoutouts" : "Song requests"}
          </div>
          <h1>{catalog.performer || "Live music"}</h1>
          {catalog.message && <p className="dim">{catalog.message}</p>}
          {methods.length > 0 ? (
            <div className="row wrap" style={{ justifyContent: "center", gap: 6, marginTop: 8 }}>
              {methods.map((m) => (
                <a key={m} className="btn" href={payLink(m, catalog.pay?.[m] ?? "", null, "Thanks for the music! 🎶")} rel="noreferrer">💸 Tip on {PAY_LABELS[m]}</a>
              ))}
            </div>
          ) : catalog.tip_url && /^https?:\/\//.test(catalog.tip_url) && (
            <a className="btn primary" href={catalog.tip_url} target="_blank" rel="noreferrer" style={{ marginTop: 8 }}>💸 Leave a tip</a>
          )}
        </div>

        {mode !== "booking" && tabs.length > 1 && (
          <div className="seg" style={{ display: "flex", marginBottom: 16 }}>
            {tabs.map(([m, label]) => (
              <button key={m} className={mode === m ? "on" : ""} style={{ flex: 1, minHeight: 46 }} onClick={() => switchMode(m)}>{label}</button>
            ))}
          </div>
        )}
        {mode === "booking" && <button className="btn ghost small" style={{ marginBottom: 10 }} onClick={() => switchMode(catalog.open ? "request" : catalog.karaoke_open ? "karaoke" : "request")}>‹ Back</button>}

        {!isOpen ? (
          <div className="card empty-state"><h2>{mode === "karaoke" ? "Karaoke sign-up is closed" : "Requests are closed right now"}</h2><p>Check back during the next set!</p></div>
        ) : state === "sent" ? (
          <div className="card empty-state">
            {mode === "karaoke" ? (
              <>
                <h2>You're on the list! 🎤</h2>
                <p>{pickedSong ? `“${pickedSong.title}”` : ""}{myPlace >= 0 ? ` — you're #${myPlace + 1} in line.` : ""} Watch the screen for your name.</p>
              </>
            ) : mode === "booking" ? (
              <>
                <h2>Thanks! 📅</h2>
                <p>We got your request and will get back to you soon.</p>
              </>
            ) : (
              <>
                <h2>{mode === "shoutout" ? "Shoutout sent! 📣" : "Request sent! 🎶"}</h2>
                <p>{mode === "shoutout" ? "We'll give it a mention." : `${sentTitle} is in the queue.`}</p>
                {sentTip && (
                  <p>
                    Didn't see {PAY_LABELS[sentTip.method]} open?{" "}
                    <a className="btn primary" href={payLink(sentTip.method, catalog.pay?.[sentTip.method] ?? "", sentTip.amount, `Request 🎶`)}>Tip {formatMoney(sentTip.amount)} on {PAY_LABELS[sentTip.method]}</a>
                  </p>
                )}
              </>
            )}
            <button className="btn" onClick={() => { setState("idle"); setPicked(null); setCustom(""); setMessage(""); setTip(null); setSentTip(null); }}>
              {mode === "karaoke" || mode === "booking" ? "Back" : mode === "shoutout" ? "Send another" : "Request another"}
            </button>
          </div>
        ) : mode === "booking" ? (
          <div className="stack">
            <label className="field"><span>Your name</span><input className="input" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} /></label>
            <label className="field"><span>Email or phone</span><input className="input" value={contact} maxLength={200} onChange={(e) => setContact(e.target.value)} placeholder="So we can get back to you" /></label>
            <label className="field"><span>Event date (if you know it)</span><input className="input" type="date" value={eventDate} onChange={(e) => setEventDate(e.target.value)} /></label>
            <label className="field"><span>Tell us about it</span>
              <textarea className="input" rows={4} value={message} maxLength={600} onChange={(e) => setMessage(e.target.value)} placeholder="Wedding, private party, brewery… where, how long, how many people" />
            </label>
            {error && <div style={{ color: "var(--danger)" }}>{error}</div>}
            <button className="btn primary" style={{ minHeight: 54, fontSize: "1.1rem" }} disabled={disabled} onClick={send}>{sendLabel}</button>
          </div>
        ) : mode === "shoutout" ? (
          <div className="stack">
            <label className="field"><span>What should we say?</span>
              <textarea className="input" rows={3} value={message} maxLength={280} onChange={(e) => setMessage(e.target.value)} placeholder="Happy 30th birthday Sarah!" />
            </label>
            <label className="field"><span>From (optional)</span><input className="input" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} /></label>
            {tipPicker}
            {error && <div style={{ color: "var(--danger)" }}>{error}</div>}
            <button className="btn primary" style={{ minHeight: 54, fontSize: "1.1rem" }} disabled={disabled} onClick={send}>{sendLabel}</button>
          </div>
        ) : (
          <div className="stack">
            {mode === "karaoke" && (
              <label className="field"><span>Your name (we'll call you up)</span>
                <input className="input" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} placeholder="First name" />
              </label>
            )}
            <input className="input" type="search" placeholder={`Search ${list.length} songs…`} value={q} onChange={(e) => setQ(e.target.value)} />
            <div className="card" style={{ padding: 0, maxHeight: "45vh", overflowY: "auto" }}>
              {list.map((s) => (
                <button key={s.id} className={`song-pick ${picked === s.id ? "on" : ""}`} onClick={() => { setPicked(picked === s.id ? null : s.id); setCustom(""); }}>
                  <span className="grow">
                    <span style={{ display: "block", fontWeight: 600 }}>{s.title}</span>
                    <span className="small dim">{s.artist}</span>
                  </span>
                  {picked === s.id && <span style={{ color: "var(--accent)", fontWeight: 700 }}>✓</span>}
                </button>
              ))}
              {list.length === 0 && <div className="dim" style={{ padding: 14 }}>{mode === "karaoke" ? "No karaoke songs match." : "Not on the list? Type it below."}</div>}
            </div>
            {mode === "request" && (
              <>
                <label className="field"><span>Or request something else</span>
                  <input className="input" placeholder="Song — Artist" value={custom} maxLength={200} onChange={(e) => { setCustom(e.target.value); if (e.target.value) setPicked(null); }} />
                </label>
                <div className="meta-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
                  <label className="field"><span>Your name (optional)</span><input className="input" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} /></label>
                  <label className="field"><span>Dedication (optional)</span><input className="input" value={message} maxLength={280} onChange={(e) => setMessage(e.target.value)} placeholder="For Sarah's birthday!" /></label>
                </div>
                {tipPicker}
              </>
            )}
            {error && <div style={{ color: "var(--danger)" }}>{error}</div>}
            <button className="btn primary" style={{ minHeight: 54, fontSize: "1.1rem" }} disabled={disabled} onClick={send}>{sendLabel}</button>
          </div>
        )}

        {mode === "karaoke" && lineup.length > 0 && (
          <div className="card" style={{ marginTop: 18 }}>
            <div className="small dim" style={{ textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>Up next</div>
            <ol style={{ margin: 0, paddingLeft: 22 }}>
              {lineup.slice(0, 10).map((l, i) => (
                <li key={i} style={{ padding: "3px 0", fontWeight: i === myPlace ? 700 : 400, color: i === myPlace ? "var(--accent)" : undefined }}>
                  {l.name} <span className="dim small">— {l.title}</span>
                </li>
              ))}
            </ol>
          </div>
        )}

        {catalog.email_list !== false && <EmailSignup token={token!} performer={catalog.performer} />}
        {catalog.booking !== false && mode !== "booking" && (
          <div style={{ textAlign: "center", marginTop: 14 }}>
            <button className="btn ghost" onClick={() => switchMode("booking")}>📅 Book {catalog.performer || "us"} for your event</button>
          </div>
        )}
      </div>
    </div>
  );
}

function EmailSignup({ token, performer }: { token: string; performer: string }) {
  const [email, setEmail] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "done">(() => (localStorage.getItem(`joined-${token}`) ? "done" : "idle"));
  const [error, setError] = useState<string | null>(null);
  const join = async () => {
    setState("sending");
    setError(null);
    const { data, error } = await supabase.rpc("join_list", { p_token: token, p_email: email, p_name: localStorage.getItem("req-name") || null });
    const res = data as { ok: boolean; error?: string } | null;
    if (error || !res?.ok) { setError(res?.error ?? "Couldn't sign you up — try again."); setState("idle"); return; }
    try { localStorage.setItem(`joined-${token}`, "1"); } catch { /* private mode */ }
    setState("done");
  };
  if (state === "done") return <div className="small dim" style={{ textAlign: "center", marginTop: 18 }}>You're on the list — we'll let you know where we're playing next. 🎶</div>;
  return (
    <div className="card stack" style={{ marginTop: 18, padding: 12 }}>
      <div style={{ fontWeight: 600 }}>Find out where {performer ? `${performer} is` : "we're"} playing next</div>
      <div className="row" style={{ gap: 6 }}>
        <input className="input grow" type="email" inputMode="email" autoComplete="email" placeholder="your@email.com" value={email} onChange={(e) => setEmail(e.target.value)} />
        <button className="btn" disabled={state === "sending" || !/.+@.+\..+/.test(email)} onClick={join}>Join</button>
      </div>
      {error && <div className="small" style={{ color: "var(--danger)" }}>{error}</div>}
    </div>
  );
}
