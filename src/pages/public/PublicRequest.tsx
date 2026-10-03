import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { supabase } from "../../lib/supabase";

interface Catalog {
  open: boolean;
  performer: string;
  message: string | null;
  tip_url: string | null;
  songs: { id: string; title: string; artist: string }[];
}

/** Audience page reached from the QR code. Works on any phone, no login. */
export function PublicRequest() {
  const { token } = useParams();
  const [catalog, setCatalog] = useState<Catalog | null | "invalid">(null);
  const [q, setQ] = useState("");
  const [picked, setPicked] = useState<string | null>(null);
  const [custom, setCustom] = useState("");
  const [name, setName] = useState(() => localStorage.getItem("req-name") ?? "");
  const [message, setMessage] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = "dark";
    supabase.rpc("request_catalog", { p_token: token }).then(({ data, error }) => {
      if (error || !data) setCatalog("invalid");
      else setCatalog(data as Catalog);
    });
  }, [token]);

  const list = useMemo(() => {
    if (!catalog || catalog === "invalid") return [];
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    return catalog.songs.filter((s) => terms.every((t) => `${s.title} ${s.artist}`.toLowerCase().includes(t)));
  }, [catalog, q]);

  const send = async () => {
    setState("sending");
    setError(null);
    localStorage.setItem("req-name", name);
    const { data, error } = await supabase.rpc("submit_request", {
      p_token: token, p_song_id: picked, p_title: picked ? null : custom, p_artist: null,
      p_name: name || null, p_message: message || null,
    });
    const res = data as { ok: boolean; error?: string } | null;
    if (error || !res?.ok) {
      setState("error");
      setError(res?.error ?? "Couldn't send — check your connection and try again.");
      return;
    }
    setState("sent");
  };

  if (catalog === null) return <div className="public"><div className="public-page dim">Loading…</div></div>;
  if (catalog === "invalid") return <div className="public"><div className="public-page empty-state"><h2>This request link isn't active</h2><p>Ask the performer for the current QR code.</p></div></div>;

  const pickedSong = catalog.songs.find((s) => s.id === picked);

  return (
    <div className="public">
      <div className="public-page">
        <div className="public-hero">
          <div className="dim small" style={{ textTransform: "uppercase", letterSpacing: "0.1em" }}>Song requests</div>
          <h1>{catalog.performer || "Live music"}</h1>
          {catalog.message && <p className="dim">{catalog.message}</p>}
          {catalog.tip_url && /^https?:\/\//.test(catalog.tip_url) && (
            <a className="btn primary" href={catalog.tip_url} target="_blank" rel="noreferrer" style={{ marginTop: 8 }}>💸 Leave a tip</a>
          )}
        </div>

        {!catalog.open ? (
          <div className="card empty-state"><h2>Requests are closed right now</h2><p>Check back during the next set!</p></div>
        ) : state === "sent" ? (
          <div className="card empty-state">
            <h2>Request sent! 🎶</h2>
            <p>{pickedSong ? `“${pickedSong.title}”` : `“${custom}”`} is in the queue.</p>
            <button className="btn" onClick={() => { setState("idle"); setPicked(null); setCustom(""); setMessage(""); }}>Request another</button>
          </div>
        ) : (
          <div className="stack">
            <input className="input" type="search" placeholder={`Search ${catalog.songs.length} songs…`} value={q} onChange={(e) => setQ(e.target.value)} />
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
              {list.length === 0 && <div className="dim" style={{ padding: 14 }}>Not on the list? Type it below.</div>}
            </div>
            <label className="field"><span>Or request something else</span>
              <input className="input" placeholder="Song — Artist" value={custom} maxLength={200} onChange={(e) => { setCustom(e.target.value); if (e.target.value) setPicked(null); }} />
            </label>
            <div className="meta-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
              <label className="field"><span>Your name (optional)</span><input className="input" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} /></label>
              <label className="field"><span>Dedication (optional)</span><input className="input" value={message} maxLength={280} onChange={(e) => setMessage(e.target.value)} placeholder="For Sarah's birthday!" /></label>
            </div>
            {error && <div style={{ color: "var(--danger)" }}>{error}</div>}
            <button className="btn primary" style={{ minHeight: 54, fontSize: "1.1rem" }} disabled={state === "sending" || (!picked && !custom.trim())} onClick={send}>
              {state === "sending" ? "Sending…" : pickedSong ? `Request “${pickedSong.title}”` : "Send request"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
