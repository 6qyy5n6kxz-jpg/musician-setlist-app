import QRCode from "qrcode";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ActsEditor, activeAct } from "../components/ActsEditor";
import { KaraokePanel } from "../components/KaraokePanel";
import { QrCode } from "../components/QrCode";
import { saveProfile } from "../lib/db";
import { useProfile } from "../lib/hooks";
import { appUrl } from "../lib/links";
import { openQueue, useRequests, type RequestStatus } from "../lib/requests";
import { useSyncStatus } from "../lib/sync";

export function RequestsPage() {
  const profile = useProfile();
  const { userEmail, phase } = useSyncStatus();
  const { requests, setStatus, remove, clearFinished } = useRequests();
  const [name, setName] = useState("");
  const [msg, setMsg] = useState("");
  const [tip, setTip] = useState("");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!profile) return;
    setName(profile.display_name ?? "");
    setMsg(profile.request_message ?? "");
    setTip(profile.tip_url ?? "");
  }, [profile?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!userEmail) {
    return (
      <div className="page">
        <h1 style={{ marginBottom: 12 }}>Requests</h1>
        <div className="card empty-state">
          <h2>Sign in to take requests</h2>
          <p>Requests travel from your audience's phones to your iPad through the cloud, so this needs your account.</p>
          <Link className="btn primary" to="/settings">Go to Settings</Link>
        </div>
      </div>
    );
  }
  if (!profile) return <div className="page dim">Loading your profile… {phase === "offline" ? "(offline — connect once to finish setup)" : ""}</div>;

  const link = appUrl(`/r/${profile.request_token}`);
  const queue = openQueue(requests);
  const active = queue.filter((r) => r.status === "new" || r.status === "queued");
  const finished = queue.filter((r) => r.status === "played" || r.status === "declined");

  const statusButtons = (id: string, status: RequestStatus) => (
    <div className="req-actions">
      {status !== "queued" && status !== "played" && <button className="btn small" onClick={() => setStatus(id, "queued")}>Queue</button>}
      {status !== "played" && <button className="btn small primary" onClick={() => setStatus(id, "played")}>Played</button>}
      {status !== "declined" && status !== "played" && <button className="btn small ghost danger" onClick={() => setStatus(id, "declined")}>Decline</button>}
      {(status === "played" || status === "declined") && <button className="btn small ghost" onClick={() => setStatus(id, "new")}>Reopen</button>}
      <button className="btn small ghost" onClick={() => remove(id)}>Delete</button>
    </div>
  );

  return (
    <div className="page">
      <div className="row" style={{ marginBottom: 14 }}>
        <h1 className="grow">Requests</h1>
        <button
          className={`btn ${profile.requests_open ? "on" : "primary"}`}
          onClick={() => saveProfile({ requests_open: !profile.requests_open })}
        >
          {profile.requests_open ? "Requests are OPEN — tap to close" : "Open requests"}
        </button>
      </div>

      <div className="editor-grid" style={{ gridTemplateColumns: "3fr 2fr" }}>
        <div>
          <h2 style={{ fontSize: "1.1rem", marginBottom: 8 }}>Queue {active.length ? `(${active.length})` : ""}</h2>
          {active.length === 0 && <div className="card dim">No requests yet tonight. {profile.requests_open ? "Put the QR code where people can see it." : "Open requests when you're ready."}</div>}
          {active.map((r) => (
            <div key={r.id} className={`req ${r.status}`}>
              <div className="row">
                <div className="grow">
                  <div className="req-title">{r.title}</div>
                  <div className="small dim">{r.artist}{r.patron_name ? ` · for ${r.patron_name}` : ""} · {new Date(r.created_at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}{!r.song_id ? " · not in your library" : ""}</div>
                </div>
                <span className={`chip ${r.status === "new" ? "accent" : ""}`}>{r.status}</span>
              </div>
              {r.message && <div className="req-msg">“{r.message}”</div>}
              {statusButtons(r.id, r.status)}
            </div>
          ))}
          {finished.length > 0 && (
            <>
              <div className="row" style={{ margin: "18px 0 8px" }}>
                <h2 className="grow" style={{ fontSize: "1.1rem" }}>Done tonight ({finished.length})</h2>
                <button className="btn small" onClick={clearFinished}>Clear</button>
              </div>
              {finished.map((r) => (
                <div key={r.id} className="req" style={{ opacity: 0.7 }}>
                  <div className="row"><div className="grow"><span className="req-title">{r.title}</span> <span className="small dim">{r.artist}</span></div><span className="chip">{r.status}</span></div>
                  {statusButtons(r.id, r.status)}
                </div>
              ))}
            </>
          )}
        </div>

        <div className="stack">
          <div className="card"><KaraokePanel /></div>
          <div className="card stack" style={{ alignItems: "center", textAlign: "center" }}>
            <QrCode value={link} />
            <div className="small dim" style={{ wordBreak: "break-all" }}>{link}</div>
            <div className="row">
              <button className="btn small" onClick={async () => { await navigator.clipboard.writeText(link); setCopied(true); setTimeout(() => setCopied(false), 1500); }}>{copied ? "Copied" : "Copy link"}</button>
              <a className="btn small" href={link} target="_blank" rel="noreferrer">Preview</a>
              <button className="btn small" onClick={() => printQrCard(link, activeAct(profile)?.name || name)}>Print sign</button>
            </div>
          </div>
          <div className="card"><ActsEditor profile={profile} /></div>
          <details className="card stack">
            <summary style={{ cursor: "pointer", fontWeight: 600 }}>Defaults (used when no act is selected)</summary>
            <label className="field" style={{ marginTop: 10 }}><span>Name on the request page</span>
              <input className="input" value={name} onChange={(e) => setName(e.target.value)} onBlur={() => saveProfile({ display_name: name || null })} placeholder="Devin Frank" />
            </label>
            <label className="field"><span>Message to the audience</span>
              <input className="input" value={msg} onChange={(e) => setMsg(e.target.value)} onBlur={() => saveProfile({ request_message: msg || null })} placeholder="Request a song! Tips appreciated 🎸" />
            </label>
            <label className="field"><span>Tip link (Venmo, Cash App, PayPal…)</span>
              <input className="input" value={tip} onChange={(e) => setTip(e.target.value)} onBlur={() => saveProfile({ tip_url: tip || null })} placeholder="https://venmo.com/u/yourname" />
            </label>
            <p className="small dim">Only songs marked “Show on the audience request page” are listed. People can also type in a song you don't have.</p>
          </details>
        </div>
      </div>
    </div>
  );
}

function printQrCard(link: string, name: string) {
  const w = window.open("", "_blank");
  if (!w) return;
  void QRCode.toDataURL(link, { margin: 1, width: 900 }).then((src) => {
    w.document.write(`<!doctype html><title>Request a song</title>
      <style>body{font-family:-apple-system,Helvetica,Arial,sans-serif;text-align:center;padding:40px}h1{font-size:54px;margin:0 0 10px}p{font-size:26px;margin:8px}img{width:420px;height:420px}</style>
      <h1>Request a song!</h1>${name ? `<p>${name.replace(/</g, "&lt;")}</p>` : ""}<img src="${src}" alt=""><p>Scan with your phone camera</p>
      <script>setTimeout(()=>print(),300)<\/script>`);
    w.document.close();
  });
}
