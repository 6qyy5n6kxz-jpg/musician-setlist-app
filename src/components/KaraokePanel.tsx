import { saveProfile } from "../lib/db";
import { useProfile } from "../lib/hooks";
import { karaokeLineup, useRequests, type SongRequest } from "../lib/requests";

/** Singer lineup for live band karaoke. In Perform mode, "Start" pulls up the singer's song. */
export function KaraokePanel({ onStart, currentSingerId }: { onStart?: (r: SongRequest) => void; currentSingerId?: string | null }) {
  const profile = useProfile();
  const { requests, setStatus, moveSinger, remove } = useRequests();
  const line = karaokeLineup(requests).filter((r) => r.id !== currentSingerId);
  const current = currentSingerId ? requests.find((r) => r.id === currentSingerId) : null;
  const open = profile?.karaoke_open ?? false;

  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="row wrap">
        <div className="grow">
          <strong>Karaoke sign-up</strong>
          <div className="small dim">{open ? "Open — singers can sign up from the request QR code." : "Closed."} Songs marked “Karaoke” are on the list.</div>
        </div>
        <button className={`btn small ${open ? "on" : ""}`} onClick={() => saveProfile({ karaoke_open: !open })}>
          {open ? "Close sign-up" : "Open sign-up"}
        </button>
      </div>

      {current && (
        <div className="req new">
          <div className="small dim" style={{ textTransform: "uppercase", letterSpacing: "0.06em" }}>Now singing</div>
          <div className="req-title">{current.patron_name}</div>
          <div className="small dim">{current.title}{current.artist ? ` — ${current.artist}` : ""}</div>
        </div>
      )}

      {line.length === 0 && <div className="dim small">No one in line{open ? " yet" : ""}.</div>}
      {line.map((r, i) => (
        <div key={r.id} className={`req ${r.status === "new" ? "new" : ""}`}>
          <div className="row">
            <span className="key-pill" style={{ minWidth: 34, height: 30 }}>{i + 1}</span>
            <div className="grow">
              <div className="req-title">{r.patron_name}</div>
              <div className="small dim">{r.title}{r.artist ? ` — ${r.artist}` : ""} · {new Date(r.created_at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}</div>
            </div>
            <button className="btn small icon" onClick={() => moveSinger(r.id, -1)} disabled={i === 0} aria-label="Move up">↑</button>
            <button className="btn small icon" onClick={() => moveSinger(r.id, 1)} disabled={i === line.length - 1} aria-label="Move down">↓</button>
          </div>
          {r.message && <div className="req-msg">“{r.message}”</div>}
          <div className="req-actions">
            {onStart && <button className="btn small primary" onClick={() => onStart(r)}>Start</button>}
            {r.status === "new" && <button className="btn small" onClick={() => setStatus(r.id, "queued")}>Approve</button>}
            <button className="btn small" onClick={() => setStatus(r.id, "played")}>Sang</button>
            <button className="btn small ghost danger" onClick={() => setStatus(r.id, "declined")}>Skip</button>
            <button className="btn small ghost" onClick={() => remove(r.id)}>Remove</button>
          </div>
        </div>
      ))}
    </div>
  );
}
