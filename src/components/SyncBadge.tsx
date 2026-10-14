import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { syncNow, useSyncStatus } from "../lib/sync";

/** "just now", "4 min ago", "3 h ago", "2 days ago". */
export function ago(iso: string | null, now = Date.now()): string {
  if (!iso) return "never";
  const s = Math.max(0, (now - Date.parse(iso)) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  const d = Math.floor(s / 86400);
  return `${d} day${d === 1 ? "" : "s"} ago`;
}

/** Signed in and online, but no successful sync for this long = something is wrong. */
const STALE_MS = 10 * 60_000;
function useNow(ms = 30_000) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), ms); return () => clearInterval(t); }, [ms]);
  return now;
}

const LABELS = {
  idle: "Synced",
  syncing: "Syncing…",
  offline: "Offline",
  "signed-out": "Not signed in",
  error: "Sync error",
} as const;

/** Small status pill; tap to open Settings (sign in / see errors). */
export function SyncBadge() {
  const { phase, pending, error, lastSynced, userEmail } = useSyncStatus();
  const now = useNow();
  const stale = !!userEmail && phase !== "offline" && phase !== "signed-out" && (!lastSynced || now - Date.parse(lastSynced) > STALE_MS) && phase !== "syncing";
  const label = phase === "offline" && pending ? `Offline · ${pending} to sync`
    : phase === "signed-out" && pending ? `Not signed in · ${pending} not synced`
    : phase === "idle" && lastSynced ? (stale ? `Last synced ${ago(lastSynced, now)}` : "Synced")
    : phase === "syncing" && !lastSynced ? "Syncing…"
    : LABELS[phase];
  const dot = (phase === "signed-out" && pending) || stale ? "error" : phase;
  return (
    <Link to="/settings" className="chip" title={`${error ?? label}${lastSynced ? ` · last synced ${ago(lastSynced, now)}` : ""}`} style={{ textDecoration: "none", minHeight: 30 }}>
      <span className={`sync-dot ${dot}`} /> {label}
    </Link>
  );
}

/**
 * Full-width warning when this device was signed out on its own (e.g. its login expired)
 * while it still holds songs or charts the other devices haven't got.
 */
export function SignedOutBanner() {
  const st = useSyncStatus();
  const now = useNow();
  const { phase, lastSynced, userEmail, error } = st;
  const stale = !!userEmail && phase !== "offline" && phase !== "signed-out" && phase !== "syncing" && (!lastSynced || now - Date.parse(lastSynced) > STALE_MS);
  if (stale || (userEmail && phase === "error")) {
    return (
      <button className="signed-out-banner no-print" style={{ width: "100%", border: 0, textAlign: "left", cursor: "pointer" }} onClick={() => void syncNow()}>
        <strong>This device isn't syncing.</strong> Last synced {ago(lastSynced, now)}{st.pending ? ` · ${st.pending} change${st.pending === 1 ? "" : "s"} waiting here` : ""}.
        {error ? ` (${error})` : ""} Tap to try again — if it keeps failing, sign out and back in under Settings.
      </button>
    );
  }
  return <SignedOutNotice />;
}

function SignedOutNotice() {
  const { phase, pending, pendingFiles, lostSession } = useSyncStatus();
  if (phase !== "signed-out" || (!lostSession && !pending)) return null;
  const what = pendingFiles
    ? `${pendingFiles} chart${pendingFiles === 1 ? "" : "s"}/file${pendingFiles === 1 ? "" : "s"}${pending > pendingFiles ? ` and ${pending - pendingFiles} other change${pending - pendingFiles === 1 ? "" : "s"}` : ""}`
    : `${pending} change${pending === 1 ? "" : "s"}`;
  return (
    <Link to="/settings" className="signed-out-banner no-print">
      <strong>{lostSession ? "This device got signed out." : "Not signed in."}</strong>{" "}
      {pending ? <>{what} are saved only on this device. </> : null}
      Sign in{lostSession ? ` again as ${lostSession}` : ""} to sync — nothing is lost.
    </Link>
  );
}
