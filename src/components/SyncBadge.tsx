import { Link } from "react-router-dom";
import { useSyncStatus } from "../lib/sync";

const LABELS = {
  idle: "Synced",
  syncing: "Syncing…",
  offline: "Offline",
  "signed-out": "Not signed in",
  error: "Sync error",
} as const;

/** Small status pill; tap to open Settings (sign in / see errors). */
export function SyncBadge() {
  const { phase, pending, error } = useSyncStatus();
  const label = phase === "offline" && pending ? `Offline · ${pending} to sync`
    : phase === "signed-out" && pending ? `Not signed in · ${pending} not synced`
    : LABELS[phase];
  return (
    <Link to="/settings" className="chip" title={error ?? label} style={{ textDecoration: "none", minHeight: 30 }}>
      <span className={`sync-dot ${phase === "signed-out" && pending ? "error" : phase}`} /> {label}
    </Link>
  );
}

/**
 * Full-width warning when this device was signed out on its own (e.g. its login expired)
 * while it still holds songs or charts the other devices haven't got.
 */
export function SignedOutBanner() {
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
