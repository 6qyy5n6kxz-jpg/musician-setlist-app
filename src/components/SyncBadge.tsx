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
  const label = phase === "offline" && pending ? `Offline · ${pending} to sync` : LABELS[phase];
  return (
    <Link to="/settings" className="chip" title={error ?? label} style={{ textDecoration: "none", minHeight: 30 }}>
      <span className={`sync-dot ${phase}`} /> {label}
    </Link>
  );
}
