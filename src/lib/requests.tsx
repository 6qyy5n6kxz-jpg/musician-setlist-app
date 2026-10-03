// Live audience request queue (needs a connection; requests come from patrons' phones).
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { getAudioContext } from "./stage";
import { supabase } from "./supabase";
import { useSyncStatus } from "./sync";

export type RequestStatus = "new" | "queued" | "played" | "declined";

export interface SongRequest {
  id: string;
  song_id: string | null;
  title: string;
  artist: string | null;
  patron_name: string | null;
  message: string | null;
  status: RequestStatus;
  kind: "request" | "karaoke";
  position: number | null;
  created_at: string;
  updated_at: string;
}

interface RequestsCtx {
  requests: SongRequest[];
  newCount: number;
  toasts: SongRequest[];
  dismissToast: (id: string) => void;
  setStatus: (id: string, status: RequestStatus) => Promise<void>;
  remove: (id: string) => Promise<void>;
  clearFinished: () => Promise<void>;
  refresh: () => Promise<void>;
  /** Move a singer up (-1) or down (+1) in the karaoke lineup. */
  moveSinger: (id: string, dir: -1 | 1) => Promise<void>;
}

const Ctx = createContext<RequestsCtx | null>(null);

export function RequestsProvider({ children }: { children: ReactNode }) {
  const { userEmail } = useSyncStatus();
  const [requests, setRequests] = useState<SongRequest[]>([]);
  const [toasts, setToasts] = useState<SongRequest[]>([]);
  const chime = useRef<HTMLAudioElement | null>(null);

  const refresh = useCallback(async () => {
    const since = new Date(Date.now() - 1000 * 60 * 60 * 18).toISOString(); // tonight's gig
    const { data } = await supabase
      .from("song_requests")
      .select("id,song_id,title,artist,patron_name,message,status,kind,position,created_at,updated_at")
      .gte("created_at", since)
      .order("created_at", { ascending: true });
    if (data) setRequests(data as SongRequest[]);
  }, []);

  useEffect(() => {
    if (!userEmail) {
      setRequests([]);
      return;
    }
    void refresh();
    const ch = supabase
      .channel("song-requests")
      .on("postgres_changes", { event: "INSERT", schema: "public", table: "song_requests" }, ({ new: row }) => {
        const r = row as SongRequest;
        setRequests((prev) => (prev.some((p) => p.id === r.id) ? prev : [...prev, r]));
        setToasts((prev) => [...prev.slice(-2), r]);
        if (navigator.vibrate) navigator.vibrate(120);
        void chime.current?.play().catch(() => {});
      })
      .on("postgres_changes", { event: "UPDATE", schema: "public", table: "song_requests" }, ({ new: row }) => {
        const r = row as SongRequest;
        setRequests((prev) => prev.map((p) => (p.id === r.id ? r : p)));
      })
      .on("postgres_changes", { event: "DELETE", schema: "public", table: "song_requests" }, ({ old }) => {
        setRequests((prev) => prev.filter((p) => p.id !== (old as { id: string }).id));
      })
      .subscribe();
    const onVis = () => document.visibilityState === "visible" && void refresh();
    document.addEventListener("visibilitychange", onVis);
    return () => {
      document.removeEventListener("visibilitychange", onVis);
      void ch.unsubscribe();
    };
  }, [userEmail, refresh]);

  // iPad Safari only plays sound that a tap started. Unlock the chime (and Web Audio for the
  // metronome) on the first touch or key press, so a request arriving mid-song can chime.
  useEffect(() => {
    const unlock = () => {
      const a = chime.current;
      if (a) {
        a.muted = true;
        void a.play().then(() => { a.pause(); a.currentTime = 0; a.muted = false; }).catch(() => { a.muted = false; });
      }
      void getAudioContext();
      window.removeEventListener("pointerdown", unlock);
      window.removeEventListener("keydown", unlock);
    };
    window.addEventListener("pointerdown", unlock);
    window.addEventListener("keydown", unlock);
    return () => {
      window.removeEventListener("pointerdown", unlock);
      window.removeEventListener("keydown", unlock);
    };
  }, []);

  useEffect(() => {
    if (!toasts.length) return;
    const t = setTimeout(() => setToasts((prev) => prev.slice(1)), 9000);
    return () => clearTimeout(t);
  }, [toasts]);

  const setStatus = useCallback(async (id: string, status: RequestStatus) => {
    setRequests((prev) => prev.map((r) => (r.id === id ? { ...r, status } : r)));
    await supabase.from("song_requests").update({ status, updated_at: new Date().toISOString() }).eq("id", id);
  }, []);

  const remove = useCallback(async (id: string) => {
    setRequests((prev) => prev.filter((r) => r.id !== id));
    await supabase.from("song_requests").delete().eq("id", id);
  }, []);

  const clearFinished = useCallback(async () => {
    const ids = requests.filter((r) => r.status === "played" || r.status === "declined").map((r) => r.id);
    setRequests((prev) => prev.filter((r) => !ids.includes(r.id)));
    if (ids.length) await supabase.from("song_requests").delete().in("id", ids);
  }, [requests]);

  const moveSinger = useCallback(async (id: string, dir: -1 | 1) => {
    const line = karaokeLineup(requests);
    const i = line.findIndex((r) => r.id === id);
    const j = i + dir;
    if (i < 0 || j < 0 || j >= line.length) return;
    const a = line[i], b = line[j];
    const pa = b.position ?? Date.parse(b.created_at) / 1000;
    const pb = a.position ?? Date.parse(a.created_at) / 1000;
    setRequests((prev) => prev.map((r) => (r.id === a.id ? { ...r, position: pa } : r.id === b.id ? { ...r, position: pb } : r)));
    await Promise.all([
      supabase.from("song_requests").update({ position: pa }).eq("id", a.id),
      supabase.from("song_requests").update({ position: pb }).eq("id", b.id),
    ]);
  }, [requests]);

  const value = useMemo<RequestsCtx>(() => ({
    requests,
    newCount: requests.filter((r) => r.status === "new").length,
    toasts,
    dismissToast: (id) => setToasts((prev) => prev.filter((t) => t.id !== id)),
    setStatus, remove, clearFinished, refresh, moveSinger,
  }), [requests, toasts, setStatus, remove, clearFinished, refresh, moveSinger]);

  return (
    <Ctx.Provider value={value}>
      {children}
      {/* short two-tone chime, generated so it works offline */}
      <audio ref={chime} src={CHIME} preload="auto" />
    </Ctx.Provider>
  );
}

export function useRequests(): RequestsCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error("useRequests outside RequestsProvider");
  return v;
}

/** Song requests (not karaoke): new first (oldest first), then queued, then finished. */
export function openQueue(requests: SongRequest[]): SongRequest[] {
  const rank = { new: 0, queued: 1, played: 2, declined: 3 } as const;
  return requests
    .filter((r) => (r.kind ?? "request") === "request")
    .sort((a, b) => rank[a.status] - rank[b.status] || a.created_at.localeCompare(b.created_at));
}

/** Singers still waiting, in lineup order. */
export function karaokeLineup(requests: SongRequest[]): SongRequest[] {
  const pos = (r: SongRequest) => r.position ?? Date.parse(r.created_at) / 1000;
  return requests
    .filter((r) => r.kind === "karaoke" && (r.status === "new" || r.status === "queued"))
    .sort((a, b) => pos(a) - pos(b));
}

/** Pop-up text for a new request or sign-up. */
export function describeRequest(r: SongRequest): { title: string; detail: string } {
  if (r.kind === "karaoke") {
    return { title: `${r.patron_name ?? "Someone"} signed up to sing`, detail: `${r.title}${r.artist ? ` — ${r.artist}` : ""}` };
  }
  return {
    title: `${r.title}${r.artist ? ` — ${r.artist}` : ""}`,
    detail: `${r.patron_name ? `from ${r.patron_name}` : "New request"}${r.message ? ` · “${r.message}”` : ""}`,
  };
}

// Tiny WAV chime built at load (two sine notes), avoids shipping an audio file.
const CHIME = (() => {
  const rate = 22050;
  const notes = [[880, 0.12], [1320, 0.18]] as const;
  const total = notes.reduce((n, [, d]) => n + Math.floor(rate * d), 0);
  const buf = new DataView(new ArrayBuffer(44 + total * 2));
  const w = (o: number, s: string) => [...s].forEach((c, i) => buf.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); buf.setUint32(4, 36 + total * 2, true); w(8, "WAVEfmt ");
  buf.setUint32(16, 16, true); buf.setUint16(20, 1, true); buf.setUint16(22, 1, true);
  buf.setUint32(24, rate, true); buf.setUint32(28, rate * 2, true); buf.setUint16(32, 2, true); buf.setUint16(34, 16, true);
  w(36, "data"); buf.setUint32(40, total * 2, true);
  let o = 44;
  for (const [f, d] of notes) {
    const n = Math.floor(rate * d);
    for (let i = 0; i < n; i++) {
      const env = Math.min(1, i / 200) * (1 - i / n);
      buf.setInt16(o, Math.sin((2 * Math.PI * f * i) / rate) * 9000 * env, true);
      o += 2;
    }
  }
  let bin = "";
  new Uint8Array(buf.buffer).forEach((b) => (bin += String.fromCharCode(b)));
  return "data:audio/wav;base64," + btoa(bin);
})();
