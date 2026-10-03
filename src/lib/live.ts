// Live session: the performer's iPad broadcasts what's on stage; the lyrics display and
// band-follow pages (no login, secret link) receive it instantly over Supabase Realtime.
// The latest state is also saved to live_sessions so a screen that joins late catches up.
import type { RealtimeChannel } from "@supabase/supabase-js";
import { useEffect, useRef, useState } from "react";
import { supabase } from "./supabase";

export interface LiveSong {
  id: string;
  title: string;
  artist: string;
  /** Key the song is written in. */
  key: string | null;
  /** Key being performed (after setlist override / transpose). */
  performKey: string | null;
  /** Semitones from written key to performed key. */
  transpose: number;
  capo: number;
  tempo: number | null;
  timeSignature: string | null;
  flow: string | null;
  content: string;
}

export interface LiveState {
  v: 1;
  at: number;
  setlist: string | null;
  song: LiveSong | null;
  /** Lyric slides of the current song, in performance order (flow applied). */
  slides: { pos: number; label: string; lines: string[] }[];
  /** Section position currently on screen (matches Slide.pos). */
  slide: number;
  /** Lyrics display shows nothing (between songs, announcements). */
  blank: boolean;
  /** Short message for the band screens ("Skip bridge", "Key change!"). */
  message: string | null;
  upNext: { title: string; artist: string } | null;
}

export const EMPTY_LIVE: LiveState = {
  v: 1, at: 0, setlist: null, song: null, slides: [], slide: 0, blank: false, message: null, upNext: null,
};

const channelName = (token: string) => `live-${token}`;

/** Performer side. Returns a publish function; state is merged into the last published state. */
export function useLivePublisher(liveToken: string | null | undefined, enabled: boolean) {
  const channelRef = useRef<RealtimeChannel | null>(null);
  const stateRef = useRef<LiveState>(EMPTY_LIVE);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => {
    if (!liveToken || !enabled) return;
    const ch = supabase.channel(channelName(liveToken), { config: { broadcast: { self: false, ack: false } } });
    // Screens that just connected ask for the current state.
    ch.on("broadcast", { event: "hello" }, () => {
      void ch.send({ type: "broadcast", event: "state", payload: stateRef.current });
    });
    ch.subscribe();
    channelRef.current = ch;
    return () => {
      void ch.unsubscribe();
      channelRef.current = null;
    };
  }, [liveToken, enabled]);

  return (patch: Partial<LiveState>) => {
    const next: LiveState = { ...stateRef.current, ...patch, v: 1, at: Date.now() };
    stateRef.current = next;
    if (!enabled) return;
    void channelRef.current?.send({ type: "broadcast", event: "state", payload: next });
    clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(async () => {
      const { data } = await supabase.auth.getSession();
      if (!data.session) return;
      await supabase.from("live_sessions").upsert({ owner_id: data.session.user.id, state: next, updated_at: new Date().toISOString() });
    }, 800);
  };
}

/** Viewer side (lyrics display, band follow). */
export function useLiveState(liveToken: string | undefined) {
  const [state, setState] = useState<LiveState | null>(null);
  const [connected, setConnected] = useState(false);
  const [invalid, setInvalid] = useState(false);

  useEffect(() => {
    if (!liveToken) return;
    let alive = true;
    const accept = (s: LiveState) => {
      if (!alive || !s || typeof s !== "object") return;
      setState((prev) => (prev && prev.at > (s.at ?? 0) ? prev : { ...EMPTY_LIVE, ...s }));
    };
    const fetchSaved = async () => {
      const { data, error } = await supabase.rpc("get_live_state", { p_token: liveToken });
      if (!alive) return;
      if (!error && data === null) setInvalid(true);
      else if (data) accept(data as LiveState);
    };
    void fetchSaved();
    const ch = supabase.channel(channelName(liveToken), { config: { broadcast: { self: false } } });
    ch.on("broadcast", { event: "state" }, ({ payload }) => accept(payload as LiveState));
    ch.subscribe((status) => {
      if (!alive) return;
      setConnected(status === "SUBSCRIBED");
      if (status === "SUBSCRIBED") void ch.send({ type: "broadcast", event: "hello", payload: {} });
    });
    // Safety net if a broadcast is missed (sleeping TV browser, flaky hotspot)
    const poll = setInterval(fetchSaved, 20_000);
    const onVis = () => document.visibilityState === "visible" && fetchSaved();
    document.addEventListener("visibilitychange", onVis);
    return () => {
      alive = false;
      clearInterval(poll);
      document.removeEventListener("visibilitychange", onVis);
      void ch.unsubscribe();
    };
  }, [liveToken]);

  return { state, connected, invalid };
}
