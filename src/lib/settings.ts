// Per-device preferences (an iPad on stage and a laptop at home can differ), kept in localStorage.
import { useSyncExternalStore } from "react";

export type Theme = "dark" | "light" | "lowlight";

export type PedalAction =
  | "pageDown" | "pageUp" | "nextSong" | "prevSong" | "toggleScroll" | "nextSlide" | "prevSlide"
  | "toggleMetronome" | "toggleTrack" | "blankDisplay";

export const PEDAL_ACTION_LABELS: Record<PedalAction, string> = {
  pageDown: "Scroll down / next page",
  pageUp: "Scroll up / previous page",
  nextSong: "Next song",
  prevSong: "Previous song",
  toggleScroll: "Start / stop autoscroll",
  nextSlide: "Lyrics display: next section",
  prevSlide: "Lyrics display: previous section",
  toggleMetronome: "Start / stop metronome",
  toggleTrack: "Play / pause backing track",
  blankDisplay: "Lyrics display: blank / show",
};

export interface Settings {
  theme: Theme;
  fontScale: number;
  showChords: boolean;
  nashville: boolean;
  columns: 1 | 2;
  /** Chord diagram panel beside the chart. */
  chordHelper: boolean;
  chordColor: string;
  /** Key name (KeyboardEvent.key) -> action. Bluetooth pedals send keys like PageDown or ArrowRight. */
  pedalMap: Record<string, PedalAction>;
  /** At the bottom of a song, "scroll down" moves to the next song in the set. */
  pageDownAdvances: boolean;
  /** Seconds before autoscroll starts moving. */
  scrollDelay: number;
  /** Used when a song has no duration. */
  defaultDurationSec: number;
  countInBars: number;
  clickAccent: boolean;
  metronomeVolume: number;
  flashOnBeat: boolean;
  /** When the performer scrolls, the lyrics display follows the section at the top of the screen. */
  displayFollowsScroll: boolean;
  /** Show the request queue pop-up when a new request arrives in Perform mode. */
  requestPopups: boolean;
  /** Send gear changes over MIDI on song change (this device only; needs a browser with Web MIDI). */
  midiOut: boolean;
  /** Name of the MIDI output port to use. */
  midiPort: string | null;
}

export const DEFAULT_SETTINGS: Settings = {
  theme: "dark",
  fontScale: 1,
  showChords: true,
  nashville: false,
  columns: 1,
  chordHelper: false,
  chordColor: "#f5b941",
  pedalMap: {
    PageDown: "pageDown",
    PageUp: "pageUp",
    ArrowDown: "pageDown",
    ArrowUp: "pageUp",
    ArrowRight: "nextSong",
    ArrowLeft: "prevSong",
    " ": "toggleScroll",
    Enter: "nextSlide",
  },
  pageDownAdvances: true,
  scrollDelay: 8,
  defaultDurationSec: 210,
  countInBars: 1,
  clickAccent: true,
  metronomeVolume: 0.7,
  flashOnBeat: true,
  displayFollowsScroll: true,
  requestPopups: true,
  midiOut: false,
  midiPort: null,
};

const KEY = "stage-settings-v1";

function load(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {
    /* private mode or corrupt value */
  }
  return DEFAULT_SETTINGS;
}

let current = load();
const listeners = new Set<() => void>();

export function getSettings(): Settings {
  return current;
}

export function updateSettings(patch: Partial<Settings>) {
  current = { ...current, ...patch };
  try {
    localStorage.setItem(KEY, JSON.stringify(current));
  } catch {
    /* ignore */
  }
  applyTheme(current.theme);
  listeners.forEach((fn) => fn());
}

export function useSettings(): Settings {
  return useSyncExternalStore(
    (fn) => (listeners.add(fn), () => listeners.delete(fn)),
    () => current,
  );
}

export function applyTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme;
  const meta = document.querySelector('meta[name="theme-color"]');
  meta?.setAttribute("content", theme === "light" ? "#f7f5f0" : theme === "lowlight" ? "#000000" : "#111214");
}
