// Stage helpers: pedal/keyboard actions, screen wake lock, metronome, autoscroll.
import { useEffect, useRef } from "react";
import { getSettings, type PedalAction } from "./settings";

/** Map key presses (Bluetooth page-turner pedals send keys) to actions. Ignores typing in fields. */
export function usePedalActions(handlers: Partial<Record<PedalAction, () => void>>, enabled = true) {
  const ref = useRef(handlers);
  ref.current = handlers;
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const action = getSettings().pedalMap[e.key];
      const fn = action && ref.current[action];
      if (fn) {
        e.preventDefault();
        fn();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled]);
}

/** Keep the screen on while performing. Re-acquired when the app comes back to the foreground. */
export function useWakeLock(active: boolean) {
  useEffect(() => {
    if (!active || !("wakeLock" in navigator)) return;
    let lock: WakeLockSentinel | null = null;
    let cancelled = false;
    const acquire = async () => {
      try {
        lock = await navigator.wakeLock.request("screen");
      } catch {
        /* denied or unsupported in this context */
      }
      if (cancelled) void lock?.release();
    };
    const onVis = () => {
      if (document.visibilityState === "visible") void acquire();
    };
    void acquire();
    document.addEventListener("visibilitychange", onVis);
    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", onVis);
      void lock?.release();
    };
  }, [active]);
}

// ------------------------------------------------------------------ metronome
let audioCtx: AudioContext | null = null;
export function getAudioContext(): AudioContext {
  if (!audioCtx) audioCtx = new AudioContext();
  if (audioCtx.state === "suspended") void audioCtx.resume();
  return audioCtx;
}

export function beatsPerBar(timeSignature: string | null | undefined): number {
  const n = parseInt((timeSignature ?? "4/4").split("/")[0], 10);
  return Number.isFinite(n) && n > 0 && n <= 16 ? n : 4;
}

/**
 * Sample-accurate metronome using a lookahead scheduler (Web Audio clock, not setInterval).
 * onBeat fires on the main thread close to each click, for visual flashes.
 */
export class Metronome {
  private nextTime = 0;
  private beat = 0;
  private timer: ReturnType<typeof setInterval> | undefined;
  private stopAfter: number | null = null;
  bpm = 120;
  beatsPerBar = 4;
  volume = 0.7;
  accent = true;
  onBeat: ((beat: number) => void) | null = null;
  onDone: (() => void) | null = null;

  get running() {
    return this.timer !== undefined;
  }

  /** Start clicking. With `bars`, stops by itself after that many bars (count-in). */
  start(bars?: number) {
    this.stop();
    const ctx = getAudioContext();
    this.beat = 0;
    this.nextTime = ctx.currentTime + 0.08;
    this.stopAfter = bars ? bars * this.beatsPerBar : null;
    this.timer = setInterval(() => this.schedule(), 25);
    this.schedule();
  }

  stop() {
    if (this.timer !== undefined) clearInterval(this.timer);
    this.timer = undefined;
  }

  private schedule() {
    const ctx = getAudioContext();
    while (this.nextTime < ctx.currentTime + 0.12) {
      if (this.stopAfter !== null && this.beat >= this.stopAfter) {
        const doneAt = this.nextTime;
        this.stop();
        setTimeout(() => this.onDone?.(), Math.max(0, (doneAt - ctx.currentTime) * 1000));
        return;
      }
      const beatInBar = this.beat % this.beatsPerBar;
      this.click(this.nextTime, this.accent && beatInBar === 0);
      const at = this.nextTime;
      const b = this.beat;
      setTimeout(() => this.onBeat?.(b % this.beatsPerBar), Math.max(0, (at - ctx.currentTime) * 1000));
      this.nextTime += 60 / this.bpm;
      this.beat++;
    }
  }

  private click(time: number, accent: boolean) {
    const ctx = getAudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.frequency.value = accent ? 1600 : 1000;
    gain.gain.setValueAtTime(0.0001, time);
    gain.gain.exponentialRampToValueAtTime(Math.max(0.001, this.volume), time + 0.002);
    gain.gain.exponentialRampToValueAtTime(0.0001, time + 0.05);
    osc.connect(gain).connect(ctx.destination);
    osc.start(time);
    osc.stop(time + 0.06);
  }
}

/** Tap tempo: returns BPM from the average of recent tap intervals. */
export function createTapTempo() {
  let taps: number[] = [];
  return () => {
    const now = performance.now();
    taps = taps.filter((t) => now - t < 3000);
    taps.push(now);
    if (taps.length < 2) return null;
    const intervals = taps.slice(1).map((t, i) => t - taps[i]);
    const avg = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    return Math.round(60000 / avg);
  };
}

// ------------------------------------------------------------------ autoscroll
/**
 * Scrolls an element from its current position to the bottom over `durationSec`,
 * after `delaySec`. Manual scrolling while running nudges the position; speed is kept.
 */
export class AutoScroller {
  private raf = 0;
  private last = 0;
  private delayLeft = 0;
  private pos = 0;
  speed = 0; // px per second
  running = false;
  onChange: ((running: boolean) => void) | null = null;

  constructor(private el: HTMLElement) {}

  start(durationSec: number, delaySec: number) {
    const distance = this.el.scrollHeight - this.el.clientHeight - this.el.scrollTop;
    if (distance <= 0) return;
    this.speed = distance / Math.max(10, durationSec - delaySec);
    this.delayLeft = delaySec;
    this.pos = this.el.scrollTop;
    this.running = true;
    this.last = performance.now();
    this.onChange?.(true);
    const step = (t: number) => {
      if (!this.running) return;
      const dt = (t - this.last) / 1000;
      this.last = t;
      if (Math.abs(this.el.scrollTop - this.pos) > 2) this.pos = this.el.scrollTop; // user scrolled
      if (this.delayLeft > 0) {
        this.delayLeft -= dt;
      } else {
        this.pos += this.speed * dt;
        this.el.scrollTop = this.pos;
        if (this.el.scrollTop + this.el.clientHeight >= this.el.scrollHeight - 1) {
          this.stop();
          return;
        }
      }
      this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }

  adjust(factor: number) {
    this.speed *= factor;
  }

  stop() {
    cancelAnimationFrame(this.raf);
    if (this.running) {
      this.running = false;
      this.onChange?.(false);
    }
  }
}

export function formatDuration(sec: number | null | undefined): string {
  if (!sec) return "";
  return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
}

export function parseDurationInput(v: string): number | null {
  const t = v.trim();
  if (!t) return null;
  const m = /^(\d+):(\d{1,2})$/.exec(t);
  if (m) return Number(m[1]) * 60 + Number(m[2]);
  const n = Number(t);
  return Number.isFinite(n) && n > 0 ? Math.round(n) : null;
}

/** Rough song length from tempo when no duration is set (matches the legacy app's estimate). */
export function estimateDuration(tempo: number | null | undefined): number {
  if (!tempo) return getSettings().defaultDurationSec;
  return Math.round(Math.min(330, Math.max(150, 240 * (100 / tempo) ** 0.35)));
}

/** Section that should be showing at track time `t` (with a small lead so lyrics appear early). */
export function timedSectionAt(marks: { pos: number; t: number }[], t: number, lead = 0.8): number {
  let pos = -1;
  for (const m of marks) if (m.t <= t + lead) pos = m.pos;
  return pos;
}
