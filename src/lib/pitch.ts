// Pitch detection for the tuner and vocal-range tools (YIN, on raw microphone samples).

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];

/**
 * Fundamental frequency of a mono buffer in Hz, or null when it's too quiet or not clearly pitched.
 * YIN difference function with cumulative-mean normalisation and parabolic interpolation.
 */
export function detectPitch(buf: Float32Array, sampleRate: number, minHz = 60, maxHz = 1500): number | null {
  let rms = 0;
  for (let i = 0; i < buf.length; i++) rms += buf[i] * buf[i];
  rms = Math.sqrt(rms / buf.length);
  if (rms < 0.01) return null; // silence / room noise

  const maxLag = Math.min(Math.floor(sampleRate / minHz), Math.floor(buf.length / 2));
  const minLag = Math.max(2, Math.floor(sampleRate / maxHz));
  const n = buf.length - maxLag;
  const d = new Float32Array(maxLag + 1);
  for (let lag = 1; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const diff = buf[i] - buf[i + lag];
      sum += diff * diff;
    }
    d[lag] = sum;
  }
  // Cumulative mean normalised difference
  const cmnd = new Float32Array(maxLag + 1);
  cmnd[0] = 1;
  let running = 0;
  for (let lag = 1; lag <= maxLag; lag++) {
    running += d[lag];
    cmnd[lag] = running ? (d[lag] * lag) / running : 1;
  }
  const threshold = 0.12;
  let tau = -1;
  for (let lag = minLag; lag <= maxLag; lag++) {
    if (cmnd[lag] < threshold) {
      while (lag + 1 <= maxLag && cmnd[lag + 1] < cmnd[lag]) lag++;
      tau = lag;
      break;
    }
  }
  if (tau < 0) return null;
  // Parabolic interpolation around the minimum for sub-sample accuracy
  const a = cmnd[tau - 1] ?? cmnd[tau], b = cmnd[tau], c = cmnd[tau + 1] ?? cmnd[tau];
  const shift = (a - c) / (2 * (a - 2 * b + c) || 1);
  const hz = sampleRate / (tau + (Number.isFinite(shift) ? shift : 0));
  return hz >= minHz && hz <= maxHz ? hz : null;
}

/** MIDI note number (A4 = 69) as a float. */
export const hzToMidi = (hz: number) => 69 + 12 * Math.log2(hz / 440);
export const midiToHz = (m: number) => 440 * 2 ** ((m - 69) / 12);

/** "A4", "C#3" (or flats). */
export function midiName(m: number, flats = false): string {
  const r = Math.round(m);
  return `${(flats ? FLAT_NAMES : NOTE_NAMES)[((r % 12) + 12) % 12]}${Math.floor(r / 12) - 1}`;
}

/** Parse "A4", "Bb3", "c#5" → MIDI number. */
export function parseMidiName(s: string): number | null {
  const m = /^\s*([A-Ga-g])([#b♯♭]?)(-?\d)\s*$/.exec(s);
  if (!m) return null;
  const base = NOTE_NAMES.indexOf(m[1].toUpperCase());
  const acc = m[2] === "#" || m[2] === "♯" ? 1 : m[2] === "b" || m[2] === "♭" ? -1 : 0;
  return (Number(m[3]) + 1) * 12 + base + acc;
}

/** Nearest note and how many cents sharp (+) or flat (−) the pitch is. */
export function tunerReading(hz: number): { note: string; midi: number; cents: number } {
  const m = hzToMidi(hz);
  const r = Math.round(m);
  return { note: midiName(r), midi: r, cents: Math.round((m - r) * 100) };
}

/**
 * Collects pitches from a live microphone. Call `start()` from a tap (iOS needs a user gesture);
 * `onPitch` fires ~30 times a second with Hz (or null for silence).
 */
export class PitchListener {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private raf = 0;
  onPitch: (hz: number | null) => void = () => {};

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    this.ctx = new Ctx();
    const src = this.ctx.createMediaStreamSource(this.stream);
    const analyser = this.ctx.createAnalyser();
    analyser.fftSize = 4096;
    src.connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    const tick = () => {
      analyser.getFloatTimeDomainData(buf);
      this.onPitch(detectPitch(buf, this.ctx!.sampleRate));
      this.raf = requestAnimationFrame(tick);
    };
    tick();
  }

  stop() {
    cancelAnimationFrame(this.raf);
    this.stream?.getTracks().forEach((t) => t.stop());
    void this.ctx?.close();
    this.ctx = null;
    this.stream = null;
  }
}

/**
 * Range a person actually sang over a stretch of pitches: ignores brief blips and octave
 * glitches by taking the 5th–95th percentile of notes held for a few frames.
 */
export function sungRange(midis: number[]): { low: number; high: number } | null {
  const notes = midis.filter((m) => Number.isFinite(m)).map((m) => Math.round(m));
  if (notes.length < 8) return null;
  // Keep notes that appear in at least 3 consecutive frames (held, not a glitch)
  const held: number[] = [];
  for (let i = 2; i < notes.length; i++) if (notes[i] === notes[i - 1] && notes[i] === notes[i - 2]) held.push(notes[i]);
  const pool = (held.length >= 6 ? held : notes).sort((a, b) => a - b);
  const at = (p: number) => pool[Math.min(pool.length - 1, Math.max(0, Math.round(p * (pool.length - 1))))];
  return { low: at(0.05), high: at(0.95) };
}
