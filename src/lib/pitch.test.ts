import { describe, expect, it } from "vitest";
import { capoFor, recommendKeys } from "./keyFinder";
import { detectPitch, hzToMidi, midiName, parseMidiName, sungRange, tunerReading } from "./pitch";

const tone = (hz: number, rate = 44100, n = 4096, harmonics = false) => {
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = (2 * Math.PI * hz * i) / rate;
    b[i] = 0.5 * Math.sin(t) + (harmonics ? 0.3 * Math.sin(2 * t) + 0.2 * Math.sin(3 * t) : 0);
  }
  return b;
};

describe("pitch detection", () => {
  it("finds the fundamental of pure and voice-like tones", () => {
    for (const hz of [82.41, 110, 220, 261.63, 440, 659.25]) {
      expect(Math.abs(hzToMidi(detectPitch(tone(hz), 44100)!) - hzToMidi(hz))).toBeLessThan(0.1);
      expect(Math.abs(hzToMidi(detectPitch(tone(hz, 44100, 4096, true), 44100)!) - hzToMidi(hz))).toBeLessThan(0.1);
    }
  });
  it("ignores silence", () => {
    expect(detectPitch(new Float32Array(4096), 44100)).toBeNull();
  });
  it("names notes and tuner offsets", () => {
    expect(midiName(69)).toBe("A4");
    expect(parseMidiName("Bb3")).toBe(58);
    expect(tunerReading(446).note).toBe("A4");
    expect(tunerReading(446).cents).toBeGreaterThan(20);
  });
  it("takes the held range and ignores glitches", () => {
    const frames = [...Array(20).fill(57), 81, ...Array(20).fill(64), ...Array(20).fill(69), 45];
    expect(sungRange(frames)).toEqual({ low: 57, high: 69 });
  });
});

describe("key finder", () => {
  // Melody A3–D5 written in G; Kendra (alto-ish) F3–C5
  const kendra = { low: 53, high: 72 };
  it("moves a song down into a lower voice", () => {
    const [best] = recommendKeys("G", { low: 57, high: 74 }, [kendra], "piano");
    expect(best.semitones).toBeLessThan(0);
    expect(best.high).toBeLessThanOrEqual(72);
    expect(best.low).toBeGreaterThanOrEqual(53);
  });
  it("finds a key that fits both singers on a duet", () => {
    const devin = { low: 45, high: 67 }; // baritone A2–G4
    const [best] = recommendKeys("C", { low: 55, high: 67 }, [kendra, devin], "acoustic");
    expect(best.fit).not.toBe("stretch");
  });
  it("suggests a capo for awkward guitar keys", () => {
    expect(capoFor("Bb")).toEqual({ fret: 1, shapes: "A" });
    expect(capoFor("G")).toBeUndefined();
  });
});

describe("key finder headroom", () => {
  it("prefers a key with room on top over one that hits the singer's ceiling", () => {
    // Jolene-like melody B3–E5 in C#m for a voice F3–C5
    const [best] = recommendKeys("C#m", { low: 59, high: 76 }, [{ low: 53, high: 72 }], "acoustic");
    expect(best.fit).toBe("comfortable");
    expect(best.high).toBeLessThan(72);
  });
});
