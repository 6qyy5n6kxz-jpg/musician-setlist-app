import { describe, expect, it } from "vitest";
import { setBalance, suggestBeatCategory, suggestGear, suggestGuitarCategory, suggestPianoCategory, suggestShows, type GearLibrary } from "./gear";

const base = { title: "", genre: null, year: null, tempo: null, time_signature: null, tags: [], instrument: null, lead_vocal: null, artist: "" };

describe("tone suggestions", () => {
  it("picks piano sounds by style", () => {
    expect(suggestPianoCategory({ ...base, genre: "Soul", tempo: 120 }).category).toBe("clav");
    expect(suggestPianoCategory({ ...base, genre: "Pop", year: 1977, tempo: 90 }).category).toBe("rhodes");
    expect(suggestPianoCategory({ ...base, genre: "Country", tempo: 130 }).category).toBe("upright");
    expect(suggestPianoCategory({ ...base, title: "White Christmas" }).category).toBe("grand");
  });
  it("picks guitar and beat styles", () => {
    expect(suggestGuitarCategory({ ...base, genre: "Alternative", year: 1994 }).category).toBe("crunch");
    expect(suggestGuitarCategory({ ...base, genre: "Country", tempo: 120 }).category).toBe("edge");
    expect(suggestBeatCategory({ ...base, time_signature: "6/8", genre: "Rock" }).category).toBe("ballad68");
    expect(suggestBeatCategory({ ...base, genre: "Country", tempo: 150 }).category).toBe("train");
  });
  it("uses the performer's presets and their usual choice per artist", () => {
    const lib: GearLibrary = {
      numa: [{ id: "g", program: 1, name: "Concert Grand", category: "grand" }, { id: "r", program: 9, name: "Suitcase", category: "rhodes" }],
    };
    const song = { ...base, artist: "Billy Joel", genre: "Pop", year: 1977, tempo: 80, instrument: "piano" as const };
    expect(suggestGear(song, lib).numa?.preset?.name).toBe("Suitcase");
    expect(suggestGear(song, lib, [{ artist: "Billy Joel", gear: { numa: "g" } }]).numa?.preset?.name).toBe("Concert Grand");
  });
});

describe("show fit and balance", () => {
  it("suggests signature shows", () => {
    expect(suggestShows({ ...base, genre: "Country", lead_vocal: "kendra" }).map((s) => s.tag)).toContain("women of country");
    expect(suggestShows({ ...base, title: "Last Christmas" }).map((s) => s.tag)).toEqual(["holiday"]);
    expect(suggestShows({ ...base, title: "Last Christmas", tags: ["holiday"] })).toEqual([]);
  });
  it("counts vocals and instrument switches", () => {
    const b = setBalance([
      { instrument: "piano", lead_vocal: "kendra" }, { instrument: "electric", lead_vocal: "devin" },
      { instrument: "electric", lead_vocal: "both" }, { instrument: null, lead_vocal: null },
    ]);
    expect(b.vocals).toEqual({ kendra: 1, devin: 1, both: 1, unset: 1 });
    expect(b.switches).toBe(1);
  });
});

import { DEFAULT_MIDI, singerKey, songMidiMessages } from "./gear";
describe("singer keys and MIDI", () => {
  it("picks the lead singer's key", () => {
    expect(singerKey({ lead_vocal: "kendra", key_kendra: "A", key_devin: "E" })).toBe("A");
    expect(singerKey({ lead_vocal: "devin", key_kendra: "A", key_devin: null })).toBeNull();
    expect(singerKey({ lead_vocal: "both", key_kendra: null, key_devin: "E" })).toBe("E");
  });
  it("builds program changes and BeatBuddy bank select", () => {
    const lib: GearLibrary = {
      numa: [{ id: "n", program: 7, name: "Upright", category: "upright" }],
      cortex: [{ id: "c", program: 3, name: "Edge", category: "edge" }],
      beatbuddy: [{ id: "b", program: 2, name: "Shuffle", category: "blues", folder: 4 }],
    };
    const msgs = songMidiMessages({ numa: "n", cortex: "c", beatbuddy: "b" }, lib, DEFAULT_MIDI, "piano");
    expect(msgs.map((m) => m.bytes)).toEqual([
      [0xc0, 6],            // Numa ch1, program 7 shown -> 6 sent
      [0xb2, 0, 0], [0xb2, 32, 3], [0xc2, 1], // BeatBuddy ch3: folder 4 -> 3, song 2 -> 1
    ]); // no Cortex: the song is a piano song
  });
});
