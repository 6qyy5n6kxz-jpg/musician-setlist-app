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
