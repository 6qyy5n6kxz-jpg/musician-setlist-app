import { describe, expect, it } from "vitest";
import { capoShape, keyDistance, normalizeKey, parseChord, toNashville, transposeChord, transposeKey } from "./chords";
import { applyFlow, lyricSlides, matchHeaderLine, parseChordPro, parseLyricLine } from "./chordpro";
import { chordProToChordsOverLyrics, importChart, isChordLine, mergeChordLine } from "./convert";

describe("chords", () => {
  it("parses chords and rejects words", () => {
    expect(parseChord("F#m7/C#")).toEqual({ root: "F#", quality: "m7", bass: "C#" });
    expect(parseChord("Bbmaj7")).toEqual({ root: "Bb", quality: "maj7", bass: undefined });
    expect(parseChord("Csus4")).not.toBeNull();
    expect(parseChord("G(add9)")).not.toBeNull();
    expect(parseChord("And")).toBeNull();
    expect(parseChord("Am I")).toBeNull();
    expect(parseChord("Ever")).toBeNull();
  });
  it("transposes with sensible spelling", () => {
    expect(transposeChord("G", 2, false)).toBe("A");
    expect(transposeChord("G/B", 3, true)).toBe("Bb/D");
    expect(transposeChord("Em7", -2, false)).toBe("Dm7");
    expect(transposeChord("C#m", 0, true)).toBe("C#m");
    expect(transposeKey("G", 3)).toBe("Bb");
    expect(transposeKey("A", 1)).toBe("Bb");
    expect(transposeKey("E", 2)).toBe("F#");
    expect(transposeKey("C", 1)).toBe("Db");
    expect(transposeKey("Em", 2)).toBe("F#m");
    expect(transposeKey("Am", 3)).toBe("Cm");
    expect(normalizeKey("f# minor")).toBe("F#m");
    expect(keyDistance("G", "A")).toBe(2);
    expect(keyDistance("C", "A")).toBe(-3);
  });
  it("capo shapes and nashville", () => {
    expect(capoShape("A", 2)).toBe("G");
    expect(capoShape("C#m", 4)).toBe("Am");
    expect(toNashville("Em", "G")).toBe("6m");
    expect(toNashville("D/F#", "G")).toBe("5/7");
    expect(toNashville("F", "G")).toBe("b7");
    expect(toNashville("Am", "Am")).toBe("1m");
    expect(toNashville("C", "Am")).toBe("3");
    expect(toNashville("B7sus4", "A")).toBe("2(7sus4)");
    expect(toNashville("Dadd9", "A")).toBe("4add9");
  });
});

describe("chordpro", () => {
  it("splits lyric lines", () => {
    expect(parseLyricLine("Amaz[A]ing grace [D]")).toEqual([
      { chord: null, lyric: "Amaz" }, { chord: "A", lyric: "ing grace " }, { chord: "D", lyric: "" },
    ]);
  });
  it("detects headers", () => {
    expect(matchHeaderLine("Verse 1:")).toBe("Verse 1");
    expect(matchHeaderLine("[Chorus]")).toBe("Chorus");
    expect(matchHeaderLine("{{title: Bridge}}")).toBe("Bridge");
    expect(matchHeaderLine("[G]")).toBeNull();
    expect(matchHeaderLine("Chorus is where we sing")).toBeNull();
    expect(matchHeaderLine("CHORUS")).toBe("CHORUS");
  });
  it("parses sections, metadata and repeats", () => {
    const song = parseChordPro(`{title: Test}
{key: G}
{start_of_verse: Verse 1}
[G]Hello [C]world
{end_of_verse}

{soc}
[D]Sing it
{eoc}
{chorus}
Bridge:
[Em]Ooh`);
    expect(song.meta).toEqual({ title: "Test", key: "G" });
    expect(song.sections.map((s) => [s.type, s.label])).toEqual([
      ["verse", "Verse 1"], ["chorus", "Chorus"], ["chorus", "Chorus"], ["bridge", "Bridge"],
    ]);
    expect(song.sections[2].lines).toHaveLength(1);
  });
  it("applies flow and builds lyric slides", () => {
    const song = parseChordPro(`Verse 1:\n[G]One\n\nChorus:\n[C]Two\n\nVerse 2:\nThree`);
    expect(applyFlow(song.sections, "V1 C V2 C").map((s) => s.label)).toEqual(["Verse 1", "Chorus", "Verse 2", "Chorus"]);
    expect(lyricSlides(song.sections)).toEqual([
      { pos: 0, label: "Verse 1", lines: ["One"] }, { pos: 1, label: "Chorus", lines: ["Two"] },
      { pos: 2, label: "Verse 2", lines: ["Three"] },
    ]);
  });
  it("blank lines split implicit paragraphs", () => {
    expect(parseChordPro("Line one\nLine two\n\nLine three").sections).toHaveLength(2);
  });
});

describe("import", () => {
  it("detects chord lines", () => {
    expect(isChordLine("G   D/F#   Em   C")).toBe(true);
    expect(isChordLine("  [A]            A/C#   [D]      [A]")).toBe(true);
    expect(isChordLine("| G . . . | C . . . |")).toBe(true);
    expect(isChordLine("A man walked in")).toBe(false);
  });
  it("merges chords over lyrics by column", () => {
    expect(mergeChordLine("G       C", "Hello my world")).toBe("[G]Hello my[C] world");
    expect(mergeChordLine("G         D", "Hi")).toBe("[G]Hi [D]");
  });
  it("converts an Ultimate Guitar paste", () => {
    const { body } = importChart(`[Verse 1]
G           D/F#
Well, today is gonna be
Em        C
the day

[Chorus]
C   D   G`);
    expect(body).toBe(`{start_of_verse: Verse 1}
[G]Well, today [D/F#]is gonna be
[Em]the day [C]
{end_of_verse}

{start_of_chorus: Chorus}
[C] [D] [G]
{end_of_chorus}`);
  });
  it("converts OnSong files", () => {
    const { meta, body } = importChart(`Amazing Grace
John Newton
Key: G
Tempo: 72
Capo: 2

Verse 1:
A[G]mazing grace`, "grace.onsong");
    expect(meta).toEqual({ title: "Amazing Grace", artist: "John Newton", key: "G", tempo: 72, capo: 2 });
    expect(body).toBe("{start_of_verse: Verse 1}\nA[G]mazing grace\n{end_of_verse}");
  });
  it("handles the legacy app's charts", () => {
    const { meta, body } = importChart(`{title: Tennessee Whiskey}  \n{artist: Chris Stapleton}  \n{key: A}  \n\n{{title: Verse 1}}  \n[A] Used to spend  \n[D] Liquor was\n\n[Intro]\n[am]`);
    expect(meta).toEqual({ title: "Tennessee Whiskey", artist: "Chris Stapleton", key: "A" });
    expect(body).toContain("{start_of_verse: Verse 1}\n[A] Used to spend\n[D] Liquor was\n{end_of_verse}");
    expect(body).toContain("{start_of_intro: Intro}\n[Am]\n{end_of_intro}");
  });
  it("round-trips to chords over lyrics", () => {
    expect(chordProToChordsOverLyrics("[G]Hello my[C] world")).toBe("G       C\nHello my world");
  });
});

import { transposeContent } from "./chordpro";
describe("transposeContent", () => {
  it("rewrites chords but not other brackets", () => {
    expect(transposeContent("[G]Hi [D/F#]there [Verse]", 3, true)).toBe("[Bb]Hi [F/A]there [Verse]");
  });
});

import { fixMojibake } from "./convert";
describe("fixMojibake", () => {
  it("repairs latin-1 decoded utf-8", () => {
    expect(fixMojibake("I\u00e2\u0080\u0099ve been")).toBe("I\u2019ve been");
    expect(fixMojibake("Caf\u00c3\u00a9")).toBe("Caf\u00e9");
    expect(fixMojibake("plain text")).toBe("plain text");
  });
});
