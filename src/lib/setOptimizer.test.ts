import { describe, expect, it } from "vitest";
import { isSlow, keyRelation, optimizeSet, scoreSet, songEnergy, type OptSong } from "./setOptimizer";

const song = (id: string, key: string, tempo: number, extra: Partial<OptSong> = {}): OptSong => ({
  id, title: id, artist: `Artist ${id}`, key, tempo, tags: [], time_signature: null, instrument: null, lead_vocal: null, ...extra,
});
const opts = { keepOpener: false, keepCloser: false, fewerSwitches: true };

describe("set optimizer", () => {
  it("relates keys", () => {
    expect(keyRelation("G", "G")).toBe("same");
    expect(keyRelation("C", "Am")).toBe("relative");
    expect(keyRelation("Em", "G")).toBe("relative");
    expect(keyRelation("G", "D")).toBe("different");
    expect(keyRelation(null, "D")).toBe("unknown");
  });

  it("rates energy from tempo and tags", () => {
    expect(isSlow(songEnergy(song("a", "C", 70)))).toBe(true);
    expect(isSlow(songEnergy(song("b", "C", 128)))).toBe(false);
    expect(songEnergy(song("c", "C", 110, { tags: ["mellow"] }))).toBeLessThan(songEnergy(song("d", "C", 110)));
  });

  it("splits up same keys and slow songs, opens and closes strong", () => {
    // Worst case: all the G songs together, all the ballads together
    const set = [
      song("1", "G", 72), song("2", "G", 70), song("3", "G", 75),
      song("4", "D", 128), song("5", "D", 132), song("6", "A", 120),
      song("7", "E", 140), song("8", "C", 68), song("9", "F", 118), song("10", "Bb", 124),
    ];
    const before = scoreSet(set, opts);
    const after = scoreSet(optimizeSet(set, opts), opts);
    expect(before.sameKey).toBeGreaterThan(0);
    expect(before.slowStacked).toBeGreaterThan(0);
    expect(after.sameKey).toBe(0);
    expect(after.slowStacked).toBe(0);
    expect(after.weakOpenOrClose).toBe(0);
    expect(after.cost).toBeLessThan(before.cost);
  });

  it("keeps a locked opener and closer and every song", () => {
    const set = Array.from({ length: 8 }, (_, i) => song(String(i), ["G", "D", "C", "A"][i % 4], 70 + i * 9));
    const out = optimizeSet(set, { ...opts, keepOpener: true, keepCloser: true });
    expect(out[0].id).toBe("0");
    expect(out[7].id).toBe("7");
    expect(out.map((s) => s.id).sort()).toEqual(set.map((s) => s.id).sort());
  });

  it("gives the same answer every time", () => {
    const set = Array.from({ length: 12 }, (_, i) => song(String(i), ["G", "D", "C"][i % 3], 60 + ((i * 37) % 90)));
    expect(optimizeSet(set, opts).map((s) => s.id)).toEqual(optimizeSet(set, opts).map((s) => s.id));
  });
});
