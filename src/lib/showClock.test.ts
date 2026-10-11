import { describe, expect, it } from "vitest";
import { showClock, stopDate, type ClockItem } from "./showClock";

const song = (id: string, minutes: number, keep = false): ClockItem => ({ id, kind: "song", title: id, seconds: minutes * 60, keep });
const at = (h: number, m: number) => { const d = new Date(2026, 9, 17, h, m, 0); return d; };

describe("show clock", () => {
  it("handles stop times after midnight", () => {
    expect(stopDate(at(22, 0), "23:00").getDate()).toBe(17);
    expect(stopDate(at(23, 30), "00:30", "21:00").getDate()).toBe(18);
    expect(stopDate(at(0, 10), "00:30", "21:00").getDate()).toBe(17);
  });

  it("reports time to spare and room for more songs", () => {
    const s = showClock(at(22, 0), "23:00", [song("a", 4), song("b", 4), song("c", 4)], 0);
    expect(s.slack).toBe(48 * 60);
    expect(s.roomForSongs).toBeGreaterThan(5);
    expect(s.cuts).toEqual([]);
  });

  it("counts breaks and suggests cuts when running long, never the closer or requested songs", () => {
    const list: ClockItem[] = [song("now", 4), song("long", 7), song("req", 6, true), { id: "brk", kind: "break", title: "Break", seconds: 0 }, song("x", 4), song("closer", 5)];
    const s = showClock(at(22, 30), "23:00", list, 1, 15);
    expect(s.slack).toBeLessThan(0);
    const ids = s.cuts.map((c) => c.id);
    expect(ids).toContain("long");
    expect(ids).not.toContain("closer");
    expect(ids).not.toContain("req");
    expect(ids).not.toContain("now");
  });
});
