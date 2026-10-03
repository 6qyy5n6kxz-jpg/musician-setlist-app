import { describe, expect, it } from "vitest";
import { timedSectionAt } from "./stage";

describe("timedSectionAt", () => {
  const marks = [{ pos: 0, t: 12 }, { pos: 1, t: 40.5 }, { pos: 2, t: 70 }];
  it("is nothing before the first mark", () => expect(timedSectionAt(marks, 5)).toBe(-1));
  it("shows a section slightly early", () => expect(timedSectionAt(marks, 11.5)).toBe(0));
  it("follows the track", () => {
    expect(timedSectionAt(marks, 39)).toBe(0);
    expect(timedSectionAt(marks, 40)).toBe(1);
    expect(timedSectionAt(marks, 200)).toBe(2);
  });
});
