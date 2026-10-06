import { describe, expect, it } from "vitest";
import { isSwipe, neighbours } from "./swipe";

describe("song swipe", () => {
  it("counts only quick, mostly-sideways drags", () => {
    expect(isSwipe(-120, 10, 200)).toBe("left");
    expect(isSwipe(120, -20, 300)).toBe("right");
    expect(isSwipe(40, 0, 200)).toBeNull(); // too short
    expect(isSwipe(100, 80, 200)).toBeNull(); // diagonal scroll
    expect(isSwipe(-150, 0, 1200)).toBeNull(); // slow drag
  });
  it("finds neighbours and stops at the ends", () => {
    expect(neighbours(["a", "b", "c"], "b")).toEqual({ prev: "a", next: "c", index: 1 });
    expect(neighbours(["a", "b", "c"], "a").prev).toBeNull();
    expect(neighbours(["a", "b", "c"], "c").next).toBeNull();
    expect(neighbours(["a"], "z").index).toBe(-1);
  });
});
