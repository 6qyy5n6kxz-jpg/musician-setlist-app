import { describe, expect, it } from "vitest";
import { countMarks, eraseAt, simplify, withPageMarks, type Mark } from "./annotations";

describe("PDF annotations", () => {
  it("simplifies strokes but keeps their ends", () => {
    const pts = [0, 0, 0.0001, 0.0001, 0.0002, 0.0002, 0.5, 0.5, 0.5001, 0.5001];
    const s = simplify(pts);
    expect(s.slice(0, 2)).toEqual([0, 0]);
    expect(s.slice(-2)).toEqual([0.5001, 0.5001]);
    expect(s.length).toBeLessThan(pts.length);
  });
  it("erases ink and text under the eraser only", () => {
    const marks: Mark[] = [
      { t: "ink", color: "#f00", w: 0.004, pts: [0.1, 0.1, 0.3, 0.1] },
      { t: "ink", color: "#00f", w: 0.004, pts: [0.1, 0.8, 0.3, 0.8] },
      { t: "text", color: "#000", x: 0.6, y: 0.5, size: 0.03, text: "Capo 2" },
    ];
    expect(eraseAt(marks, 0.2, 0.1, 0.01, 1.3).map((m) => m.color)).toEqual(["#00f", "#000"]);
    expect(eraseAt(marks, 0.62, 0.49, 0.01, 1.3).map((m) => m.t)).toEqual(["ink", "ink"]);
    expect(eraseAt(marks, 0.9, 0.3, 0.01, 1.3)).toHaveLength(3);
  });
  it("stores marks per page and drops empty pages", () => {
    const a = withPageMarks({}, 2, [{ t: "text", color: "#000", x: 0, y: 0, size: 0.03, text: "x" }]);
    expect(countMarks(a)).toBe(1);
    expect(withPageMarks(a, 2, []).pages).toEqual({});
  });
});
