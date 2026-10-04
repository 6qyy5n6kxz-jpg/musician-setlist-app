import { describe, expect, it } from "vitest";
import { clefsFor, findLedgerNotes, findStaves, noteName, type DarkImage } from "./omr";

// Draw a synthetic score: staff lines, ledger lines, noteheads, a lyric blob and a tab staff.
function score() {
  const w = 900, h = 520;
  const dark = new Uint8Array(w * h);
  const rect = (x0: number, y0: number, x1: number, y1: number) => {
    for (let y = Math.round(y0); y <= Math.round(y1); y++) for (let x = Math.round(x0); x <= Math.round(x1); x++) if (x >= 0 && y >= 0 && x < w && y < h) dark[y * w + x] = 1;
  };
  const ellipse = (cx: number, cy: number, rx: number, ry: number, hollow = false) => {
    for (let y = -ry; y <= ry; y++) for (let x = -rx; x <= rx; x++) {
      const d = (x * x) / (rx * rx) + (y * y) / (ry * ry);
      if (d <= 1 && (!hollow || d >= 0.45)) dark[Math.round(cy + y) * w + Math.round(cx + x)] = 1;
    }
  };
  const s = 14; // staff space
  const top = 120;
  const lineY = (i: number) => top + i * s; // i = 0..4 top->bottom
  for (let i = 0; i < 5; i++) rect(60, lineY(i), 840, lineY(i) + 1);
  const bottom = lineY(4) + 0.5;
  const yOf = (step: number) => bottom - (step * s) / 2;
  const ledger = (cx: number, step: number) => rect(cx - s * 0.95, yOf(step) - 0.5, cx + s * 0.95, yOf(step) + 0.5);
  const stem = (cx: number, cy: number) => rect(cx + s * 0.55, cy - s * 3, cx + s * 0.62, cy);
  // A5 on first ledger above (step 10)
  ledger(200, 10); ellipse(200, yOf(10), 8, 6); stem(200, yOf(10));
  // C6 on second ledger above (step 12): two ledger lines
  ledger(320, 10); ledger(320, 12); ellipse(320, yOf(12), 8, 6); stem(320, yOf(12));
  // B5 hollow half note in the space above the first ledger (step 11)
  ledger(440, 10); ellipse(440, yOf(11), 8, 6, true); stem(440, yOf(11));
  // C4 on first ledger below (step -2)
  ledger(560, -2); ellipse(560, yOf(-2), 8, 6); stem(560, yOf(-2));
  // G4 inside the staff (step 2): not a ledger note
  ellipse(680, yOf(2), 8, 6); stem(680, yOf(2));
  // "Lyric" blob below the staff, no ledger lines: must be ignored
  rect(740, yOf(-6) - 5, 760, yOf(-6) + 5);
  // Tab staff (6 lines, wider spacing) further down: must be skipped
  for (let i = 0; i < 6; i++) rect(60, 360 + i * 18, 840, 361 + i * 18);
  return { img: { w, h, dark } as DarkImage, s };
}

describe("ledger-note reading help", () => {
  it("finds the music staff and skips tab staves", () => {
    const { img, s } = score();
    const staves = findStaves(img);
    expect(staves).toHaveLength(1);
    expect(Math.abs(staves[0].space - s)).toBeLessThan(1);
  });
  it("finds notes on ledger lines only, and names them", () => {
    const { img } = score();
    const staves = findStaves(img);
    const notes = findLedgerNotes(img, staves).sort((a, b) => a.x - b.x);
    expect(notes.map((n) => [Math.round(n.x / 10) * 10, n.step, noteName(n.step, "treble", true)])).toEqual([
      [200, 10, "A5"], [320, 12, "C6"], [440, 11, "B5"], [560, -2, "C4"],
    ]);
  });
  it("names notes in bass clef and grand staff", () => {
    expect(noteName(-2, "bass", true)).toBe("E2");
    expect(noteName(10, "bass", true)).toBe("C4");
    expect(noteName(0, "treble")).toBe("E");
    expect(clefsFor(4, "grand")).toEqual(["treble", "bass", "treble", "bass"]);
  });
});

import { integral, pairedStaves, placeLabel } from "./omr";
describe("label quality", () => {
  function blank(w = 700, h = 420) {
    const dark = new Uint8Array(w * h);
    const rect = (x0: number, y0: number, x1: number, y1: number) => {
      for (let y = Math.round(y0); y <= Math.round(y1); y++) for (let x = Math.round(x0); x <= Math.round(x1); x++) dark[y * w + x] = 1;
    };
    const ellipse = (cx: number, cy: number, rx: number, ry: number) => {
      for (let y = -ry; y <= ry; y++) for (let x = -rx; x <= rx; x++) if ((x * x) / (rx * rx) + (y * y) / (ry * ry) <= 1) dark[Math.round(cy + y) * w + Math.round(cx + x)] = 1;
    };
    return { w, h, dark, rect, ellipse };
  }
  it("gives a filled note just beyond a ledger line one name, not two", () => {
    const b = blank();
    const s = 14, top = 100;
    for (let i = 0; i < 5; i++) b.rect(40, top + i * s, 660, top + i * s + 1);
    const bottom = top + 4 * s + 0.5, yOf = (k: number) => bottom - (k * s) / 2;
    b.rect(200 - 13, yOf(10) - 0.5, 200 + 13, yOf(10) + 0.5); // first ledger above
    b.ellipse(200, yOf(11), 8, 6); // B5, filled, in the space above the ledger
    b.rect(200 - 8, yOf(11), 200 - 7, yOf(11) + 3 * s); // stem down
    const img = { w: b.w, h: b.h, dark: b.dark };
    const notes = findLedgerNotes(img, findStaves(img));
    expect(notes.map((n) => noteName(n.step, "treble"))).toEqual(["B"]);
  });
  it("recognises a piano grand staff by the line joining the staves", () => {
    const b = blank();
    const s = 12;
    for (let i = 0; i < 5; i++) b.rect(40, 60 + i * s, 660, 61 + i * s);
    for (let i = 0; i < 5; i++) b.rect(40, 200 + i * s, 660, 201 + i * s);
    b.rect(40, 60, 41, 200 + 4 * s + 1); // system line joining both staves
    const img = { w: b.w, h: b.h, dark: b.dark };
    const staves = findStaves(img);
    const paired = pairedStaves(img, staves);
    expect(paired).toEqual([true, true]);
    expect(clefsFor(2, "auto", paired)).toEqual(["treble", "bass"]);
    expect(clefsFor(1, "auto", [false])).toEqual(["treble"]);
  });
  it("moves a label off ink and off other labels", () => {
    const b = blank();
    const s = 14;
    b.rect(180, 60, 220, 92); // something (lyrics, a beam) right above the note
    const img = { w: b.w, h: b.h, dark: b.dark };
    const ii = integral(img);
    const note = { x: 200, y: 110, step: 11, staffIndex: 0, score: 1 };
    const placed: { x0: number; y0: number; x1: number; y1: number }[] = [];
    const first = placeLabel(ii, img, note, s, "B", placed);
    expect(first.y0).toBeGreaterThan(note.y); // went below instead of onto the ink above
    const second = placeLabel(ii, img, { ...note, x: 202 }, s, "C", placed);
    expect(second.x0 === first.x0 && second.y0 === first.y0).toBe(false); // not stacked on the first
  });
});
