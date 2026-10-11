import { describe, expect, it } from "vitest";
import { nameChord, pdfToChart, type PdfLayout, type TextItem } from "./pdfChart";

const item = (x: number, y: number, str: string, extra: Partial<TextItem> = {}): TextItem =>
  ({ page: 1, x, y, w: str.length * 6, h: 10, str, font: "f1", ...extra });

describe("PDF to chart", () => {
  it("names chords from notes", () => {
    expect(nameChord([7, 11, 2], 7)).toBe("G");
    expect(nameChord([9, 0, 4], 9)).toBe("Am");
    expect(nameChord([2, 6, 9, 0], 2)).toBe("D7");
    expect(nameChord([0, 4, 7, 2], 0)).toBe("Cadd9");
    expect(nameChord([7, 11, 2], 11)).toBe("G/B");
    expect(nameChord([10, 2, 5], 10)).toBe("Bb");
    expect(nameChord([4, 11], 4)).toBe("E5");
    expect(nameChord([4], 4)).toBeNull();
  });

  it("reads a chord sheet (chords above lyrics)", () => {
    const layout: PdfLayout = {
      pages: [{ width: 600, height: 800 }],
      vlines: [],
      items: [
        item(40, 40, "Free Fallin'", { h: 20, w: 140 }),
        item(40, 70, "Tom Petty", { h: 14, w: 80 }),
        item(40, 120, "[Verse]"),
        item(40, 140, "F"), item(100, 140, "Bb"), item(160, 140, "F"), item(220, 140, "C"),
        item(40, 152, "She's a good girl, loves her mama", { w: 33 * 6 }),
        item(40, 760, "Page 1/1"),
      ],
    };
    const r = pdfToChart(layout);
    expect(r.kind).toBe("chords");
    expect(r.body).toMatch(/\[F\]She's/);
    expect(r.body).toMatch(/\[Bb\]/);
    expect(r.body).not.toMatch(/Page 1/);
  });

  it("works out chords from a tab staff and keeps the lyrics", () => {
    // One system: lyrics above a 6-line staff (top 100, bottom 150, lines every 10), two bars
    const fret = (x: number, string: number, n: string) => item(x, 100 + string * 10 + 3.5, n, { w: 6 });
    const staff = (y0: number, y1: number, xs: number[]) => xs.map((x) => ({ page: 1, x, y0, y1 }));
    const layout: PdfLayout = {
      pages: [{ width: 600, height: 800 }],
      items: [
        item(40, 20, "Song", { h: 24 }), item(40, 40, "Tuning : E A D G B E", { h: 9 }),
        item(40, 60, "Verse 1", { font: "label" }),
        item(60, 85, "Hello"), item(220, 85, "world"),
        // Bar 1: G chord 320003 (strings: 0 = high e … 5 = low E)
        fret(60, 5, "3"), fret(66, 4, "2"), fret(72, 3, "0"), fret(78, 2, "0"), fret(84, 1, "0"), fret(90, 0, "3"),
        // Bar 2: C chord x32010
        fret(220, 4, "3"), fret(226, 3, "2"), fret(232, 2, "0"), fret(238, 1, "1"), fret(244, 0, "0"),
        // Second system so it's recognised as a score
        fret(60, 5, "0").str ? { ...fret(60, 5, "0"), y: 300 + 50 + 3.5 } : fret(60, 5, "0"),
        { ...fret(66, 4, "2"), y: 300 + 40 + 3.5 }, { ...fret(72, 3, "2"), y: 300 + 30 + 3.5 },
      ],
      vlines: [...staff(100, 150, [50, 200, 400]), ...staff(300, 350, [50, 400])],
    };
    const r = pdfToChart(layout);
    expect(r.kind).toBe("tab");
    expect(r.body).toMatch(/\[G\]Hello/);
    expect(r.body).toMatch(/\[C\]world/);
    expect(r.body).toMatch(/start_of_verse/);
  });
});
