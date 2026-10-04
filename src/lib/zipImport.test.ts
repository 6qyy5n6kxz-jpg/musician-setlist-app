import { strToU8, zipSync } from "fflate";
import { describe, expect, it } from "vitest";
import { chartEntriesFromZip } from "./zipImport";

describe("zip import", () => {
  it("keeps chart files from nested folders and skips macOS junk", () => {
    const zip = zipSync({
      "Library/Wonderwall.onsong": strToU8("Wonderwall\nOasis\nKey: F#m\n\nVerse 1:\n[Em]Today"),
      "Library/Jolene.chopro": strToU8("{title: Jolene}\n[Am]Jolene"),
      "Library/Charts/Piano Man.pdf": new Uint8Array([37, 80, 68, 70]),
      "__MACOSX/Library/._Jolene.chopro": strToU8("junk"),
      "Library/._Wonderwall.onsong": strToU8("junk"),
      "Library/OnSong.sqlite3": new Uint8Array([1, 2, 3]),
      "Library/notes.docx": strToU8("x"),
    });
    const names = chartEntriesFromZip(zip.buffer as ArrayBuffer).map((e) => e.name);
    expect(names).toEqual(["Jolene.chopro", "Piano Man.pdf", "Wonderwall.onsong"]);
  });
});
