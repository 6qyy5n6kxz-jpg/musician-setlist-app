// Unpack a zip of charts (OnSong/ChordPro library exports, a folder of PDFs) into importable files.
import { unzipSync } from "fflate";

const CHART_FILE = /\.(cho|chopro|chordpro|pro|crd|onsong|txt|pdf|csv)$/i;
// macOS/ExFAT metadata and hidden files
const JUNK = /(^|\/)(\._|__MACOSX\/|\.DS_Store)/;

export interface ZipEntry {
  name: string;
  data: Uint8Array;
}

export function chartEntriesFromZip(buffer: ArrayBuffer): ZipEntry[] {
  const entries = unzipSync(new Uint8Array(buffer), {
    filter: (f) => !JUNK.test(f.name) && CHART_FILE.test(f.name) && f.originalSize > 0,
  });
  return Object.entries(entries)
    .map(([path, data]) => ({ name: path.split("/").pop() ?? path, data }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

/** Zips become their chart files; other files pass through. */
export async function expandZips(files: File[]): Promise<File[]> {
  const out: File[] = [];
  for (const f of files) {
    if (!/\.zip$/i.test(f.name)) {
      out.push(f);
      continue;
    }
    for (const e of chartEntriesFromZip(await f.arrayBuffer())) {
      const copy = new Uint8Array(e.data); // own the bytes (fflate may share the buffer)
      out.push(new File([copy], e.name, { type: /\.pdf$/i.test(e.name) ? "application/pdf" : "text/plain" }));
    }
  }
  return out;
}
