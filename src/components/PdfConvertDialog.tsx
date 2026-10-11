import type { PDFDocumentProxy } from "pdfjs-dist";
import { useEffect, useMemo, useState } from "react";
import { db, patchRow, type Song } from "../lib/db";
import { guessKey } from "../lib/music/chords";
import { allChords, parseChordPro } from "../lib/music/chordpro";
import { pdfToChart, readPdfLayout, type PdfChartResult } from "../lib/pdfChart";
import { ChartView } from "./ChartView";

const KIND_TEXT: Record<PdfChartResult["kind"], string> = {
  chords: "Chord sheet — chords and lyrics read straight from the PDF.",
  tab: "Tab score — lyrics and sections read from the PDF; chords worked out from the tab.",
  lyrics: "Sheet music — lyrics and sections only (there are no chords or tab to read).",
};

/**
 * Converts the song's PDF into an editable chart: shows the result next to a live preview so it
 * can be fixed up, then saves it as the song's chart. The PDF stays attached.
 */
export function PdfConvertDialog({ song, doc, onClose }: { song: Song; doc: PDFDocumentProxy; onClose: () => void }) {
  const [result, setResult] = useState<PdfChartResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [body, setBody] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { OPS } = await import("pdfjs-dist");
        const layout = await readPdfLayout(doc as never, OPS as unknown as Record<string, number>);
        const r = pdfToChart(layout);
        if (cancelled) return;
        setResult(r);
        setBody(r.body);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => { cancelled = true; };
  }, [doc]);

  const parsed = useMemo(() => parseChordPro(body), [body]);
  const key = useMemo(() => guessKey(allChords(parsed)), [parsed]);

  const save = async () => {
    if (song.content.trim() && !confirm("Replace this song's current chart with the converted one?")) return;
    setSaving(true);
    await patchRow(db.songs, song.id, {
      content: body,
      song_key: song.song_key || key || null,
      capo: song.capo || result?.meta.capo || 0,
      tempo: song.tempo || result?.meta.tempo || null,
    });
    setSaving(false);
    onClose();
  };

  return (
    <div className="convert-overlay" role="dialog" aria-label="Make an editable chart from this PDF">
      <div className="convert-head">
        <strong className="grow">Make editable chart from PDF</strong>
        <button className="btn small ghost" onClick={onClose}>Cancel</button>
        <button className="btn small primary" onClick={save} disabled={!result || !body.trim() || saving}>Save as chart</button>
      </div>
      {error && <div className="card" style={{ margin: 12, color: "var(--danger)" }}>Couldn't read this PDF: {error}</div>}
      {!result && !error && <div className="empty-state">Reading the PDF…</div>}
      {result && (
        <>
          <div className="convert-notes small">
            <div><strong>{KIND_TEXT[result.kind]}</strong>{key ? ` Key looks like ${key}.` : ""}{result.meta.capo ? ` Capo ${result.meta.capo}.` : ""}{result.meta.tempo ? ` ${result.meta.tempo} bpm.` : ""}</div>
            {result.warnings.map((w) => <div key={w} className="dim">• {w}</div>)}
            <div className="dim">Fix anything below — chords go in [brackets] right before the syllable. The PDF stays attached; switch with the PDF button.</div>
          </div>
          <div className="convert-body">
            <textarea className="input mono convert-text" value={body} onChange={(e) => setBody(e.target.value)} spellCheck={false} aria-label="Chart text" />
            <div className="convert-preview">
              <ChartView sections={parsed.sections} songKey={key} transpose={0} capo={0} showChords nashville={false} columns={1} fontScale={0.85} activePos={null} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
