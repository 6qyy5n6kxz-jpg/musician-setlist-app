import type { PDFDocumentProxy } from "pdfjs-dist";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  countMarks, drawMarks, eraseAt, HIGHLIGHT_COLOR, pageMarks, PEN_COLORS, simplify, withPageMarks,
  type Mark, type PdfAnnotations,
} from "../lib/annotations";

type Tool = { kind: "pen"; color: string } | { kind: "highlight" } | { kind: "text"; color: string } | { kind: "erase" };

interface Props {
  blob: Blob;
  annotations?: PdfAnnotations;
  /** When provided, the Annotate toolbar is available and changes are reported here. */
  onAnnotationsChange?: (a: PdfAnnotations) => void;
}

/**
 * Renders every page of a PDF to canvases (iOS Safari only shows page 1 in an iframe), with a
 * notes layer on top: pen, highlighter, text and eraser. The PDF file itself is never modified.
 */
export function PdfView({ blob, annotations, onAnnotationsChange }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const [doc, setDoc] = useState<PDFDocumentProxy | null>(null);
  const [pages, setPages] = useState<{ num: number; ratio: number }[]>([]);
  const [width, setWidth] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [annotating, setAnnotating] = useState(false);
  const [tool, setTool] = useState<Tool>({ kind: "pen", color: PEN_COLORS[0] });
  const [history, setHistory] = useState<PdfAnnotations[]>([]);
  // Once an Apple Pencil is used, fingers scroll and only the Pencil draws (palm rejection).
  const [pencilSeen, setPencilSeen] = useState(false);
  const current = useRef<PdfAnnotations>(annotations ?? {});
  current.current = annotations ?? {};

  useEffect(() => {
    let cancelled = false;
    setDoc(null);
    setError(null);
    (async () => {
      try {
        const pdfjs = await import("pdfjs-dist");
        const workerUrl = (await import("pdfjs-dist/build/pdf.worker.min.mjs?url")).default;
        pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
        const d = await pdfjs.getDocument({ data: new Uint8Array(await blob.arrayBuffer()) }).promise;
        const info: { num: number; ratio: number }[] = [];
        for (let i = 1; i <= d.numPages; i++) {
          const vp = (await d.getPage(i)).getViewport({ scale: 1 });
          info.push({ num: i, ratio: vp.height / vp.width });
        }
        if (!cancelled) { setDoc(d); setPages(info); }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => { cancelled = true; };
  }, [blob]);

  // Re-render pages at the right resolution when the width changes (rotation, split view)
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(Math.round(el.clientWidth)));
    ro.observe(el);
    setWidth(Math.round(el.clientWidth));
    return () => ro.disconnect();
  }, []);

  const commit = useCallback((page: number, marks: Mark[]) => {
    if (!onAnnotationsChange) return;
    setHistory((h) => [...h.slice(-49), current.current]);
    onAnnotationsChange(withPageMarks(current.current, page, marks));
  }, [onAnnotationsChange]);

  const undo = () => {
    const prev = history[history.length - 1];
    if (!prev || !onAnnotationsChange) return;
    setHistory((h) => h.slice(0, -1));
    onAnnotationsChange(prev);
  };

  const total = countMarks(annotations);
  const [clefMode, setClefMode] = useState<"auto" | "treble" | "bass" | "grand">(() => {
    try { return (localStorage.getItem("ledger-clef-v2") as "auto" | "treble" | "bass" | "grand") || "auto"; } catch { return "auto"; }
  });
  const [labeling, setLabeling] = useState<string | null>(null);
  const autoCount = Object.values(annotations?.pages ?? {}).flat().filter((m) => m.t === "text" && m.auto).length;

  /** Reading help: find notes on ledger lines and add their names as labels. */
  const nameLedgerNotes = async () => {
    if (!doc || !onAnnotationsChange) return;
    try { localStorage.setItem("ledger-clef-v2", clefMode); } catch { /* ignore */ }
    setLabeling("Reading the music…");
    const { toDark, findStaves, findLedgerNotes, noteName, clefsFor, pairedStaves, placeLabel, integral, groupChords } = await import("../lib/omr");
    let next: PdfAnnotations = current.current;
    let labeled = 0, pagesWith = 0, staffCount = 0;
    for (const p of pages) {
      const page = await doc.getPage(p.num);
      const base = page.getViewport({ scale: 1 });
      const scale = 2000 / base.width; // enough resolution to measure staff lines reliably
      const viewport = page.getViewport({ scale });
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(viewport.width);
      canvas.height = Math.round(viewport.height);
      const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      // "print" intent renders without requestAnimationFrame, so it finishes even if the app is backgrounded
      await page.render({ canvas, viewport, intent: "print" }).promise;
      const img = toDark(ctx.getImageData(0, 0, canvas.width, canvas.height).data, canvas.width, canvas.height);
      const staves = findStaves(img);
      staffCount += staves.length;
      const clefs = clefsFor(staves.length, clefMode, pairedStaves(img, staves));
      const notes = findLedgerNotes(img, staves);
      const W = canvas.width, H = canvas.height;
      const ii = integral(img);
      const placed: { x0: number; y0: number; x1: number; y1: number }[] = [];
      // One column of names per chord (top to bottom, matching the notes), placed in the clearest spot
      const labels: Mark[] = [];
      for (const group of groupChords(notes, staves)) {
        const names = group.map((n) => noteName(n.step, clefs[n.staffIndex]));
        const box = placeLabel(ii, img, group, staves[group[0].staffIndex].space, names, placed);
        names.forEach((name, i) => labels.push({
          t: "text", auto: true, color: "#1e63d6", text: name, size: box.fontPx / W,
          x: (box.x0 + 1) / W,
          y: (box.y0 + box.fontPx * 0.78 + i * box.lineGap) / H,
        }));
      }
      const kept = pageMarks(next, p.num).filter((m) => !(m.t === "text" && m.auto));
      next = withPageMarks(next, p.num, [...kept, ...labels]);
      labeled += labels.length;
      if (labels.length) pagesWith++;
      canvas.width = canvas.height = 0; // free memory on iPad
    }
    setHistory((h) => [...h.slice(-49), current.current]);
    onAnnotationsChange(next);
    setLabeling(
      labeled
        ? `Named ${labeled} ledger note${labeled === 1 ? "" : "s"} on ${pagesWith} page${pagesWith === 1 ? "" : "s"}. Erase any that are wrong.`
        : staffCount
          ? "Found the staves but no notes on ledger lines."
          : "No music staves found — this works on computer-made sheet music (e.g. Ultimate Guitar Pro), not chord charts or photos.",
    );
  };

  const removeAuto = () => {
    if (!onAnnotationsChange) return;
    let next: PdfAnnotations = current.current;
    for (const p of pages) next = withPageMarks(next, p.num, pageMarks(next, p.num).filter((m) => !(m.t === "text" && m.auto)));
    setHistory((h) => [...h.slice(-49), current.current]);
    onAnnotationsChange(next);
    setLabeling(null);
  };
  const toolBtn = (active: boolean) => `btn small ${active ? "on" : ""}`;

  return (
    <div>
      {onAnnotationsChange && (
        <div className="pdf-toolbar no-print">
          {!annotating ? (
            <>
              <button className="btn small" onClick={() => setAnnotating(true)}>✎ Annotate{total ? ` (${total})` : ""}</button>
              <button className="btn small" onClick={nameLedgerNotes} disabled={!doc || labeling === "Reading the music…"} title="Label notes on ledger lines with their names">♪ Name ledger notes</button>
              <select className="select" style={{ width: "auto", minHeight: 34, padding: "0 8px" }} value={clefMode}
                onChange={(e) => setClefMode(e.target.value as "auto" | "treble" | "bass" | "grand")} aria-label="Clef">
                <option value="auto">Auto clef</option>
                <option value="treble">Treble</option>
                <option value="bass">Bass</option>
                <option value="grand">Grand staff</option>
              </select>
              {autoCount > 0 && <button className="btn small ghost" onClick={removeAuto}>Remove note labels</button>}
              {labeling && <span className="small dim" style={{ flexBasis: "100%" }}>{labeling}</span>}
            </>
          ) : (
            <>
              {PEN_COLORS.map((c) => (
                <button key={c} className={toolBtn(tool.kind === "pen" && tool.color === c)} onClick={() => setTool({ kind: "pen", color: c })} aria-label="Pen">
                  <span className="swatch" style={{ background: c }} />
                </button>
              ))}
              <button className={toolBtn(tool.kind === "highlight")} onClick={() => setTool({ kind: "highlight" })}>
                <span className="swatch" style={{ background: HIGHLIGHT_COLOR }} /> Highlight
              </button>
              <button className={toolBtn(tool.kind === "text")} onClick={() => setTool({ kind: "text", color: tool.kind === "pen" ? tool.color : PEN_COLORS[0] })}>T Text</button>
              <button className={toolBtn(tool.kind === "erase")} onClick={() => setTool({ kind: "erase" })}>Eraser</button>
              <button className="btn small" onClick={undo} disabled={!history.length}>Undo</button>
              {total > 0 && (
                <button className="btn small ghost danger" onClick={() => { if (confirm("Remove all notes from this PDF?")) { setHistory((h) => [...h, current.current]); onAnnotationsChange({}); } }}>Clear all</button>
              )}
              <span className="spacer" />
              <span className="small dim">{pencilSeen ? "Pencil draws · finger scrolls" : "Draw with a finger or Apple Pencil"}</span>
              <button className="btn small primary" onClick={() => setAnnotating(false)}>Done</button>
            </>
          )}
        </div>
      )}
      {error && <div className="card dim">Couldn't open this PDF: {error}</div>}
      <div ref={host}>
        {doc && width > 0 && pages.map((p) => (
          <PdfPage
            key={p.num}
            doc={doc}
            num={p.num}
            ratio={p.ratio}
            width={width}
            marks={pageMarks(annotations, p.num)}
            annotating={annotating}
            tool={tool}
            pencilSeen={pencilSeen}
            onPencil={() => setPencilSeen(true)}
            onMarks={(marks) => commit(p.num, marks)}
          />
        ))}
      </div>
    </div>
  );
}

function PdfPage({ doc, num, ratio, width, marks, annotating, tool, pencilSeen, onPencil, onMarks }: {
  doc: PDFDocumentProxy; num: number; ratio: number; width: number; marks: Mark[];
  annotating: boolean; tool: Tool; pencilSeen: boolean; onPencil: () => void; onMarks: (marks: Mark[]) => void;
}) {
  const pageCanvas = useRef<HTMLCanvasElement>(null);
  const inkCanvas = useRef<HTMLCanvasElement>(null);
  const drawing = useRef<{ id: number; pts: number[] } | null>(null);
  const erased = useRef<Mark[] | null>(null);
  const height = Math.round(width * ratio);
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);

  // Render the PDF page
  useEffect(() => {
    let cancelled = false;
    let task: { cancel: () => void } | null = null;
    (async () => {
      const page = await doc.getPage(num);
      const base = page.getViewport({ scale: 1 });
      const viewport = page.getViewport({ scale: (width / base.width) * dpr });
      const canvas = pageCanvas.current;
      if (!canvas || cancelled) return;
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      const t = page.render({ canvas, viewport });
      task = t;
      await t.promise.catch(() => {});
    })();
    return () => { cancelled = true; task?.cancel(); };
  }, [doc, num, width, dpr]);

  // Draw saved marks
  const redraw = useCallback((list: Mark[]) => {
    const c = inkCanvas.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    drawMarks(ctx, list, c.width, c.height);
  }, []);
  useEffect(() => {
    const c = inkCanvas.current;
    if (!c) return;
    c.width = Math.round(width * dpr);
    c.height = Math.round(height * dpr);
    redraw(marks);
  }, [width, height, dpr, marks, redraw]);

  const point = (e: React.PointerEvent) => {
    const r = inkCanvas.current!.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
  };
  const accepts = (e: React.PointerEvent) => !(pencilSeen && e.pointerType === "touch");

  const inkStyle = (): Omit<Extract<Mark, { t: "ink" }>, "pts"> =>
    tool.kind === "highlight" ? { t: "ink", color: HIGHLIGHT_COLOR, w: 0.022, hl: true } : { t: "ink", color: tool.kind === "pen" ? tool.color : "#111", w: 0.0035 };

  const onDown = (e: React.PointerEvent) => {
    if (!annotating) return;
    if (e.pointerType === "pen") onPencil();
    if (!accepts(e)) return;
    e.preventDefault();
    const [x, y] = point(e);
    if (tool.kind === "text") {
      const text = window.prompt("Note")?.trim();
      if (text) onMarks([...marks, { t: "text", color: tool.color, x, y, size: 0.028, text }]);
      return;
    }
    try {
      (e.target as Element).setPointerCapture(e.pointerId); // keep the stroke if the pen leaves the page
    } catch {
      /* not capturable (e.g. synthetic events) — drawing still works */
    }
    if (tool.kind === "erase") {
      erased.current = eraseAt(marks, x, y, 0.015, ratio);
      redraw(erased.current);
      return;
    }
    drawing.current = { id: e.pointerId, pts: [x, y] };
  };

  const onMove = (e: React.PointerEvent) => {
    if (!annotating || !accepts(e)) return;
    const [x, y] = point(e);
    if (tool.kind === "erase" && erased.current) {
      const next = eraseAt(erased.current, x, y, 0.015, ratio);
      if (next.length !== erased.current.length) { erased.current = next; redraw(next); }
      return;
    }
    const d = drawing.current;
    if (!d || d.id !== e.pointerId) return;
    d.pts.push(x, y);
    // Draw the stroke as it grows (saved marks + the live stroke)
    const c = inkCanvas.current!;
    drawMarks(c.getContext("2d")!, [...marks, { ...inkStyle(), pts: d.pts }], c.width, c.height);
  };

  const onUp = (e: React.PointerEvent) => {
    if (tool.kind === "erase" && erased.current) {
      if (erased.current.length !== marks.length) onMarks(erased.current);
      erased.current = null;
      return;
    }
    const d = drawing.current;
    if (!d || d.id !== e.pointerId) return;
    drawing.current = null;
    onMarks([...marks, { ...inkStyle(), pts: simplify(d.pts) }]);
  };

  return (
    <div className="pdf-page" style={{ width, height }}>
      <canvas ref={pageCanvas} style={{ width, height }} />
      <canvas
        ref={inkCanvas}
        className="pdf-ink"
        data-no-swipe={annotating ? "" : undefined}
        style={{
          width, height,
          pointerEvents: annotating ? "auto" : "none",
          touchAction: annotating ? (pencilSeen ? "pan-x pan-y pinch-zoom" : "none") : "auto",
          cursor: annotating ? (tool.kind === "text" ? "text" : "crosshair") : "default",
        }}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
      />
    </div>
  );
}
