import { useEffect, useRef, useState } from "react";

/** Renders every page of a PDF to canvases (iOS Safari only shows page 1 in an iframe). */
export function PdfView({ blob }: { blob: Blob }) {
  const host = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const el = host.current;
    if (!el) return;
    el.innerHTML = "";
    (async () => {
      try {
        const pdfjs = await import("pdfjs-dist");
        const workerUrl = (await import("pdfjs-dist/build/pdf.worker.min.mjs?url")).default;
        pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
        const doc = await pdfjs.getDocument({ data: new Uint8Array(await blob.arrayBuffer()) }).promise;
        const width = el.clientWidth || 800;
        const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
        for (let i = 1; i <= doc.numPages; i++) {
          if (cancelled) return;
          const page = await doc.getPage(i);
          const base = page.getViewport({ scale: 1 });
          const viewport = page.getViewport({ scale: (width / base.width) * dpr });
          const canvas = document.createElement("canvas");
          canvas.width = viewport.width;
          canvas.height = viewport.height;
          canvas.style.width = "100%";
          canvas.style.display = "block";
          canvas.style.marginBottom = "12px";
          canvas.style.background = "#fff";
          canvas.style.borderRadius = "4px";
          el.appendChild(canvas);
          await page.render({ canvas, viewport }).promise;
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [blob]);

  return (
    <div>
      {error && <div className="card dim">Couldn't open this PDF: {error}</div>}
      <div ref={host} />
    </div>
  );
}
