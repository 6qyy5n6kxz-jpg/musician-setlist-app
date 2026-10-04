// Notes drawn over PDF pages. Coordinates are fractions of the page (0..1) so marks stay in place
// at any zoom, screen size or orientation. Stored on song_files.annotations.

export type InkMark = {
  t: "ink";
  color: string;
  /** Stroke width as a fraction of page width. */
  w: number;
  /** Highlighter: drawn translucent and underneath text visually. */
  hl?: boolean;
  /** Flattened x,y pairs (fractions of page width/height). */
  pts: number[];
};

export type TextMark = {
  t: "text";
  color: string;
  x: number;
  y: number;
  /** Font size as a fraction of page width. */
  size: number;
  text: string;
};

export type Mark = InkMark | TextMark;

export interface PdfAnnotations {
  /** Page number (1-based) -> marks. */
  pages?: Record<string, Mark[]>;
}

export const PEN_COLORS = ["#e53935", "#1e63d6", "#111111", "#2e7d32"];
export const HIGHLIGHT_COLOR = "#ffe234";

/** Drop points closer together than `eps` (fraction of page) to keep strokes small. */
export function simplify(pts: number[], eps = 0.0015): number[] {
  if (pts.length <= 4) return pts;
  const out = [pts[0], pts[1]];
  for (let i = 2; i < pts.length - 2; i += 2) {
    const dx = pts[i] - out[out.length - 2];
    const dy = pts[i + 1] - out[out.length - 1];
    if (dx * dx + dy * dy >= eps * eps) out.push(pts[i], pts[i + 1]);
  }
  out.push(pts[pts.length - 2], pts[pts.length - 1]);
  return out.map((n) => Math.round(n * 10000) / 10000);
}

function distToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  const t = len2 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0;
  const x = ax + t * dx, y = ay + t * dy;
  return Math.hypot(px - x, py - y);
}

/**
 * Remove marks under the eraser at (x, y). `aspect` = page height / width, so distances are
 * measured in real proportions rather than squashed fractions.
 */
export function eraseAt(marks: Mark[], x: number, y: number, radius: number, aspect: number): Mark[] {
  return marks.filter((m) => {
    if (m.t === "text") {
      const width = m.size * 0.55 * m.text.length;
      const height = (m.size * 1.3) / aspect;
      return !(x >= m.x - radius && x <= m.x + width + radius && y >= m.y - height - radius / aspect && y <= m.y + radius / aspect);
    }
    const p = m.pts;
    if (p.length === 2) return Math.hypot(x - p[0], (y - p[1]) * aspect) > radius + m.w;
    for (let i = 0; i < p.length - 2; i += 2) {
      if (distToSegment(x, y * aspect, p[i], p[i + 1] * aspect, p[i + 2], p[i + 3] * aspect) <= radius + m.w / 2) return false;
    }
    return true;
  });
}

/** Draw marks onto a canvas of size w x h (device pixels). */
export function drawMarks(ctx: CanvasRenderingContext2D, marks: Mark[], w: number, h: number) {
  ctx.clearRect(0, 0, w, h);
  // Highlighter first so pen ink and text sit on top of it
  const ordered = [...marks.filter((m) => m.t === "ink" && m.hl), ...marks.filter((m) => !(m.t === "ink" && m.hl))];
  for (const m of ordered) {
    if (m.t === "ink") {
      ctx.save();
      ctx.strokeStyle = m.color;
      ctx.lineWidth = Math.max(1, m.w * w);
      ctx.lineCap = m.hl ? "butt" : "round";
      ctx.lineJoin = "round";
      ctx.globalAlpha = m.hl ? 0.35 : 1;
      if (m.hl) ctx.globalCompositeOperation = "multiply";
      ctx.beginPath();
      ctx.moveTo(m.pts[0] * w, m.pts[1] * h);
      if (m.pts.length === 2) ctx.lineTo(m.pts[0] * w + 0.1, m.pts[1] * h);
      for (let i = 2; i < m.pts.length; i += 2) ctx.lineTo(m.pts[i] * w, m.pts[i + 1] * h);
      ctx.stroke();
      ctx.restore();
    } else {
      ctx.save();
      const size = Math.max(10, m.size * w);
      ctx.font = `600 ${size}px -apple-system, "Helvetica Neue", Arial, sans-serif`;
      ctx.fillStyle = m.color;
      // White halo keeps notes readable over printed notation
      ctx.lineWidth = Math.max(2, size * 0.18);
      ctx.strokeStyle = "rgba(255,255,255,0.9)";
      ctx.lineJoin = "round";
      ctx.strokeText(m.text, m.x * w, m.y * h);
      ctx.fillText(m.text, m.x * w, m.y * h);
      ctx.restore();
    }
  }
}

export function pageMarks(a: PdfAnnotations | null | undefined, page: number): Mark[] {
  return a?.pages?.[String(page)] ?? [];
}

export function withPageMarks(a: PdfAnnotations | null | undefined, page: number, marks: Mark[]): PdfAnnotations {
  const pages = { ...(a?.pages ?? {}) };
  if (marks.length) pages[String(page)] = marks;
  else delete pages[String(page)];
  return { ...(a ?? {}), pages };
}

export function countMarks(a: PdfAnnotations | null | undefined): number {
  return Object.values(a?.pages ?? {}).reduce((n, m) => n + m.length, 0);
}
