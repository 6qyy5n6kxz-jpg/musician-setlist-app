import { useRef } from "react";

/** Library order the song page swipes through (the list as last filtered/sorted on the Songs screen). */
const ORDER_KEY = "lib-order";

export function rememberSongOrder(ids: string[]) {
  try { sessionStorage.setItem(ORDER_KEY, JSON.stringify(ids)); } catch { /* private mode */ }
}

export function songOrder(): string[] | null {
  try {
    const v = JSON.parse(sessionStorage.getItem(ORDER_KEY) ?? "null");
    return Array.isArray(v) && v.length ? v : null;
  } catch {
    return null;
  }
}

/** Neighbours of `id` in `order`; null at the ends. */
export function neighbours(order: string[], id: string): { prev: string | null; next: string | null; index: number } {
  const index = order.indexOf(id);
  if (index < 0) return { prev: null, next: null, index };
  return { prev: order[index - 1] ?? null, next: order[index + 1] ?? null, index };
}

export type SwipeDir = "left" | "right";

/** True when a finished touch counts as a deliberate sideways swipe. */
export function isSwipe(dx: number, dy: number, ms: number): SwipeDir | null {
  if (ms > 800 || Math.abs(dx) < 70 || Math.abs(dx) < Math.abs(dy) * 1.8) return null;
  return dx < 0 ? "left" : "right";
}

function blocksSwipe(target: EventTarget | null): boolean {
  for (let el = target as HTMLElement | null; el && el !== document.body; el = el.parentElement) {
    if (el.dataset?.noSwipe !== undefined) return true;
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName) || el.isContentEditable) return true;
  }
  return false;
}

/** Ancestors that can scroll sideways, with their current position. */
function sideScrollers(target: EventTarget | null): [HTMLElement, number][] {
  const out: [HTMLElement, number][] = [];
  for (let el = target as HTMLElement | null; el && el !== document.body; el = el.parentElement) {
    if (el.scrollWidth > el.clientWidth + 2) out.push([el, el.scrollLeft]);
  }
  return out;
}

/**
 * Touch handlers for swiping left (next) / right (previous). Ignores pinches, pinch-zoomed
 * pages, form fields, touches that scrolled content sideways and anything marked `data-no-swipe`
 * (e.g. the PDF canvas while annotating).
 */
export function useSwipe(onSwipe: (dir: SwipeDir) => void) {
  const start = useRef<{ x: number; y: number; t: number; scrollers: [HTMLElement, number][] } | null>(null);
  return {
    onTouchStart: (e: React.TouchEvent) => {
      const zoomed = (window.visualViewport?.scale ?? 1) > 1.05;
      start.current = e.touches.length === 1 && !zoomed && !blocksSwipe(e.target)
        ? { x: e.touches[0].clientX, y: e.touches[0].clientY, t: Date.now(), scrollers: sideScrollers(e.target) }
        : null;
    },
    onTouchMove: (e: React.TouchEvent) => { if (e.touches.length > 1) start.current = null; },
    onTouchEnd: (e: React.TouchEvent) => {
      const s = start.current;
      start.current = null;
      const t = e.changedTouches[0];
      if (!s || !t) return;
      // The finger scrolled something sideways (wide tab line, zoomed PDF) — that wasn't a swipe
      if (s.scrollers.some(([el, left]) => Math.abs(el.scrollLeft - left) > 4)) return;
      const dir = isSwipe(t.clientX - s.x, t.clientY - s.y, Date.now() - s.t);
      if (dir) onSwipe(dir);
    },
  };
}
