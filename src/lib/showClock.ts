// Show clock: are we on pace to finish the set list by the stop time?

export interface ClockItem {
  id: string;
  kind: "song" | "break";
  title: string;
  /** Expected length in seconds (songs), ignored for breaks. */
  seconds: number;
  /** Requested tonight / locked in — never suggested as a cut. */
  keep?: boolean;
}

export interface ClockStatus {
  /** When the remaining list would finish at this pace. */
  projectedEnd: Date;
  /** Hard stop as a Date (today, or after midnight). */
  stop: Date;
  /** Seconds until the stop time. */
  secondsLeft: number;
  /** Positive = time to spare, negative = running over (seconds). */
  slack: number;
  /** Songs to skip to finish on time (when running over). */
  cuts: ClockItem[];
  /** Roughly how many more songs fit (when there's time to spare). */
  roomForSongs: number;
}

/** "23:00" on the gig day (rolls past midnight when the stop is earlier than the start, e.g. 00:30). */
export function stopDate(now: Date, end: string, start?: string | null): Date {
  const [h, m] = end.split(":").map(Number);
  const d = new Date(now);
  d.setHours(h, m, 0, 0);
  const [sh] = (start ?? "").split(":").map(Number);
  const startsBefore = Number.isFinite(sh) && start ? sh > h : false;
  // A stop that's already hours behind us (or earlier than the start) means after midnight
  if (startsBefore && now.getHours() >= sh - 1) d.setDate(d.getDate() + (d < now ? 1 : 0));
  else if (d.getTime() < now.getTime() - 3 * 3600_000) d.setDate(d.getDate() + 1);
  return d;
}

/**
 * Where the show stands. `remaining` starts with the song on screen; `currentElapsed` is how long
 * it has been playing. Breaks still to come count `breakMinutes` each.
 */
export function showClock(now: Date, end: string, remaining: ClockItem[], currentElapsed: number, breakMinutes = 15, start?: string | null): ClockStatus {
  const stop = stopDate(now, end, start);
  let seconds = 0;
  remaining.forEach((it, i) => {
    if (it.kind === "break") seconds += breakMinutes * 60;
    else seconds += i === 0 ? Math.max(30, it.seconds - currentElapsed) : it.seconds;
  });
  const projectedEnd = new Date(now.getTime() + seconds * 1000);
  const secondsLeft = Math.round((stop.getTime() - now.getTime()) / 1000);
  const slack = Math.round((stop.getTime() - projectedEnd.getTime()) / 1000);

  const songs = remaining.filter((r) => r.kind === "song");
  const cuts: ClockItem[] = [];
  if (slack < -60) {
    // Never the song on screen or the closer; skip unrequested songs, longest first, until it fits
    const closer = songs[songs.length - 1]?.id;
    const candidates = songs.slice(1).filter((s) => s.id !== closer && !s.keep).sort((a, b) => b.seconds - a.seconds);
    let over = -slack;
    for (const c of candidates) {
      if (over <= 30) break;
      cuts.push(c);
      over -= c.seconds;
    }
  }
  const avg = songs.length ? songs.reduce((a, s) => a + s.seconds, 0) / songs.length : 210;
  return { projectedEnd, stop, secondsLeft, slack, cuts, roomForSongs: slack > 0 ? Math.floor(slack / Math.max(120, avg)) : 0 };
}

export function formatClock(sec: number): string {
  const s = Math.abs(Math.round(sec));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}:${String(m).padStart(2, "0")}` : `${m} min`;
}
