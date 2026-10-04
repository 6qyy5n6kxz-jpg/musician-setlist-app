// Gig log: Perform mode records each song that was actually played, per gig (setlist + date).
import { db, live, newId, nowIso, saveRow, type Gig, type Setlist } from "./db";

export const todayIso = () => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
};

/** The gig for this setlist today, created on first use. */
export async function ensureGig(setlist: Setlist, actId: string | null): Promise<Gig> {
  const today = todayIso();
  const existing = live(await db.gigs.where("setlist_id").equals(setlist.id).toArray()).find((g) => g.gig_date === today);
  if (existing) return existing;
  const t = nowIso();
  return saveRow(db.gigs, {
    id: newId(), created_at: t, updated_at: t, deleted_at: null, dirty: 1,
    setlist_id: setlist.id, act_id: setlist.act_id ?? actId, name: setlist.name,
    venue: setlist.venue, gig_date: today, notes: null,
  });
}

/** Record a played song once per gig. */
export async function logPlayed(gigId: string, songId: string, fromRequest: boolean) {
  const already = live(await db.gig_songs.where("gig_id").equals(gigId).toArray()).some((g) => g.song_id === songId);
  if (already) return;
  const t = nowIso();
  await saveRow(db.gig_songs, {
    id: newId(), created_at: t, updated_at: t, deleted_at: null, dirty: 1,
    gig_id: gigId, song_id: songId, played_at: t, from_request: fromRequest,
  });
}

export interface PlayStats {
  count: number;
  requests: number;
  last?: { date: string; venue: string | null };
}

/** Times each song was played (and requested), with the most recent gig. */
export async function playStats(): Promise<Map<string, PlayStats>> {
  const gigs = new Map(live(await db.gigs.toArray()).map((g) => [g.id, g]));
  const out = new Map<string, PlayStats>();
  for (const gs of live(await db.gig_songs.toArray())) {
    const gig = gigs.get(gs.gig_id);
    if (!gig || !gs.song_id) continue;
    const s = out.get(gs.song_id) ?? { count: 0, requests: 0 };
    s.count++;
    if (gs.from_request) s.requests++;
    if (!s.last || gig.gig_date > s.last.date) s.last = { date: gig.gig_date, venue: gig.venue };
    out.set(gs.song_id, s);
  }
  return out;
}

/** Songs played at a venue before (most recent date per song), for "played here last time" warnings. */
export async function playedAtVenue(venue: string, beforeDate = todayIso()): Promise<Map<string, string>> {
  const v = venue.trim().toLowerCase();
  const out = new Map<string, string>();
  if (!v) return out;
  const gigs = live(await db.gigs.toArray()).filter((g) => (g.venue ?? "").trim().toLowerCase() === v && g.gig_date < beforeDate);
  const byId = new Map(gigs.map((g) => [g.id, g]));
  for (const gs of live(await db.gig_songs.toArray())) {
    const gig = byId.get(gs.gig_id);
    if (!gig || !gs.song_id) continue;
    const prev = out.get(gs.song_id);
    if (!prev || gig.gig_date > prev) out.set(gs.song_id, gig.gig_date);
  }
  return out;
}

export function daysAgo(date: string): number {
  return Math.round((Date.parse(todayIso()) - Date.parse(date)) / 86_400_000);
}
