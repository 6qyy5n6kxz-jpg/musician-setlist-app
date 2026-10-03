import { useLiveQuery } from "dexie-react-hooks";
import { db, live, type Setlist, type SetlistItem, type Song } from "./db";

export function useSongs(): Song[] | undefined {
  return useLiveQuery(async () => live(await db.songs.toArray()).sort(bySongTitle), []);
}

export function useSong(id: string | undefined): Song | null | undefined {
  return useLiveQuery(async () => (id ? (await db.songs.get(id)) ?? null : null), [id]);
}

export function useSetlists(): Setlist[] | undefined {
  return useLiveQuery(async () =>
    live(await db.setlists.toArray()).sort((a, b) =>
      (b.event_date ?? "").localeCompare(a.event_date ?? "") || b.updated_at.localeCompare(a.updated_at),
    ), []);
}

export function useSetlistItems(setlistId: string | undefined): SetlistItem[] | undefined {
  return useLiveQuery(async () => {
    if (!setlistId) return [];
    const rows = await db.setlist_items.where("setlist_id").equals(setlistId).toArray();
    return live(rows).sort((a, b) => a.position - b.position);
  }, [setlistId]);
}

export function useSongFiles(songId: string | undefined) {
  return useLiveQuery(async () => {
    if (!songId) return [];
    return live(await db.song_files.where("song_id").equals(songId).toArray());
  }, [songId]);
}

export function useProfile() {
  return useLiveQuery(() => db.profile.toCollection().first(), []);
}

/** Library order: ignore a leading "The"/"A" like a record store. */
export function sortTitle(t: string): string {
  return t.trim().toLowerCase().replace(/^(the|a|an)\s+/, "");
}

export function bySongTitle(a: Song, b: Song): number {
  return sortTitle(a.title).localeCompare(sortTitle(b.title));
}
