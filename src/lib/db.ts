// Local database (IndexedDB via Dexie). The app always reads and writes here first;
// sync.ts mirrors it to Supabase whenever there's a connection.
import Dexie, { type Table } from "dexie";
import type { GearLibrary, Instrument, LeadVocal, SongGear } from "./gear";

export interface SyncFields {
  id: string;
  created_at: string;
  updated_at: string;
  deleted_at: string | null;
  /** 1 = changed locally and not yet pushed. */
  dirty: 0 | 1;
}

export interface Song extends SyncFields {
  title: string;
  artist: string;
  song_key: string | null;
  tempo: number | null;
  time_signature: string | null;
  duration_sec: number | null;
  capo: number;
  tags: string[];
  genre: string | null;
  year: number | null;
  ccli: string | null;
  content: string;
  notes: string | null;
  flow: string | null;
  requestable: boolean;
  /** Available on the karaoke sign-up list. */
  karaoke: boolean;
  /** Section start times recorded against the song's backing track. */
  timings: SongTimings | null;
  /** The one instrument Devin plays on this song. */
  instrument: Instrument | null;
  /** Who sings lead. */
  lead_vocal: LeadVocal | null;
  /** Numa X / Nano Cortex / BeatBuddy settings for this song. */
  gear: SongGear;
  /** Key to perform in when Kendra / Devin sings lead (null = the written key). */
  key_kendra: string | null;
  key_devin: string | null;
}

export interface Gig extends SyncFields {
  setlist_id: string | null;
  act_id: string | null;
  name: string;
  venue: string | null;
  gig_date: string;
  notes: string | null;
}

export interface GigSong extends SyncFields {
  gig_id: string;
  song_id: string | null;
  played_at: string;
  from_request: boolean;
}

export interface SongTimings {
  /** The flow string the marks were recorded with (null = sections as written). */
  flow: string | null;
  /** Section position (in the arranged list) and the track time it starts at, in seconds. */
  marks: { pos: number; t: number }[];
}

export type FileKind = "pdf" | "audio" | "image";

export interface SongFile extends SyncFields {
  song_id: string;
  kind: FileKind;
  name: string;
  mime: string | null;
  size: number | null;
  storage_path: string | null;
}

export interface Setlist extends SyncFields {
  name: string;
  /** Which act this set is for (Profile.acts id). */
  act_id: string | null;
  /** A permanent signature show: pinned, protected from deletion, duplicated for each gig. */
  signature: boolean;
  event_date: string | null;
  venue: string | null;
  notes: string | null;
}

export interface SetlistItem extends SyncFields {
  setlist_id: string;
  song_id: string | null;
  kind: "song" | "break";
  label: string | null;
  position: number;
  key_override: string | null;
  capo_override: number | null;
  notes: string | null;
}

export interface Profile {
  id: string;
  display_name: string | null;
  request_token: string;
  requests_open: boolean;
  request_message: string | null;
  tip_url: string | null;
  live_token: string;
  karaoke_open: boolean;
  acts: Act[];
  /** The performer's own preset lists for the Numa X, Nano Cortex and BeatBuddy. */
  gear_library: GearLibrary;
  /** The act currently performing: its name, message and tip link show on the request page. */
  active_act: string | null;
  settings: Record<string, unknown>;
  updated_at: string;
  dirty: 0 | 1;
}

export interface Act {
  id: string;
  name: string;
  message: string;
  tip_url: string;
}

export interface StoredBlob {
  id: string; // song_files.id
  blob: Blob;
  /** 1 = needs uploading to storage */
  dirty: 0 | 1;
  /** Upload failed for good (e.g. the stored file read back empty) — needs re-attaching. */
  failed?: string;
}

export interface KV {
  key: string;
  value: unknown;
}

class StageDB extends Dexie {
  songs!: Table<Song, string>;
  song_files!: Table<SongFile, string>;
  setlists!: Table<Setlist, string>;
  setlist_items!: Table<SetlistItem, string>;
  gigs!: Table<Gig, string>;
  gig_songs!: Table<GigSong, string>;
  profile!: Table<Profile, string>;
  blobs!: Table<StoredBlob, string>;
  kv!: Table<KV, string>;

  constructor() {
    super("setlist-stage");
    this.version(1).stores({
      songs: "id, title, artist, dirty, updated_at",
      song_files: "id, song_id, dirty",
      setlists: "id, dirty, event_date, updated_at",
      setlist_items: "id, setlist_id, song_id, dirty",
      profile: "id",
      blobs: "id, dirty",
      kv: "key",
    });
    this.version(2).stores({
      gigs: "id, setlist_id, gig_date, dirty",
      gig_songs: "id, gig_id, song_id, dirty",
    });
  }
}

export const db = new StageDB();

// Order matters for pushes: parents before children (foreign keys).
export const SYNC_TABLES = ["songs", "song_files", "setlists", "setlist_items", "gigs", "gig_songs"] as const;
export type SyncTable = (typeof SYNC_TABLES)[number];

export const nowIso = () => new Date().toISOString();
export const newId = () => crypto.randomUUID();

type Listener = () => void;
const changeListeners = new Set<Listener>();
/** sync.ts subscribes so local edits get pushed soon after they happen. */
export function onLocalChange(fn: Listener) {
  changeListeners.add(fn);
  return () => changeListeners.delete(fn);
}
function notify() {
  changeListeners.forEach((fn) => fn());
}

function baseRow(): SyncFields {
  const t = nowIso();
  return { id: newId(), created_at: t, updated_at: t, deleted_at: null, dirty: 1 };
}

export function blankSong(partial: Partial<Song> = {}): Song {
  return {
    ...baseRow(),
    title: "", artist: "", song_key: null, tempo: null, time_signature: null, duration_sec: null,
    capo: 0, tags: [], genre: null, year: null, ccli: null, content: "", notes: null, flow: null,
    requestable: true, karaoke: false, timings: null, instrument: null, lead_vocal: null, gear: {},
    key_kendra: null, key_devin: null,
    ...partial,
  };
}

export function blankSetlist(partial: Partial<Setlist> = {}): Setlist {
  return { ...baseRow(), name: "", act_id: null, signature: false, event_date: null, venue: null, notes: null, ...partial };
}

export function blankItem(partial: Partial<SetlistItem> & { setlist_id: string }): SetlistItem {
  return {
    ...baseRow(), song_id: null, kind: "song", label: null, position: 0,
    key_override: null, capo_override: null, notes: null, ...partial,
  };
}

/** Insert or update a row locally and queue it for sync. */
export async function saveRow<T extends SyncFields>(table: Table<T, string>, row: T): Promise<T> {
  const next = { ...row, updated_at: nowIso(), dirty: 1 as const };
  await table.put(next);
  notify();
  return next;
}

export async function patchRow<T extends SyncFields>(table: Table<T, string>, id: string, patch: Partial<T>) {
  const row = await table.get(id);
  if (!row) return;
  return saveRow(table, { ...row, ...patch });
}

export async function softDelete<T extends SyncFields>(table: Table<T, string>, id: string) {
  return patchRow(table, id, { deleted_at: nowIso() } as Partial<T>);
}

export async function saveProfile(patch: Partial<Profile>) {
  const p = await db.profile.toCollection().first();
  if (!p) return;
  await db.profile.put({ ...p, ...patch, updated_at: nowIso(), dirty: 1 });
  notify();
}

export async function addSongFile(songId: string, file: File): Promise<SongFile> {
  const kind: FileKind = file.type.startsWith("audio/") || /\.(mp3|m4a|wav|aac|ogg)$/i.test(file.name)
    ? "audio"
    : file.type.startsWith("image/") ? "image" : "pdf";
  const row: SongFile = {
    ...baseRow(), song_id: songId, kind, name: file.name, mime: file.type || null, size: file.size, storage_path: null,
  };
  // Store a copy of the bytes, not the picked File: iPad Safari can hand back a File from the Files
  // app that later reads as empty once the picker's temporary copy is cleaned up.
  const bytes = await file.arrayBuffer();
  const blob = new Blob([bytes], { type: file.type || "application/octet-stream" });
  await db.transaction("rw", db.song_files, db.blobs, async () => {
    await db.song_files.put({ ...row, size: bytes.byteLength });
    await db.blobs.put({ id: row.id, blob, dirty: 1 });
  });
  notify();
  return row;
}

/** Position halfway between neighbours, so reordering only rewrites one row. */
export function positionBetween(before: number | undefined, after: number | undefined): number {
  if (before === undefined && after === undefined) return 1000;
  if (before === undefined) return after! - 1000;
  if (after === undefined) return before + 1000;
  return (before + after) / 2;
}

export const live = <T extends { deleted_at: string | null }>(rows: T[]) => rows.filter((r) => !r.deleted_at);
