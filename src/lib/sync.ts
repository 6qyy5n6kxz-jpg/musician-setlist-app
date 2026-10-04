// Two-way sync between the local Dexie database and Supabase.
//
// Push: rows with dirty=1 are upserted; the server keeps whichever copy has the newer
// updated_at (see sync_row trigger) and returns the stored row, which we write back.
// Pull: rows whose server_updated_at is past our cursor are fetched; a local row that
// is still dirty and newer wins until it is pushed.
import type { RealtimeChannel, Session } from "@supabase/supabase-js";
import { useSyncExternalStore } from "react";
import { db, onLocalChange, SYNC_TABLES, type Profile, type SyncTable } from "./db";
import { FILE_BUCKET, supabase } from "./supabase";

export type SyncPhase = "offline" | "signed-out" | "idle" | "syncing" | "error";

interface SyncStatus {
  phase: SyncPhase;
  lastSynced: string | null;
  error: string | null;
  pending: number;
  userEmail: string | null;
}

let status: SyncStatus = { phase: "idle", lastSynced: null, error: null, pending: 0, userEmail: null };
const statusListeners = new Set<() => void>();
function setStatus(patch: Partial<SyncStatus>) {
  status = { ...status, ...patch };
  statusListeners.forEach((fn) => fn());
}
export function useSyncStatus(): SyncStatus {
  return useSyncExternalStore(
    (fn) => (statusListeners.add(fn), () => statusListeners.delete(fn)),
    () => status,
  );
}

const LOCAL_ONLY = ["dirty"] as const;
const OVERLAP_MS = 60_000; // re-read a minute back so late-committing writes aren't missed

/**
 * Defaults for every column, per table. Rows saved before a column existed lack the field, and the
 * API rejects a batch whose rows don't all have the same keys ("All object keys must match"), so
 * every pushed row is filled out to the full shape.
 */
const COLUMN_DEFAULTS: Record<SyncTable, Record<string, unknown>> = {
  songs: {
    title: "", artist: "", song_key: null, tempo: null, time_signature: null, duration_sec: null, capo: 0, tags: [],
    genre: null, year: null, ccli: null, content: "", notes: null, flow: null, requestable: true, karaoke: false,
    timings: null, instrument: null, lead_vocal: null, gear: {}, key_kendra: null, key_devin: null,
  },
  song_files: { song_id: null, kind: "pdf", name: "", mime: null, size: null, storage_path: null },
  setlists: { name: "", act_id: null, signature: false, event_date: null, venue: null, notes: null },
  setlist_items: {
    setlist_id: null, song_id: null, kind: "song", label: null, position: 0, key_override: null, capo_override: null, notes: null,
  },
  gigs: { setlist_id: null, act_id: null, name: "", venue: null, gig_date: null, notes: null },
  gig_songs: { gig_id: null, song_id: null, played_at: null, from_request: false },
};
const SYNC_DEFAULTS = { created_at: null, updated_at: null, deleted_at: null };

export function toServerRow(table: SyncTable, row: Record<string, unknown>) {
  const out = toServer({ ...SYNC_DEFAULTS, ...COLUMN_DEFAULTS[table], ...row });
  // Only send known columns (drops anything stale a very old client might have stored)
  const allowed = new Set(["id", ...Object.keys(SYNC_DEFAULTS), ...Object.keys(COLUMN_DEFAULTS[table])]);
  for (const k of Object.keys(out)) if (!allowed.has(k)) delete out[k];
  // Keep every row the same shape: fill missing timestamps rather than dropping the key
  const now = new Date().toISOString();
  for (const k of ["created_at", "updated_at"]) out[k] ??= now;
  if (table === "gig_songs") out.played_at ??= now;
  if (table === "gigs") out.gig_date ??= now.slice(0, 10);
  return out;
}

function toServer<T extends Record<string, unknown>>(row: T) {
  const out: Record<string, unknown> = { ...row };
  for (const k of LOCAL_ONLY) delete out[k];
  delete out.owner_id;
  delete out.server_updated_at;
  return out;
}

function fromServer(row: Record<string, unknown>) {
  const { owner_id: _o, server_updated_at: _s, ...rest } = row;
  void _o;
  void _s;
  return { ...rest, dirty: 0 as const };
}

async function countPending(): Promise<number> {
  let n = 0;
  for (const t of SYNC_TABLES) n += await db[t].where("dirty").equals(1).count();
  n += await db.blobs.where("dirty").equals(1).count();
  return n;
}

let session: Session | null = null;
let running = false;
let again = false;
let timer: ReturnType<typeof setTimeout> | undefined;

export function scheduleSync(delayMs = 1200) {
  clearTimeout(timer);
  timer = setTimeout(() => void syncNow(), delayMs);
}

export async function syncNow(): Promise<void> {
  setStatus({ pending: await countPending() });
  if (!navigator.onLine) return setStatus({ phase: "offline" });
  if (!session) return setStatus({ phase: "signed-out" });
  if (running) {
    again = true;
    return;
  }
  running = true;
  setStatus({ phase: "syncing", error: null });
  try {
    // Songs/setlists first: an attachment problem must never hold up the rest of the library.
    for (const t of SYNC_TABLES) await pushTable(t);
    const fileErrors = await uploadBlobs(session.user.id);
    await pushProfile();
    for (const t of SYNC_TABLES) await pullTable(t);
    await pullProfile(session.user.id);
    await downloadBlobs();
    setStatus({
      phase: fileErrors.length ? "error" : "idle",
      error: fileErrors.length ? `Couldn't upload: ${fileErrors.join(", ")} — remove and attach again` : null,
      lastSynced: new Date().toISOString(),
      pending: await countPending(),
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    setStatus({ phase: navigator.onLine ? "error" : "offline", error: msg });
  } finally {
    running = false;
    if (again) {
      again = false;
      scheduleSync(300);
    }
  }
}

async function pushTable(table: SyncTable) {
  const dirty = await db[table].where("dirty").equals(1).toArray();
  for (let i = 0; i < dirty.length; i += 200) {
    const chunk = dirty.slice(i, i + 200);
    const { data, error } = await supabase
      .from(table)
      .upsert(chunk.map((r) => toServerRow(table, r as unknown as Record<string, unknown>)))
      .select();
    if (error) throw new Error(`${table}: ${error.message}`);
    // Write back the server's copy, unless the row was edited again while we were pushing.
    await db.transaction("rw", db[table], async () => {
      for (const serverRow of data ?? []) {
        const pushed = chunk.find((r) => r.id === serverRow.id);
        const current = await db[table].get(serverRow.id as string);
        if (current && pushed && current.updated_at !== pushed.updated_at) continue;
        await db[table].put(fromServer(serverRow) as never);
      }
    });
  }
}

async function pullTable(table: SyncTable) {
  const cursorKey = `cursor:${table}`;
  const cursor = ((await db.kv.get(cursorKey))?.value as string | undefined) ?? "1970-01-01T00:00:00Z";
  const since = new Date(new Date(cursor).getTime() - OVERLAP_MS).toISOString();
  let maxSeen = cursor;
  for (let from = 0; ; from += 1000) {
    const { data, error } = await supabase
      .from(table)
      .select("*")
      .gt("server_updated_at", since)
      .order("server_updated_at", { ascending: true })
      .range(from, from + 999);
    if (error) throw new Error(`${table}: ${error.message}`);
    if (!data?.length) break;
    await db.transaction("rw", db[table], async () => {
      for (const row of data) {
        const local = await db[table].get(row.id as string);
        if (local?.dirty && local.updated_at > (row.updated_at as string)) continue;
        await db[table].put(fromServer(row) as never);
        if ((row.server_updated_at as string) > maxSeen) maxSeen = row.server_updated_at as string;
      }
    });
    if (data.length < 1000) break;
  }
  await db.kv.put({ key: cursorKey, value: maxSeen });
}

async function pushProfile() {
  const p = await db.profile.toCollection().first();
  if (!p?.dirty) return;
  const { error } = await supabase
    .from("profiles")
    .update({
      display_name: p.display_name, requests_open: p.requests_open, karaoke_open: p.karaoke_open ?? false,
      acts: p.acts ?? [], active_act: p.active_act ?? null, gear_library: p.gear_library ?? {},
      request_message: p.request_message,
      tip_url: p.tip_url, settings: p.settings, updated_at: p.updated_at,
    })
    .eq("id", p.id);
  if (error) throw new Error(`profile: ${error.message}`);
  const current = await db.profile.get(p.id);
  if (current?.updated_at === p.updated_at) await db.profile.put({ ...current, dirty: 0 });
}

async function pullProfile(userId: string) {
  const { data, error } = await supabase.from("profiles").select("*").eq("id", userId).maybeSingle();
  if (error) throw new Error(`profile: ${error.message}`);
  if (!data) return;
  const local = await db.profile.get(userId);
  if (local?.dirty && local.updated_at > data.updated_at) return;
  const { server_updated_at: _s, ...rest } = data;
  void _s;
  await db.transaction("rw", db.profile, async () => {
    await db.profile.where("id").notEqual(userId).delete();
    await db.profile.put({ ...(rest as Omit<Profile, "dirty">), dirty: 0 });
  });
}

/** Upload pending attachments one by one. Returns names of files that failed (others still upload). */
async function uploadBlobs(userId: string): Promise<string[]> {
  const pending = await db.blobs.where("dirty").equals(1).toArray();
  const failed: string[] = [];
  let uploaded = 0;
  for (const b of pending) {
    const file = await db.song_files.get(b.id);
    const song = file ? await db.songs.get(file.song_id) : undefined;
    if (!file || file.deleted_at || !song || song.deleted_at) {
      await db.blobs.update(b.id, { dirty: 0 });
      continue;
    }
    // A stored file that reads back empty can never upload — flag it instead of retrying forever.
    let size = 0;
    try {
      size = (await b.blob.arrayBuffer()).byteLength;
    } catch {
      size = 0;
    }
    if (!size) {
      await db.blobs.update(b.id, { dirty: 0, failed: "The saved copy of this file is empty." });
      failed.push(file.name);
      continue;
    }
    const path = `${userId}/${file.id}`;
    const { error } = await supabase.storage
      .from(FILE_BUCKET)
      .upload(path, b.blob, { upsert: true, contentType: file.mime ?? undefined });
    if (error) {
      failed.push(file.name);
      // The server rejected the file itself (too big, bad request): stop retrying until it's re-attached.
      // Network trouble or server outages stay pending and retry on the next sync.
      const status = Number((error as { statusCode?: string | number }).statusCode ?? 0);
      if (status >= 400 && status < 500 && status !== 401 && status !== 408 && status !== 429) {
        await db.blobs.update(b.id, { dirty: 0, failed: error.message });
      }
      continue;
    }
    await db.blobs.update(b.id, { dirty: 0, failed: undefined });
    await db.song_files.put({ ...file, storage_path: path, updated_at: new Date().toISOString(), dirty: 1 });
    uploaded++;
  }
  if (uploaded) await pushTable("song_files");
  return failed;
}

/** Keep every chart PDF and backing track on the device so gigs work offline. */
async function downloadBlobs() {
  const files = await db.song_files.filter((f) => !f.deleted_at && !!f.storage_path).toArray();
  for (const f of files) {
    if (await db.blobs.get(f.id)) continue;
    const { data, error } = await supabase.storage.from(FILE_BUCKET).download(f.storage_path!);
    if (error || !data) continue; // try again next sync
    await db.blobs.put({ id: f.id, blob: data, dirty: 0 });
  }
}

// ------------------------------------------------------------------ lifecycle
let realtime: RealtimeChannel | null = null;

function subscribeRealtime() {
  realtime?.unsubscribe();
  realtime = null;
  if (!session) return;
  const ch = supabase.channel("sync-tables");
  for (const t of [...SYNC_TABLES, "profiles"]) {
    ch.on("postgres_changes", { event: "*", schema: "public", table: t }, () => scheduleSync(500));
  }
  realtime = ch.subscribe();
}

let started = false;
export function startSync() {
  if (started) return;
  started = true;
  supabase.auth.getSession().then(({ data }) => {
    session = data.session;
    setStatus({ userEmail: session?.user.email ?? null });
    subscribeRealtime();
    void syncNow();
  });
  supabase.auth.onAuthStateChange((event, s) => {
    const changedUser = s?.user.id !== session?.user.id;
    session = s;
    setStatus({ userEmail: s?.user.email ?? null });
    if (changedUser || event === "SIGNED_IN") {
      subscribeRealtime();
      void syncNow();
    }
  });
  onLocalChange(() => scheduleSync());
  window.addEventListener("online", () => void syncNow());
  window.addEventListener("offline", () => setStatus({ phase: "offline" }));
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") void syncNow();
  });
  setInterval(() => void syncNow(), 60_000);
}

export function currentUserId(): string | null {
  return session?.user.id ?? null;
}
