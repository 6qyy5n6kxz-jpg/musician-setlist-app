import { createClient } from "@supabase/supabase-js";

// Publishable values: safe to ship in the app. Row Level Security protects the data.
export const SUPABASE_URL = "https://bovujlijcemlkbncflkv.supabase.co";
export const SUPABASE_KEY = "sb_publishable_QSiLy3W32RjMSKTIbJ_ZjQ_uMJII3_n";

export const supabase = createClient(SUPABASE_URL, SUPABASE_KEY, {
  // PKCE puts auth codes in ?code= (not the #hash, which the app router uses)
  auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true, flowType: "pkce" },
  realtime: { params: { eventsPerSecond: 20 } },
  // No request may hang forever (a dead hotspot connection would otherwise stall sync for good).
  // File uploads/downloads get longer than API calls.
  global: {
    fetch: (input, init) => {
      if (init?.signal || typeof AbortSignal === "undefined" || !("timeout" in AbortSignal)) return fetch(input, init);
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      return fetch(input, { ...init, signal: AbortSignal.timeout(/\/storage\/v1\//.test(url) ? 180_000 : 30_000) });
    },
  },
});

export const FILE_BUCKET = "song-files";
