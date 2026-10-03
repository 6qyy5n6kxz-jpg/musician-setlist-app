import { createClient } from "@supabase/supabase-js";

// Publishable values: safe to ship in the app. Row Level Security protects the data.
export const SUPABASE_URL = "https://bovujlijcemlkbncflkv.supabase.co";
export const SUPABASE_KEY = "sb_publishable_QSiLy3W32RjMSKTIbJ_ZjQ_uMJII3_n";

export const supabase = createClient(SUPABASE_URL, SUPABASE_KEY, {
  // PKCE puts auth codes in ?code= (not the #hash, which the app router uses)
  auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true, flowType: "pkce" },
  realtime: { params: { eventsPerSecond: 20 } },
});

export const FILE_BUCKET = "song-files";
