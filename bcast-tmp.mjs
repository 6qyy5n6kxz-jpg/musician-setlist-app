import { createClient } from "@supabase/supabase-js";
const sb = createClient("https://bovujlijcemlkbncflkv.supabase.co", "sb_publishable_QSiLy3W32RjMSKTIbJ_ZjQ_uMJII3_n");
const ch = sb.channel("live-qcdgcfch5b3fqpb4tvaz", { config: { broadcast: { self: false } } });
await new Promise((res) => ch.subscribe((s) => s === "SUBSCRIBED" && res()));
const base = { v: 1, setlist: "Karaoke Night", blank: false, message: null, upNext: null,
  nextSingers: [{ name: "Jess", title: "Shallow" }, { name: "Mike", title: "Mr. Brightside" }] };
await ch.send({ type: "broadcast", event: "state", payload: { ...base, at: Date.now(), song: null, slides: [], slide: 0, singer: null } });
console.log("between songs");
