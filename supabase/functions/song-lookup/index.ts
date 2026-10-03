// Look up factual song details for the Tune-up screen:
//   iTunes Search API -> length, release year, genre (accurate, official catalog)
//   Deezer API        -> tempo (BPM), when Deezer has analysed the track
// Neither service needs an API key. Requires a signed-in user (verify_jwt).
import "jsr:@supabase/functions-js/edge-runtime.d.ts";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

const norm = (s: string) =>
  s.toLowerCase().replace(/\(.*?\)|\[.*?\]/g, "").replace(/feat\.?.*$|ft\.?.*$/, "").replace(/[^a-z0-9]+/g, " ").trim();

/** Rough similarity: share of query words found in the candidate. */
function score(query: string, candidate: string): number {
  const q = norm(query).split(" ").filter(Boolean);
  const c = norm(candidate);
  if (!q.length) return 0;
  return q.filter((w) => c.includes(w)).length / q.length;
}

interface ItunesTrack {
  trackName: string;
  artistName: string;
  trackTimeMillis?: number;
  releaseDate?: string;
  primaryGenreName?: string;
  trackViewUrl?: string;
}

async function itunes(title: string, artist: string) {
  const url = `https://itunes.apple.com/search?media=music&entity=song&limit=15&term=${encodeURIComponent(`${title} ${artist}`)}`;
  const res = await fetch(url);
  if (!res.ok) return null;
  const { results } = (await res.json()) as { results: ItunesTrack[] };
  let best: ItunesTrack | null = null;
  let bestScore = 0;
  for (const r of results ?? []) {
    const s = score(title, r.trackName) * 2 + (artist ? score(artist, r.artistName) : 1);
    // Prefer the original over live/karaoke/remix versions on ties
    const penalty = /live|karaoke|remix|instrumental|cover|tribute/i.test(`${r.trackName} ${r.artistName}`) ? 0.5 : 0;
    if (s - penalty > bestScore) {
      best = r;
      bestScore = s - penalty;
    }
  }
  if (!best || bestScore < 2) return null; // title must match well
  return {
    title: best.trackName,
    artist: best.artistName,
    duration_sec: best.trackTimeMillis ? Math.round(best.trackTimeMillis / 1000) : null,
    year: best.releaseDate ? Number(best.releaseDate.slice(0, 4)) : null,
    genre: best.primaryGenreName ?? null,
    url: best.trackViewUrl ?? null,
  };
}

async function deezerBpm(title: string, artist: string): Promise<number | null> {
  // Deezer's advanced query syntax no longer matches; a plain search does.
  const res = await fetch(`https://api.deezer.com/search?limit=10&q=${encodeURIComponent(`${title} ${artist}`)}`);
  if (!res.ok) return null;
  const { data } = (await res.json()) as { data?: { id: number; title: string; artist: { name: string } }[] };
  const hits = (data ?? [])
    .filter((d) => score(title, d.title) >= 0.75 && (!artist || score(artist, d.artist.name) >= 0.5))
    // originals first, then live/remix versions
    .sort((x, y) => Number(/remix|mix\)|live|version|edit/i.test(x.title)) - Number(/remix|mix\)|live|version|edit/i.test(y.title)))
    .slice(0, 5);
  for (const h of hits) {
    const track = await fetch(`https://api.deezer.com/track/${h.id}`).then((r) => r.json()).catch(() => null);
    const bpm = Number(track?.bpm);
    if (bpm > 30 && bpm < 300) return Math.round(bpm);
  }
  return null;
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });
  try {
    const { title, artist } = await req.json();
    if (typeof title !== "string" || !title.trim()) {
      return Response.json({ error: "title required" }, { status: 400, headers: CORS });
    }
    const a = typeof artist === "string" ? artist : "";
    const [catalog, bpm] = await Promise.all([itunes(title, a).catch(() => null), deezerBpm(title, a).catch(() => null)]);
    return Response.json({ catalog, bpm }, { headers: CORS });
  } catch (e) {
    return Response.json({ error: String(e) }, { status: 500, headers: CORS });
  }
});
