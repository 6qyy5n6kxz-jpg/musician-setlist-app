# Setlist Stage

A personal OnSong-style app for performing musicians, built to run on an iPad at gigs — with or without venue internet.

- **Song library** — ChordPro charts with chords above lyrics, key/tempo/time/length/capo, tags, flow (arrangement), sticky notes, PDF charts and backing tracks.
- **Import anything** — paste from Ultimate Guitar or any chords-over-lyrics text, OnSong files, ChordPro files, PDFs, CSV song lists, backups.
- **Transpose & capo** — per song or per setlist entry, with sensible sharp/flat spelling; Nashville numbers; lyrics-only mode; two columns; three themes (dark, low light, light).
- **Setlists** — drag to reorder, set breaks, per-song key overrides, running times per set, auto-build to a target length, printable stage list.
- **Perform mode** — full-screen, Bluetooth foot-pedal control (learnable keys), autoscroll timed to the song or backing track, metronome with count-in and silent visual beat, screen kept awake.
- **Live requests** — audience scans a QR code, picks from your library (or types a song), adds a dedication and a tip link; requests pop up on stage in a queue.
- **Lyrics display (karaoke)** — a link you open on whatever drives the TV; follows your section as you scroll or press the pedal.
- **Band follow** — bandmates open a link and see your current song with chords, each in their own key/capo/view.

## How it works offline

The app is a Progressive Web App. Installed on the iPad (Safari → Share → **Add to Home Screen**), the app itself, every song, setlist, PDF and backing track live on the device in IndexedDB. Edits are saved locally first and synced to Supabase whenever there's a connection (last edit wins per item).

Requests, the lyrics display and band screens need a connection between devices: put the iPad on your phone's hotspot; patrons use their own cell data.

## Stack

- React + TypeScript + Vite, `vite-plugin-pwa` (Workbox) for offline, Dexie for local storage, PDF.js for charts
- Supabase: Postgres with row-level security, Realtime (requests + live screens), Storage (PDFs, audio), Auth (single owner account)
- Hosted on GitHub Pages (`.github/workflows/deploy.yml`); a scheduled workflow pings Supabase so the free project never pauses

## Develop

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # music engine tests (chords, ChordPro, importers)
npm run build
```

Database schema lives in `supabase/migrations/`. Only the first account can register (personal app).

`scripts/legacy-export.mts` converts the old Flask app's `musician.db` into a backup file you can restore from **Import**. The old Flask app is on the `legacy-flask` branch.
