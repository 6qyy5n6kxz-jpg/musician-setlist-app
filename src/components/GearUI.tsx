import { useMemo } from "react";
import type { Song } from "../lib/db";
import {
  BEAT_CATEGORIES, GUITAR_CATEGORIES, INSTRUMENT_LABELS, PIANO_CATEGORIES, presetLabel, suggestGear, suggestShows,
  VOCAL_LABELS, type GearLibrary, type Instrument, type LeadVocal, type Preset, type SongGear,
} from "../lib/gear";

const INSTRUMENT_ICON: Record<Instrument, string> = { piano: "🎹", electric: "⚡", acoustic: "🎸" };
const VOCAL_ICON: Record<LeadVocal, string> = { kendra: "K", devin: "D", both: "K+D" };

/** Small inline badges: lead vocal + instrument. */
export function DuoBadges({ song }: { song: Pick<Song, "instrument" | "lead_vocal"> }) {
  if (!song.instrument && !song.lead_vocal) return null;
  return (
    <>
      {song.lead_vocal && <span className="chip" title={`${VOCAL_LABELS[song.lead_vocal]} sings lead`}>🎤 {VOCAL_ICON[song.lead_vocal]}</span>}
      {song.instrument && <span className="chip" title={INSTRUMENT_LABELS[song.instrument]}>{INSTRUMENT_ICON[song.instrument]} {INSTRUMENT_LABELS[song.instrument]}</span>}
    </>
  );
}

/** Instrument + lead vocal pickers (segmented buttons; tap the active one again to clear). */
export function DuoPickers({ song, onChange, compact = false }: {
  song: Pick<Song, "instrument" | "lead_vocal">;
  onChange: (patch: Partial<Pick<Song, "instrument" | "lead_vocal">>) => void;
  compact?: boolean;
}) {
  return (
    <div className="row wrap" style={{ gap: compact ? 6 : 10 }}>
      <div className="seg" role="group" aria-label="Instrument">
        {(Object.keys(INSTRUMENT_LABELS) as Instrument[]).map((i) => (
          <button key={i} className={song.instrument === i ? "on" : ""} style={compact ? { minHeight: 32, padding: "0 8px" } : undefined}
            onClick={() => onChange({ instrument: song.instrument === i ? null : i })}>
            {INSTRUMENT_ICON[i]} {compact ? "" : INSTRUMENT_LABELS[i]}
          </button>
        ))}
      </div>
      <div className="seg" role="group" aria-label="Lead vocal">
        {(Object.keys(VOCAL_LABELS) as LeadVocal[]).map((v) => (
          <button key={v} className={song.lead_vocal === v ? "on" : ""} style={compact ? { minHeight: 32, padding: "0 8px" } : undefined}
            onClick={() => onChange({ lead_vocal: song.lead_vocal === v ? null : v })}>
            {compact ? VOCAL_ICON[v] : VOCAL_LABELS[v]}
          </button>
        ))}
      </div>
    </div>
  );
}

function PresetSelect({ label, presets, value, onChange, empty }: {
  label: string; presets: Preset[] | undefined; value: string | null | undefined; onChange: (id: string | null) => void; empty: string;
}) {
  return (
    <label className="field"><span>{label}</span>
      <select className="select" value={value ?? ""} onChange={(e) => onChange(e.target.value || null)}>
        <option value="">{presets?.length ? "—" : empty}</option>
        {presets?.map((p) => <option key={p.id} value={p.id}>{presetLabel(p)}{p.captain ? `  (${p.captain})` : ""}</option>)}
      </select>
    </label>
  );
}

/** Song editor section: categories, gear and smart suggestions. */
export function SongGearEditor({ song, library, history, onChange }: {
  song: Song;
  library: GearLibrary;
  history: { artist: string; gear: SongGear }[];
  onChange: (patch: Partial<Song>) => void;
}) {
  const gear = song.gear ?? {};
  const setGear = (patch: Partial<SongGear>) => onChange({ gear: { ...gear, ...patch } });
  const facts = { ...song, tags: song.tags ?? [], instrument: song.instrument ?? null, lead_vocal: song.lead_vocal ?? null };
  const sugg = useMemo(() => suggestGear(facts, library, history), [song, library, history]); // eslint-disable-line react-hooks/exhaustive-deps
  const shows = useMemo(() => suggestShows(facts), [song]); // eslint-disable-line react-hooks/exhaustive-deps

  const Suggest = ({ s, current, apply, cats }: { s?: { preset?: Preset; category: string; why: string }; current?: string | null; apply: (id: string) => void; cats: Record<string, string> }) => {
    if (!s) return null;
    if (s.preset && s.preset.id === current) return <span className="small dim">✓ matches the suggestion</span>;
    return (
      <span className="small dim">
        Suggested: <strong>{s.preset ? presetLabel(s.preset) : cats[s.category]}</strong> ({s.why}){" "}
        {s.preset ? <button className="chip accent" onClick={() => apply(s.preset!.id)}>Use</button> : <em>— add a “{cats[s.category]}” preset in Settings → Gear</em>}
      </span>
    );
  };

  return (
    <div className="stack">
      <DuoPickers song={song} onChange={onChange} />
      {shows.length > 0 && (
        <div className="row wrap small">
          <span className="dim">Fits:</span>
          {shows.map((s) => (
            <button key={s.tag} className="chip accent" title={s.why} onClick={() => onChange({ tags: [...(song.tags ?? []), s.tag] })}>
              + {s.label}
            </button>
          ))}
        </div>
      )}
      <div className="meta-grid" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))" }}>
        {song.instrument === "piano" && (
          <div className="stack" style={{ gap: 4 }}>
            <PresetSelect label="Numa X sound" presets={library.numa} value={gear.numa} onChange={(id) => setGear({ numa: id })} empty="Add presets in Settings → Gear" />
            <Suggest s={sugg.numa} current={gear.numa} apply={(id) => setGear({ numa: id })} cats={PIANO_CATEGORIES} />
          </div>
        )}
        {song.instrument === "electric" && (
          <div className="stack" style={{ gap: 4 }}>
            <PresetSelect label="Nano Cortex preset" presets={library.cortex} value={gear.cortex} onChange={(id) => setGear({ cortex: id })} empty="Add presets in Settings → Gear" />
            <Suggest s={sugg.cortex} current={gear.cortex} apply={(id) => setGear({ cortex: id })} cats={GUITAR_CATEGORIES} />
          </div>
        )}
        <div className="stack" style={{ gap: 4 }}>
          <PresetSelect label="BeatBuddy" presets={library.beatbuddy} value={gear.beatbuddy} onChange={(id) => setGear({ beatbuddy: id })} empty="Add beats in Settings → Gear" />
          <Suggest s={sugg.beatbuddy} current={gear.beatbuddy} apply={(id) => setGear({ beatbuddy: id })} cats={BEAT_CATEGORIES} />
        </div>
        {gear.beatbuddy && (
          <label className="field"><span>BeatBuddy tempo</span>
            <input className="input" inputMode="numeric" placeholder={song.tempo ? String(song.tempo) : "bpm"} value={gear.bbTempo ?? ""}
              onChange={(e) => setGear({ bbTempo: Number(e.target.value) || null })} />
          </label>
        )}
      </div>
    </div>
  );
}

/** Perform-mode strip: what to set for this song, and a heads-up for the next one. */
export function GearStrip({ song, next, library }: { song: Song; next?: Song; library: GearLibrary }) {
  const find = (list: Preset[] | undefined, id?: string | null) => list?.find((p) => p.id === id);
  const g = song.gear ?? {};
  const numa = song.instrument === "piano" ? find(library.numa, g.numa) : undefined;
  const cortex = song.instrument === "electric" ? find(library.cortex, g.cortex) : undefined;
  const bb = library.beatbuddy?.find((p) => p.id === g.beatbuddy);
  const items: string[] = [];
  if (song.lead_vocal) items.push(`🎤 ${VOCAL_LABELS[song.lead_vocal]}`);
  if (song.instrument) items.push(`${INSTRUMENT_ICON[song.instrument]} ${INSTRUMENT_LABELS[song.instrument]}`);
  if (numa) items.push(`Numa ${presetLabel(numa)}${numa.captain ? ` · ${numa.captain}` : ""}`);
  if (cortex) items.push(`Cortex ${presetLabel(cortex)}${cortex.captain ? ` · ${cortex.captain}` : ""}`);
  if (bb) items.push(`BB ${bb.folder ? `${bb.folder}/` : ""}${presetLabel(bb)}${g.bbTempo || song.tempo ? ` · ${g.bbTempo || song.tempo} bpm` : ""}${bb.captain ? ` · ${bb.captain}` : ""}`);
  const switching = next?.instrument && song.instrument && next.instrument !== song.instrument;
  if (!items.length && !switching) return null;
  return (
    <div className="gear-strip">
      {items.map((t, i) => <span key={i} className="chip">{t}</span>)}
      {switching && <span className="chip accent">Next: switch to {INSTRUMENT_LABELS[next!.instrument!].toLowerCase()}</span>}
    </div>
  );
}
