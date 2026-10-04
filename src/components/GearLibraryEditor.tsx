import { useEffect, useState } from "react";
import { newId, saveProfile, type Profile } from "../lib/db";
import { BEAT_CATEGORIES, GUITAR_CATEGORIES, PIANO_CATEGORIES, type BeatPreset, type GearLibrary, type Preset } from "../lib/gear";

type Device = "numa" | "cortex" | "beatbuddy";

const DEVICES: { key: Device; title: string; hint: string; cats: Record<string, string>; folder?: boolean }[] = [
  { key: "numa", title: "Numa X Piano 73 sounds", hint: "The sounds/presets you use. The program number is what the MIDI Captain sends.", cats: PIANO_CATEGORIES },
  { key: "cortex", title: "Nano Cortex presets", hint: "Your saved presets/captures and the program number that recalls each one.", cats: GUITAR_CATEGORIES },
  { key: "beatbuddy", title: "BeatBuddy beats", hint: "Folder (bank) and song number for each beat you use, plus its style.", cats: BEAT_CATEGORIES, folder: true },
];

/** Enter your presets once; songs then pick from them and get smart suggestions. */
export function GearLibraryEditor({ profile }: { profile: Profile }) {
  const [lib, setLib] = useState<GearLibrary>(profile.gear_library ?? {});
  useEffect(() => setLib(profile.gear_library ?? {}), [profile.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const save = (next: GearLibrary) => { setLib(next); void saveProfile({ gear_library: next }); };
  const edit = (d: Device, id: string, patch: Partial<BeatPreset>) =>
    setLib((l) => ({ ...l, [d]: (l[d] ?? []).map((p) => (p.id === id ? { ...p, ...patch } : p)) }));
  const add = (d: Device, cats: Record<string, string>) => {
    const p: BeatPreset = { id: newId(), program: null, name: "", category: Object.keys(cats)[0], captain: "", folder: null };
    save({ ...lib, [d]: [...(lib[d] ?? []), p] });
  };
  const remove = (d: Device, id: string) => save({ ...lib, [d]: (lib[d] ?? []).filter((p) => p.id !== id) });
  const num = (v: string) => (v.trim() === "" ? null : Number.isFinite(Number(v)) ? Number(v) : null);

  return (
    <div className="stack" style={{ gap: 18 }}>
      {DEVICES.map((dev) => (
        <div key={dev.key} className="stack" style={{ gap: 8 }}>
          <div>
            <strong>{dev.title}</strong>
            <div className="small dim">{dev.hint}</div>
          </div>
          {(lib[dev.key] ?? []).map((p: Preset | BeatPreset) => (
            <div key={p.id} className="row wrap" style={{ gap: 6 }}>
              {dev.folder && (
                <input className="input" style={{ width: 70, minHeight: 36 }} placeholder="Folder" inputMode="numeric"
                  value={(p as BeatPreset).folder ?? ""} onChange={(e) => edit(dev.key, p.id, { folder: num(e.target.value) })} onBlur={() => save(lib)} />
              )}
              <input className="input" style={{ width: 70, minHeight: 36 }} placeholder={dev.folder ? "Song" : "Prog #"} inputMode="numeric"
                value={p.program ?? ""} onChange={(e) => edit(dev.key, p.id, { program: num(e.target.value) })} onBlur={() => save(lib)} />
              <input className="input grow" style={{ minHeight: 36, minWidth: 150 }} placeholder="Name (as on the device)"
                value={p.name} onChange={(e) => edit(dev.key, p.id, { name: e.target.value })} onBlur={() => save(lib)} />
              <select className="select" style={{ width: 170, minHeight: 36 }} value={p.category}
                onChange={(e) => save({ ...lib, [dev.key]: (lib[dev.key] ?? []).map((x) => (x.id === p.id ? { ...x, category: e.target.value } : x)) })}>
                {Object.entries(dev.cats).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
              <input className="input" style={{ width: 130, minHeight: 36 }} placeholder="Captain switch"
                title="Which MIDI Captain page/switch calls this up, e.g. P2·B"
                value={p.captain ?? ""} onChange={(e) => edit(dev.key, p.id, { captain: e.target.value })} onBlur={() => save(lib)} />
              <button className="btn small ghost danger" onClick={() => remove(dev.key, p.id)} aria-label="Remove">✕</button>
            </div>
          ))}
          <div><button className="btn small" onClick={() => add(dev.key, dev.cats)}>+ Add {dev.key === "beatbuddy" ? "beat" : "preset"}</button></div>
        </div>
      ))}
      <p className="small dim" style={{ margin: 0 }}>
        The category is what powers suggestions: label a song “Piano” and it suggests one of your sounds in the right category
        (e.g. a Rhodes for 70s soft rock). Once you've picked gear for a few songs by an artist, it suggests your usual for that artist.
      </p>
    </div>
  );
}
