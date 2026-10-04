import { saveProfile, type Profile } from "../lib/db";
import { DEFAULT_MIDI, type MidiConfig, type MidiMessage } from "../lib/gear";
import { sendMidi, useMidiOut } from "../lib/midi";
import { updateSettings, useSettings } from "../lib/settings";

const DEVICES: { key: keyof MidiConfig; label: string }[] = [
  { key: "numa", label: "Numa X Piano 73" },
  { key: "cortex", label: "Nano Cortex" },
  { key: "beatbuddy", label: "BeatBuddy" },
];

/** Automatic gear changes over MIDI (where the browser supports it). */
export function MidiOutPanel({ profile }: { profile: Profile }) {
  const settings = useSettings();
  const { supported, ports, error, refresh } = useMidiOut();
  const lib = profile.gear_library ?? {};
  const cfg: MidiConfig = { ...DEFAULT_MIDI, ...(lib.midi ?? {}) };
  const setCfg = (key: keyof MidiConfig, patch: Partial<MidiConfig["numa"]>) =>
    void saveProfile({ gear_library: { ...lib, midi: { ...cfg, [key]: { ...cfg[key], ...patch } } } });

  const test = async () => {
    const msgs: MidiMessage[] = [];
    const first = { numa: lib.numa?.[0], cortex: lib.cortex?.[0], beatbuddy: lib.beatbuddy?.[0] };
    for (const d of DEVICES) {
      const p = first[d.key];
      if (p?.program == null) continue;
      const ch = cfg[d.key].channel - 1;
      msgs.push({ device: d.key, bytes: [0xc0 | ch, Math.max(0, p.program - (cfg[d.key].oneBased ? 1 : 0))], describe: `${d.label} → ${p.name}` });
    }
    const n = await sendMidi(msgs);
    alert(n ? `Sent ${n} program change${n === 1 ? "" : "s"}:\n${msgs.map((m) => m.describe).join("\n")}` : "Nothing sent — add presets with program numbers, and pick an output.");
  };

  return (
    <div className="stack" style={{ gap: 10 }}>
      <strong>Automatic gear changes (MIDI out)</strong>
      {!supported ? (
        <p className="small dim" style={{ margin: 0 }}>
          This browser can't send MIDI — iPad Safari doesn't allow it, so on the iPad your MIDI Captain does the switching using the
          gear strip in Perform mode. On a Mac or PC, open the app in <strong>Chrome</strong> with a USB MIDI interface connected and
          it can change sounds automatically on every song change.
        </p>
      ) : (
        <>
          <label className="check">
            <input type="checkbox" checked={settings.midiOut} onChange={(e) => updateSettings({ midiOut: e.target.checked })} />
            Send each song's gear when it comes up in Perform mode (this device)
          </label>
          {settings.midiOut && (
            <>
              <div className="row wrap">
                <label className="field grow"><span>MIDI output</span>
                  <select className="select" value={settings.midiPort ?? ""} onChange={(e) => updateSettings({ midiPort: e.target.value || null })}>
                    <option value="">{ports.length ? "First available" : "No MIDI outputs found"}</option>
                    {ports.map((p) => <option key={p}>{p}</option>)}
                  </select>
                </label>
                <button className="btn small" onClick={refresh}>Rescan</button>
                <button className="btn small" onClick={test}>Send test</button>
              </div>
              {error && <div className="small" style={{ color: "var(--danger)" }}>{error}</div>}
            </>
          )}
        </>
      )}
      <div className="stack" style={{ gap: 6 }}>
        <span className="small dim">MIDI channel for each device (match the channel set on the device; also used by a future native iPad version):</span>
        {DEVICES.map((d) => (
          <div key={d.key} className="row wrap small" style={{ gap: 8 }}>
            <span style={{ width: 140 }}>{d.label}</span>
            <select className="select" style={{ width: 90, minHeight: 34 }} value={cfg[d.key].channel} onChange={(e) => setCfg(d.key, { channel: Number(e.target.value) })}>
              {Array.from({ length: 16 }, (_, i) => <option key={i} value={i + 1}>Ch {i + 1}</option>)}
            </select>
            <label className="check" style={{ minHeight: 34 }}>
              <input type="checkbox" checked={cfg[d.key].oneBased} onChange={(e) => setCfg(d.key, { oneBased: e.target.checked })} /> numbers start at 1 on the device
            </label>
          </div>
        ))}
      </div>
    </div>
  );
}
