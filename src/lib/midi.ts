// MIDI out through the Web MIDI API. Works in Chrome/Edge on Mac, Windows, Android and ChromeOS.
// iPad Safari has no Web MIDI, so on the iPad the MIDI Captain does the switching (or a future
// native shell provides the same send() function).
import { useCallback, useEffect, useState } from "react";
import type { MidiMessage } from "./gear";
import { getSettings, useSettings } from "./settings";

export const midiSupported = () => typeof navigator !== "undefined" && "requestMIDIAccess" in navigator;

let access: MIDIAccess | null = null;
let pending: Promise<MIDIAccess | null> | null = null;

async function getAccess(): Promise<MIDIAccess | null> {
  if (access) return access;
  if (!midiSupported()) return null;
  pending ??= navigator.requestMIDIAccess({ sysex: false }).then((a) => (access = a)).catch(() => null);
  return pending;
}

/** Output ports (name list) and a send() for MIDI messages to the chosen port. */
export function useMidiOut() {
  const settings = useSettings();
  const [ports, setPorts] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const a = await getAccess();
    if (!a) {
      setError(midiSupported() ? "MIDI access was blocked — allow MIDI devices when the browser asks, or turn it on in the site settings (lock icon in the address bar)." : null);
      return;
    }
    setPorts([...a.outputs.values()].map((o) => o.name ?? o.id));
    a.onstatechange = () => setPorts([...a.outputs.values()].map((o) => o.name ?? o.id));
  }, []);

  useEffect(() => {
    if (settings.midiOut) void refresh();
  }, [settings.midiOut, refresh]);

  return { supported: midiSupported(), ports, error, refresh };
}

/** Send messages to the configured port. Returns how many were sent. */
export async function sendMidi(messages: MidiMessage[]): Promise<number> {
  const { midiOut, midiPort } = getSettings();
  if (!midiOut || !messages.length) return 0;
  const a = await getAccess();
  if (!a) return 0;
  const outputs = [...a.outputs.values()];
  const out = outputs.find((o) => (o.name ?? o.id) === midiPort) ?? outputs[0];
  if (!out) return 0;
  let t = performance.now();
  for (const m of messages) {
    out.send(m.bytes, t);
    t += 15; // small gap so devices that process slowly (e.g. bank + program) keep up
  }
  return messages.length;
}
