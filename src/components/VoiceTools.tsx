import { useEffect, useMemo, useRef, useState } from "react";
import { db, patchRow, saveProfile, type Profile, type Song } from "../lib/db";
import { describeOption, recommendKeys, type Range, type VocalRanges } from "../lib/keyFinder";
import { guessKey } from "../lib/music/chords";
import { allChords, parseChordPro } from "../lib/music/chordpro";
import { hzToMidi, midiName, PitchListener, sungRange, tunerReading } from "../lib/pitch";

type Singer = "kendra" | "devin";
const SINGER_NAMES: Record<Singer, string> = { kendra: "Kendra", devin: "Devin" };

/** Live microphone pitch while `on`. Returns the latest Hz (or null) and an error message if the mic is blocked. */
function useMicPitch(on: boolean, onFrame?: (hz: number | null) => void) {
  const [hz, setHz] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const frame = useRef(onFrame);
  frame.current = onFrame;
  useEffect(() => {
    if (!on) return;
    const l = new PitchListener();
    let smooth: number | null = null;
    l.onPitch = (p) => {
      // Light smoothing so the display doesn't jitter
      smooth = p == null ? null : smooth == null || Math.abs(hzToMidi(p) - hzToMidi(smooth)) > 0.7 ? p : smooth * 0.7 + p * 0.3;
      setHz(smooth);
      frame.current?.(p);
    };
    l.start().catch((e) => setError(/denied|permission|NotAllowed/i.test(String(e)) ? "Microphone access is blocked — allow it for this site in Settings → Safari → Microphone." : String(e?.message ?? e)));
    return () => l.stop();
  }, [on]);
  return { hz, error };
}

/** Chromatic tuner. */
export function Tuner({ onClose }: { onClose: () => void }) {
  const { hz, error } = useMicPitch(true);
  const r = hz ? tunerReading(hz) : null;
  const inTune = r && Math.abs(r.cents) <= 5;
  return (
    <div className="voice-modal" role="dialog" aria-label="Tuner">
      <div className="voice-card">
        <div className="row"><strong className="grow">Tuner</strong><button className="btn small ghost" onClick={onClose}>Close</button></div>
        {error ? <div style={{ color: "var(--danger)" }}>{error}</div> : (
          <>
            <div className={`tuner-note ${inTune ? "in" : ""}`}>{r ? r.note.replace(/-?\d+$/, "") : "–"}<span className="tuner-oct">{r ? r.note.match(/-?\d+$/)?.[0] : ""}</span></div>
            <div className="tuner-meter">
              <div className="tuner-center" />
              {r && <div className="tuner-needle" style={{ left: `${50 + Math.max(-50, Math.min(50, r.cents))}%`, background: inTune ? "var(--ok)" : "var(--accent)" }} />}
            </div>
            <div className="small dim" style={{ textAlign: "center" }}>{r ? `${r.cents > 0 ? "+" : ""}${r.cents} cents · ${hz!.toFixed(1)} Hz` : "Play a note"}</div>
          </>
        )}
      </div>
    </div>
  );
}

const NOTE_CHOICES = Array.from({ length: 50 }, (_, i) => 36 + i); // C2 … C#6
export function NoteSelect({ value, onChange, label }: { value: number | null | undefined; onChange: (m: number | null) => void; label: string }) {
  return (
    <label className="field"><span>{label}</span>
      <select className="select" value={value ?? ""} onChange={(e) => onChange(e.target.value ? Number(e.target.value) : null)}>
        <option value="">—</option>
        {NOTE_CHOICES.map((m) => <option key={m} value={m}>{midiName(m)}</option>)}
      </select>
    </label>
  );
}

/**
 * Capture a range by singing: listens while active and keeps the held notes; the result is the
 * 5th–95th percentile so a crack or a breath doesn't count.
 */
function useSungCapture() {
  const [listening, setListening] = useState(false);
  const frames = useRef<number[]>([]);
  const { hz, error } = useMicPitch(listening, (p) => { if (p) frames.current.push(hzToMidi(p)); });
  const start = () => { frames.current = []; setListening(true); };
  const stop = () => { setListening(false); return sungRange(frames.current); };
  return { listening, hz, error, start, stop };
}

/** One singer's comfortable range: sing low, then high (or pick the notes). */
export function RangeTest({ profile, singer }: { profile: Profile; singer: Singer }) {
  const ranges: VocalRanges = profile.vocal_ranges ?? {};
  const current = ranges[singer];
  const [step, setStep] = useState<"idle" | "low" | "high">("idle");
  const [low, setLow] = useState<number | null>(current?.low ?? null);
  const cap = useSungCapture();
  const save = (r: Partial<Range>) => {
    const next = { ...(ranges[singer] ?? { low: 48, high: 72 }), ...r };
    if (next.low > next.high) [next.low, next.high] = [next.high, next.low];
    void saveProfile({ vocal_ranges: { ...ranges, [singer]: next } });
  };
  const finish = () => {
    const r = cap.stop();
    if (!r) { setStep("idle"); return; }
    if (step === "low") { setLow(r.low); setStep("high"); setTimeout(cap.start, 300); }
    else { save({ low: low ?? r.low, high: r.high }); setStep("idle"); }
  };
  const live = cap.hz ? midiName(hzToMidi(cap.hz)) : "…";
  return (
    <div className="card stack" style={{ background: "var(--bg-sunken)", padding: 12 }}>
      <div className="row">
        <strong className="grow">{SINGER_NAMES[singer]}</strong>
        <span className="small dim">{current ? `${midiName(current.low)} – ${midiName(current.high)}` : "not tested yet"}</span>
      </div>
      {step === "idle" ? (
        <>
          <div className="meta-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
            <NoteSelect label="Lowest comfortable" value={current?.low} onChange={(m) => m != null && save({ low: m })} />
            <NoteSelect label="Highest comfortable" value={current?.high} onChange={(m) => m != null && save({ high: m })} />
          </div>
          <button className="btn small" onClick={() => { setStep("low"); cap.start(); }}>🎤 Test by singing</button>
        </>
      ) : (
        <div className="stack">
          <div>{step === "low" ? "Sing down to your lowest comfortable note and hold it (not your absolute bottom)." : "Now slide up to your highest comfortable note — one you could sing in a song — and hold it."}</div>
          <div className="tuner-note small-note">{live}</div>
          {cap.error && <div style={{ color: "var(--danger)" }}>{cap.error}</div>}
          <div className="row">
            <button className="btn primary" onClick={finish}>{step === "low" ? "Got my low note" : "Got my high note"}</button>
            <button className="btn ghost" onClick={() => { cap.stop(); setStep("idle"); }}>Cancel</button>
          </div>
        </div>
      )}
    </div>
  );
}

/** Key finder for one song: melody range + singer ranges → best keys, one tap to use. */
export function KeyFinderDialog({ song, profile, onClose }: { song: Song; profile: Profile; onClose: () => void }) {
  const written = song.song_key || guessKey(allChords(parseChordPro(song.content)));
  const ranges: VocalRanges = profile.vocal_ranges ?? {};
  const [measuring, setMeasuring] = useState(false);
  const cap = useSungCapture();
  const melody = song.melody_low != null && song.melody_high != null ? { low: song.melody_low, high: song.melody_high } : null;
  const singers: Singer[] = song.lead_vocal === "kendra" ? ["kendra"] : song.lead_vocal === "devin" ? ["devin"] : ["kendra", "devin"];
  const missing = singers.filter((s) => !ranges[s]);

  const options = useMemo(() => {
    if (!written || !melody) return [];
    const each = singers.filter((s) => ranges[s]).map((s) => ({ who: SINGER_NAMES[s], singer: s as Singer | "both", list: recommendKeys(written, melody, [ranges[s]!], song.instrument).slice(0, 3) }));
    if (song.lead_vocal === "both" && ranges.kendra && ranges.devin) {
      each.unshift({ who: "Both (duet)", singer: "both", list: recommendKeys(written, melody, [ranges.kendra, ranges.devin], song.instrument).slice(0, 3) });
    }
    return each;
  }, [written, melody?.low, melody?.high, ranges.kendra?.low, ranges.kendra?.high, ranges.devin?.low, ranges.devin?.high, song.lead_vocal, song.instrument]); // eslint-disable-line react-hooks/exhaustive-deps

  const use = (singer: Singer | "both", key: string) => {
    if (singer === "kendra" || singer === "both") void patchRow(db.songs, song.id, { key_kendra: key === written ? null : key });
    if (singer === "devin" || singer === "both") void patchRow(db.songs, song.id, { key_devin: key === written ? null : key });
  };
  const current = (s: Singer | "both") => s === "devin" ? song.key_devin : song.key_kendra;

  return (
    <div className="voice-modal" role="dialog" aria-label="Key finder">
      <div className="voice-card wide">
        <div className="row"><strong className="grow">🎤 Key finder — {song.title}</strong><button className="btn small ghost" onClick={onClose}>Done</button></div>
        {!written ? <div className="dim">Set the song's key first (or add chords) so the key finder knows where it starts.</div> : (
          <>
            <div className="small dim">Written in <strong>{written}</strong>. The melody range is the lowest and highest note sung in that key.</div>
            {!measuring ? (
              <div className="row wrap" style={{ alignItems: "flex-end" }}>
                <NoteSelect label="Melody low" value={song.melody_low} onChange={(m) => patchRow(db.songs, song.id, { melody_low: m })} />
                <NoteSelect label="Melody high" value={song.melody_high} onChange={(m) => patchRow(db.songs, song.id, { melody_high: m })} />
                <button className="btn" onClick={() => { setMeasuring(true); cap.start(); }}>🎤 Measure by singing</button>
              </div>
            ) : (
              <div className="card stack" style={{ padding: 12 }}>
                <div>Sing the melody <strong>in the original key</strong> (along with the recording is easiest) — include the lowest verse line and the highest chorus note.</div>
                <div className="tuner-note small-note">{cap.hz ? midiName(hzToMidi(cap.hz)) : "…"}</div>
                {cap.error && <div style={{ color: "var(--danger)" }}>{cap.error}</div>}
                <div className="row">
                  <button className="btn primary" onClick={() => { const r = cap.stop(); setMeasuring(false); if (r) void patchRow(db.songs, song.id, { melody_low: r.low, melody_high: r.high }); }}>Done singing</button>
                  <button className="btn ghost" onClick={() => { cap.stop(); setMeasuring(false); }}>Cancel</button>
                </div>
              </div>
            )}
            {missing.length > 0 && (
              <div className="stack">
                <div className="small" style={{ color: "var(--accent)" }}>Test {missing.map((s) => SINGER_NAMES[s]).join(" and ")}'s range once and every song can use it:</div>
                {missing.map((s) => <RangeTest key={s} profile={profile} singer={s} />)}
              </div>
            )}
            {!melody && <div className="small dim">Add the melody range above to see key suggestions.</div>}
            {options.map((o) => (
              <div key={o.who} className="stack">
                <strong>{o.who}</strong>
                {o.list.map((k, i) => (
                  <div key={k.key} className={`key-option ${k.fit}`}>
                    <span className="key-pill" style={{ minWidth: 46 }}>{k.key}</span>
                    <div className="grow small">
                      <div>{i === 0 ? "Best · " : ""}{k.semitones === 0 ? "original key" : `${k.semitones > 0 ? "up" : "down"} ${Math.abs(k.semitones)}`} · {describeOption(k)}</div>
                      {k.capo && <div className="dim">Guitar: capo {k.capo.fret}, play {k.capo.shapes} shapes</div>}
                    </div>
                    {current(o.singer) === k.key || (!current(o.singer) && k.key === written) ? <span className="chip accent">in use</span> : (
                      <button className="btn small" onClick={() => use(o.singer, k.key)}>Use for {o.singer === "both" ? "both" : SINGER_NAMES[o.singer]}</button>
                    )}
                  </div>
                ))}
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}
