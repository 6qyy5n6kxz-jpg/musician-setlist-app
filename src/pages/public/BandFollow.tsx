import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { ChartView } from "../../components/ChartView";
import { useLiveState } from "../../lib/live";
import { ALL_KEYS, keyDistance, transposeKey } from "../../lib/music/chords";
import { arrangeSong } from "../../lib/music/chordpro";
import { applyTheme, updateSettings, useSettings } from "../../lib/settings";

/** Bandmate view: follows the performer's song and section; each viewer sets their own key/capo/view. */
export function BandFollow() {
  const { token } = useParams();
  const { state, connected, invalid } = useLiveState(token);
  const settings = useSettings();
  const [myKey, setMyKey] = useState<string>(""); // "" = follow performer's key
  const [capo, setCapo] = useState(0);
  const [follow, setFollow] = useState(true);
  useEffect(() => applyTheme(settings.theme), [settings.theme]);

  const song = state?.song;
  const sections = useMemo(() => (song ? arrangeSong(song.content, song.flow, !!song.flow) : []), [song?.content, song?.flow]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => setMyKey(""), [song?.id]);

  useEffect(() => {
    if (!follow || state?.slide === undefined) return;
    document.querySelector(`[data-pos="${state.slide}"]`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [state?.slide, follow, song?.id]);

  if (invalid) return <div className="public"><div className="public-page empty-state"><h2>Link not active</h2></div></div>;

  const performKey = song?.performKey ?? song?.key ?? null;
  const viewKey = myKey || performKey;
  const transpose = song?.key && viewKey ? keyDistance(song.key, viewKey) : 0;

  return (
    <div className="song-view" style={{ height: "100dvh" }}>
      <div className="topbar">
        <span className={`sync-dot ${connected ? "idle" : "offline"}`} title={connected ? "Live" : "Reconnecting"} />
        <div className="grow" style={{ minWidth: 0 }}>
          <div className="truncate" style={{ fontWeight: 700 }}>{song ? song.title : "Waiting for the performer…"}</div>
          <div className="small dim truncate">
            {song?.artist}{performKey ? ` · Key ${performKey}` : ""}{song?.tempo ? ` · ${song.tempo} bpm` : ""}
            {state?.upNext ? ` · Next: ${state.upNext.title}` : ""}
          </div>
        </div>
      </div>
      {state?.message && (
        <div style={{ background: "var(--accent)", color: "var(--accent-ink)", padding: "10px 16px", fontWeight: 700, fontSize: "1.2rem", textAlign: "center" }}>
          {state.message}
        </div>
      )}
      <div className="chart-scroll">
        {song ? (
          <ChartView
            sections={sections}
            songKey={song.key}
            transpose={transpose}
            capo={capo}
            showChords={settings.showChords}
            nashville={settings.nashville}
            columns={settings.columns}
            fontScale={settings.fontScale}
            activePos={state?.slide ?? null}
          />
        ) : (
          <div className="empty-state">When the performer starts a song in Perform mode, it appears here.</div>
        )}
      </div>
      <div className="controls">
        <select className="select" style={{ width: 120, minHeight: 36 }} value={myKey} onChange={(e) => setMyKey(e.target.value)} aria-label="My key">
          <option value="">{performKey ? `Key ${performKey}` : "Key"}</option>
          {(song?.key?.endsWith("m") ? ALL_KEYS.filter((k) => k.endsWith("m")) : ALL_KEYS.filter((k) => !k.endsWith("m"))).map((k) => <option key={k} value={k}>{k}</option>)}
        </select>
        <select className="select" style={{ width: 110, minHeight: 36 }} value={capo} onChange={(e) => setCapo(Number(e.target.value))} aria-label="Capo">
          {Array.from({ length: 10 }, (_, i) => <option key={i} value={i}>{i ? `Capo ${i}` : "No capo"}</option>)}
        </select>
        {capo > 0 && viewKey && <span className="small dim">{transposeKey(viewKey, -capo)} shapes</span>}
        <button className={`btn small ${settings.showChords ? "on" : ""}`} onClick={() => updateSettings({ showChords: !settings.showChords })}>Chords</button>
        <button className={`btn small ${settings.nashville ? "on" : ""}`} onClick={() => updateSettings({ nashville: !settings.nashville })}>123</button>
        <button className={`btn small ${follow ? "on" : ""}`} onClick={() => setFollow(!follow)}>Follow</button>
        <button className="btn small icon" onClick={() => updateSettings({ fontScale: Math.max(0.6, settings.fontScale - 0.1) })}>A−</button>
        <button className="btn small icon" onClick={() => updateSettings({ fontScale: Math.min(2.6, settings.fontScale + 0.1) })}>A+</button>
        <div className="seg">
          {(["dark", "lowlight", "light"] as const).map((t) => (
            <button key={t} className={settings.theme === t ? "on" : ""} onClick={() => updateSettings({ theme: t })}>{t === "lowlight" ? "Dim" : t[0].toUpperCase() + t.slice(1)}</button>
          ))}
        </div>
      </div>
    </div>
  );
}
