import { useEffect, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useLiveState } from "../../lib/live";

/**
 * Audience / karaoke lyrics screen. Open on whatever drives the TV.
 * Options via URL: ?size=6 (text size in vw), ?next=0 (hide "up next"), ?title=0 (hide song title).
 */
export function LyricsDisplay() {
  const { token } = useParams();
  const [params] = useSearchParams();
  const { state, connected, invalid } = useLiveState(token);
  const [cursorHidden, setCursorHidden] = useState(true);
  const size = Number(params.get("size")) || null;
  const showNext = params.get("next") !== "0";
  const showTitle = params.get("title") !== "0";

  // Keep the TV awake where supported, and offer fullscreen on first tap.
  useEffect(() => {
    let lock: WakeLockSentinel | null = null;
    void navigator.wakeLock?.request("screen").then((l) => (lock = l)).catch(() => {});
    return () => void lock?.release();
  }, []);

  if (invalid) return <div className="display"><div className="d-splash"><h1>Link not active</h1><p>Get the current display link from the performer's Settings.</p></div></div>;

  const slides = state?.slides ?? [];
  const exact = slides.find((s) => s.pos === state?.slide);
  // If the performer is on a section with no lyrics (solo, instrumental), keep the previous lines up.
  const current = exact ?? [...slides].reverse().find((s) => s.pos < (state?.slide ?? 0));
  const instrumental = !exact && state?.song;
  const nextSlide = current ? slides[slides.indexOf(current) + 1] : slides[0];
  const style = size ? ({ "--display-size": `${size}vw` } as React.CSSProperties) : undefined;

  return (
    <div
      className="display"
      style={{ cursor: cursorHidden ? "none" : "default" }}
      onClick={() => { void document.documentElement.requestFullscreen?.().catch(() => {}); setCursorHidden(false); setTimeout(() => setCursorHidden(true), 3000); }}
    >
      <div className="d-status">{connected ? "" : "reconnecting…"}</div>
      {!state?.song || state.blank ? (
        state?.blank ? null : (
          <div className="d-splash">
            {state?.nextSingers?.length ? (
              <>
                <p style={{ textTransform: "uppercase", letterSpacing: "0.12em" }}>🎤 Up next to sing</p>
                {state.nextSingers.map((n, i) => (
                  <h1 key={i} style={{ fontSize: i === 0 ? "6vw" : "3.4vw", margin: "1vh 0", color: i === 0 ? "#fff" : "#9a9890" }}>
                    {n.name} <span style={{ fontWeight: 400, fontSize: "0.5em", color: "#9a9890" }}>— {n.title}</span>
                  </h1>
                ))}
              </>
            ) : (
              <>
                <h1>{state?.act || state?.setlist || "Live music"}</h1>
                <p>Lyrics will appear here</p>
              </>
            )}
          </div>
        )
      ) : (
        <>
          {state.singer && <div className="d-singer">🎤 {state.singer}</div>}
          {showTitle && <div className="d-title">{state.song.title}{state.song.artist ? ` — ${state.song.artist}` : ""}</div>}
          {current && !instrumental ? (
            <div key={`${state.song.id}-${current.pos}`} className="d-lines fade" style={style}>
              {current.lines.map((l, i) => <div key={i}>{l}</div>)}
            </div>
          ) : (
            <div key="inst" className="d-lines fade" style={{ ...style, color: "#6c6a63" }}>♪</div>
          )}
          {showNext && nextSlide && !instrumental && (
            <div className="d-next">{nextSlide.lines[0]}</div>
          )}
          {showNext && !nextSlide && (state.nextSingers?.[0] || state.upNext) && (
            <div className="d-next">
              {state.nextSingers?.[0] ? `🎤 Next singer: ${state.nextSingers[0].name} — ${state.nextSingers[0].title}` : `Up next: ${state.upNext!.title}`}
            </div>
          )}
        </>
      )}
    </div>
  );
}
