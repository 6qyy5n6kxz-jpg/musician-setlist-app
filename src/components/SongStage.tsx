import { useLiveQuery } from "dexie-react-hooks";
import { useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState, type ReactNode, type Ref } from "react";
import { db, patchRow, type Song } from "../lib/db";
import { useSongFiles } from "../lib/hooks";
import { ALL_KEYS, guessKey, keyDistance, transposeKey } from "../lib/music/chords";
import { allChords, applyFlow, parseChordPro, type Section } from "../lib/music/chordpro";
import { updateSettings, useSettings, type PedalAction } from "../lib/settings";
import { AutoScroller, beatsPerBar, createTapTempo, estimateDuration, formatDuration, Metronome, timedSectionAt, usePedalActions } from "../lib/stage";
import { ChartView } from "./ChartView";
import { IconMetronome, IconMusic, IconPause, IconPlay, IconScroll } from "./Icons";
import { PdfView } from "./PdfView";

export interface StageHandle {
  scrollToPos: (pos: number) => void;
  sections: Section[];
}

interface Props {
  song: Song;
  /** Key to perform in (e.g. from the setlist). Defaults to the song's key. */
  performKey?: string | null;
  capoOverride?: number | null;
  onPerformKeyChange?: (key: string | null) => void;
  onCapoChange?: (capo: number) => void;
  onActiveSection?: (pos: number, sections: Section[]) => void;
  /** Called whenever the arranged sections change (song change, flow toggle). */
  onSectionsChange?: (sections: Section[], flowUsed: boolean) => void;
  /** Timed lyrics moved to a new section while the backing track plays. */
  onTimedSection?: (pos: number) => void;
  onReachEnd?: () => void;
  onReachStart?: () => void;
  pedalHandlers?: Partial<Record<PedalAction, () => void>>;
  activePos?: number | null;
  handleRef?: Ref<StageHandle>;
  /** Extra buttons at the right of the control bar. */
  extraControls?: ReactNode;
  /** Shown above the title (e.g. "Song 3 of 14"). */
  kicker?: ReactNode;
  pedalsEnabled?: boolean;
}

export function SongStage(props: Props) {
  const { song, performKey, capoOverride, onPerformKeyChange, onCapoChange, onActiveSection, onReachEnd, onReachStart,
    pedalHandlers, activePos, handleRef, extraControls, kicker, pedalsEnabled = true, onSectionsChange, onTimedSection } = props;
  const settings = useSettings();
  const scrollRef = useRef<HTMLDivElement>(null);

  const parsed = useMemo(() => parseChordPro(song.content), [song.content]);
  const writtenKey = song.song_key || guessKey(allChords(parsed));

  const [transpose, setTranspose] = useState(0);
  const [capo, setCapo] = useState(0);
  const [useFlow, setUseFlow] = useState(true);
  const [view, setView] = useState<"chart" | "pdf">("chart");

  // Reset per song
  useEffect(() => {
    setTranspose(performKey && writtenKey ? keyDistance(writtenKey, performKey) : 0);
    setCapo(capoOverride ?? song.capo ?? 0);
    setUseFlow(true);
    scrollRef.current?.scrollTo({ top: 0 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [song.id, performKey, capoOverride]);

  const sections = useMemo(
    () => (useFlow && song.flow ? applyFlow(parsed.sections, song.flow) : parsed.sections),
    [parsed, useFlow, song.flow],
  );
  const shownKey = writtenKey ? transposeKey(writtenKey, transpose) : null;
  const flowUsed = useFlow && !!song.flow;
  useEffect(() => { onSectionsChange?.(sections, flowUsed); }, [sections, flowUsed, onSectionsChange]);

  // ---------------------------------------------------------- files
  const files = useSongFiles(song.id);
  const pdf = files?.find((f) => f.kind === "pdf");
  const audio = files?.find((f) => f.kind === "audio");
  const pdfBlob = useLiveQuery(async () => (pdf ? (await db.blobs.get(pdf.id))?.blob ?? null : null), [pdf?.id]);
  const audioBlob = useLiveQuery(async () => (audio ? (await db.blobs.get(audio.id))?.blob ?? null : null), [audio?.id]);
  const hasChart = parsed.sections.length > 0;
  useEffect(() => setView(hasChart || !pdf ? "chart" : "pdf"), [song.id, hasChart, pdf]);

  const audioUrl = useMemo(() => (audioBlob ? URL.createObjectURL(audioBlob) : null), [audioBlob]);
  useEffect(() => () => { if (audioUrl) URL.revokeObjectURL(audioUrl); }, [audioUrl]);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [trackTime, setTrackTime] = useState(0);
  const [trackDur, setTrackDur] = useState(0);

  // ---------------------------------------------------------- autoscroll
  const scroller = useRef<AutoScroller | null>(null);
  const [scrolling, setScrolling] = useState(false);
  useEffect(() => {
    if (!scrollRef.current) return;
    const s = new AutoScroller(scrollRef.current);
    s.onChange = setScrolling;
    scroller.current = s;
    return () => s.stop();
  }, []);
  useEffect(() => scroller.current?.stop(), [song.id]);

  const songDuration = song.duration_sec || estimateDuration(song.tempo);
  const startScroll = useCallback((durationSec?: number, delay?: number) => {
    scroller.current?.start(durationSec ?? songDuration, delay ?? settings.scrollDelay);
  }, [songDuration, settings.scrollDelay]);
  const toggleScroll = useCallback(() => {
    if (scroller.current?.running) scroller.current.stop();
    else startScroll(playing && trackDur ? trackDur - trackTime : undefined, playing ? 0 : undefined);
  }, [startScroll, playing, trackDur, trackTime]);

  // ---------------------------------------------------------- metronome
  const metro = useRef<Metronome>(new Metronome());
  const [metroOn, setMetroOn] = useState(false);
  const [bpm, setBpm] = useState(song.tempo ?? 100);
  const [flash, setFlash] = useState<"" | "on" | "on accent">("");
  const tap = useRef(createTapTempo());
  useEffect(() => setBpm(song.tempo ?? 100), [song.id, song.tempo]);
  useEffect(() => {
    const m = metro.current;
    m.bpm = bpm;
    m.beatsPerBar = beatsPerBar(song.time_signature);
    m.volume = settings.metronomeVolume;
    m.accent = settings.clickAccent;
    m.onBeat = (beat) => {
      if (!settings.flashOnBeat) return;
      setFlash(beat === 0 ? "on accent" : "on");
      setTimeout(() => setFlash(""), 90);
    };
  }, [bpm, song.time_signature, settings.metronomeVolume, settings.clickAccent, settings.flashOnBeat]);
  useEffect(() => () => metro.current.stop(), []);
  useEffect(() => { metro.current.stop(); setMetroOn(false); }, [song.id]);

  const toggleMetronome = useCallback(() => {
    const m = metro.current;
    if (m.running) { m.stop(); setMetroOn(false); }
    else { m.onDone = null; m.start(); setMetroOn(true); }
  }, []);

  // ---------------------------------------------------------- timed lyrics
  // Marks are section start times on the backing track, recorded by tapping along once.
  const flowKey = flowUsed ? song.flow : null;
  const timings = song.timings && (song.timings.flow ?? null) === flowKey && song.timings.marks.length ? song.timings : null;
  const [recording, setRecording] = useState(false);
  const [draftMarks, setDraftMarks] = useState<{ pos: number; t: number }[]>([]);
  useEffect(() => { setRecording(false); setDraftMarks([]); }, [song.id]);
  const mark = useCallback(() => {
    const a = audioRef.current;
    if (!a) return;
    setDraftMarks((m) => (m.length >= sections.length ? m : [...m, { pos: m.length, t: Math.round(a.currentTime * 10) / 10 }]));
  }, [sections.length]);
  const saveTimings = useCallback(async (marks: { pos: number; t: number }[]) => {
    setRecording(false);
    if (marks.length) await patchRow(db.songs, song.id, { timings: { flow: flowKey, marks } });
  }, [song.id, flowKey]);

  const toggleTrack = useCallback(() => {
    const a = audioRef.current;
    if (!a) return;
    if (!a.paused) { a.pause(); return; }
    const go = () => {
      void a.play();
      // With recorded timings the chart jumps section by section instead of scrolling linearly
      if (!timings && !recording) startScroll(Math.max(10, (a.duration || songDuration) - a.currentTime), settings.scrollDelay);
    };
    // Count-in clicks before the track starts (only from the top)
    if (settings.countInBars > 0 && a.currentTime < 0.5 && song.tempo) {
      const m = metro.current;
      m.onDone = () => { setMetroOn(false); go(); };
      m.start(settings.countInBars);
      setMetroOn(true);
    } else go();
  }, [startScroll, songDuration, settings.scrollDelay, settings.countInBars, song.tempo, timings, recording]);

  // ---------------------------------------------------------- paging & sections
  const pageDown = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    // The chart has bottom padding so the last lines can scroll up; "the end" is the last section's bottom.
    const secs = el.querySelectorAll<HTMLElement>("[data-pos]");
    const last = secs[secs.length - 1];
    const contentBottom = last ? last.offsetTop + last.offsetHeight : el.scrollHeight;
    const atEnd = el.scrollTop + el.clientHeight >= contentBottom + 8;
    if (atEnd && onReachEnd && settings.pageDownAdvances) onReachEnd();
    else el.scrollBy({ top: el.clientHeight * 0.8, behavior: "smooth" });
  }, [onReachEnd, settings.pageDownAdvances]);

  const pageUp = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (el.scrollTop <= 2 && onReachStart) onReachStart();
    else el.scrollBy({ top: -el.clientHeight * 0.8, behavior: "smooth" });
  }, [onReachStart]);

  const scrollToPos = useCallback((pos: number) => {
    const el = scrollRef.current?.querySelector<HTMLElement>(`[data-pos="${pos}"]`);
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);
  useImperativeHandle(handleRef, () => ({ scrollToPos, sections }), [scrollToPos, sections]);

  const timedPos = useRef(-1);
  useEffect(() => { timedPos.current = -1; }, [song.id]);
  useEffect(() => {
    if (!timings || recording || !playing) return;
    const pos = timedSectionAt(timings.marks, trackTime);
    if (pos >= 0 && pos !== timedPos.current) {
      timedPos.current = pos;
      scrollToPos(pos);
      onTimedSection?.(pos);
    }
  }, [trackTime, timings, recording, playing, scrollToPos, onTimedSection]);

  const lastActive = useRef(-1);
  useEffect(() => { lastActive.current = -1; }, [song.id, useFlow]);
  const onScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el || !onActiveSection) return;
    const line = el.scrollTop + el.clientHeight * 0.22;
    let pos = 0;
    el.querySelectorAll<HTMLElement>("[data-pos]").forEach((s) => {
      if (s.offsetTop <= line) pos = Number(s.dataset.pos);
    });
    if (pos !== lastActive.current) {
      lastActive.current = pos;
      onActiveSection(pos, sections);
    }
  }, [onActiveSection, sections]);
  useEffect(() => { onScroll(); }, [sections, onScroll]);

  usePedalActions({
    pageDown, pageUp, toggleScroll, toggleMetronome,
    toggleTrack: audio ? toggleTrack : undefined,
    ...pedalHandlers,
    ...(recording ? { nextSlide: mark } : {}),
  }, pedalsEnabled);

  // ---------------------------------------------------------- key / capo controls
  const changeTranspose = (semis: number) => {
    let t = semis;
    while (t > 6) t -= 12;
    while (t < -6) t += 12;
    setTranspose(t);
    if (onPerformKeyChange && writtenKey) onPerformKeyChange(t === 0 ? null : transposeKey(writtenKey, t));
  };
  const changeCapo = (c: number) => {
    const v = Math.max(0, Math.min(11, c));
    setCapo(v);
    onCapoChange?.(v);
  };

  const fontStep = (d: number) => updateSettings({ fontScale: Math.max(0.6, Math.min(2.6, Math.round((settings.fontScale + d) * 100) / 100)) });

  return (
    <div className="stage">
      <div className={`beat-flash ${flash}`} />
      <div className="chart-scroll" ref={scrollRef} onScroll={onScroll}>
        <div className="song-head" style={{ padding: 0, marginBottom: 14 }}>
          {kicker}
          <h1>{song.title || "Untitled"}</h1>
          <div className="dim">{song.artist}</div>
          <div className="song-facts">
            {shownKey && <span className="chip accent">Key {shownKey}{transpose ? ` (from ${writtenKey})` : ""}</span>}
            {capo > 0 && shownKey && <span className="chip accent">Capo {capo} · {transposeKey(shownKey, -capo)} shapes</span>}
            {song.tempo && <span className="chip">{song.tempo} bpm</span>}
            {song.time_signature && <span className="chip">{song.time_signature}</span>}
            <span className="chip">{formatDuration(song.duration_sec) || `~${formatDuration(songDuration)}`}</span>
            {song.flow && useFlow && <span className="chip">{song.flow}</span>}
          </div>
        </div>
        {song.notes && <div className="sticky-note">{song.notes}</div>}
        {view === "pdf" && pdfBlob ? (
          <PdfView
            blob={pdfBlob}
            annotations={pdf?.annotations}
            onAnnotationsChange={pdf ? (a) => void patchRow(db.song_files, pdf.id, { annotations: a }) : undefined}
          />
        ) : view === "pdf" && pdf && !pdfBlob ? (
          <div className="card dim">This PDF hasn't downloaded to this device yet. Connect once and it will be saved for offline use.</div>
        ) : hasChart ? (
          <ChartView
            sections={sections}
            songKey={writtenKey}
            transpose={transpose}
            capo={capo}
            showChords={settings.showChords}
            nashville={settings.nashville}
            columns={settings.columns}
            fontScale={settings.fontScale}
            activePos={activePos}
          />
        ) : (
          <div className="empty-state">No chart yet. Edit the song to add chords and lyrics, or attach a PDF.</div>
        )}
      </div>

      {audioUrl && (
        <div className="audio-bar">
          <button className="btn small icon" onClick={toggleTrack} aria-label={playing ? "Pause track" : "Play track"}>
            {playing ? <IconPause size={18} /> : <IconPlay size={18} />}
          </button>
          <span className="small mono">{formatDuration(Math.floor(trackTime)) || "0:00"}</span>
          <input
            type="range" min={0} max={trackDur || 1} step={0.5} value={trackTime}
            onChange={(e) => { if (audioRef.current) audioRef.current.currentTime = Number(e.target.value); }}
            aria-label="Track position"
          />
          <span className="small mono">{formatDuration(Math.floor(trackDur))}</span>
          {recording ? (
            <>
              <button className="btn small primary" onClick={mark} disabled={!playing || draftMarks.length >= sections.length}>
                Mark {sections[draftMarks.length]?.label || (draftMarks.length >= sections.length ? "(done)" : `section ${draftMarks.length + 1}`)}
              </button>
              <button className="btn small" onClick={() => setDraftMarks((m) => m.slice(0, -1))} disabled={!draftMarks.length}>Undo</button>
              <button className="btn small" onClick={() => saveTimings(draftMarks)} disabled={!draftMarks.length}>Save ({draftMarks.length}/{sections.length})</button>
              <button className="btn small ghost" onClick={() => { setRecording(false); setDraftMarks([]); }}>Cancel</button>
            </>
          ) : (
            <>
              {timings && <span className="chip accent" title="Chart and lyrics display follow the track">timed ✓</span>}
              <button className="btn small ghost" disabled={!hasChart} title="Tap along once to time the lyrics to this track"
                onClick={() => {
                  setDraftMarks([]);
                  setRecording(true);
                  scroller.current?.stop();
                  if (audioRef.current) audioRef.current.currentTime = 0;
                }}>
                {timings ? "Re-time" : "Time lyrics"}
              </button>
              <span className="small dim truncate" style={{ maxWidth: 140 }}>{audio?.name}</span>
            </>
          )}
          <audio
            ref={audioRef} src={audioUrl} preload="auto"
            onPlay={() => setPlaying(true)}
            onPause={() => { setPlaying(false); scroller.current?.stop(); }}
            onEnded={() => { setPlaying(false); if (recording) void saveTimings(draftMarks); }}
            onTimeUpdate={(e) => setTrackTime(e.currentTarget.currentTime)}
            onLoadedMetadata={(e) => setTrackDur(e.currentTarget.duration)}
          />
        </div>
      )}

      <div className="controls no-print">
        <div className="group" aria-label="Key">
          <button className="btn small icon" onClick={() => changeTranspose(transpose - 1)} aria-label="Transpose down">−</button>
          <select
            className="select" style={{ minHeight: 34, width: 86, padding: "0 6px" }}
            value={shownKey ?? ""} disabled={!writtenKey}
            onChange={(e) => writtenKey && changeTranspose(keyDistance(writtenKey, e.target.value))}
            aria-label="Key"
          >
            {!writtenKey && <option value="">Key</option>}
            {(writtenKey?.endsWith("m") ? ALL_KEYS.filter((k) => k.endsWith("m")) : ALL_KEYS.filter((k) => !k.endsWith("m"))).map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
          <button className="btn small icon" onClick={() => changeTranspose(transpose + 1)} aria-label="Transpose up">+</button>
        </div>
        <div className="group" aria-label="Capo">
          <button className="btn small icon" onClick={() => changeCapo(capo - 1)} aria-label="Capo down">−</button>
          <span className="val">Capo {capo}</span>
          <button className="btn small icon" onClick={() => changeCapo(capo + 1)} aria-label="Capo up">+</button>
        </div>
        <div className="group">
          <button className={`btn small ${settings.showChords ? "on" : ""}`} onClick={() => updateSettings({ showChords: !settings.showChords })} title="Show chords">Chords</button>
          <button className={`btn small ${settings.nashville ? "on" : ""}`} onClick={() => updateSettings({ nashville: !settings.nashville })} title="Nashville numbers">123</button>
          {song.flow && <button className={`btn small ${useFlow ? "on" : ""}`} onClick={() => setUseFlow(!useFlow)} title="Play in flow order">Flow</button>}
          {pdf && hasChart && <button className={`btn small ${view === "pdf" ? "on" : ""}`} onClick={() => setView(view === "pdf" ? "chart" : "pdf")}>PDF</button>}
        </div>
        <div className="group">
          <button className="btn small icon" onClick={() => fontStep(-0.1)} aria-label="Smaller text">A−</button>
          <button className="btn small icon" onClick={() => fontStep(0.1)} aria-label="Larger text">A+</button>
          <button className={`btn small ${settings.columns === 2 ? "on" : ""}`} onClick={() => updateSettings({ columns: settings.columns === 2 ? 1 : 2 })} title="Two columns">2 col</button>
        </div>
        <div className="group">
          <button className={`btn small ${scrolling ? "on" : ""}`} onClick={toggleScroll} title="Autoscroll"><IconScroll size={18} /></button>
          {scrolling && (
            <>
              <button className="btn small icon" onClick={() => scroller.current?.adjust(0.85)} aria-label="Slower">−</button>
              <button className="btn small icon" onClick={() => scroller.current?.adjust(1.18)} aria-label="Faster">+</button>
            </>
          )}
          <button className={`btn small ${metroOn ? "on" : ""}`} onClick={toggleMetronome} title="Metronome"><IconMetronome size={18} /></button>
          <button
            className="btn small" title="Tap tempo"
            onClick={() => { const t = tap.current(); if (t && t > 30 && t < 300) setBpm(t); }}
          >{bpm} bpm</button>
          {audio && !audioUrl && <span className="chip" title="Track not downloaded yet"><IconMusic size={14} /> …</span>}
        </div>
        <div className="spacer" />
        {extraControls}
      </div>
    </div>
  );
}
