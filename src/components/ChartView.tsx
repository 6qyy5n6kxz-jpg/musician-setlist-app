import { memo, useMemo } from "react";
import { keyPrefersFlats, parseChord, toNashville, transposeChord, transposeKey } from "../lib/music/chords";
import type { Line, Section, Segment } from "../lib/music/chordpro";
import { lineText } from "../lib/music/chordpro";

export interface ChartOptions {
  songKey: string | null;
  transpose: number;
  capo: number;
  showChords: boolean;
  nashville: boolean;
  columns: 1 | 2;
  fontScale: number;
}

interface Props extends ChartOptions {
  sections: Section[];
  activePos?: number | null;
  onSectionClick?: (pos: number) => void;
}

/** Chord as it should be shown: transposed to the performed key, then shifted for the capo. */
export function useChordFormatter({ songKey, transpose, capo, nashville }: ChartOptions) {
  return useMemo(() => {
    const performKey = songKey ? transposeKey(songKey, transpose) : null;
    const shapeKey = performKey && capo ? transposeKey(performKey, -capo) : performKey;
    const flats = shapeKey ? keyPrefersFlats(shapeKey) : false;
    return (chord: string) => {
      if (!parseChord(chord)) return chord;
      if (nashville && songKey) return toNashville(chord, songKey);
      return transposeChord(chord, transpose - capo, flats);
    };
  }, [songKey, transpose, capo, nashville]);
}

type Piece = { kind: "space"; text: string } | { kind: "word"; units: { chord: string | null; text: string }[] };

/** Group segments into unbreakable words so a line wraps between words, never inside one. */
function toPieces(segments: Segment[]): Piece[] {
  const pieces: Piece[] = [];
  let word: { chord: string | null; text: string }[] = [];
  const flush = () => {
    if (word.length) pieces.push({ kind: "word", units: word });
    word = [];
  };
  for (const seg of segments) {
    const tokens = seg.lyric.split(/(\s+)/);
    let chord = seg.chord;
    if (tokens.length === 1 && tokens[0] === "") {
      word.push({ chord, text: "" });
      continue;
    }
    for (const tok of tokens) {
      if (tok === "") {
        if (chord !== null) {
          word.push({ chord, text: "" });
          chord = null;
        }
        continue;
      }
      if (/^\s+$/.test(tok)) {
        flush();
        pieces.push({ kind: "space", text: tok });
        continue;
      }
      word.push({ chord, text: tok });
      chord = null;
    }
  }
  flush();
  return pieces;
}

function LyricLine({ line, fmt, showChords }: { line: Line & { kind: "lyrics" }; fmt: (c: string) => string; showChords: boolean }) {
  const hasChords = showChords && line.segments.some((s) => s.chord);
  if (!hasChords) {
    const text = lineText(line);
    return text ? <div className="ln lyric-only">{text}</div> : null;
  }
  const lyricless = line.segments.every((s) => !s.lyric.trim());
  if (lyricless) {
    return (
      <div className="ln chords-only">
        {line.segments.filter((s) => s.chord).map((s, i) => (
          <span key={i} className="ch">{fmt(s.chord!)}</span>
        ))}
      </div>
    );
  }
  return (
    <div className="ln">
      {toPieces(line.segments).map((p, i) =>
        p.kind === "space" ? (
          <span key={i} className="sp">{p.text}</span>
        ) : (
          <span key={i} className="wd">
            {p.units.map((u, j) => (
              <span key={j} className="u">
                <span className="ch">{u.chord ? fmt(u.chord) : " "}</span>
                <span className="ly">{u.text || " "}</span>
              </span>
            ))}
          </span>
        ),
      )}
    </div>
  );
}

function ChartViewInner(props: Props) {
  const { sections, showChords, columns, fontScale, activePos, onSectionClick } = props;
  const fmt = useChordFormatter(props);
  return (
    <div className={`chart cols-${columns}`} style={{ fontSize: `${fontScale * 1.25}rem` }}>
      {sections.map((s, pos) => (
        <section
          key={pos}
          data-pos={pos}
          className={`sec sec-${s.type}${activePos === pos ? " active" : ""}`}
          onClick={onSectionClick ? () => onSectionClick(pos) : undefined}
        >
          {s.label && <h3 className="sec-label">{s.label}</h3>}
          {s.lines.map((l, i) => {
            if (l.kind === "lyrics") return <LyricLine key={i} line={l} fmt={fmt} showChords={showChords} />;
            if (l.kind === "comment") return <div key={i} className="ln comment">{l.text}</div>;
            if (l.kind === "tab") return showChords ? <pre key={i} className="ln tab">{l.text}</pre> : null;
            return <div key={i} className="ln empty" />;
          })}
        </section>
      ))}
    </div>
  );
}

export const ChartView = memo(ChartViewInner);
