import { useLiveQuery } from "dexie-react-hooks";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ChartView } from "../components/ChartView";
import { IconBack, IconFile, IconMusic, IconTrash } from "../components/Icons";
import { addSongFile, db, deleteSong, saveRow, softDelete, type Song } from "../lib/db";
import { useProfile, useSong, useSongFiles, useSongs } from "../lib/hooks";
import { SongGearEditor } from "../components/GearUI";
import { ALL_KEYS, guessKey } from "../lib/music/chords";
import { allChords, parseChordPro, sectionAbbrev } from "../lib/music/chordpro";
import { importChart } from "../lib/music/convert";
import { useSettings } from "../lib/settings";
import { formatDuration, parseDurationInput } from "../lib/stage";

const SECTION_SNIPPETS: [string, string][] = [
  ["Verse", "{start_of_verse: Verse}\n\n{end_of_verse}"],
  ["Chorus", "{start_of_chorus: Chorus}\n\n{end_of_chorus}"],
  ["Pre-Chorus", "{start_of_prechorus: Pre-Chorus}\n\n{end_of_prechorus}"],
  ["Bridge", "{start_of_bridge: Bridge}\n\n{end_of_bridge}"],
  ["Intro", "{start_of_intro: Intro}\n\n{end_of_intro}"],
  ["Outro", "{start_of_outro: Outro}\n\n{end_of_outro}"],
  ["Comment", "{comment: }"],
  ["Repeat chorus", "{chorus}"],
];

export function SongEditor() {
  const { id } = useParams();
  const stored = useSong(id);
  const navigate = useNavigate();
  const settings = useSettings();
  const [draft, setDraft] = useState<Song | null>(null);
  const [durationText, setDurationText] = useState("");
  const [tagText, setTagText] = useState("");
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const files = useSongFiles(id);
  const failedFiles = useLiveQuery(async () => new Set((await db.blobs.toArray()).filter((b) => b.failed).map((b) => b.id)), []);
  const profile = useProfile();
  const allSongs = useSongs();
  const gearHistory = useMemo(
    () => (allSongs ?? []).filter((s) => s.id !== id && s.gear && Object.keys(s.gear).length).map((s) => ({ artist: s.artist, gear: s.gear })),
    [allSongs, id],
  );

  // Load once; after that the draft is the source of truth while editing.
  useEffect(() => {
    if (stored && (!draft || draft.id !== stored.id)) {
      setDraft(stored);
      setDurationText(formatDuration(stored.duration_sec));
      setTagText(stored.tags.join(", "));
    }
  }, [stored, draft]);

  const update = (patch: Partial<Song>) => {
    setDraft((d) => {
      if (!d) return d;
      const next = { ...d, ...patch };
      clearTimeout(saveTimer.current);
      saveTimer.current = setTimeout(async () => {
        await saveRow(db.songs, next);
        setSavedAt(new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }));
      }, 500);
      return next;
    });
  };
  // Flush on leave
  useEffect(() => () => clearTimeout(saveTimer.current), []);

  const parsed = useMemo(() => parseChordPro(draft?.content ?? ""), [draft?.content]);
  const detectedKey = useMemo(() => guessKey(allChords(parsed)), [parsed]);

  if (stored === undefined || !draft) return null;
  if (stored === null) return <div className="page empty-state">Song not found.</div>;

  const insert = (snippet: string) => {
    const ta = textRef.current;
    const content = draft.content;
    const at = ta ? ta.selectionStart : content.length;
    const before = content.slice(0, at);
    const pre = before && !before.endsWith("\n\n") ? (before.endsWith("\n") ? "\n" : "\n\n") : "";
    update({ content: before + pre + snippet + "\n" + content.slice(at) });
    requestAnimationFrame(() => ta?.focus());
  };

  const cleanUp = () => {
    const { meta, body } = importChart(draft.content);
    update({
      content: body,
      title: draft.title || meta.title || "",
      artist: draft.artist || meta.artist || "",
      song_key: draft.song_key || meta.key || null,
      tempo: draft.tempo ?? meta.tempo ?? null,
      time_signature: draft.time_signature || meta.time || null,
      capo: draft.capo || meta.capo || 0,
      duration_sec: draft.duration_sec ?? meta.duration_sec ?? null,
      flow: draft.flow || meta.flow || null,
    });
    if (meta.duration_sec && !draft.duration_sec) setDurationText(formatDuration(meta.duration_sec));
  };

  const remove = async () => {
    if (!confirm(`Delete “${draft.title || "this song"}”? It will also be removed from setlists.`)) return;
    await deleteSong(draft.id);
    navigate("/");
  };

  const num = (v: string) => (v.trim() === "" ? null : Number.isFinite(Number(v)) ? Number(v) : null);
  const abbrevs = parsed.sections.filter((s) => s.label).map((s) => sectionAbbrev(s.label, s.type));

  return (
    <div className="page" style={{ maxWidth: 1400 }}>
      <div className="row" style={{ marginBottom: 12 }}>
        <button className="btn ghost icon" onClick={() => navigate(-1)} aria-label="Back"><IconBack /></button>
        <h1 className="grow truncate">{draft.title || "New song"}</h1>
        <span className="small dim">{savedAt ? `Saved ${savedAt}` : "Changes save automatically"}</span>
        <Link className="btn primary" to={`/song/${draft.id}`}>Done</Link>
      </div>

      <div className="card stack" style={{ marginBottom: 16 }}>
        <div className="meta-grid" style={{ gridTemplateColumns: "2fr 2fr" }}>
          <label className="field"><span>Title</span>
            <input className="input" value={draft.title} onChange={(e) => update({ title: e.target.value })} autoFocus={!draft.title} />
          </label>
          <label className="field"><span>Artist</span>
            <input className="input" value={draft.artist} onChange={(e) => update({ artist: e.target.value })} />
          </label>
        </div>
        <div className="meta-grid">
          <label className="field"><span>Key (as written)</span>
            <select className="select" value={draft.song_key ?? ""} onChange={(e) => update({ song_key: e.target.value || null })}>
              <option value="">{detectedKey ? `Detect (${detectedKey})` : "—"}</option>
              {ALL_KEYS.map((k) => <option key={k}>{k}</option>)}
            </select>
          </label>
          <label className="field"><span>Tempo (bpm)</span>
            <input className="input" inputMode="numeric" value={draft.tempo ?? ""} onChange={(e) => update({ tempo: num(e.target.value) })} />
          </label>
          <label className="field"><span>Time</span>
            <select className="select" value={draft.time_signature ?? ""} onChange={(e) => update({ time_signature: e.target.value || null })}>
              <option value="">—</option>
              {["4/4", "3/4", "6/8", "2/4", "12/8", "5/4", "7/8"].map((t) => <option key={t}>{t}</option>)}
            </select>
          </label>
          <label className="field"><span>Length (m:ss)</span>
            <input className="input" placeholder="3:45" value={durationText}
              onChange={(e) => { setDurationText(e.target.value); update({ duration_sec: parseDurationInput(e.target.value) }); }} />
          </label>
          <label className="field"><span>Capo</span>
            <select className="select" value={draft.capo} onChange={(e) => update({ capo: Number(e.target.value) })}>
              {Array.from({ length: 12 }, (_, i) => <option key={i} value={i}>{i === 0 ? "None" : i}</option>)}
            </select>
          </label>
          <label className="field"><span>Year</span>
            <input className="input" inputMode="numeric" value={draft.year ?? ""} onChange={(e) => update({ year: num(e.target.value) })} />
          </label>
        </div>
        <div className="meta-grid" style={{ gridTemplateColumns: "2fr 2fr 1fr" }}>
          <label className="field"><span>Tags (comma separated)</span>
            <input className="input" placeholder="country, upbeat, wedding" value={tagText}
              onChange={(e) => {
                setTagText(e.target.value);
                update({ tags: e.target.value.split(",").map((t) => t.trim().toLowerCase()).filter(Boolean) });
              }} />
          </label>
          <label className="field"><span>Flow {abbrevs.length ? <span className="dim">({abbrevs.join(" ")})</span> : null}</span>
            <input className="input" placeholder="I V1 C V2 C B C C" value={draft.flow ?? ""} onChange={(e) => update({ flow: e.target.value || null })} />
          </label>
          <label className="field"><span>Genre</span>
            <input className="input" value={draft.genre ?? ""} onChange={(e) => update({ genre: e.target.value || null })} />
          </label>
        </div>
        <label className="field"><span>Sticky note (shown above the chart)</span>
          <textarea className="textarea" style={{ minHeight: 60 }} placeholder="Capo 2 · start on the chorus · watch the drummer for the stop at 2:45"
            value={draft.notes ?? ""} onChange={(e) => update({ notes: e.target.value || null })} />
        </label>
        <div className="row wrap">
          <label className="check"><input type="checkbox" checked={draft.requestable} onChange={(e) => update({ requestable: e.target.checked })} /> Show on the audience request page</label>
          <label className="check"><input type="checkbox" checked={draft.karaoke ?? false} onChange={(e) => update({ karaoke: e.target.checked })} /> Available for karaoke sign-up</label>
        </div>
      </div>

      <div className="card stack" style={{ marginBottom: 16 }}>
        <h2 style={{ fontSize: "1.1rem" }}>Duo & gear</h2>
        <SongGearEditor song={draft} library={profile?.gear_library ?? {}} history={gearHistory} onChange={update} />
      </div>

      <div className="editor-grid">
        <div>
          <div className="insert-bar">
            {SECTION_SNIPPETS.map(([label, snip]) => (
              <button key={label} className="btn small" onClick={() => insert(snip)}>+ {label}</button>
            ))}
            <button className="btn small primary" onClick={cleanUp} title="Convert chords-over-lyrics, Ultimate Guitar or OnSong text to ChordPro">
              Clean up pasted chart
            </button>
          </div>
          <textarea
            ref={textRef}
            className="code-area"
            spellCheck={false}
            autoCapitalize="off"
            autoCorrect="off"
            placeholder={"Paste a chart from anywhere, then tap “Clean up pasted chart”.\n\nOr write ChordPro directly:\n\n{start_of_verse: Verse 1}\n[G]Well today is [D]gonna be the day\n{end_of_verse}"}
            value={draft.content}
            onChange={(e) => update({ content: e.target.value })}
          />
          <p className="small dim">
            Chords go in brackets right before the syllable: <span className="mono">Amaz[G]ing [C]grace</span>.
            Sections: <span className="mono">{"{start_of_chorus}"}</span> … <span className="mono">{"{end_of_chorus}"}</span>, or a line like <span className="mono">Verse 2:</span>.
          </p>
        </div>
        <div>
          <div className="small dim" style={{ marginBottom: 8 }}>Preview</div>
          <div className="preview-pane">
            {parsed.sections.length ? (
              <ChartView sections={parsed.sections} songKey={draft.song_key || detectedKey} transpose={0} capo={0}
                showChords nashville={false} columns={1} fontScale={settings.fontScale * 0.85} />
            ) : <div className="dim">Your chart will appear here.</div>}
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="row" style={{ marginBottom: 10 }}>
          <h2 className="grow" style={{ fontSize: "1.1rem" }}>Attachments</h2>
          <label className="btn small">
            <IconFile size={18} /> Add PDF / image
            <input type="file" accept="application/pdf,image/*" hidden onChange={async (e) => {
              const f = e.target.files?.[0];
              if (f) await addSongFile(draft.id, f);
              e.target.value = "";
            }} />
          </label>
          <label className="btn small">
            <IconMusic size={18} /> Add backing track
            <input type="file" accept="audio/*,.mp3,.m4a,.wav,.aac" hidden onChange={async (e) => {
              const f = e.target.files?.[0];
              if (f) await addSongFile(draft.id, f);
              e.target.value = "";
            }} />
          </label>
        </div>
        {files?.length ? (
          <ul className="list">
            {files.map((f) => (
              <li key={f.id} className="list-item">
                {f.kind === "audio" ? <IconMusic /> : <IconFile />}
                <div className="grow">
                  <div className="truncate">{f.name}</div>
                  <div className="small dim">
                    {f.kind} · {f.size ? `${(f.size / 1048576).toFixed(1)} MB` : ""}{" "}
                    {failedFiles?.has(f.id)
                      ? <span style={{ color: "var(--danger)" }}>· upload failed — remove it and attach the file again</span>
                      : f.storage_path ? "· synced" : "· waiting to upload"}
                  </div>
                </div>
                <button className="btn small danger" onClick={() => softDelete(db.song_files, f.id)} aria-label="Remove attachment"><IconTrash size={18} /></button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="dim small">Attach a PDF chart (shown when there's no typed chart, or with the PDF button) or a backing track (MP3/M4A) to play along with — both are saved on the device for offline gigs.</p>
        )}
      </div>

      <div className="row" style={{ marginTop: 20 }}>
        <span className="spacer" />
        <button className="btn danger" onClick={remove}><IconTrash size={18} /> Delete song</button>
      </div>
    </div>
  );
}
