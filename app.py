from flask import Flask, request, redirect, url_for, render_template_string, abort, Response, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from datetime import datetime
import os, csv
from io import StringIO, BytesIO
import hashlib, requests  # spotify + demo AI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit
import csv, io
from flask import request
from collections import deque

# --- CSV helpers ---
def parse_mmss_to_seconds(text):
    """Return total seconds from 'mm:ss' (or int seconds). None if blank/invalid."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        if ':' in s:
            mm, ss = s.split(':', 1)
            return int(mm) * 60 + int(ss)
        # allow plain seconds like "185" or "185.0"
        return int(float(s))
    except Exception:
        return None

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
app.url_map.strict_slashes = False

# --- Database config (uses DATABASE_URL if set; else SQLite) ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_SQLITE = "sqlite:///" + os.path.join(BASE_DIR, "musician.db")
db_url = os.getenv("DATABASE_URL", DEFAULT_SQLITE)

# Render/Railway sometimes prefix with postgres:// – SQLAlchemy accepts postgresql://
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# --- Song model ---
class Song(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    artist = db.Column(db.String(200), nullable=False)
    tempo_bpm = db.Column(db.Integer, nullable=True)
    musical_key = db.Column(db.String(20), nullable=True)   # e.g., "C major"
    genre = db.Column(db.String(100), nullable=True)
    tags = db.Column(db.String(500), nullable=True)         # comma-separated
    # NEW: manual duration override stored in seconds
    duration_override_sec = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# --- Setlist + join table ---
class Setlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    event_type = db.Column(db.String(100), nullable=True)
    venue_type = db.Column(db.String(100), nullable=True)
    target_minutes = db.Column(db.Integer, nullable=True)
    notes = db.Column(db.String(1000), nullable=True)
    no_repeat_artists = db.Column(db.Boolean, default=True)  # NEW: avoid duplicate artists in this show
    share_token = db.Column(db.String(64), unique=True, nullable=True)  # secret share link token
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    songs = db.relationship(
        "SetlistSong",
        backref="setlist",
        cascade="all, delete-orphan",
        order_by="SetlistSong.position",
    )

class SetlistSong(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    setlist_id = db.Column(db.Integer, db.ForeignKey("setlist.id"), nullable=False)
    song_id = db.Column(db.Integer, db.ForeignKey("song.id"), nullable=False)
    position = db.Column(db.Integer, nullable=False, default=0)
    notes = db.Column(db.String(500), nullable=True)  # NEW: per-song notes for this setlist row

    song = db.relationship("Song")

# ---- Simple in-memory Undo for setlist ordering (Step 45) ----

# {setlist_id: deque([ [song_id, song_id, ...] ], maxlen=1)}
_UNDO_STACK = {}

def _snapshot_order(setlist_id):
    items = (SetlistSong.query
             .filter_by(setlist_id=setlist_id)
             .order_by(SetlistSong.position.asc())
             .all())
    return [it.song_id for it in items]

def stash_order(setlist_id):
    """Save current order so we can undo once."""
    stack = _UNDO_STACK.setdefault(setlist_id, deque(maxlen=1))
    stack.append(_snapshot_order(setlist_id))

def restore_last_order(setlist_id):
    """Restore most recently stashed order. Returns True if restored."""
    stack = _UNDO_STACK.get(setlist_id)
    if not stack:
        return False
    last = stack.pop()
    # Rebuild positions from the saved song_id sequence
    rows = SetlistSong.query.filter_by(setlist_id=setlist_id).all()
    by_song_id = {r.song_id: r for r in rows}
    pos = 1
    for sid in last:
        row = by_song_id.get(sid)
        if row:
            row.position = pos
            pos += 1
    # Any new songs not in snapshot get appended after
    for r in rows:
        if r.song_id not in last:
            r.position = pos
            pos += 1
    db.session.commit()
    return True

# TEMP convenience routes for testing (we'll auto-stash next step)
@app.get("/setlists/<int:setlist_id>/stash-order")
def stash_order_route(setlist_id):
    stash_order(setlist_id)
    return redirect(url_for("edit_setlist", setlist_id=setlist_id))

@app.get("/setlists/<int:setlist_id>/undo-order")
def undo_order_route(setlist_id):
    ok = restore_last_order(setlist_id)
    flash("Order restored to last change." if ok else "Nothing to undo yet — make a move first.")
    return redirect(url_for("edit_setlist", setlist_id=setlist_id))
# ---- /Undo (Step 45) ----

# --- schema helper (adds column if the DB already exists) ---
def ensure_schema():
    db.create_all()

    # --- Song table retrofits ---
    info = db.session.execute(text("PRAGMA table_info(song);")).fetchall()
    cols = [row[1] for row in info]
    if "duration_override_sec" not in cols:
        db.session.execute(text("ALTER TABLE song ADD COLUMN duration_override_sec INTEGER"))
        db.session.commit()

        # --- Setlist table retrofits ---
    info3 = db.session.execute(text("PRAGMA table_info(setlist);")).fetchall()
    cols3 = [row[1] for row in info3]
    if "no_repeat_artists" not in cols3:
        db.session.execute(text("ALTER TABLE setlist ADD COLUMN no_repeat_artists BOOLEAN DEFAULT 1"))
        db.session.commit()    

    # --- SetlistSong table retrofits ---
    # SQLAlchemy default table name for SetlistSong is "setlist_song"
    sls_table = "setlist_song"

    # Only try to alter if the table already exists (older DBs)
    tables = db.session.execute(
        text("SELECT name FROM sqlite_master WHERE type='table'")
    ).fetchall()
    table_names = {t[0] for t in tables}

    if sls_table in table_names:
        info2 = db.session.execute(text(f"PRAGMA table_info({sls_table});")).fetchall()
        cols2 = [row[1] for row in info2]
        if "notes" not in cols2:
            db.session.execute(text(f"ALTER TABLE {sls_table} ADD COLUMN notes VARCHAR(500)"))
            db.session.commit()
    # If the table didn't exist, db.create_all() above just created it
    # with the current model (which already includes notes).

        # --- Setlist: add share_token if missing
        info4 = db.session.execute(text("PRAGMA table_info(setlist);")).fetchall()
        cols4 = [row[1] for row in info4]
        if "share_token" not in cols4:
            db.session.execute(text("ALTER TABLE setlist ADD COLUMN share_token VARCHAR(64)"))
            db.session.commit()

# --- helpers ---
def normalize_positions(setlist: Setlist):
    """Make positions contiguous starting at 1."""
    rows = (SetlistSong.query
            .filter_by(setlist_id=setlist.id)
            .order_by(SetlistSong.position.asc(), SetlistSong.id.asc())
            .all())
    for idx, row in enumerate(rows, start=1):
        row.position = idx
    db.session.commit()

import secrets

def get_or_create_share_token(sl: Setlist) -> str:
    """Return existing share token or create a new one if missing."""
    if not sl.share_token:
        sl.share_token = secrets.token_hex(16)  # 32-char hex
        db.session.commit()
    return sl.share_token

def estimate_duration_seconds(bpm: int | None) -> int:
    """
    Heuristic: ~210s at 120 BPM, scaled by tempo, clamped to 2:00–6:00.
    """
    baseline_bpm = 120
    baseline_sec = 210
    if not bpm or bpm <= 0:
        return baseline_sec
    est = int(round(baseline_sec * baseline_bpm / bpm))
    return max(120, min(est, 360))

def fmt_mmss(total_sec: int | None) -> str:
    if total_sec is None:
        return ""
    m = total_sec // 60
    s = total_sec % 60
    return f"{m}:{s:02d}"

def parse_mmss(s: str | None) -> int | None:
    """
    Accepts 'mm:ss' (e.g., 3:30). Also accepts '210' as seconds.
    Returns total seconds or None if empty/invalid.
    """
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        if ":" in s:
            m, sec = s.split(":", 1)
            return int(m) * 60 + int(sec)
        # digits only: treat as seconds
        return int(s)
    except ValueError:
        return None
    
def song_duration(song: "Song") -> int:
    """Manual override takes priority; else estimate from BPM."""
    if song.duration_override_sec and song.duration_override_sec > 0:
        return song.duration_override_sec
    return estimate_duration_seconds(song.tempo_bpm)

def select_songs_for_target(all_songs, target_minutes: int | None, vibe: str = "mixed",
                            required_tags: list[str] | None = None,
                            required_genres: list[str] | None = None,
                            avoid_same_artist: bool = True):
    """Greedy fill to (about) target minutes with simple vibe/filters."""
    required_tags = [t.strip().lower() for t in (required_tags or []) if t.strip()]
    required_genres = [g.strip().lower() for g in (required_genres or []) if g.strip()]

    def tags_set(txt: str | None):
        return {t.strip().lower() for t in (txt or "").split(",") if t.strip()}

    def passes_filters(s: "Song"):
        if required_tags and not (tags_set(s.tags) & set(required_tags)):
            return False
        if required_genres:
            sg = (s.genre or "").lower()
            if not any(g in sg for g in required_genres):
                return False
        return True

    candidates = [s for s in all_songs if passes_filters(s)]

    def tempo_of(s): return s.tempo_bpm or 120
    if vibe == "chill":
        candidates.sort(key=tempo_of)                      # low → high
    elif vibe == "energetic":
        candidates.sort(key=tempo_of, reverse=True)        # high → low
    else:  # mixed: bias toward mid-tempo first
        candidates.sort(key=lambda s: abs(tempo_of(s) - 120))

    chosen, total_sec, used_artists = [], 0, set()
    for s in candidates:
        if avoid_same_artist and s.artist.lower() in used_artists:
            continue
        chosen.append(s)
        total_sec += song_duration(s)
        used_artists.add(s.artist.lower())
        if target_minutes and total_sec >= target_minutes * 60:
            break
    return chosen

# --- Demo AI (offline, deterministic) ---
def ai_guess_metadata(title: str, artist: str):
    h = int(hashlib.sha256(f"{title}|{artist}".lower().encode("utf-8")).hexdigest(), 16)
    tempos = range(70, 160)
    keys = [
        "C major","G major","D major","A major","E major","F major","Bb major","Eb major",
        "A minor","E minor","D minor","G minor","C minor","F# minor","Bb minor"
    ]
    genres = ["Pop","Rock","Country","R&B","Hip-Hop","Folk","Jazz","Blues","EDM","Reggae","Latin","Indie"]
    tempo_bpm = list(tempos)[h % len(list(tempos))]
    musical_key = keys[(h >> 8) % len(keys)]
    genre = genres[(h >> 16) % len(genres)]
    tags = []
    tl = title.lower()
    if any(w in tl for w in ("love","heart")): tags.append("love")
    if any(w in tl for w in ("dance","party","club")): tags.append("dance")
    if any(w in tl for w in ("blues","blue")): tags.append("bluesy")
    if any(w in tl for w in ("acoustic","unplugged")): tags.append("acoustic")
    if not tags: tags.append("general")
    eras = ["60s","70s","80s","90s","2000s","2010s","2020s"]
    tags.append(eras[(h >> 24) % len(eras)])
    tags.append("auto")
    return {"tempo_bpm": tempo_bpm, "musical_key": musical_key, "genre": genre, "tags": ", ".join(tags)}

# --- Live lookup via Spotify (optional env vars) ---
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
_PITCH_NAMES = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]

def _spotify_token():
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        print("Spotify: missing SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET; using demo.")
        return None
    try:
        r = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("access_token")
        print("Spotify token error:", r.text)
    except Exception as e:
        print("Spotify token exception:", e)
    return None

def _audio_key_name(key_num: int, mode: int | None):
    if key_num is None or key_num < 0:
        return None
    root = _PITCH_NAMES[key_num % 12]
    quality = "major" if mode == 1 else "minor"
    return f"{root} {quality}"

def spotify_lookup(title: str, artist: str):
    token = _spotify_token()
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}"}

    def _search(q):
        return requests.get(
            "https://api.spotify.com/v1/search",
            headers=headers,
            params={"q": q, "type": "track", "limit": 1},
            timeout=10,
        )
    r = _search(f'track:{title} artist:{artist}')
    if r.status_code != 200 or not r.json().get("tracks", {}).get("items"):
        r = _search(title)
        if r.status_code != 200 or not r.json().get("tracks", {}).get("items"):
            print("Spotify: no results for", title, artist)
            return None
    track = r.json()["tracks"]["items"][0]
    track_id = track["id"]
    af = requests.get(f"https://api.spotify.com/v1/audio-features/{track_id}", headers=headers, timeout=10)
    tempo_bpm = musical_key = None
    if af.status_code == 200:
        data = af.json()
        tempo = data.get("tempo")
        key_num = data.get("key")
        mode = data.get("mode")
        tempo_bpm = round(tempo) if isinstance(tempo, (int, float)) else None
        musical_key = _audio_key_name(key_num, mode)
    genres_list = []
    try:
        artist_id = track["artists"][0]["id"]
        ar = requests.get(f"https://api.spotify.com/v1/artists/{artist_id}", headers=headers, timeout=10)
        if ar.status_code == 200:
            genres_list = ar.json().get("genres", []) or []
    except Exception as e:
        print("Spotify artist lookup exception:", e)
    genre = (genres_list[0].title() if genres_list else None)
    tags = ", ".join((genres_list[:3] + ["spotify"])) if genres_list else "spotify"
    if not any([tempo_bpm, musical_key, genre, tags]):
        return None
    return {"tempo_bpm": tempo_bpm, "musical_key": musical_key, "genre": genre, "tags": tags}

# --- super simple layout ---
BASE_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Setlist Genie</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 16px; }
    a { text-decoration: none; }
    .btn { display: inline-block; padding: 8px 12px; border: 1px solid #ccc; border-radius: 8px; }
    .btn-danger { color: #a00; border-color: #a00; }
    .row { padding: 10px 0; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .muted { color: #666; font-size: 0.9em; }
    input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 8px; }
    label { font-weight: 600; }
    details { margin-top: 8px; }
    .right { text-align: right; display:flex; gap:6px; align-items:center; }
    form.inline { display: inline; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .section { padding: 10px 0; border-top: 1px solid #eee; margin-top: 10px; }
    .iconbtn { padding: 6px 8px; line-height: 1; }
        /* Reorder mode visuals */
    body.has-reorder .dnd-item { border: 1px dashed #aaa; cursor: grab; }
    body.has-reorder .dnd-item:active { cursor: grabbing; }
        /* Toast */
    .toast {
      position: fixed; left: 50%; bottom: 24px; transform: translateX(-50%);
      background: rgba(0,0,0,0.85); color: #fff; padding: 10px 14px; border-radius: 10px;
      font-size: 14px; opacity: 0; pointer-events: none; transition: opacity .2s ease;
      z-index: 9999;
    }
    .toast.show { opacity: 1; }
  </style>
</head>
<body>
  {% with msgs = get_flashed_messages(with_categories=true) %}
    {% if msgs %}
      <div style="margin-bottom:12px;">
        {% for cat, msg in msgs %}
          <div style="padding:8px 12px; border-radius:8px; margin-bottom:6px; border:1px solid #ddd;">
            {{ msg }}
          </div>
        {% endfor %}
      </div>
    {% endif %}
  {% endwith %}
    <div id="toast" class="toast" role="status" aria-live="polite"></div>
  <script>
    function showToast(msg) {
      var el = document.getElementById('toast');
      if (!el) return;
      el.textContent = msg;
      el.classList.add('show');
      clearTimeout(window.__toastTimer);
      window.__toastTimer = setTimeout(function(){ el.classList.remove('show'); }, 1500);
    }
  </script>
  <h1>Setlist Genie</h1>
  <p>
    <a class="btn" href="{{ url_for('home') }}">Home</a>
    <a class="btn" href="{{ url_for('list_songs') }}">Songs</a>
    <a class="btn" href="{{ url_for('new_song') }}">Add Song</a>
    <a class="btn" href="{{ url_for('list_setlists') }}">Setlists</a>
    <a class="btn" href="{{ url_for('new_setlist') }}">New Setlist</a>
  </p>
  {{ content|safe }}
</body>
</html>
"""

# ---------- SONG PAGES ----------
LIST_HTML = """
<h2>Songs</h2>

<form method="get" action="{{ url_for('list_songs') }}" style="margin: 12px 0;">
  <input name="q" placeholder="Search title, artist, genre, key, tags…" value="{{ q or '' }}" />
     <p style="margin-top:8px;">
    <button class="btn" type="submit">Search</button>
    <a class="btn" href="{{ url_for('list_songs') }}">Clear</a>
    <a class="btn" href="{{ url_for('export_songs_csv', q=q) }}">⬇️ Export CSV</a>
    <form class="inline" method="post" action="{{ url_for('autofill_all_songs', q=q) }}" style="display:inline;">
      <input type="hidden" name="source" value="demo" />
      <button class="btn" type="submit" title="Fill missing tempo/key/genre/tags for all filtered songs"
              onclick="return confirm('Auto-fill missing info for ALL currently filtered songs?');">
        ✨ Autofill (all filtered)
      </button>
    </form>
  </p>
  <p style="margin:8px 0 16px;">
    <a class="btn" href="{{ url_for('songs_import_form') }}">Import CSV</a>
    <a class="btn" href="{{ url_for('songs_template_csv') }}">Download Template</a>
    </p>
</form>
<form method="post" action="{{ url_for('create_song') }}" style="margin: 8px 0 16px;">
  <div class="grid">
    <p><input name="title" placeholder="Quick add — Title" required /></p>
    <p><input name="artist" placeholder="Artist" required /></p>
  </div>
  <p>
    <button class="btn" type="submit">＋ Quick Add</button>
    <span class="muted">Adds with just title & artist (you can edit details later).</span>
  </p>
</form>
{% if songs %}
  {% for s in songs %}
    <div class="row">
      <div>
        <div><strong>{{ s.title }}</strong> — {{ s.artist }}</div>
        <div class="muted">
            {% if s.tempo_bpm %}Tempo: {{ s.tempo_bpm }} BPM · {% endif %}
            {% if s.musical_key %}Key: {{ s.musical_key }} · {% endif %}
            {% if s.genre %}Genre: {{ s.genre }} · {% endif %}
            {% if s.tags %}Tags: {{ s.tags }} · {% endif %}
            {% set dur = fmt_mmss(s.duration_override_sec) if s.duration_override_sec else fmt_mmss(estimate(s.tempo_bpm)) %}
            Duration: {{ dur }}
        </div>
      </div>
            <div class="right">
        <form class="inline" method="post" action="{{ url_for('ai_enrich_song', song_id=s.id) }}">
          <input type="hidden" name="source" value="demo" />
          <button class="btn" type="submit" title="Fill tempo/key/genre/tags (demo)">✨ AI Autofill</button>
        </form>
        <a class="btn" href="{{ url_for('edit_song', song_id=s.id) }}">Edit</a>
        <form class="inline" method="post" action="{{ url_for('delete_song', song_id=s.id) }}" onsubmit="return confirm('Delete this song?');">
          <button class="btn btn-danger" type="submit">Delete</button>
        </form>
      </div>
    </div>
  {% endfor %}
{% else %}
  <p>No songs found. <a href="{{ url_for('new_song') }}">Add one</a>.</p>
{% endif %}
"""

NEW_HTML = """
<h2>Add a Song</h2>
<form method="post" action="{{ url_for('create_song') }}">
  <p><label>Title</label><br/><input name="title" required /></p>
  <p><label>Artist</label><br/><input name="artist" required /></p>
  <details>
    <summary>Optional details (we’ll auto-fill later with AI)</summary>
    <p><label>Tempo (BPM)</label><br/><input name="tempo_bpm" type="number" min="1" /></p>
    <p><label>Key (e.g., C major, G minor)</label><br/><input name="musical_key" /></p>
    <p><label>Genre</label><br/><input name="genre" /></p>
    <p><label>Tags (comma-separated)</label><br/><input name="tags" placeholder="wedding, upbeat, 90s" /></p>
    <p><label>Manual duration (mm:ss)</label><br/><input name="duration_override" placeholder="e.g., 3:30" /></p>
  </details>
  <p><button class="btn" type="submit">Save Song</button></p>
</form>
"""

EDIT_HTML = """
<h2>Edit Song</h2>
<form method="post" action="{{ url_for('update_song', song_id=song.id) }}">
  <p><label>Title</label><br/><input name="title" required value="{{ song.title }}" /></p>
  <p><label>Artist</label><br/><input name="artist" required value="{{ song.artist }}" /></p>
  <details open>
    <summary>Optional details</summary>
    <p><label>Tempo (BPM)</label><br/><input name="tempo_bpm" type="number" min="1" value="{{ song.tempo_bpm or '' }}" /></p>
    <p><label>Key (e.g., C major, G minor)</label><br/><input name="musical_key" value="{{ song.musical_key or '' }}" /></p>
    <p><label>Genre</label><br/><input name="genre" value="{{ song.genre or '' }}" /></p>
    <p><label>Tags (comma-separated)</label><br/><input name="tags" value="{{ song.tags or '' }}" /></p>
    <p><label>Manual duration (mm:ss)</label><br/><input name="duration_override" value="{{ fmt_mmss(song.duration_override_sec) if song.duration_override_sec else '' }}" placeholder="e.g., 3:30" /></p>
  </details>
  <p>
    <button class="btn" type="submit">Update Song</button>
    <a class="btn" href="{{ url_for('list_songs') }}">Cancel</a>
  </p>
</form>

<form method="post" action="{{ url_for('ai_enrich_song', song_id=song.id) }}">
  <p class="muted" style="margin-top:6px;">Auto-fill tempo, key, genre, and tags.</p>
  <p>
    <label>Source</label><br/>
    <select name="source">
      <option value="demo">Demo (local, no internet)</option>
      <option value="spotify">Live (Spotify)</option>
    </select>
  </p>
  <label style="font-weight:400;"><input type="checkbox" name="force" value="1" /> Overwrite existing values</label>
  <p style="margin-top:8px;"><button class="btn" type="submit">Auto-fill</button></p>
</form>
"""

# ---------- SETLIST PAGES ----------
SETLISTS_LIST_HTML = """
<h2>Setlists</h2>
<p><a class="btn" href="{{ url_for('new_setlist') }}">Create New Setlist</a></p>
{% if setlists %}
  {% for sl in setlists %}
    <div class="row">
      <div>
        <div><strong><a href="{{ url_for('edit_setlist', setlist_id=sl.id) }}">{{ sl.name }}</a></strong></div>
        <div class="muted">
          {% if sl.event_type %}Event: {{ sl.event_type }} · {% endif %}
          {% if sl.venue_type %}Venue: {{ sl.venue_type }} · {% endif %}
          {% if sl.target_minutes %}Target: {{ sl.target_minutes }} min · {% endif %}
          {{ (sl.songs|length) }} songs
        </div>
      </div>
           <div class="right">
        <a class="btn" href="{{ url_for('edit_setlist', setlist_id=sl.id) }}">Open</a>
        <a class="btn" href="{{ url_for('view_setlist', setlist_id=sl.id) }}" target="_blank">🔗 Share</a>
        <a class="btn" href="{{ url_for('print_setlist', setlist_id=sl.id) }}" target="_blank">🖨️ Print</a>
        <form class="inline" method="post" action="{{ url_for('duplicate_setlist', setlist_id=sl.id) }}" style="display:inline;">
          <button class="btn" type="submit">📄 Duplicate</button>
        </form>
        <form class="inline" method="post" action="{{ url_for('delete_setlist', setlist_id=sl.id) }}" onsubmit="return confirm('Delete this setlist?');" style="display:inline;">
          <button class="btn btn-danger" type="submit">Delete</button>
        </form>
      </div>
    </div>
  {% endfor %}
{% else %}
  <p>No setlists yet. <a href="{{ url_for('new_setlist') }}">Create one</a>.</p>
{% endif %}
"""

SETLIST_NEW_HTML = """
<h2>New Setlist</h2>
<form method="post" action="{{ url_for('create_setlist') }}">
  <div class="grid">
    <p><label>Name</label><br/><input name="name" required /></p>
    <p><label>Target Minutes</label><br/><input name="target_minutes" type="number" min="1" /></p>
  </div>
  <div class="grid">
    <p><label>Event Type</label><br/><input name="event_type" placeholder="Wedding, Bar, Festival..." /></p>
    <p><label>Venue Type</label><br/><input name="venue_type" placeholder="Indoor, Outdoor..." /></p>
  </div>
  <p><label>Notes</label><br/><textarea name="notes" rows="3" placeholder="Any notes..."></textarea></p>
  <p><button class="btn" type="submit">Create Setlist</button></p>
</form>
"""

SETLIST_EDIT_HTML = """
<h2>Edit Setlist</h2>
<form method="post" action="{{ url_for('update_setlist', setlist_id=setlist.id) }}">
  <div class="grid">
    <p><label>Name</label><br/><input name="name" required value="{{ setlist.name }}" /></p>
    <p><label>Target Minutes</label><br/><input name="target_minutes" type="number" min="1" value="{{ setlist.target_minutes or '' }}" /></p>
  </div>
  <div class="grid">
    <p><label>Event Type</label><br/><input name="event_type" value="{{ setlist.event_type or '' }}" placeholder="Wedding, Bar, Festival..." /></p>
    <p><label>Venue Type</label><br/><input name="venue_type" value="{{ setlist.venue_type or '' }}" placeholder="Indoor, Outdoor..." /></p>
  </div>
    <p><label>Notes</label><br/><textarea name="notes" rows="2" placeholder="Any notes...">{{ setlist.notes or '' }}</textarea></p>
  <p>
    <label style="display:inline-flex; gap:8px; align-items:center;">
      <input type="checkbox" name="no_repeat_artists" value="1" {{ 'checked' if setlist.no_repeat_artists else '' }} />
      Don’t repeat artists across this whole show
    </label>
  </p>
  <p>
    <button class="btn" type="submit">Save Details</button>
  </p>
</form>
<div class="section" style="padding-top:0;">
  <h3>Event Presets</h3>
  <div style="display:flex; gap:8px; flex-wrap:wrap;">
    <!-- Wedding preset -->
    <form class="inline" method="post" action="{{ url_for('update_setlist', setlist_id=setlist.id) }}">
      <input type="hidden" name="name" value="{{ setlist.name }}" />
      <input type="hidden" name="target_minutes" value="120" />
      <input type="hidden" name="event_type" value="Wedding" />
      <input type="hidden" name="venue_type" value="Banquet Hall" />
      <input type="hidden" name="notes" value="{{ setlist.notes or '' }}" />
      <button class="btn" type="submit" title="120 min · Wedding · Banquet Hall">💍 Wedding</button>
    </form>

    <!-- Bar/Night preset -->
    <form class="inline" method="post" action="{{ url_for('update_setlist', setlist_id=setlist.id) }}">
      <input type="hidden" name="name" value="{{ setlist.name }}" />
      <input type="hidden" name="target_minutes" value="90" />
      <input type="hidden" name="event_type" value="Bar/Night" />
      <input type="hidden" name="venue_type" value="Bar" />
      <input type="hidden" name="notes" value="{{ setlist.notes or '' }}" />
      <button class="btn" type="submit" title="90 min · Bar/Night · Bar">🍻 Bar/Night</button>
    </form>

    <!-- Festival preset -->
    <form class="inline" method="post" action="{{ url_for('update_setlist', setlist_id=setlist.id) }}">
      <input type="hidden" name="name" value="{{ setlist.name }}" />
      <input type="hidden" name="target_minutes" value="45" />
      <input type="hidden" name="event_type" value="Festival" />
      <input type="hidden" name="venue_type" value="Outdoor Stage" />
      <input type="hidden" name="notes" value="{{ setlist.notes or '' }}" />
      <button class="btn" type="submit" title="45 min · Festival · Outdoor Stage">🎪 Festival</button>
    </form>

    <!-- Coffeehouse / Chill preset -->
    <form class="inline" method="post" action="{{ url_for('update_setlist', setlist_id=setlist.id) }}">
      <input type="hidden" name="name" value="{{ setlist.name }}" />
      <input type="hidden" name="target_minutes" value="60" />
      <input type="hidden" name="event_type" value="Coffeehouse" />
      <input type="hidden" name="venue_type" value="Cafe" />
      <input type="hidden" name="notes" value="{{ setlist.notes or '' }}" />
      <button class="btn" type="submit" title="60 min · Coffeehouse · Cafe">☕ Coffeehouse</button>
    </form>
  </div>
</div>
<p>
  <a class="btn" href="{{ url_for('export_setlist_csv', setlist_id=setlist.id) }}">⬇️ Download CSV</a>
  <a class="btn" href="{{ url_for('export_setlist_pdf', setlist_id=setlist.id) }}">⬇️ Download PDF</a>
  <a class="btn" href="{{ url_for('print_setlist', setlist_id=setlist.id) }}" target="_blank">🖨️ Print View</a>
  <a class="btn" href="{{ url_for('view_setlist', setlist_id=setlist.id) }}" target="_blank">🔗 Share View</a>
  <a class="btn" href="{{ url_for('create_share_link', setlist_id=setlist.id) }}">🔒 Create/Copy Secret Link</a>
  <a class="btn" href="{{ url_for('rotate_share_link', setlist_id=setlist.id) }}" onclick="return confirm('Rotate (revoke) the existing secret link and create a new one?');">♻️ Rotate Secret Link</a>
  <form class="inline" method="post" action="{{ url_for('duplicate_setlist', setlist_id=setlist.id) }}" style="display:inline;">
    <button class="btn" type="submit">📄 Duplicate</button>
  </form>
  <form class="inline" method="post" action="{{ url_for('clear_setlist', setlist_id=setlist.id) }}" style="display:inline;" onsubmit="return confirm('Clear ALL songs from this setlist?');">
    <button class="btn btn-danger" type="submit">🧹 Clear Setlist</button>
    {% if setlist.share_token %}
  <div class="section" style="margin-top:8px;">
    <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
      <input id="shareLink" readonly
             value="{{ url_for('view_setlist_by_token', token=setlist.share_token, _external=True) }}"
             style="flex:1; min-width:280px; padding:8px; border:1px solid #ccc; border-radius:8px;" />
      <button class="btn" type="button" onclick="(async()=>{try{
  const v=document.getElementById('shareLink').value;
  await navigator.clipboard.writeText(v);
  showToast('Link copied!');
} catch(e){
  // Fallback: select text
  const inp=document.getElementById('shareLink');
  inp.select(); inp.setSelectionRange(0, 99999);
  showToast('Select + copy the link');
}})();">
  📋 Copy
</button>
      <a class="btn btn-danger" href="{{ url_for('disable_share_link', setlist_id=setlist.id) }}"
         onclick="return confirm('Disable this secret link? The current URL will stop working.');">
        🚫 Disable Secret Link
      </a>
    </div>
    <p class="muted" style="margin-top:6px;">Anyone with this link can view your read-only setlist.</p>
  </div>
{% endif %}
  </form>
</p>
<div class="section">
  <h3>Auto-build from your library</h3>
  <form method="post" action="{{ url_for('autobuild_setlist', setlist_id=setlist.id) }}">
    <div class="grid">
      <p>
        <label>Target length (minutes)</label><br/>
        <input name="target_minutes" type="number" min="5" value="{{ setlist.target_minutes or 45 }}" />
      </p>
      <p>
        <label>Vibe</label><br/>
        <select name="vibe">
          <option value="mixed">Mixed</option>
          <option value="chill">Chill</option>
          <option value="energetic">Energetic</option>
        </select>
      </p>
    </div>
    <div class="grid">
      <p>
        <label>Require tags (comma-separated)</label><br/>
        <input name="tags" placeholder="wedding, upbeat" />
      </p>
      <p>
        <label>Require genres (comma-separated)</label><br/>
        <input name="genres" placeholder="pop, rock" />
      </p>
    </div>
    <p>
      <label><input type="checkbox" name="avoid_same_artist" value="1" checked /> Avoid repeating the same artist</label>
      &nbsp;&nbsp;
      <label><input type="checkbox" name="clear_first" value="1" /> Clear current setlist first</label>
    </p>
    <p><button class="btn" type="submit">Auto-build</button></p>
  </form>
</div>

<div class="section">
  <h3>Add songs to this setlist</h3>
    {% if setlist.no_repeat_artists %}
    <p class="muted" style="margin:6px 0 2px;">No-repeat artists is ON — songs by artists already in this show will be skipped.</p>
  {% endif %}
  <form method="get" action="{{ url_for('edit_setlist', setlist_id=setlist.id) }}">
    <input name="q" placeholder="Search your songs…" value="{{ q or '' }}" />
    <p style="margin-top:8px;">
      <button class="btn" type="submit">Search</button>
      <a class="btn" href="{{ url_for('edit_setlist', setlist_id=setlist.id) }}">Clear</a>
    </p>
  </form>

  <form method="post" action="{{ url_for('add_songs_to_setlist', setlist_id=setlist.id) }}">
    {% if songs %}
      {% for s in songs %}
        <div class="row" style="border:none;">
          <label style="display:flex; align-items:center; gap:8px;">
            <input type="checkbox" name="song_ids" value="{{ s.id }}" />
            <span><strong>{{ s.title }}</strong> — {{ s.artist }}</span>
          </label>
          <div class="muted">
            {% if s.musical_key %}Key: {{ s.musical_key }} · {% endif %}
            {% if s.genre %}{{ s.genre }}{% endif %}
          </div>
        </div>
      {% endfor %}
      <p><button class="btn" type="submit">Add Selected</button></p>
    {% else %}
      <p class="muted">No songs match your search.</p>
    {% endif %}
  </form>
</div>

<div class="section">
  <h3>Current order ({{ setlist.songs|length }} songs) · ~Total {{ total_str }}</h3>
    <p><a class="btn" href="{{ url_for('undo_order_route', setlist_id=setlist.id) }}">↩️ Undo Last Change</a></p>
   <p><button class="btn" type="button" id="toggleReorder">🧲 Reorder mode: Off</button></p>
  {% if setlist.songs %}
    {% for ss in setlist.songs %}
      <div class="row dnd-item" data-song-id="{{ ss.song.id }}">
        <div style="flex:1;">
            #{{ ss.position }} — <strong>{{ ss.song.title }}</strong> — {{ ss.song.artist }}
            <div class="muted">
                ~{{ estimates[ss.song.id] }}
                {% if ss.notes %} · <em>Notes:</em> {{ ss.notes }}{% endif %}
            </div>
            <form class="inline" method="post" action="{{ url_for('update_setlist_song_notes', setlist_id=setlist.id, song_id=ss.song.id) }}" style="margin-top:6px; display:flex; gap:6px; align-items:center;">
                <input name="notes" placeholder="capo 2 · start on chorus · stop at 2:45" value="{{ ss.notes or '' }}" style="max-width:420px;" />
                <button class="btn" type="submit" title="Save notes for this song">💾 Save</button>
            </form>
        </div>
        <div class="right">
            <form class="inline" method="post" action="{{ url_for('move_song_top', setlist_id=setlist.id, song_id=ss.song.id) }}">
                <button class="btn iconbtn" title="Move to Top" type="submit">⤒</button>
            </form>
            <form class="inline" method="post" action="{{ url_for('move_song_up', setlist_id=setlist.id, song_id=ss.song.id) }}">
                <button class="btn iconbtn" title="Move Up" type="submit">↑</button>
             </form>
             <form class="inline" method="post" action="{{ url_for('move_song_down', setlist_id=setlist.id, song_id=ss.song.id) }}">
                <button class="btn iconbtn" title="Move Down" type="submit">↓</button>
            </form>
            <form class="inline" method="post" action="{{ url_for('move_song_bottom', setlist_id=setlist.id, song_id=ss.song.id) }}">
                <button class="btn iconbtn" title="Move to Bottom" type="submit">⤓</button>
            </form>
            <form class="inline" method="post" action="{{ url_for('remove_song_from_setlist', setlist_id=setlist.id, song_id=ss.song.id) }}" onsubmit="return confirm('Remove this song from the setlist?');">
                <button class="btn btn-danger" type="submit">Remove</button>
            </form>
        </div>
    </div>

    {% endfor %}
  {% else %}
    <p class="muted">This setlist is empty. Use the form above to add songs or auto-build.</p>
  {% endif %}
</div>
<script>
(function () {
  const rows = Array.from(document.querySelectorAll('.dnd-item'));
  if (!rows.length) return;

  const body = document.body;
  const btn = document.getElementById('toggleReorder');
  let enabled = false;
  let dragging = null;

  function setEnabled(on) {
    enabled = !!on;
    body.classList.toggle('has-reorder', enabled);
    rows.forEach(r => r.draggable = enabled);
    if (btn) btn.textContent = enabled ? '🧲 Reorder mode: On' : '🧲 Reorder mode: Off';
  }

  if (btn) {
    btn.addEventListener('click', () => setEnabled(!enabled));
  }
  // Start disabled
  setEnabled(false);

  // Make inputs inside rows not start drags
  rows.forEach(row => {
    row.querySelectorAll('input, textarea, button, select, a').forEach(el => {
      el.addEventListener('mousedown', e => e.stopPropagation());
      el.addEventListener('touchstart', e => e.stopPropagation(), {passive:true});
      el.addEventListener('dragstart', e => e.preventDefault());
    });
  });

  rows.forEach(row => {
    row.addEventListener('dragstart', e => {
      if (!enabled) { e.preventDefault(); return; }
      dragging = row;
      e.dataTransfer.effectAllowed = 'move';
      row.style.opacity = '0.6';
    });

    row.addEventListener('dragend', () => {
      if (dragging) dragging.style.opacity = '';
      dragging = null;
    });

    row.addEventListener('dragover', e => {
      if (!enabled) return;
      e.preventDefault(); // allow drop
      if (!dragging || dragging === row) return;
      const rect = row.getBoundingClientRect();
      const before = (e.clientY - rect.top) < rect.height / 2;
      const parent = row.parentNode;
      if (before) parent.insertBefore(dragging, row);
      else parent.insertBefore(dragging, row.nextSibling);
    });

    row.addEventListener('drop', async e => {
      if (!enabled) return;
      e.preventDefault();

      // Build new order from current DOM order
      const order = Array.from(document.querySelectorAll('.dnd-item'))
        .map(el => el.dataset.songId)
        .join(',');

      const form = new FormData();
      form.append('order', order);
      try {
        await fetch('{{ url_for("reorder_setlist", setlist_id=setlist.id) }}', {
          method: 'POST',
          body: form
        });
      } catch (err) {
        console.error('Reorder failed', err);
      }
      location.reload();
    });
  });
})();
</script>
"""

# --- routes: Home ---
@app.get("/")
def home():
    inner = """
    <h2>Welcome</h2>
    <p>This is your local Setlist Genie dev app. Use the buttons above to manage songs and setlists.</p>
    """
    return render_template_string(BASE_HTML, content=inner)

# --- routes: Songs ---
@app.get("/songs")
def list_songs():
    q = request.args.get("q", "").strip()
    query = Song.query
    if q:
        like = f"%{q}%"
        query = query.filter(
            db.or_(
                Song.title.ilike(like),
                Song.artist.ilike(like),
                Song.genre.ilike(like),
                Song.musical_key.ilike(like),
                Song.tags.ilike(like),
            )
        )
    songs = query.order_by(Song.created_at.desc()).all()
    inner = render_template_string(LIST_HTML, songs=songs, q=q, fmt_mmss=fmt_mmss, estimate=estimate_duration_seconds)
    return render_template_string(BASE_HTML, content=inner)

@app.get("/songs/new")
def new_song():
    inner = render_template_string(NEW_HTML)
    return render_template_string(BASE_HTML, content=inner)

@app.post("/songs")
def create_song():
    title = request.form.get("title", "").strip()
    artist = request.form.get("artist", "").strip()
    tempo_bpm = request.form.get("tempo_bpm", "").strip() or None
    musical_key = request.form.get("musical_key", "").strip() or None
    genre = request.form.get("genre", "").strip() or None
    tags = request.form.get("tags", "").strip() or None
    duration_override = parse_mmss(request.form.get("duration_override"))

    if not title or not artist:
        return "Title and Artist are required.", 400

    song = Song(
        title=title,
        artist=artist,
        tempo_bpm=int(tempo_bpm) if tempo_bpm else None,
        musical_key=musical_key,
        genre=genre,
        tags=tags,
        duration_override_sec=duration_override,
    )
    db.session.add(song)
    db.session.commit()
    return redirect(url_for("list_songs"))

@app.get("/songs/<int:song_id>/edit")
def edit_song(song_id):
    song = Song.query.get_or_404(song_id)
    inner = render_template_string(EDIT_HTML, song=song, fmt_mmss=fmt_mmss)
    return render_template_string(BASE_HTML, content=inner)

@app.post("/songs/<int:song_id>")
def update_song(song_id):
    song = Song.query.get_or_404(song_id)
    title = request.form.get("title", "").strip()
    artist = request.form.get("artist", "").strip()
    tempo_bpm = request.form.get("tempo_bpm", "").strip() or None
    musical_key = request.form.get("musical_key", "").strip() or None
    genre = request.form.get("genre", "").strip() or None
    tags = request.form.get("tags", "").strip() or None
    duration_override = parse_mmss(request.form.get("duration_override"))

    if not title or not artist:
        return "Title and Artist are required.", 400

    song.title = title
    song.artist = artist
    song.tempo_bpm = int(tempo_bpm) if tempo_bpm else None
    song.musical_key = musical_key
    song.genre = genre
    song.tags = tags
    song.duration_override_sec = duration_override
    db.session.commit()
    return redirect(url_for("list_songs"))

@app.post("/songs/<int:song_id>/delete")
def delete_song(song_id):
    song = Song.query.get_or_404(song_id)
    db.session.delete(song)
    db.session.commit()
    return redirect(url_for("list_songs"))

@app.post("/songs/<int:song_id>/ai")
def ai_enrich_song(song_id):
    song = Song.query.get_or_404(song_id)
    force = request.form.get("force") == "1"
    source = (request.form.get("source") or "demo").lower()

    meta = None
    if source == "spotify":
        meta = spotify_lookup(song.title, song.artist)
        if not meta:
            print("Spotify lookup failed or not configured; falling back to demo.")
    if not meta:
        meta = ai_guess_metadata(song.title, song.artist)

    if force or song.tempo_bpm is None:
        if meta.get("tempo_bpm") is not None:
            song.tempo_bpm = meta["tempo_bpm"]
    if force or not song.musical_key:
        if meta.get("musical_key"):
            song.musical_key = meta["musical_key"]
    if force or not song.genre:
        if meta.get("genre"):
            song.genre = meta["genre"]
    if force or not song.tags:
        if meta.get("tags"):
            song.tags = meta["tags"]

    db.session.commit()
    return redirect(url_for("edit_song", song_id=song.id))

@app.route("/songs/import", methods=["GET"])
def songs_import_form():
    # Simple upload form for CSV import
    return """
    <h1>Import Songs (CSV)</h1>
    <form action="/songs/import" method="post" enctype="multipart/form-data">
      <p><input type="file" name="file" accept=".csv" required></p>
      <p><button type="submit">Upload CSV</button></p>
    </form>
    <p><strong>Expected columns</strong> (header row): 
       title, artist, tempo_bpm, musical_key, genre, tags, duration
       <br>(<em>duration</em> can be <code>mm:ss</code> or plain seconds)
    </p>
    <p><a href="/songs">Back to songs</a></p>
    """

@app.route("/songs/import", methods=["POST"])
def songs_import_post():
    # Handle CSV upload and create/update songs
    file = request.files.get("file")
    if not file or file.filename == "":
        return "No file uploaded.", 400

    try:
        raw = file.read()
        text = raw.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
    except Exception as e:
        return f"Failed to read CSV: {e}", 400

    created = 0
    updated = 0
    skipped = 0
    errors = []

    for i, row in enumerate(reader, start=2):  # header is line 1
        title = (row.get("title") or "").strip()
        artist = (row.get("artist") or "").strip()
        if not title or not artist:
            skipped += 1
            errors.append(f"Row {i}: missing title or artist")
            continue

        tempo_str = (row.get("tempo_bpm") or "").strip()
        try:
            tempo_bpm = int(tempo_str) if tempo_str else None
        except ValueError:
            tempo_bpm = None
            errors.append(f"Row {i}: invalid tempo_bpm '{tempo_str}'")

        musical_key = (row.get("musical_key") or "").strip() or None
        genre = (row.get("genre") or "").strip() or None
        tags = (row.get("tags") or "").strip() or None
        duration = (row.get("duration") or "").strip()
        duration_override_sec = parse_mmss_to_seconds(duration) if duration else None

        # Upsert by (title, artist)
        existing = Song.query.filter_by(title=title, artist=artist).first()
        if existing:
            if tempo_bpm is not None:
                existing.tempo_bpm = tempo_bpm
            if musical_key is not None:
                existing.musical_key = musical_key
            if genre is not None:
                existing.genre = genre
            if tags is not None:
                existing.tags = tags
            if duration_override_sec is not None:
                existing.duration_override_sec = duration_override_sec
            updated += 1
        else:
            s = Song(
                title=title,
                artist=artist,
                tempo_bpm=tempo_bpm,
                musical_key=musical_key,
                genre=genre,
                tags=tags,
                duration_override_sec=duration_override_sec,
            )
            db.session.add(s)
            created += 1

    db.session.commit()

    # Pretty result page
    notes = ""
    if errors:
        notes = "<h3>Notes</h3><ul>" + "".join(f"<li>{e}</li>" for e in errors) + "</ul>"

    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Import Complete</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 2rem; }}
        .card {{ max-width: 720px; border: 1px solid #ddd; border-radius: 12px; padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.04); }}
        h1 {{ margin-top: 0; }}
        .stats {{ font-size: 1.1rem; margin: 0.5rem 0 1rem; }}
        .btns a {{ display: inline-block; margin-right: .5rem; margin-bottom: .5rem; padding: .6rem .9rem; border-radius: 8px; text-decoration: none; border: 1px solid #ccc; }}
        .primary {{ background: #f5f5f5; }}
        ul {{ line-height: 1.5; }}
      </style>
    </head>
    <body>
      <div class="card">
        <h1>Import Complete</h1>
        <div class="stats">
          <strong>Created:</strong> {created} &nbsp;|&nbsp;
          <strong>Updated:</strong> {updated} &nbsp;|&nbsp;
          <strong>Skipped:</strong> {skipped}
        </div>
        {notes}
        <div class="btns">
          <a class="primary" href="/songs">Back to Songs</a>
          <a href="/songs/import">Import More</a>
          <a href="/songs/template.csv">Download Template CSV</a>
        </div>
      </div>
    </body>
    </html>
    """

@app.route("/songs/template.csv")
def songs_template_csv():
    """Downloadable blank template for imports."""
    sample = (
        "title,artist,tempo_bpm,musical_key,genre,tags,duration\n"
        "Imagine,John Lennon,75,C major,Pop,classic; mellow,03:10\n"
        "Brown Eyed Girl,Van Morrison,148,G major,Classic Rock,upbeat; singalong,03:05\n"
    )
    return (
        sample,
        200,
        {
            "Content-Type": "text/csv",
            "Content-Disposition": 'attachment; filename="songs_template.csv"',
        },
    )

@app.get("/songs/export.csv")
def export_songs_csv():
    # optional search filter, same as list_songs()
    q = request.args.get("q", "").strip()
    query = Song.query
    if q:
        like = f"%{q}%"
        query = query.filter(
            db.or_(
                Song.title.ilike(like),
                Song.artist.ilike(like),
                Song.genre.ilike(like),
                Song.musical_key.ilike(like),
                Song.tags.ilike(like),
            )
        )
    songs = query.order_by(Song.created_at.desc()).all()

    # build CSV
    out = StringIO()
    w = csv.writer(out)
    w.writerow(["Title", "Artist", "Tempo (BPM)", "Key", "Genre", "Tags", "Duration (mm:ss)"])
    for s in songs:
        dur_sec = s.duration_override_sec if s.duration_override_sec else estimate_duration_seconds(s.tempo_bpm)
        w.writerow([
            s.title,
            s.artist,
            s.tempo_bpm or "",
            s.musical_key or "",
            s.genre or "",
            s.tags or "",
            fmt_mmss(dur_sec),
        ])

    resp = Response(out.getvalue(), mimetype="text/csv")
    resp.headers["Content-Disposition"] = 'attachment; filename="songs_export.csv"'
    return resp

@app.post("/songs/autofill_all")
def autofill_all_songs():
    # respect the same search filter as /songs
    q = request.args.get("q", "").strip()
    source = (request.form.get("source") or "demo").lower()

    query = Song.query
    if q:
        like = f"%{q}%"
        query = query.filter(
            db.or_(
                Song.title.ilike(like),
                Song.artist.ilike(like),
                Song.genre.ilike(like),
                Song.musical_key.ilike(like),
                Song.tags.ilike(like),
            )
        )
    songs = query.order_by(Song.created_at.desc()).all()

    updated = 0
    for s in songs:
        meta = None
        if source == "spotify":
            meta = spotify_lookup(s.title, s.artist)
        if not meta:
            meta = ai_guess_metadata(s.title, s.artist)

        # only fill fields that are currently missing
        changed = False
        if s.tempo_bpm is None and meta.get("tempo_bpm") is not None:
            s.tempo_bpm = meta["tempo_bpm"]; changed = True
        if not s.musical_key and meta.get("musical_key"):
            s.musical_key = meta["musical_key"]; changed = True
        if not s.genre and meta.get("genre"):
            s.genre = meta["genre"]; changed = True
        if not s.tags and meta.get("tags"):
            s.tags = meta["tags"]; changed = True

        if changed:
            updated += 1

    if updated:
        db.session.commit()

    # back to the songs page, preserving the search
    return redirect(url_for("list_songs", q=q))

# --- routes: Setlists ---
@app.post("/setlists/<int:setlist_id>/autobuild")
def autobuild_setlist(setlist_id):
    setlist = Setlist.query.get_or_404(setlist_id)

    target_minutes = request.form.get("target_minutes", "").strip()
    vibe = (request.form.get("vibe") or "mixed").lower()
    req_tags = [t.strip() for t in (request.form.get("tags") or "").split(",") if t.strip()]
    req_genres = [g.strip() for g in (request.form.get("genres") or "").split(",") if g.strip()]
    avoid_same_artist = request.form.get("avoid_same_artist") == "1"
    clear_first = request.form.get("clear_first") == "1"
    
        # If the setlist's toggle is on, force no repeats regardless of the form checkbox
    if setlist.no_repeat_artists:
        avoid_same_artist = True


    # pool: all songs
    all_songs = Song.query.order_by(Song.created_at.desc()).all()

    # maybe clear existing rows first
    if clear_first:
        SetlistSong.query.filter_by(setlist_id=setlist.id).delete()
        db.session.commit()

    # if not clearing, avoid repeating songs already in this setlist
    existing_song_ids = {ss.song_id for ss in setlist.songs}
    selected = select_songs_for_target(
        all_songs=all_songs,
        target_minutes=int(target_minutes) if target_minutes else (setlist.target_minutes or 45),
        vibe=vibe,
        required_tags=req_tags,
        required_genres=req_genres,
        avoid_same_artist=avoid_same_artist,
    )
    if not clear_first:
        selected = [s for s in selected if s.id not in existing_song_ids]
        if avoid_same_artist:
            existing_artists = {ss.song.artist.lower() for ss in setlist.songs}
            selected = [s for s in selected if s.artist.lower() not in existing_artists]

    # append to setlist
    current_max = db.session.query(db.func.max(SetlistSong.position)).filter_by(setlist_id=setlist.id).scalar() or 0
    pos = current_max
    for s in selected:
        pos += 1
        db.session.add(SetlistSong(setlist_id=setlist.id, song_id=s.id, position=pos))
    db.session.commit()
    normalize_positions(setlist)
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

@app.get("/setlists")
def list_setlists():
    setlists = Setlist.query.order_by(Setlist.created_at.desc()).all()
    inner = render_template_string(SETLISTS_LIST_HTML, setlists=setlists)
    return render_template_string(BASE_HTML, content=inner)

@app.get("/s/<token>")
def view_setlist_by_token(token):
    sl = Setlist.query.filter_by(share_token=token).first_or_404()
    # Reuse your existing builder from view_setlist:
    rows = []
    total_sec = 0
    for ss in sl.songs:
        dur_sec = ss.song.duration_override_sec or estimate_duration_seconds(ss.song.tempo_bpm)
        total_sec += dur_sec
        rows.append({
            "pos": ss.position,
            "title": ss.song.title,
            "artist": ss.song.artist,
            "key": ss.song.musical_key or "",
            "bpm": ss.song.tempo_bpm or "",
            "dur": fmt_mmss(dur_sec),
            "notes": getattr(ss, "notes", "") or "",
        })

    rows_html = []
    for r in rows:
        rows_html.append(f"""
        <div class="row">
          <div>
            <strong>#{r['pos']} — {r['title']}</strong> — {r['artist']}
            <div class="muted">
              {('Key: ' + r['key'] + ' · ') if r['key'] else ''}
              {('BPM: ' + str(r['bpm']) + ' · ') if r['bpm'] else ''}
              Dur: {r['dur']}
            </div>
            {('<div class="muted"><em>Notes:</em> ' + r['notes'] + '</div>') if r['notes'] else ''}
          </div>
        </div>
        """)

    meta_bits = [p for p in [
        f"Event: {sl.event_type}" if sl.event_type else "",
        f"Venue: {sl.venue_type}" if sl.venue_type else "",
        f"Target: {sl.target_minutes} min" if sl.target_minutes else "",
        f"Songs: {len(rows)}",
        f"Approx Total: {fmt_mmss(total_sec)}"
    ] if p]

    inner = f"""
    <h2>{sl.name}</h2>
    <p class="muted">{' · '.join(meta_bits)}</p>
    <div class="section" style="padding-top:0;">
      {''.join(rows_html) or "<p class='muted'>This setlist is empty.</p>"}
    </div>
    """
    return render_template_string(BASE_HTML, content=inner)

@app.get("/setlists/new")
def new_setlist():
    inner = render_template_string(SETLIST_NEW_HTML)
    return render_template_string(BASE_HTML, content=inner)

@app.post("/setlists")
def create_setlist():
    name = request.form.get("name", "").strip()
    if not name:
        return "Name is required.", 400
    target_minutes = request.form.get("target_minutes", "").strip() or None
    event_type = request.form.get("event_type", "").strip() or None
    venue_type = request.form.get("venue_type", "").strip() or None
    notes = request.form.get("notes", "").strip() or None

    sl = Setlist(
        name=name,
        target_minutes=int(target_minutes) if target_minutes else None,
        event_type=event_type,
        venue_type=venue_type,
        notes=notes,
    )
    db.session.add(sl)
    db.session.commit()
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.get("/setlists/<int:setlist_id>/edit")
def edit_setlist(setlist_id):
    setlist = Setlist.query.get_or_404(setlist_id)

    # search pool for the “Add songs” section
    q = request.args.get("q", "").strip()
    query = Song.query
    if q:
        like = f"%{q}%"
        query = query.filter(
            db.or_(
                Song.title.ilike(like),
                Song.artist.ilike(like),
                Song.genre.ilike(like),
                Song.musical_key.ilike(like),
                Song.tags.ilike(like),
            )
        )
    songs = query.order_by(Song.created_at.desc()).all()

    # durations (manual override first, else BPM estimate)
    estimates = {}
    total_sec = 0
    for ss in setlist.songs:
        if ss.song.duration_override_sec and ss.song.duration_override_sec > 0:
            seconds = ss.song.duration_override_sec
        else:
            seconds = estimate_duration_seconds(ss.song.tempo_bpm)
        estimates[ss.song.id] = fmt_mmss(seconds)
        total_sec += seconds

    inner = render_template_string(
        SETLIST_EDIT_HTML,
        setlist=setlist,
        songs=songs,
        q=q,
        estimates=estimates,
        total_str=fmt_mmss(total_sec),
    )
    return render_template_string(BASE_HTML, content=inner)

@app.post("/setlists/<int:setlist_id>/add_songs")
def add_songs_to_setlist(setlist_id):
    setlist = Setlist.query.get_or_404(setlist_id)
    ids = request.form.getlist("song_ids")
    if not ids:
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))

    # Enforce no-repeat-artists if enabled
    no_repeat = bool(getattr(setlist, "no_repeat_artists", False))
    existing_artists = {ss.song.artist.lower() for ss in setlist.songs} if no_repeat else set()

    current_max = db.session.query(db.func.max(SetlistSong.position)).filter_by(setlist_id=setlist.id).scalar() or 0
    pos = current_max

    added = 0
    skipped_existing = 0
    skipped_artist_dup = 0
    skipped_missing = 0

    for raw_id in ids:
        try:
            sid = int(raw_id)
        except (TypeError, ValueError):
            skipped_missing += 1
            continue

        if SetlistSong.query.filter_by(setlist_id=setlist.id, song_id=sid).first():
            skipped_existing += 1
            continue

        song = Song.query.get(sid)
        if not song:
            skipped_missing += 1
            continue

        if no_repeat and song.artist and song.artist.lower() in existing_artists:
            skipped_artist_dup += 1
            continue

        pos += 1
        db.session.add(SetlistSong(setlist_id=setlist.id, song_id=sid, position=pos))
        added += 1
        if no_repeat and song.artist:
            existing_artists.add(song.artist.lower())

    if added:
        db.session.commit()

    # Build a concise message
    parts = [f"Added {added}"]
    if skipped_existing:
        parts.append(f"skipped {skipped_existing} (already in setlist)")
    if skipped_artist_dup and no_repeat:
        parts.append(f"skipped {skipped_artist_dup} (duplicate artist)")
    if skipped_missing:
        parts.append(f"skipped {skipped_missing} (not found)")

    flash(" · ".join(parts) if parts else "No changes.")
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

@app.post("/setlists/<int:setlist_id>/remove_song/<int:song_id>")
def remove_song_from_setlist(setlist_id, song_id):
    setlist = Setlist.query.get_or_404(setlist_id)
    row = SetlistSong.query.filter_by(setlist_id=setlist.id, song_id=song_id).first()
    if row:
        db.session.delete(row)
        db.session.commit()
        normalize_positions(setlist)
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

@app.post("/setlists/<int:setlist_id>/notes/<int:song_id>")
def update_setlist_song_notes(setlist_id, song_id):
    setlist = Setlist.query.get_or_404(setlist_id)
    row = SetlistSong.query.filter_by(setlist_id=setlist.id, song_id=song_id).first_or_404()
    row.notes = (request.form.get("notes") or "").strip() or None
    db.session.commit()
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

@app.post("/setlists/<int:setlist_id>/move/<int:song_id>/up")
def move_song_up(setlist_id, song_id):
    setlist = Setlist.query.get_or_404(setlist_id) 
    stash_order(setlist.id)
    row = SetlistSong.query.filter_by(setlist_id=setlist.id, song_id=song_id).first_or_404()
    if row.position <= 1:
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))
    above = (SetlistSong.query
             .filter(SetlistSong.setlist_id == setlist.id, SetlistSong.position < row.position)
             .order_by(SetlistSong.position.desc())
             .first())
    if above:
        row.position, above.position = above.position, row.position
        db.session.commit()
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

@app.post("/setlists/<int:setlist_id>/move/<int:song_id>/down")
def move_song_down(setlist_id, song_id):
    setlist = Setlist.query.get_or_404(setlist_id) 
    stash_order(setlist.id)
    row = SetlistSong.query.filter_by(setlist_id=setlist.id, song_id=song_id).first_or_404()
    below = (SetlistSong.query
             .filter(SetlistSong.setlist_id == setlist.id, SetlistSong.position > row.position)
             .order_by(SetlistSong.position.asc())
             .first())
    if below:
        row.position, below.position = below.position, row.position
        db.session.commit()
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

@app.post("/setlists/<int:setlist_id>/delete")
def delete_setlist(setlist_id):
    setlist = Setlist.query.get_or_404(setlist_id)
    db.session.delete(setlist)
    db.session.commit()
    return redirect(url_for("list_setlists"))

@app.get("/setlists/<int:setlist_id>/print")
def print_setlist(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)

    # build simple rows with durations (override first, else estimate)
    rows = []
    total_sec = 0
    for ss in sl.songs:
        dur_sec = ss.song.duration_override_sec or estimate_duration_seconds(ss.song.tempo_bpm)
        total_sec += dur_sec
        rows.append({
            "pos": ss.position,
            "title": ss.song.title,
            "artist": ss.song.artist,
            "key": ss.song.musical_key or "",
            "bpm": ss.song.tempo_bpm or "",
            "dur": fmt_mmss(dur_sec),
            "notes": ss.notes or "",
        })

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>{sl.name} — Print View</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 24px; }}
        h1 {{ margin: 0 0 6px; }}
        .meta {{ color:#555; margin-bottom:16px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border-bottom: 1px solid #eee; padding: 8px 6px; text-align:left; }}
        th {{ background:#fafafa; }}
        @media print {{
          .noprint {{ display:none; }}
          body {{ margin: 8px; }}
        }}
      </style>
    </head>
    <body>
      <div class="noprint" style="margin-bottom:12px;">
        <a href="{url_for('edit_setlist', setlist_id=sl.id)}">← Back</a>
        &nbsp;·&nbsp;
        <a href="{url_for('export_setlist_pdf', setlist_id=sl.id)}">Download PDF</a>
      </div>
      <h1>{sl.name}</h1>
      <div class="meta">
        {" · ".join([p for p in [
          f"Event: {sl.event_type}" if sl.event_type else "",
          f"Venue: {sl.venue_type}" if sl.venue_type else "",
          f"Target: {sl.target_minutes} min" if sl.target_minutes else "",
          f"Songs: {len(rows)}",
          f"Approx Total: {fmt_mmss(total_sec)}"
        ] if p])}
      </div>

      <table>
        <thead>
          <tr><th>#</th><th>Song</th><th>Artist</th><th>Key</th><th>BPM</th><th>Dur</th><th>Notes</th></tr>
        </thead>
        <tbody>
          {''.join(f"<tr><td>{r['pos']}</td><td>{r['title']}</td><td>{r['artist']}</td><td>{r['key']}</td><td>{r['bpm']}</td><td>{r['dur']}</td><td>{r['notes']}</td></tr>" for r in rows) or "<tr><td colspan='7'>No songs yet.</td></tr>"}
        </tbody>
      </table>
    </body>
    </html>
    """
    return html

@app.get("/setlists/<int:setlist_id>")
def view_setlist(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)

    # build rows + total
    rows = []
    total_sec = 0
    for ss in sl.songs:
        dur_sec = ss.song.duration_override_sec or estimate_duration_seconds(ss.song.tempo_bpm)
        total_sec += dur_sec
        rows.append({
    "pos": ss.position,
    "title": ss.song.title,
    "artist": ss.song.artist,
    "key": ss.song.musical_key or "",
    "bpm": ss.song.tempo_bpm or "",
    "dur": fmt_mmss(dur_sec),
    "notes": ss.notes or "",
})

    # build the rows HTML safely using single quotes inside f-strings
    rows_html = []
    for r in rows:
        rows_html.append(f"""
        <div class="row">
          <div>
            <strong>#{r['pos']} — {r['title']}</strong> — {r['artist']}
            <div class="muted">
              {('Key: ' + r['key'] + ' · ') if r['key'] else ''}
              {('BPM: ' + str(r['bpm']) + ' · ') if r['bpm'] else ''}
              Dur: {r['dur']}
            </div>
            {('<div class="muted"><em>Notes:</em> ' + r['notes'] + '</div>') if r['notes'] else ''}
          </div>
        </div>
        """)

    meta_bits = [p for p in [
        f"Event: {sl.event_type}" if sl.event_type else "",
        f"Venue: {sl.venue_type}" if sl.venue_type else "",
        f"Target: {sl.target_minutes} min" if sl.target_minutes else "",
        f"Songs: {len(rows)}",
        f"Approx Total: {fmt_mmss(total_sec)}"
    ] if p]

    inner = f"""
    <h2>{sl.name}</h2>
    <p class="muted">{' · '.join(meta_bits)}</p>

    <div class="section" style="padding-top:0;">
      {''.join(rows_html) or "<p class='muted'>This setlist is empty.</p>"}
    </div>

    <p>
      <a class="btn" href="{url_for('print_setlist', setlist_id=sl.id)}" target="_blank">🖨️ Print View</a>
      <a class="btn" href="{url_for('edit_setlist', setlist_id=sl.id)}">Edit Setlist</a>
    </p>
    """
    return render_template_string(BASE_HTML, content=inner)

@app.post("/setlists/<int:setlist_id>/duplicate")
def duplicate_setlist(setlist_id):
    orig = Setlist.query.get_or_404(setlist_id)

    # create clone of header fields
    new = Setlist(
        name=f"{orig.name} (Copy)",
        event_type=orig.event_type,
        venue_type=orig.venue_type,
        target_minutes=orig.target_minutes,
        notes=orig.notes,
    )
    db.session.add(new)
    db.session.flush()  # get new.id before inserting songs

    # copy songs with same order
    for ss in orig.songs:
        db.session.add(SetlistSong(setlist_id=new.id, song_id=ss.song_id, position=ss.position))

    db.session.commit()
    return redirect(url_for("edit_setlist", setlist_id=new.id))

@app.post("/setlists/<int:setlist_id>/clear")
def clear_setlist(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    SetlistSong.query.filter_by(setlist_id=sl.id).delete()
    db.session.commit()
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.post("/setlists/<int:setlist_id>/update")
def update_setlist(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    sl.name = (request.form.get("name") or sl.name).strip()
    tm = (request.form.get("target_minutes") or "").strip()
    sl.target_minutes = int(tm) if tm else None
    sl.event_type = (request.form.get("event_type") or "").strip() or None
    sl.venue_type = (request.form.get("venue_type") or "").strip() or None
    sl.notes = (request.form.get("notes") or "").strip() or None
    sl.no_repeat_artists = (request.form.get("no_repeat_artists") == "1")
    db.session.commit()
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.post("/setlists/<int:setlist_id>/move/<int:song_id>/top")
def move_song_top(setlist_id, song_id):
    setlist = Setlist.query.get_or_404(setlist_id) 
    stash_order(setlist.id)
    row = SetlistSong.query.filter_by(setlist_id=setlist.id, song_id=song_id).first_or_404()
    row.position = 0  # temporarily smallest
    db.session.commit()
    normalize_positions(setlist)
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

@app.post("/setlists/<int:setlist_id>/move/<int:song_id>/bottom")
def move_song_bottom(setlist_id, song_id):
    setlist = Setlist.query.get_or_404(setlist_id) 
    stash_order(setlist.id)
    row = SetlistSong.query.filter_by(setlist_id=setlist.id, song_id=song_id).first_or_404()
    maxpos = db.session.query(db.func.max(SetlistSong.position)).filter_by(setlist_id=setlist.id).scalar() or 0
    row.position = maxpos + 1
    db.session.commit()
    normalize_positions(setlist)
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

@app.route("/setlists/<int:setlist_id>/reorder", methods=["POST", "GET"])
def reorder_setlist(setlist_id):
    """
    Reorder songs by a comma-separated list of song IDs in their NEW order.
    Accepts:
      - POST form field 'order'
      - or GET query param ?order=1,5,3,...
    """
    setlist = Setlist.query.get_or_404(setlist_id)

    # Read order string
    order_str = (request.form.get("order") or request.args.get("order") or "").strip()
    if not order_str:
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))

    # Parse IDs and keep only those actually in this setlist
    try:
        new_ids = [int(x) for x in order_str.split(",") if x.strip()]
    except ValueError:
        # Ignore bad input
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))

    # Build a map of current rows in this setlist
    rows = (SetlistSong.query
            .filter_by(setlist_id=setlist.id)
            .order_by(SetlistSong.position.asc())
            .all())
    by_song_id = {r.song_id: r for r in rows}

    # Filter to valid song_ids
    new_ids = [sid for sid in new_ids if sid in by_song_id]

    if not new_ids:
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))

    # Snapshot for Undo (so ↩️ works)
    stash_order(setlist.id)

    # Apply new positions in given order; append any leftovers after
    pos = 1
    for sid in new_ids:
        by_song_id[sid].position = pos
        pos += 1

    # Any songs not mentioned stay, appended in their previous relative order
    for r in rows:
        if r.song_id not in new_ids:
            r.position = pos
            pos += 1

    db.session.commit()
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

@app.get("/setlists/<int:setlist_id>/share/create")
def create_share_link(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    token = get_or_create_share_token(sl)
    # Build absolute URL for convenience
    url = url_for("view_setlist_by_token", token=token, _external=True)
    flash(f"Secret share link created: {url}")
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.get("/setlists/<int:setlist_id>/share/rotate")
def rotate_share_link(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    # Always create a fresh token
    sl.share_token = secrets.token_hex(16)
    db.session.commit()
    url = url_for("view_setlist_by_token", token=sl.share_token, _external=True)
    flash(f"Secret share link rotated: {url}")
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.get("/setlists/<int:setlist_id>/share/disable")
def disable_share_link(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    sl.share_token = None
    db.session.commit()
    flash("Secret share link disabled. The old URL no longer works.")
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

# --- export CSV ---
@app.get("/setlists/<int:setlist_id>/export.csv")
def export_setlist_csv(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Position", "Title", "Artist", "Tempo (BPM)", "Key", "Genre", "Tags", "Duration (mm:ss)", "Notes"])
    for ss in sl.songs:
        dur = ss.song.duration_override_sec if ss.song.duration_override_sec else estimate_duration_seconds(ss.song.tempo_bpm)
        writer.writerow([
    ss.position,
    ss.song.title,
    ss.song.artist,
    ss.song.tempo_bpm or "",
    ss.song.musical_key or "",
    ss.song.genre or "",
    ss.song.tags or "",
    fmt_mmss(dur),
    ss.notes or "",
])
    csv_data = output.getvalue()
    output.close()
    filename = f'{sl.name.replace(" ", "_")}.csv'
    resp = Response(csv_data, mimetype="text/csv")
    resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp

@app.get("/setlists/<int:setlist_id>/export.pdf")
def export_setlist_pdf(setlist_id):
    from datetime import datetime

    sl = Setlist.query.get_or_404(setlist_id)

    # build rows with duration + notes
    total_sec = 0
    rows = []
    for ss in sl.songs:
        dur_sec = ss.song.duration_override_sec or estimate_duration_seconds(ss.song.tempo_bpm)
        total_sec += dur_sec
        rows.append({
            "pos": ss.position,
            "title": ss.song.title,
            "artist": ss.song.artist,
            "key": ss.song.musical_key or "",
            "bpm": str(ss.song.tempo_bpm) if ss.song.tempo_bpm else "",
            "dur": fmt_mmss(dur_sec),
            "notes": ss.notes or "",
        })

    # --- PDF layout ---
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 0.75 * inch

    # columns
    col_dur_w = 0.8 * inch
    col_bpm_w = 0.7 * inch
    col_key_w = 1.0 * inch
    x_dur = width - margin - col_dur_w
    x_bpm = x_dur - col_bpm_w
    x_key = x_bpm - col_key_w
    x_title = margin
    title_w = x_key - margin - 6

    # row metrics
    title_line_h = 14  # Helvetica 11
    notes_line_h = 12  # Helvetica-Oblique 9
    row_gap = 2

    printed_str = "Printed " + datetime.now().strftime("%b %d, %Y")

    def draw_footer():
        """Draw footer with date and page number (current of total)."""
        # ReportLab fills %(pageNumber)s automatically; we’ll use a placeholder for total later
        c.setFont("Helvetica", 9)
        y = margin - 0.45 * inch
        if y < 0.3 * inch:
            y = 0.3 * inch
        # Left: printed date
        c.drawString(margin, y, printed_str)
        # Right: page number (total filled by canvas on save)
        page_text = "Page %d" % c.getPageNumber()
        c.drawRightString(width - margin, y, page_text)

    def draw_page_header(title_suffix=""):
        """Draw header + column labels; returns starting y."""
        y = height - margin
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y, f"Setlist: {sl.name}{title_suffix}")
        y -= 18

        c.setFont("Helvetica", 10)
        meta_bits = []
        if sl.event_type: meta_bits.append(f"Event: {sl.event_type}")
        if sl.venue_type: meta_bits.append(f"Venue: {sl.venue_type}")
        if sl.target_minutes: meta_bits.append(f"Target: {sl.target_minutes} min")
        c.drawString(margin, y, "  ·  ".join(meta_bits) if meta_bits else "")
        y -= 14
        c.drawString(margin, y, f"Songs: {len(rows)}    Approx Total: {fmt_mmss(total_sec)}")
        y -= 18

        # column labels
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x_title, y, "Song")
        c.drawRightString(x_key + col_key_w - 2, y, "Key")
        c.drawRightString(x_bpm + col_bpm_w - 2, y, "BPM")
        c.drawRightString(x_dur + col_dur_w - 2, y, "Dur")
        y -= 10
        c.setLineWidth(0.5)
        c.line(margin, y, width - margin, y)
        y -= 8
        return y

    def new_page(cont_suffix=" (cont.)"):
        # footer for the page we’re finishing
        draw_footer()
        c.showPage()
        return draw_page_header(cont_suffix)

    # first page
    y = draw_page_header("")

    # draw each song row (title + optional notes line)
    for r in rows:
        # calculate needed height for this row
        title_text = f"#{r['pos']} — {r['title']} — {r['artist']}"
        title_wrapped = simpleSplit(title_text, "Helvetica", 11, title_w)
        notes_wrapped = simpleSplit(f"Notes: {r['notes']}", "Helvetica-Oblique", 9, title_w) if r["notes"] else []

        needed = len(title_wrapped) * title_line_h + (len(notes_wrapped) * notes_line_h) + row_gap
        # make room
        if y - needed < margin + 20:
            y = new_page()

        # draw title lines
        c.setFont("Helvetica", 11)
        for i, line in enumerate(title_wrapped):
            c.drawString(x_title, y, line)
            if i == 0:
                # right-side columns align with first title line
                c.drawRightString(x_key + col_key_w - 2, y, r["key"])
                c.drawRightString(x_bpm + col_bpm_w - 2, y, r["bpm"])
                c.drawRightString(x_dur + col_dur_w - 2, y, r["dur"])
            y -= title_line_h

        # draw notes (if any)
        if notes_wrapped:
            c.setFont("Helvetica-Oblique", 9)
            for line in notes_wrapped:
                c.drawString(x_title, y, line)
                y -= notes_line_h

        y -= row_gap

    # final footer then save
    draw_footer()
    c.save()

    pdf_bytes = buf.getvalue()
    buf.close()

    filename = f'{sl.name.replace(" ", "_")}.pdf'
    resp = Response(pdf_bytes, mimetype="application/pdf")
    resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp

    # --- PDF layout ---
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 0.75 * inch

    # columns
    col_dur_w = 0.8 * inch
    col_bpm_w = 0.7 * inch
    col_key_w = 1.0 * inch
    x_dur = width - margin - col_dur_w
    x_bpm = x_dur - col_bpm_w
    x_key = x_bpm - col_key_w
    x_title = margin
    title_w = x_key - margin - 6

    # row metrics
    title_line_h = 14  # Helvetica 11
    notes_line_h = 12  # Helvetica-Oblique 9
    row_gap = 2

    def draw_page_header(title_suffix=""):
        """Draw header + column labels; returns starting y."""
        y = height - margin
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y, f"Setlist: {sl.name}{title_suffix}")
        y -= 18

        c.setFont("Helvetica", 10)
        meta_bits = []
        if sl.event_type: meta_bits.append(f"Event: {sl.event_type}")
        if sl.venue_type: meta_bits.append(f"Venue: {sl.venue_type}")
        if sl.target_minutes: meta_bits.append(f"Target: {sl.target_minutes} min")
        c.drawString(margin, y, "  ·  ".join(meta_bits) if meta_bits else "")
        y -= 14
        c.drawString(margin, y, f"Songs: {len(rows)}    Approx Total: {fmt_mmss(total_sec)}")
        y -= 18

        # column labels
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x_title, y, "Song")
        c.drawRightString(x_key + col_key_w - 2, y, "Key")
        c.drawRightString(x_bpm + col_bpm_w - 2, y, "BPM")
        c.drawRightString(x_dur + col_dur_w - 2, y, "Dur")
        y -= 10
        c.setLineWidth(0.5)
        c.line(margin, y, width - margin, y)
        y -= 8
        return y

    def new_page(cont_suffix=" (cont.)"):
        c.showPage()
        return draw_page_header(cont_suffix)

    # first page
    y = draw_page_header("")

    # draw each song row (title + optional notes line)
    for r in rows:
        # calculate needed height for this row
        title_text = f"#{r['pos']} — {r['title']} — {r['artist']}"
        title_wrapped = simpleSplit(title_text, "Helvetica", 11, title_w)
        notes_wrapped = simpleSplit(f"Notes: {r['notes']}", "Helvetica-Oblique", 9, title_w) if r["notes"] else []

        needed = len(title_wrapped) * title_line_h + (len(notes_wrapped) * notes_line_h) + row_gap
        # make room
        if y - needed < margin + 20:
            y = new_page()

        # draw title lines
        c.setFont("Helvetica", 11)
        for i, line in enumerate(title_wrapped):
            c.drawString(x_title, y, line)
            if i == 0:
                # right-side columns align with first title line
                c.drawRightString(x_key + col_key_w - 2, y, r["key"])
                c.drawRightString(x_bpm + col_bpm_w - 2, y, r["bpm"])
                c.drawRightString(x_dur + col_dur_w - 2, y, r["dur"])
            y -= title_line_h

        # draw notes (if any)
        if notes_wrapped:
            c.setFont("Helvetica-Oblique", 9)
            for line in notes_wrapped:
                c.drawString(x_title, y, line)
                y -= notes_line_h

        y -= row_gap

    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()

    filename = f'{sl.name.replace(" ", "_")}.pdf'
    resp = Response(pdf_bytes, mimetype="application/pdf")
    resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp

# --- init / migrate route ---
@app.get("/initdb")
def initdb():
    ensure_schema()
    return "Database initialized / migrated."

# Ensure DB schema exists when the app is imported (e.g., Gunicorn/Render/Railway)
def _init_schema_on_import():
    try:
        with app.app_context():
            ensure_schema()
    except Exception as e:
        # Safe print — won’t crash the dyno if first boot races the DB
        print("ensure_schema on import warning:", e)

_init_schema_on_import()

if __name__ == "__main__":
    # Local dev server
    app.run(debug=True, port=5055)
