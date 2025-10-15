from collections import deque
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Callable
import csv
import hashlib
import io
import mimetypes
import os
import secrets
import uuid
import re
import unicodedata
import zipfile

import qrcode
import requests  # spotify + demo AI
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    abort,
    flash,
    redirect,
    render_template_string,
    request,
    send_file,
    send_from_directory,
    url_for,
    jsonify,
)
from flask_sqlalchemy import SQLAlchemy
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas
from sqlalchemy import text
from werkzeug.utils import secure_filename

load_dotenv()

def _run_section_preset(setlist_id: int, preset_func: Callable[[int], object]) -> None:
    """Invoke preset routines safely outside a real request."""
    with app.app_context():
        if not Setlist.query.get(setlist_id):
            return
        with app.test_request_context():
            preset_func(setlist_id)


# --- Helpers for applying section presets programmatically ---
def apply_section_preset_basic(setlist_id: int) -> None:
    _run_section_preset(setlist_id, preset_sections_basic)


def apply_section_preset_three_sets(setlist_id: int) -> None:
    _run_section_preset(setlist_id, preset_sections_three_sets)


def apply_section_preset_by_chunk(setlist_id: int) -> None:
    _run_section_preset(setlist_id, preset_sections_by_chunk)

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

@app.context_processor
def inject_utils():
    return dict(
        fmt_mmss=fmt_mmss,
        estimate=estimate_duration_seconds,
        song_has_pdf=song_has_pdf,  # ← add this
    )

# --- File uploads (PDF attachments) ---

UPLOAD_DIR = Path(app.instance_path) / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_MIME = {"application/pdf"}
# Max upload: 16 MB (Flask will 413 if exceeded)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

REQUEST_STATUS_CHOICES = ("new", "queued", "done", "declined")

def wants_json_response() -> bool:
    """
    Lightweight helper to see if the caller prefers a JSON response.
    Looks at Accept header, X-Requested-With, or JSON body.
    """
    accept = (request.headers.get("Accept") or "").lower()
    if "application/json" in accept:
        return True
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return True
    if request.is_json:
        return True
    return False

def _allowed_file(file_storage):
    if not file_storage or not file_storage.filename:
        return False
    mt = (file_storage.mimetype or "").lower()
    # Fallback: guess from filename if mimetype missing
    if not mt:
        mt = mimetypes.guess_type(file_storage.filename)[0] or ""
    return mt in ALLOWED_MIME

# --- Song model ---
class Song(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    artist = db.Column(db.String(200), nullable=False)
    tempo_bpm = db.Column(db.Integer, nullable=True)
    musical_key = db.Column(db.String(20), nullable=True)   # e.g., "C major"
    genre = db.Column(db.String(100), nullable=True)
    tags = db.Column(db.String(500), nullable=True)         # comma-separated
    duration_override_sec = db.Column(db.Integer, nullable=True)
    release_year = db.Column(db.Integer, nullable=True)
    chord_chart = db.Column(db.Text, nullable=True)  # ChordPro / plain chart text for Live Mode
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Optional: song-level default chart (used in Live Mode fallback)
    default_file_id = db.Column(
        "default_attachment_id",  # reuse existing column during transition
        db.Integer,
        db.ForeignKey("attachment.id"),
        nullable=True,
    )
    default_file = db.relationship(
        "SongFile",
        foreign_keys=[default_file_id],
        uselist=False,
        lazy="joined",
        backref="default_for_songs",
    )

    # Keep the relationship INSIDE the class; specify foreign_keys explicitly
    # All attachments that belong to this song
    files = db.relationship(
        "SongFile",
        backref="song",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="SongFile.id.desc()",
        foreign_keys="SongFile.song_id",
    )

    # Backwards-compatibility aliases while the UI migrates
    @property
    def default_attachment_id(self):
        return self.default_file_id

    @default_attachment_id.setter
    def default_attachment_id(self, value):
        self.default_file_id = value

    @property
    def default_attachment(self):
        return self.default_file

    @default_attachment.setter
    def default_attachment(self, value):
        self.default_file = value

    @property
    def attachments(self):
        return self.files

# --- Song files (PDFs, audio, etc.) ---
class SongFile(db.Model):
    __tablename__ = "attachment"

    id = db.Column(db.Integer, primary_key=True)
    song_id = db.Column(db.Integer, db.ForeignKey("song.id"), nullable=False, index=True)
    kind = db.Column(db.String(20), nullable=False, default="pdf")  # pdf, chordpro, audio, image

    original_name = db.Column(db.String(255), nullable=True)
    stored_name = db.Column(db.String(255), nullable=True)
    mimetype = db.Column(db.String(120), nullable=True)
    size_bytes = db.Column(db.Integer, nullable=True)
    pages = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, nullable=True, default=datetime.utcnow)

    # Legacy support for the previous attachments table name/columns.
    orig_name = db.Column(db.String(255))

    @property
    def path(self) -> str:
        return str(UPLOAD_DIR / (self.stored_name or ""))

    @property
    def filename(self) -> str:
        return self.original_name or self.orig_name

    @property
    def is_pdf(self) -> bool:
        mt = (self.mimetype or "").lower()
        return self.kind == "pdf" or mt.startswith("application/pdf")

    def __repr__(self) -> str:
        return f"<SongFile id={self.id} song_id={self.song_id} kind={self.kind} name={self.original_name!r}>"

# --- Setlist + join table ---
class Setlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    event_type = db.Column(db.String(100), nullable=True)
    venue_type = db.Column(db.String(100), nullable=True)
    target_minutes = db.Column(db.Integer, nullable=True)
    notes = db.Column(db.String(1000), nullable=True)
    no_repeat_artists = db.Column(db.Boolean, default=True)  # avoid duplicate artists in this show
    share_token = db.Column(db.String(64), unique=True, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    reset_numbering_per_section = db.Column(db.Boolean, default=False)

    songs = db.relationship(
    "SetlistSong",
    backref="setlist",
    cascade="all, delete-orphan",
    order_by="SetlistSong.position.asc()",
    lazy="joined",
)

class SetlistSong(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    setlist_id = db.Column(db.Integer, db.ForeignKey("setlist.id"), nullable=False)
    song_id    = db.Column(db.Integer, db.ForeignKey("song.id"),    nullable=False)

    position   = db.Column(db.Integer, nullable=False)
    notes      = db.Column(db.Text, nullable=True)
    section_name = db.Column(db.String(120), nullable=True)
    # NEW: lock a row in the setlist (UI toggle)
    locked = db.Column(db.Boolean, default=False, nullable=False)

    # Optional preferred attachment for this setlist row
    preferred_file_id = db.Column(
        "preferred_attachment_id",  # reuse existing column
        db.Integer,
        db.ForeignKey("attachment.id"),
        nullable=True,
    )

    # Relationships
    song = db.relationship("Song", backref="setlist_links", lazy="joined")
    preferred_file = db.relationship(
        "SongFile",
        foreign_keys=[preferred_file_id],
        lazy="joined",
        uselist=False,
        viewonly=True,
    )

    @property
    def preferred_attachment_id(self):
        return self.preferred_file_id

    @preferred_attachment_id.setter
    def preferred_attachment_id(self, value):
        self.preferred_file_id = value

    @property
    def preferred_attachment(self):
        return self.preferred_file


class PatronRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    setlist_id = db.Column(db.Integer, db.ForeignKey("setlist.id"), nullable=True, index=True)
    song_id = db.Column(db.Integer, db.ForeignKey("song.id"), nullable=True, index=True)
    free_text_title = db.Column(db.String(255), nullable=True)
    from_name = db.Column(db.String(120), nullable=True)
    from_contact = db.Column(db.String(120), nullable=True)
    status = db.Column(db.String(20), nullable=False, default="new")
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    setlist = db.relationship("Setlist", backref=db.backref("requests", lazy="dynamic"))
    song = db.relationship("Song")

    def label(self) -> str:
        if self.song:
            return f"{self.song.title} — {self.song.artist}"
        return self.free_text_title or "(Untitled request)"

def serialize_patron_request(req: "PatronRequest") -> dict:
    """Return a compact dict for JSON responses and Live Mode drawer."""
    song = req.song
    return {
        "id": req.id,
        "status": req.status,
        "statusLabel": req.status.title(),
        "label": req.label(),
        "songId": song.id if song else None,
        "songTitle": song.title if song else None,
        "songArtist": song.artist if song else None,
        "hasSong": bool(song),
        "freeText": req.free_text_title,
        "fromName": req.from_name,
        "fromContact": req.from_contact,
        "createdAt": req.created_at.isoformat() if req.created_at else None,
        "updatedAt": req.updated_at.isoformat() if req.updated_at else None,
    }

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


def _ensure_schema_columns() -> None:
    """Backfill columns when running against older SQLite databases."""
    with app.app_context():
        bind = db.session.get_bind()
        if not bind or bind.dialect.name != "sqlite":
            return

        tables = db.session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
        table_names = {t[0] for t in tables}

        # Song table
        info_song = db.session.execute(text("PRAGMA table_info(song);")).fetchall()
        song_cols = [row[1] for row in info_song]
        if "duration_override_sec" not in song_cols:
            db.session.execute(text("ALTER TABLE song ADD COLUMN duration_override_sec INTEGER"))
            db.session.commit()
        if "default_attachment_id" not in song_cols:
            db.session.execute(text("ALTER TABLE song ADD COLUMN default_attachment_id INTEGER"))
            db.session.commit()
        if "release_year" not in song_cols:
            db.session.execute(text("ALTER TABLE song ADD COLUMN release_year INTEGER"))
            db.session.commit()
        if "chord_chart" not in song_cols:
            db.session.execute(text("ALTER TABLE song ADD COLUMN chord_chart TEXT"))
            db.session.commit()

        sls_table = "setlist_song"

        if sls_table in table_names:
            info2 = db.session.execute(text(f"PRAGMA table_info({sls_table});")).fetchall()
            cols2 = [row[1] for row in info2]
            if "notes" not in cols2:
                db.session.execute(text(f"ALTER TABLE {sls_table} ADD COLUMN notes VARCHAR(500)"))
                db.session.commit()
            if "section_name" not in cols2:
                db.session.execute(text(f"ALTER TABLE {sls_table} ADD COLUMN section_name VARCHAR(100)"))
                db.session.commit()
            if "preferred_attachment_id" not in cols2:
                db.session.execute(text(f"ALTER TABLE {sls_table} ADD COLUMN preferred_attachment_id INTEGER"))
                db.session.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_setlist_song_preferred_attachment_id "
                    "ON setlist_song(preferred_attachment_id)"
                ))
                db.session.commit()
            if "locked" not in cols2:
                db.session.execute(text(f"ALTER TABLE {sls_table} ADD COLUMN locked BOOLEAN DEFAULT 0"))
                db.session.commit()

        # Attachment/SongFile table retrofits
        att_table = "attachment"
        if att_table in table_names:
            info_att = db.session.execute(text(f"PRAGMA table_info({att_table});")).fetchall()
            att_cols = [row[1] for row in info_att]
            if "original_name" not in att_cols:
                db.session.execute(text(f"ALTER TABLE {att_table} ADD COLUMN original_name VARCHAR(255)"))
                db.session.commit()
            if "stored_name" not in att_cols:
                db.session.execute(text(f"ALTER TABLE {att_table} ADD COLUMN stored_name VARCHAR(255)"))
                db.session.commit()
            if "mimetype" not in att_cols:
                db.session.execute(text(f"ALTER TABLE {att_table} ADD COLUMN mimetype VARCHAR(120)"))
                db.session.commit()
            if "size_bytes" not in att_cols:
                db.session.execute(text(f"ALTER TABLE {att_table} ADD COLUMN size_bytes INTEGER"))
                db.session.commit()
            if "created_at" not in att_cols:
                db.session.execute(text(f"ALTER TABLE {att_table} ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP"))
                db.session.commit()
            if "kind" not in att_cols:
                db.session.execute(text(f"ALTER TABLE {att_table} ADD COLUMN kind VARCHAR(20) DEFAULT 'pdf'"))
                db.session.commit()
            if "pages" not in att_cols:
                db.session.execute(text(f"ALTER TABLE {att_table} ADD COLUMN pages INTEGER"))
                db.session.commit()

        pr_table = "patron_request"
        if pr_table in table_names:
            info_pr = db.session.execute(text(f"PRAGMA table_info({pr_table});")).fetchall()
            pr_cols = [row[1] for row in info_pr]
            if "setlist_id" not in pr_cols:
                db.session.execute(text(f"ALTER TABLE {pr_table} ADD COLUMN setlist_id INTEGER"))
                db.session.commit()
            if "song_id" not in pr_cols:
                db.session.execute(text(f"ALTER TABLE {pr_table} ADD COLUMN song_id INTEGER"))
                db.session.commit()
            if "free_text_title" not in pr_cols:
                db.session.execute(text(f"ALTER TABLE {pr_table} ADD COLUMN free_text_title VARCHAR(255)"))
                db.session.commit()
            if "from_name" not in pr_cols:
                db.session.execute(text(f"ALTER TABLE {pr_table} ADD COLUMN from_name VARCHAR(120)"))
                db.session.commit()
            if "from_contact" not in pr_cols:
                db.session.execute(text(f"ALTER TABLE {pr_table} ADD COLUMN from_contact VARCHAR(120)"))
                db.session.commit()
            if "status" not in pr_cols:
                db.session.execute(text(f"ALTER TABLE {pr_table} ADD COLUMN status VARCHAR(20) DEFAULT 'new'"))
                db.session.commit()
            if "created_at" not in pr_cols:
                db.session.execute(text(f"ALTER TABLE {pr_table} ADD COLUMN created_at DATETIME"))
                db.session.commit()
            if "updated_at" not in pr_cols:
                db.session.execute(text(f"ALTER TABLE {pr_table} ADD COLUMN updated_at DATETIME"))
                db.session.commit()

        info3 = db.session.execute(text("PRAGMA table_info(setlist);")).fetchall()
        cols3 = [row[1] for row in info3]
        if "no_repeat_artists" not in cols3:
            db.session.execute(text("ALTER TABLE setlist ADD COLUMN no_repeat_artists BOOLEAN DEFAULT 1"))
            db.session.commit()
        if "share_token" not in cols3:
            db.session.execute(text("ALTER TABLE setlist ADD COLUMN share_token VARCHAR(64)"))
            db.session.commit()
        if "reset_numbering_per_section" not in cols3:
            db.session.execute(text("ALTER TABLE setlist ADD COLUMN reset_numbering_per_section BOOLEAN DEFAULT 0"))
            db.session.commit()


_ensure_schema_columns()

# --- helpers ---

def _ext(fn): 
    return (os.path.splitext(fn or "")[1] or "").lower()

def _store_file(file_storage) -> tuple[str, str, int, str]:
    """Return (stored_name, original_name, size_bytes, mimetype)."""
    orig = (file_storage.filename or "file").strip()
    safe = secure_filename(orig) or "file"
    # make unique
    stem, ext = os.path.splitext(safe)
    ext = ext or _ext(orig) or ""
    rand = secrets.token_hex(8)
    stored = f"{stem}.{rand}{ext}"
    path = UPLOAD_DIR / stored
    file_storage.save(path)
    size = path.stat().st_size
    mimetype = file_storage.mimetype or "application/octet-stream"
    return stored, orig, size, mimetype

def _song_file_abs_path(sf: SongFile) -> str:
    return str(UPLOAD_DIR / sf.stored_name)

def normalize_positions(setlist: Setlist):
    """Make positions contiguous starting at 1."""
    rows = (SetlistSong.query
            .filter_by(setlist_id=setlist.id)
            .order_by(SetlistSong.position.asc(), SetlistSong.id.asc())
            .all())
    for idx, row in enumerate(rows, start=1):
        row.position = idx
    db.session.commit()

def get_or_create_share_token(sl: Setlist) -> str:
    """Return existing share token or create a new one if missing."""
    if not sl.share_token:
        sl.share_token = secrets.token_hex(16)  # 32-char hex
        db.session.commit()
    return sl.share_token


def _slugify(value: str, fallback: str = "item") -> str:
    value = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")
    return value or fallback

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

def song_has_pdf(song: "Song") -> bool:
    """True if the song has at least one PDF attachment."""
    try:
        return any(sf.is_pdf for sf in (song.files or []))
    except Exception:
        return False

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

# --- Metadata helpers --------------------------------------------------------
def _fetch_song_metadata(song: "Song") -> dict:
    meta = spotify_lookup(song.title, song.artist)
    source = None
    if meta:
        source = "spotify"
        needs = [field for field in ("tempo_bpm", "musical_key", "genre", "tags", "release_year")
                 if not meta.get(field)]
        if needs:
            alt = openai_song_metadata(song.title, song.artist)
            if alt:
                for field in needs:
                    val = alt.get(field)
                    if val:
                        meta[field] = val
                source = "spotify+openai" if any(meta.get(field) for field in needs) else source
    else:
        meta = openai_song_metadata(song.title, song.artist)
        if meta:
            source = "openai"
    if meta and source:
        meta["_source"] = source
    return meta or {}


def _apply_song_metadata(song: "Song", meta: dict, force: bool = False) -> bool:
    """Apply metadata dict to song; returns True if any field changed."""
    if not meta:
        return False
    changed = False
    tempo = meta.get("tempo_bpm")
    if tempo is not None and (force or song.tempo_bpm is None):
        song.tempo_bpm = int(tempo)
        changed = True
    key = meta.get("musical_key") or meta.get("key")
    if key and (force or not song.musical_key):
        song.musical_key = key
        changed = True
    genre = meta.get("genre")
    if genre and (force or not song.genre):
        song.genre = genre
        changed = True
    tags = meta.get("tags")
    if isinstance(tags, (list, tuple)):
        tags = ", ".join(tags)
    if tags and (force or not song.tags):
        song.tags = tags
        changed = True
    year = meta.get("release_year")
    if year and (force or not song.release_year):
        try:
            song.release_year = int(str(year)[:4])
            changed = True
        except Exception:
            pass
    return changed

# --- Live lookup via Spotify (optional env vars) ---
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
_PITCH_NAMES = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]

def _spotify_token():
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        print("Spotify: missing SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET; skipping lookup.")
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
    release_year = None
    try:
        release = track.get("album", {}).get("release_date")
        if release:
            release_year = int(release.split("-")[0])
    except Exception:
        release_year = None
    if not any([tempo_bpm, musical_key, genre, tags, release_year]):
        return None
    return {
        "tempo_bpm": tempo_bpm,
        "musical_key": musical_key,
        "genre": genre,
        "tags": tags,
        "release_year": release_year,
    }


def openai_song_metadata(title: str, artist: str) -> dict | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    safe_title = (title or "").replace("{", "[").replace("}", "]")
    safe_artist = (artist or "").replace("{", "[").replace("}", "]")
    prompt = (
        f"Provide structured metadata for the song '{safe_title}' by '{safe_artist}'. "
        "Respond as JSON with keys tempo_bpm (integer), musical_key (string), "
        "genre (string), tags (comma-separated string), release_year (integer). "
        "If you are unsure, give your best musical estimate rather than null."
    )
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a music metadata assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
            timeout=20,
        )
        if resp.status_code != 200:
            print("OpenAI metadata error:", resp.text)
            return None
        payload = resp.json()
        content = payload.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            return None
        import json

        meta = json.loads(content)
        # Normalize fields
        try:
            if meta.get("tempo_bpm") is not None:
                meta["tempo_bpm"] = int(meta["tempo_bpm"])
        except Exception:
            meta["tempo_bpm"] = None
        try:
            if meta.get("release_year") is not None:
                meta["release_year"] = int(str(meta["release_year"])[:4])
        except Exception:
            meta["release_year"] = None
        if isinstance(meta.get("tags"), list):
            meta["tags"] = ", ".join(meta["tags"])
        return meta
    except Exception as exc:
        print("OpenAI metadata exception:", exc)
        return None

def openai_generate_chord_chart(title: str, artist: str) -> tuple[str | None, str | None]:
    """
    Ask OpenAI for a ChordPro-style chord chart.
    Returns (chart_text, error_message). One of them will be None.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OpenAI API key is not configured."

    safe_title = (title or "").replace("{", "[").replace("}", "]")
    safe_artist = (artist or "").replace("{", "[").replace("}", "]")
    prompt = (
        f"Draft a concise ChordPro-style chord chart for the song '{safe_title}' by '{safe_artist}'. "
        "Use bracketed chords (e.g., [C]) inline with lyrics. "
        "Include common sections like {{title: Verse 1}}, {{title: Chorus}}. "
        "Do not use START_OF_/END_OF_ markers, Markdown code fences, or TITLE/ARTIST metadata lines. "
        "Keep it performance-ready, and avoid prose explanations."
    )

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You craft live-performance chord charts using ChordPro formatting."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.5,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return None, f"OpenAI error {resp.status_code}: {resp.text[:200]}"
        payload = resp.json()
        content = payload.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            return None, "OpenAI returned an empty response."
        chart = sanitize_ai_chart(content)
        if not chart:
            return None, "OpenAI returned a blank chart."
        return chart, None
    except Exception as exc:
        print("OpenAI chord chart exception:", exc)
        return None, str(exc)

def sanitize_ai_chart(raw: str | None) -> str | None:
    if not raw:
        return None
    import re

    text = raw.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text).strip()

    lines: list[str] = []
    BAR_TOKENS = {":", "|", "||", "||:", ":||", "|:", ":|", "‖", "x4", "x2"}

    def is_chord_token(token: str) -> bool:
        if not token:
            return False
        tok = token.strip()
        if tok.upper() == "NC":
            return True
        tok = tok.replace("♭", "b").replace("♯", "#")
        return bool(re.match(r"^[A-G][#b]?((add|sus|maj|min|aug|dim|m|M)?\d{0,2}|m|maj7|m7|m9|7|9|11|13|sus2|sus4|add9|6|\\+|-|°|ø)*([/][A-G][#b]?)?$", tok))

    def wrap_inline_chords(line: str) -> str:
        parts = re.split(r"(\s+)", line)
        changed = False
        for idx, part in enumerate(parts):
            if not part or part.isspace():
                continue
            token = part.strip()
            if token.startswith("[") and token.endswith("]"):
                continue
            if is_chord_token(token):
                parts[idx] = part.replace(token, f"[{token}]")
                changed = True
        return "".join(parts) if changed else line

    def convert_chord_line(line: str) -> str:
        result = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch.isspace():
                result.append(ch)
                i += 1
                continue
            start = i
            while i < len(line) and not line[i].isspace():
                i += 1
            token = line[start:i]
            if is_chord_token(token):
                result.append(f"[{token}]")
            else:
                result.append(token)
        return "".join(result)

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        upper = stripped.upper()
        if re.fullmatch(r"`{2,}", stripped):
            continue
        if upper.startswith("TITLE:") or upper.startswith("ARTIST:"):
            continue
        if upper.startswith("START_OF_"):
            section = stripped[len("START_OF_"):].strip()
            if ":" in section:
                section = section.split(":", 1)[1]
            section = section.replace("_", " ").strip().title()
            lines.append(f"{{title: {section}}}")
            continue
        if upper.startswith("END_OF_"):
            continue
        if upper.endswith(":") and len(stripped.split()) <= 4:
            base = upper.strip(":").replace("_", " ").strip()
            if base and all(part.isalnum() for part in base.split()):
                section = base.title()
                lines.append(f"{{title: {section}}}")
                continue
        formatted = convert_chord_line(line)
        lines.append(formatted)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or None

# --- super simple layout ---
BASE_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Setlist Genie</title>

  <!-- Base app styles (unchanged from yours) -->
  <style>
    details.rowmore > summary { list-style: none; cursor: pointer; }
    details.rowmore[open] > summary { opacity: .7; }
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
    body.has-reorder .dnd-item { border: 1px dashed #aaa; cursor: grab; transition: background .15s ease; }
    body.has-reorder .dnd-item.dragging { opacity: 0.6; }
    body.has-reorder .dnd-item:active { cursor: grabbing; }
    .dnd-item[data-locked="1"] { opacity: 0.82; position: relative; }
    body.has-reorder .dnd-item[data-locked="1"] { cursor: not-allowed; border-style: solid; }
    body.has-reorder .dnd-item[data-locked="1"]::after {
      content: "Locked";
      position: absolute;
      top: 8px;
      right: 12px;
      font-size: 0.7rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      background: rgba(0,0,0,0.1);
      padding: 2px 6px;
      border-radius: 999px;
    }
    #reorderStatus.is-success { color: #1b7f3b; }
    #reorderStatus.is-error { color: #9c1c1c; }
    #reorderStatus.is-busy { color: #444; font-style: italic; }

    /* Toast */
    .toast {
      position: fixed; left: 50%; bottom: 24px; transform: translateX(-50%);
      background: rgba(0,0,0,0.85); color: #fff; padding: 10px 14px; border-radius: 10px;
      font-size: 14px; opacity: 0; pointer-events: none; transition: opacity .2s ease;
      z-index: 9999;
    }
    .toast.show { opacity: 1; }
  </style>

  <!-- THEME VARIABLES (global, single source of truth) -->
  <style>
    :root {
      --bg:#ffffff; --text:#111111; --muted:#555555;
      --surface:#f7f7f8; --border:#dddddd;
      --btn-bg:#ffffff; --btn-text:#111111; --btn-border:#bbbbbb;
      --link:#0a58ca;
      --field-bg:#ffffff; --field-border:#cdd0d6;
      --card-bg:#ffffff; --card-border:var(--border);
      --card-shadow:0 10px 26px rgba(15,23,42,0.08);
      --focus-ring:rgba(13,110,253,0.18);
    }
    [data-theme="dark"]{
      --bg:#0f1115; --text:#e6e6e6; --muted:#9aa4b2;
      --surface:#141925; --border:#2b3042;
      --btn-bg:#171a23; --btn-text:#e6e6e6; --btn-border:#3a4157;
      --link:#9ecbff;
      --field-bg:#1c2230; --field-border:#313a51;
      --card-bg:#151b27; --card-border:#283144;
      --card-shadow:0 18px 36px rgba(0,0,0,0.55);
      --focus-ring:rgba(158,203,255,0.25);
    }
    html,body{ background:var(--bg); color:var(--text); }
    a { color: var(--link); }
    .muted{ color:var(--muted); }
    .section{
      background:var(--card-bg);
      border:1px solid var(--card-border);
      border-radius:14px;
      padding:14px 16px;
      margin-top:14px;
      box-shadow:var(--card-shadow);
    }
    .row{ border-bottom:1px solid var(--border); }
    .btn{ background:var(--btn-bg); color:var(--btn-text); border-color:var(--btn-border); }
    .btn.btn-danger { color:#ff9aa0; border-color:#7a2f34; }
    input,textarea,select{
      background:var(--field-bg);
      color:var(--text);
      border:1px solid var(--field-border);
      border-radius:10px;
      padding:8px 10px;
      box-shadow:inset 0 1px 1px rgba(0,0,0,0.03);
      transition:border-color .15s ease, box-shadow .15s ease;
    }
    input:focus,textarea:focus,select:focus{
      outline:none;
      border-color:var(--btn-border);
      box-shadow:0 0 0 2px var(--focus-ring);
    }
    .grid p{ margin-bottom:12px; }
    .grid p:last-child{ margin-bottom:0; }
    .section > summary{
      margin:-14px -16px 12px;
      padding:14px 16px;
      background:linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.74));
      border-bottom:1px solid var(--border);
      border-radius:14px 14px 0 0;
      font-weight:600;
      font-size:1.05rem;
      display:flex;
      align-items:center;
      gap:8px;
      cursor:pointer;
    }
    [data-theme="dark"] .section > summary{
      background:linear-gradient(180deg, rgba(33,41,58,0.95), rgba(21,27,39,0.92));
    }
    .section:not([open]) > summary{ margin-bottom:0; border-radius:14px; border-bottom:none; }
    .section > summary::-webkit-details-marker{ display:none; }
    .section[open] > summary::before{ content:"▾"; margin-right:4px; }
    .section:not([open]) > summary::before{ content:"▸"; margin-right:4px; }
    .summary-note{ font-weight:400; opacity:0.85; }
    .section-divider{
      background:var(--card-bg);
      border:1px solid var(--card-border);
      border-radius:10px;
      padding:12px 14px;
      box-shadow:var(--card-shadow);
    }
    .section-divider__inner{
      font-weight:700;
      display:flex;
      gap:8px;
      align-items:center;
      justify-content:space-between;
      width:100%;
      flex-wrap:wrap;
    }
    .section-divider__label{
      display:flex;
      gap:8px;
      align-items:center;
      flex-wrap:wrap;
    }
    .section-divider__actions{
      display:flex;
      gap:6px;
      align-items:center;
    }
    .section-divider__name{
      font-weight:700;
    }
    .secmeta{
      font-weight:400;
      color:var(--muted);
      font-size:0.9em;
    }

    /* Items you previously forced for dark theme – now keyed to [data-theme="dark"] */
    [data-theme="dark"] tr.section td,
    [data-theme="dark"] .row[style*="background:#fafafa"],
    [data-theme="dark"] .chip,
    [data-theme="dark"] .chips .chip,
    [data-theme="dark"] .row[style*="background:#fafafa;"] {
      background:#141925 !important; border-color:#2b3042 !important; color:#e6e6e6;
    }
    [data-theme="dark"] .chip { border:1px solid #2b3042; color:#d7dbe7; }

    @media print{
      html{background:#fff !important;color:#000 !important;}
      [data-theme="dark"]{color-scheme:light;}
    }
  </style>

  <!-- Early paint: set theme before page renders to avoid flash -->
  <script>
    (function(){
      try{
        const KEY='sg:theme';
        const saved=(localStorage.getItem(KEY)||'').toLowerCase();
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        const initial = saved || (prefersDark ? 'dark':'light');
        document.documentElement.setAttribute('data-theme', initial);
      }catch(_){}
    
  </script>
</head>

<body>
  {% with msgs = get_flashed_messages(with_categories=true) %}
    <div id="flash-messages" style="display:none;">
      {% if msgs %}
        {% for cat, msg in msgs %}
          <div data-cat="{{ cat|e }}" data-msg="{{ msg|e }}"></div>
        {% endfor %}
      {% endif %}
    </div>
  {% endwith %}

  <div id="toast" class="toast" role="status" aria-live="polite"></div>

  <!-- Toast util + auto-toast flashes + action nudges -->
  <script>
    function showToast(msg) {
      var el = document.getElementById('toast');
      if (!el) return;
      el.textContent = msg;
      el.classList.add('show');
      clearTimeout(window.__toastTimer);
      window.__toastTimer = setTimeout(function(){ el.classList.remove('show'); }, 1600);
    }
    (function () {
      var wrap = document.getElementById('flash-messages');
      if (!wrap) return;
      var items = Array.from(wrap.querySelectorAll('div[data-msg]')).map(function(n){ return n.getAttribute('data-msg'); });
      if (items.length) showToast(items[items.length - 1]);
    })();
    (function () {
      document.addEventListener('click', function (e) {
        var aSongs = e.target.closest('a[href*="/songs/export.csv"]');
        if (aSongs) { showToast('Preparing songs CSV…'); }
        var aSetCsv = e.target.closest('a[href$="/export.csv"]');
        if (aSetCsv && aSetCsv.href.indexOf('/songs/') === -1) { showToast('Preparing setlist CSV…'); }
        var aPdf = e.target.closest('a[href$="/export.pdf"]');
        if (aPdf) { showToast('Generating PDF…'); }
      });
      document.addEventListener('submit', function (e) {
        var f = e.target;
        if (f && f.matches('form[action*="/songs/autofill_all"]')) showToast('Autofilling missing fields…');
        if (f && f.matches('form[action*="/songs/"][action$="/ai"]')) showToast('Autofilling…');
        if (f && f.matches('form[action="/songs"]')) showToast('Adding song…');
        if (f && f.matches('form[action*="/songs/"][action$="/delete"]')) showToast('Deleting…');
        if (f && f.matches('form[action*="/setlists/"]')) showToast('Working…');
      });
    })();
  </script>

  <!-- Top nav with a single theme toggle button -->
  <nav class="topnav" style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
    <a class="btn" href="{{ url_for('home') }}">Home</a>
    <a class="btn" href="{{ url_for('list_songs') }}">Songs</a>
    <a class="btn" href="{{ url_for('new_song') }}">Add Song</a>
    <a class="btn" href="{{ url_for('list_setlists') }}">Setlists</a>
    <a class="btn" href="{{ url_for('new_setlist') }}">New Setlist</a>
    <span style="flex:1;"></span>
    <button id="themeToggle" class="btn" type="button" title="Toggle dark / light">🌗 Theme</button>
  </nav>

  <h1>Setlist Genie</h1>

  {{ content|safe }}

  <!-- Theme toggle wiring (single source) -->
  <script>
    (function(){
      const KEY='sg:theme';
      const root=document.documentElement;
      const btn=document.getElementById('themeToggle');
      if(!btn) return;

      function apply(next){
        root.setAttribute('data-theme', next);
        try{ localStorage.setItem(KEY,next);}catch(_){}
        if (typeof showToast==='function') showToast(next==='dark'?'Dark theme':'Light theme');
      }

      btn.addEventListener('click',()=>{
        const isDark = root.getAttribute('data-theme')==='dark';
        apply(isDark ? 'light' : 'dark');
      });
    })();
  </script>
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
      <button class="btn" type="submit" title="Fill missing metadata for all filtered songs"
              onclick="return confirm('Auto-fill missing info for ALL currently filtered songs? This may take a moment.');">
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
  <p class="muted" style="margin-top:6px;">
    <label style="display:flex; gap:8px; align-items:center;">
      <input type="checkbox" name="generate_ai_chart" value="1" />
      <span>Generate draft chord chart with AI (accuracy not guaranteed).</span>
    </label>
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
            {% if s.release_year %}Year: {{ s.release_year }} · {% endif %}
            {% if s.tags %}Tags: {{ s.tags }} · {% endif %}
            {% set dur = fmt_mmss(s.duration_override_sec) if s.duration_override_sec else fmt_mmss(estimate(s.tempo_bpm)) %}
            Duration: {{ dur }}{% if song_has_pdf(s) %} · 📄 PDF{% endif %}{% if s.chord_chart %} · 🎼 Chart{% endif %}
        </div>
      </div>
            <div class="right">
        <form class="inline" method="post" action="{{ url_for('ai_enrich_song', song_id=s.id) }}">
          <button class="btn" type="submit" title="Fetch tempo/key/genre/tags/year (Spotify → OpenAI)">✨ AI Autofill</button>
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

CHORD_CHART_SCRIPT = r"""
<script>
(function(){
  const textarea = document.getElementById('chordChartInput');
  const convertBtn = document.getElementById('chordChartConvertBtn');
  if (!textarea || !convertBtn) return;

  const sectionRegex = /^(?:\s*)((?:pre[\s-]?chorus|verse|chorus|bridge|intro|outro|tag|solo|vamp|ending|coda)(?:[\s:-]*\d+|\s*[A-Za-z]*)?)[\s:.!-]*$/i;

  function normalizeLineEndings(text) {
    return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  }

  function toTitleCase(str) {
    return str.replace(/\w\S*/g, function(txt){
      return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
    }).replace(/\s+/g, ' ').trim();
  }

  function looksLikeChord(rawToken) {
    if (!rawToken) return false;
    const token = rawToken.replace(/[\u266d\u266f]/g, match => match === '\u266d' ? 'b' : '#');
    if (token.includes('[') || token.includes(']')) return false;
    if (/^[|:.,]+$/.test(token)) return false;
    if (/^N\.?C\.?$/i.test(token)) return true;
    const match = token.match(/^([A-Ga-g][#b]?)(.*)$/);
    if (!match) return false;

    const rest = match[2] || '';
    if (!rest) return true;
    if (!/^[0-9A-Za-z+#\/()\-b♭♯.]*$/.test(rest)) return false;
    // Avoid matching simple words like "Am" or "Do" followed by lowercase vowel
    if (token.length === 2 && token[0] === token[0].toUpperCase() && token[1] === token[1].toLowerCase() && !/[#b]/.test(token[1])) {
      return false;
    }
    return true;
  }

  function isBarToken(token) {
    return /^[:|\\-]+$/.test(token);
  }

  function isChordLine(line) {
    const trimmed = line.trim();
    if (!trimmed) return false;
    const tokens = trimmed.split(/\s+/).filter(Boolean);
    if (!tokens.length) return false;
    let chordCount = 0;
    for (const token of tokens) {
      if (isBarToken(token)) continue;
      if (looksLikeChord(token)) {
        chordCount += 1;
      } else {
        return false;
      }
    }
    return chordCount > 0;
  }

  function convertChordLyricPair(chordLine, lyricLine) {
    const chords = [];
    let i = 0;
    while (i < chordLine.length) {
      if (chordLine[i] === ' ' || chordLine[i] === '\t') {
        i += 1;
        continue;
      }
      const start = i;
      while (i < chordLine.length && chordLine[i] !== ' ' && chordLine[i] !== '\t') {
        i += 1;
      }
      const token = chordLine.slice(start, i).trim();
      if (looksLikeChord(token)) {
        chords.push({ token, pos: start });
      }
    }
    if (!chords.length) {
      return chordLine;
    }
    const lyric = lyricLine || '';
    if (!lyric.trim()) {
      return chords.map(ch => '[' + ch.token + ']').join(' ');
    }
    let result = '';
    let pointer = 0;
    chords.forEach(ch => {
      const insertPos = Math.min(ch.pos, lyric.length);
      if (insertPos > pointer) {
        result += lyric.slice(pointer, insertPos);
        pointer = insertPos;
      } else if (insertPos < pointer) {
        result += '';
      }
      result += '[' + ch.token + ']';
    });
    result += lyric.slice(pointer);
    return result;
  }

  function wrapInlineChords(line) {
    if (!line || line.includes('[')) return line;
    const parts = line.split(/(\s+)/);
    let wrapped = 0;
    const rebuilt = parts.map(part => {
      if (!part.trim() || /\s+/.test(part)) return part;
      const token = part.trim();
      if (looksLikeChord(token)) {
        wrapped += 1;
        return part.replace(token, '[' + token + ']');
      }
      return part;
    }).join('');
    return wrapped ? rebuilt : line;
  }

  function maybeConvertSection(line) {
    const match = line.match(sectionRegex);
    if (!match) return null;
    const label = toTitleCase(match[1] + (match[2] || ''));
    return `{title: ${label}}`;
  }

  function autoFormatChart(text) {
    const lines = normalizeLineEndings(text).split(/\\r?\\n/);
    const output = [];
    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];
      const trimmed = line.trim();
      if (!trimmed) {
        output.push('');
        continue;
      }
      if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
        output.push(trimmed);
        continue;
      }
      const section = maybeConvertSection(trimmed);
      if (section) {
        output.push(section);
        continue;
      }
      if (isChordLine(line)) {
        const nextLine = (i + 1 < lines.length) ? lines[i + 1] : '';
        if (nextLine && !isChordLine(nextLine) && nextLine.trim()) {
          output.push(convertChordLyricPair(line, nextLine));
          i += 1;
        } else {
          output.push(wrapInlineChords(line));
        }
        continue;
      }
      output.push(wrapInlineChords(line));
    }
    return output.join('\n').replace(/\n{3,}/g, '\n\n');
  }

  convertBtn.addEventListener('click', () => {
    const source = textarea.value || '';
    if (!source.trim()) {
      if (typeof showToast === 'function') showToast('Nothing to format.');
      return;
    }
    const formatted = autoFormatChart(source);
    textarea.value = formatted;
    if (typeof showToast === 'function') {
      showToast('Chord chart formatted');
    }
  });
})();
</script>
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
    <p><label>Release year</label><br/><input name="release_year" type="number" min="1900" max="2100" /></p>
    <p><label>Manual duration (mm:ss)</label><br/><input name="duration_override" placeholder="e.g., 3:30" /></p>
    <p><label>Chord chart (ChordPro or plain text)</label><br/>
      <textarea id="chordChartInput" name="chord_chart" rows="12" style="width:100%; font-family:Menlo,Consolas,monospace; resize:vertical;" placeholder="[C]Amazing [G]Grace\n[F]How sweet..."></textarea>
    </p>
    <div style="margin-top:6px; display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
      <button class="btn" type="button" id="chordChartConvertBtn">⚡ Auto-format chords</button>
      <span class="muted">Paste plain text; we’ll wrap chords & section titles for you.</span>
    </div>
    <label class="muted" style="display:flex; gap:8px; align-items:center; margin-top:6px;">
      <input type="checkbox" name="generate_ai_chart" value="1" />
      <span>Also draft a chord chart with AI (accuracy not guaranteed).</span>
    </label>
  </details>
  <p><button class="btn" type="submit">Save Song</button></p>
</form>
""" + CHORD_CHART_SCRIPT

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
    <p><label>Release year</label><br/><input name="release_year" type="number" min="1900" max="2100" value="{{ song.release_year or '' }}" /></p>
    <p><label>Manual duration (mm:ss)</label><br/><input name="duration_override" value="{{ fmt_mmss(song.duration_override_sec) if song.duration_override_sec else '' }}" placeholder="e.g., 3:30" /></p>
  </details>
  <details class="section" open>
    <summary>🎸 Chord chart (Live Mode)</summary>
    <p class="muted" style="margin-bottom:6px;">
      Paste or type a <a href="https://www.chordpro.org/chordpro/chordpro-introduction/" target="_blank">ChordPro-style</a> chart. Wrap chords in brackets (e.g., <code>[G]</code>) so we can highlight and transpose them on stage. You can also include section headings like <code>{title: Verse 1}</code>.
    </p>
    <textarea id="chordChartInput" name="chord_chart" rows="18" style="width:100%; font-family:Menlo,Consolas,monospace; resize:vertical;" placeholder="[C]Amazing [G]Grace\n[F]How sweet...">{{ song.chord_chart or '' }}</textarea>
    <div style="margin-top:6px; display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
      <button class="btn" type="button" id="chordChartConvertBtn">⚡ Auto-format chords</button>
      <span class="muted">Paste plain text; we’ll wrap chords & section titles for you.</span>
    </div>
    <p class="muted" style="margin-top:6px;">Leave blank if you prefer to rely on PDFs for this song.</p>
</details>
<p>
  <button class="btn" type="submit">Update Song</button>
  <a class="btn" href="{{ url_for('list_songs') }}">Cancel</a>
</p>
</form>

<form method="post" action="{{ url_for('song_generate_ai_chart', song_id=song.id) }}" style="margin-top:6px; display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
  <button class="btn" type="submit">🤖 Regenerate chord chart with AI</button>
  <span class="muted">Grabs a new draft (accuracy not guaranteed) — review before saving.</span>
</form>

<form method="post" action="{{ url_for('ai_enrich_song', song_id=song.id) }}">
  <p class="muted" style="margin-top:6px;">Pulls fresh metadata from Spotify with an OpenAI fallback. Tick overwrite to replace existing entries.</p>
  <label style="font-weight:400;"><input type="checkbox" name="force" value="1" /> Overwrite existing values</label>
  <p style="margin-top:8px;"><button class="btn" type="submit">✨ Refresh Metadata</button></p>
</form>
<details class="section">
  <summary>📄 Attachments (PDF)</summary>
  <div style="margin-top:8px;">
    <h3 style="margin-top:18px;">📎 Attachments (PDF)</h3>

    <form action="{{ url_for('upload_attachment', song_id=song.id) }}" method="post" enctype="multipart/form-data" style="margin: .5rem 0 1rem;">
      <input type="file" name="file" accept=".pdf" required>
      <button class="btn" type="submit">Upload PDF</button>
    </form>

    {% set files = song.files or [] %}
    {% if not files %}
      <p class="muted">No files yet.</p>
    {% else %}
    <div style="overflow:auto;">
      <table style="width:100%; border-collapse:collapse;">
        <thead>
          <tr>
            <th style="text-align:left; border-bottom:1px solid #ddd; padding:6px;">File</th>
            <th style="text-align:left; border-bottom:1px solid #ddd; padding:6px;">Kind</th>
            <th style="text-align:right; border-bottom:1px solid #ddd; padding:6px;">Size</th>
            <th style="text-align:left; border-bottom:1px solid #ddd; padding:6px;">Added</th>
            <th style="text-align:left; border-bottom:1px solid #ddd; padding:6px;">Actions</th>
          </tr>
        </thead>
        <tbody>
          {% for file in files %}
          <tr>
            <td style="padding:6px;">
              {{ file.original_name or file.stored_name }}
              {% if song.default_file_id == file.id %}
                <span class="muted">· default</span> 🔖
              {% endif %}
            </td>
            <td style="padding:6px;">{{ file.kind or "—" }}</td>
            <td style="padding:6px; text-align:right;">
              {% if file.size_bytes %}{{ "{:,}".format(file.size_bytes) }} B{% else %}—{% endif %}
            </td>
            <td style="padding:6px;">
              {% if file.created_at %}{{ file.created_at.strftime("%Y-%m-%d %H:%M") }}{% else %}—{% endif %}
            </td>
            <td style="padding:6px;">
              <a class="btn" href="{{ url_for('view_attachment', att_id=file.id) }}" target="_blank" title="Open">Open</a>
              <a class="btn" href="{{ url_for('download_attachment', att_id=file.id) }}" title="Download">Download</a>
              {% if song.default_file_id != file.id %}
              <form action="{{ url_for('song_set_default_file', song_id=song.id, file_id=file.id) }}" method="post" style="display:inline;">
                <button class="btn" type="submit" title="Set as default">Make Default</button>
              </form>
              {% endif %}
              <form action="{{ url_for('delete_attachment', att_id=file.id) }}" method="post" style="display:inline;" onsubmit="return confirm('Delete this file?');">
                <button class="btn" type="submit">Delete</button>
              </form>
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% endif %}
  </div>
</details>
""" + CHORD_CHART_SCRIPT

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
        <form class="inline" method="post" action="{{ url_for('duplicate_setlist', setlist_id=sl.id) }}">
          <button class="btn" type="submit">📄 Duplicate</button>
        </form>
        <form class="inline" method="post" action="{{ url_for('delete_setlist', setlist_id=sl.id) }}" onsubmit="return confirm('Delete this setlist?');">
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
<form method="post" action="{{ url_for('create_setlist') }}" id="newSetForm">
  <div class="grid">
    <p><label>Name</label><br/><input name="name" required /></p>
    <p><label>Target Minutes</label><br/><input name="target_minutes" type="number" min="1" /></p>
  </div>
  <div class="grid">
    <p><label>Event Type</label><br/><input name="event_type" placeholder="Wedding, Bar, Festival..." /></p>
    <p><label>Venue Type</label><br/><input name="venue_type" placeholder="Indoor, Outdoor..." /></p>
  </div>
  <p><label>Notes</label><br/><textarea name="notes" rows="3" placeholder="Any notes..."></textarea></p>

  <!-- Build tools (moved here) -->
  <details class="section" id="new-buildtools" open>
    <summary>⚙️ Build tools <span class="muted" style="font-weight:400;">(presets & optional auto-build)</span></summary>

    <style>
      #new-buildtools .tabs { display:flex; gap:6px; border-bottom:1px solid var(--border,#ddd); margin:6px 0 10px; }
      #new-buildtools .tab { padding:6px 10px; border:1px solid var(--border,#ddd); border-bottom:none; border-radius:8px 8px 0 0; cursor:pointer; background:var(--btn-bg,#fff); }
      #new-buildtools .tab[aria-selected="true"] { font-weight:600; }
      #new-buildtools .tabpanel { display:none; }
      #new-buildtools .tabpanel.active { display:block; }
      [data-theme="dark"] #new-buildtools .tab { background:#2a2b2e; border-color:#4c4f54; }
    </style>

    <div class="tabs" role="tablist" aria-label="Build tools">
      <button class="tab" id="tab-presets-new"  role="tab" aria-controls="panel-presets-new"  aria-selected="true">Event presets</button>
      <button class="tab" id="tab-autobuild-new" role="tab" aria-controls="panel-autobuild-new" aria-selected="false">Auto-build</button>
    </div>

    <!-- PANEL: Presets (these set the form fields above) -->
    <div id="panel-presets-new" class="tabpanel active" role="tabpanel" aria-labelledby="tab-presets-new">
      <div style="margin-top:2px; display:flex; gap:8px; flex-wrap:wrap;">
        <button class="btn" type="button" data-preset='{"target":120,"event":"Wedding","venue":"Banquet Hall"}' title="120 min · Wedding · Banquet Hall">💍 Wedding</button>
        <button class="btn" type="button" data-preset='{"target":90,"event":"Bar/Night","venue":"Bar"}' title="90 min · Bar/Night · Bar">🍻 Bar/Night</button>
        <button class="btn" type="button" data-preset='{"target":45,"event":"Festival","venue":"Outdoor Stage"}' title="45 min · Festival · Outdoor Stage">🎪 Festival</button>
        <button class="btn" type="button" data-preset='{"target":60,"event":"Coffeehouse","venue":"Cafe"}' title="60 min · Coffeehouse · Cafe">☕ Coffeehouse</button>
      </div>
      <p class="muted" style="margin-top:8px;">Presets fill the fields above; you can tweak them before creating.</p>
    </div>

    <!-- PANEL: Auto-build options -->
    <div id="panel-autobuild-new" class="tabpanel" role="tabpanel" aria-labelledby="tab-autobuild-new">
      <div class="grid" style="margin-top:2px;">
        <p>
          <label>Vibe</label><br/>
          <select name="ab_vibe">
            <option value="mixed">Mixed</option>
            <option value="chill">Chill</option>
            <option value="energetic">Energetic</option>
          </select>
        </p>
        <p>
          <label>Avoid repeating the same artist</label><br/>
          <label style="display:inline-flex; gap:8px; align-items:center;">
            <input type="checkbox" name="ab_avoid_same_artist" value="1" checked />
            No repeat artists
          </label>
        </p>
      </div>
      <div class="grid">
        <p>
          <label>Require tags (comma-separated)</label><br/>
          <input name="ab_tags" placeholder="wedding, upbeat" />
        </p>
        <p>
          <label>Require genres (comma-separated)</label><br/>
          <input name="ab_genres" placeholder="pop, rock" />
        </p>
      </div>
      <label style="display:inline-flex; gap:8px; align-items:center;">
        <input type="checkbox" name="ab_do" value="1" checked />
        <strong>Create & auto-build now</strong>
      </label>
      <p class="muted" style="margin-top:4px;">If checked, we’ll build the setlist right after creating it.</p>
            <div id="ab_preset_block" style="margin-top:10px;">
        <label><strong>Optional section preset after auto-build</strong></label><br/>
        <select name="ab_preset" style="max-width:300px;">
          <option value="">— None —</option>
          <option value="basic">Set 1 / Break / Set 2 / Encore</option>
          <option value="three">Three-set layout (1 / Break / 2 / Break / 3 / Encore)</option>
          <option value="chunk">Chunked (by 8 songs per set)</option>
        </select>
        <p class="muted" style="margin-top:4px;">We’ll label sections using this pattern immediately after the build.</p>
      </div>
    </div>

    <script>
      (function(){
        // Tabs
        const tabs = [
          {btn: document.getElementById('tab-presets-new'),  panel: document.getElementById('panel-presets-new'),  id:'presets'},
          {btn: document.getElementById('tab-autobuild-new'),panel: document.getElementById('panel-autobuild-new'),id:'autobuild'}
        ];
        function select(which){
          tabs.forEach(t => {
            const on = (t.id === which);
            t.btn.setAttribute('aria-selected', String(on));
            t.panel.classList.toggle('active', on);
          });
        }
        tabs.forEach(t => t.btn && t.btn.addEventListener('click', () => select(t.id)));
        // Preset buttons write to the main form fields
        const form = document.getElementById('newSetForm');
        function setVal(name, val){ const el=form.querySelector('[name="'+name+'"]'); if (el) el.value = val; }
        document.querySelectorAll('#panel-presets-new .btn[data-preset]').forEach(btn=>{
          btn.addEventListener('click', ()=>{
            try{
              const p = JSON.parse(btn.getAttribute('data-preset')||'{}');
              if (p.target) setVal('target_minutes', p.target);
              if (p.event)  setVal('event_type', p.event);
              if (p.venue)  setVal('venue_type', p.venue);
              if (typeof showToast==='function') showToast('Preset applied');
            }catch(_){}
          });
        });
      })();
    </script>
  </details>

  <p style="margin-top:10px;">
    <button class="btn" type="submit">Create Setlist</button>
  </p>
</form>
"""

SETLIST_EDIT_HTML = """
<h2>Edit Setlist</h2>
<div id="slTopAnchor"></div>
<p style="margin:6px 0 12px;"><a class="btn" href="{{ url_for('setlist_requests', setlist_id=setlist.id) }}">Audience Requests{% if pending_requests %} ({{ pending_requests }}){% endif %}</a></p>
<style>.viewopts{display:none!important;}</style>
<style>
/* View Options toolbar */
.viewopts {
  display:flex; gap:8px; flex-wrap:wrap; align-items:center;
  margin:6px 0 8px;
}
.viewopts .chip {
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 10px; border:1px solid var(--btn-border, #bbb);
  border-radius:999px; background:var(--btn-bg, #fff); color:var(--text, #111);
  font-size:.92rem; line-height:1.2; cursor:pointer; user-select:none;
}
.viewopts .chip input[type="checkbox"]{ margin:0; }
.viewopts .spacer { flex:1; }

/* Dark theme awareness */
[data-theme="dark"] .viewopts .chip { border-color:#4c4f54; background:#2a2b2e; color:#e9e9ea; }
</style>
<div id="editFormBlock">
<form method="post" action="{{ url_for('update_setlist', setlist_id=setlist.id) }}">
  <style>
  #editFormBlock form{
    background:var(--card-bg);
    border:1px solid var(--card-border);
    border-radius:16px;
    padding:18px 20px 16px;
    box-shadow:var(--card-shadow);
    display:flex;
    flex-direction:column;
    gap:12px;
  }
  #editFormBlock label{ font-weight:600; color:var(--text); }
  #editFormBlock textarea{ min-height:100px; resize:vertical; }
  #editFormBlock .grid{ gap:14px; }
  @media (max-width:720px){
    #editFormBlock form{ padding:16px 14px; }
  }
  </style>
  <div class="grid">
    <p><label>Name</label><br/><input name="name" required value="{{ setlist.name }}" /></p>
    <p><label>Target Minutes</label><br/><input name="target_minutes" type="number" min="1" value="{{ setlist.target_minutes or '' }}" /></p>
  </div>
  <div class="grid">
    <p><label>Event Type</label><br/><input name="event_type" value="{{ setlist.event_type or '' }}" placeholder="Wedding, Bar, Festival..." /></p>
    <p><label>Venue Type</label><br/><input name="venue_type" value="{{ setlist.venue_type or '' }}" placeholder="Indoor, Outdoor..." /></p>
  </div>
    <p><label>Notes</label><br/><textarea name="notes" rows="2" placeholder="Any notes...">{{ setlist.notes or '' }}</textarea></p>
  <style>
/* compact, aligned control row */
.formline {
  display:flex;
  gap:12px;
  align-items:center;
  flex-wrap:wrap;
  margin:4px 0 0;
  padding-top:12px;
  border-top:1px solid var(--border);
}
.formline .spacer { flex:1; }
.formline .switch { display:inline-flex; gap:8px; align-items:center; font-weight:500; }
.formline .switch input { margin:0; }
</style>
<div class="formline">
  <label class="switch" title="Skip adding songs when the same artist is already in this setlist">
    <input type="checkbox" name="no_repeat_artists" value="1" {{ 'checked' if setlist.no_repeat_artists else '' }} />
    No repeat artists
  </label>

  <label class="switch" title="In Print/PDF, numbering restarts at each section header">
    <input type="checkbox" name="reset_numbering_per_section" value="1" {% if setlist.reset_numbering_per_section %}checked{% endif %} />
    Reset numbering per section
  </label>

  <span class="spacer"></span>
  <button class="btn" type="submit">Save Details</button>
</div>
</form>
</div>
<div class="viewopts" role="group" aria-label="View options">
  <label class="chip" title="Move the current setlist card above the builder panels">
    <input type="checkbox" id="setlistFirstToggle" />
    Setlist at top
  </label>

  <label class="chip" title="Keep Event Presets & Auto-build collapsed by default">
    <input type="checkbox" id="compactModeToggle" />
    Compact extras
  </label>

  <label class="chip" title="Hide per-song notes in Print/PDF exports">
    <input type="checkbox" id="hideNotesToggle" />
    Hide notes (print)
  </label>

  <span class="spacer" aria-hidden="true"></span>
  <a class="btn" href="#current-setlist" title="Jump down to the current setlist">🎯 Current setlist</a>
</div>

<div class="section" style="margin-top:16px;">
  <h3>Export & Share</h3>
  <form method="get" action="{{ url_for('export_setlist_pdf', setlist_id=setlist.id) }}" target="_blank" style="display:flex; flex-wrap:wrap; gap:12px; align-items:center;">
    <label style="display:flex; gap:6px; align-items:center;">
      <input type="checkbox" name="include_charts" value="1" checked /> Append charts
    </label>
    <label style="display:flex; gap:6px; align-items:center;">
      <input type="checkbox" name="include_requests" value="1" {% if pending_requests %}checked{% endif %} /> Include request summary
    </label>
    <label style="display:flex; gap:6px; align-items:center;">
      <input type="checkbox" name="hide_notes" value="1" /> Hide notes
    </label>
    <button class="btn" type="submit">⬇️ PDF</button>
    <span class="muted" style="flex:1; min-width:220px;">PDF always includes the song list summary; charts are appended when available.</span>
  </form>
  <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;">
    <a class="btn" href="{{ url_for('export_songs_csv') }}">⬇️ Songs CSV</a>
    <a class="btn" href="{{ url_for('export_setlist_csv', setlist_id=setlist.id) }}">⬇️ Setlist CSV</a>
    <a class="btn" href="{{ url_for('export_setlist_chopro', setlist_id=setlist.id) }}">ChordPro Text</a>
    <a class="btn" href="{{ url_for('export_setlist_chopro_zip', setlist_id=setlist.id) }}">ChordPro Pack (.zip)</a>
    <a class="btn" href="{{ url_for('setlist_requests', setlist_id=setlist.id) }}" target="_blank">Requests Console{% if pending_requests %} ({{ pending_requests }}){% endif %}</a>
    <a class="btn" href="{{ url_for('view_setlist', setlist_id=setlist.id) }}" target="_blank">Viewer</a>
  </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function(){
  const SETLIST_ID = {{ setlist.id }};

  // -------------------------
  // Compact Mode (persisting)
  // -------------------------
  (function(){
    const KEY = 'sg:compact:' + String(SETLIST_ID);
    const cb = document.getElementById('compactModeToggle');
    const panels = [
  document.getElementById('sec-buildtools'),
  document.getElementById('sec-addsongs'),
  document.getElementById('tools-panel'),
].filter(Boolean);

    function apply(compact){
      if (!panels.length) return;
      if (compact){
        // Close collapsible sections when compact is ON
        panels.forEach(p => p.removeAttribute('open'));
      }
      // When OFF we respect the user's manual open/close state
    }

    let saved = false;
    try { saved = JSON.parse(localStorage.getItem(KEY) || 'false'); } catch(_){}
    cb.checked = !!saved;
    apply(cb.checked);

    cb.addEventListener('change', () => {
      try { localStorage.setItem(KEY, JSON.stringify(cb.checked)); } catch(_){}
      apply(cb.checked);
      if (typeof showToast === 'function') showToast(cb.checked ? 'Compact mode on' : 'Compact mode off');
    });
  })();

   // -----------------------------------------
  // Show Current Setlist at top (persisting)
  // -----------------------------------------
  (function(){
    const KEY = 'sg:setlistFirst:' + String(SETLIST_ID);
    const cb = document.getElementById('setlistFirstToggle');

    function targetBeforeNode(){
      // Insert before the first of these sections if present
      return document.getElementById('sec-event-presets')
          || document.getElementById('sec-autobuild')
          || document.getElementById('sec-addsongs')
          || null;
    }

    function apply(atTop){
      const block = document.getElementById('current-setlist');
      if (!block || !block.parentNode) return;

      if (atTop){
        const before = targetBeforeNode();
        if (before && before.parentNode){
          before.parentNode.insertBefore(block, before);
        } else {
          block.parentNode.insertBefore(block, block.parentNode.firstChild);
        }
      } else {
        // Restore to just after Add songs → else Auto-build → else Event Presets
        const afterA = document.getElementById('sec-addsongs');
        const afterB = document.getElementById('sec-autobuild');
        const afterC = document.getElementById('sec-event-presets');
        const anchor = afterA || afterB || afterC;
        if (anchor && anchor.parentNode){
          anchor.parentNode.insertBefore(block, anchor.nextSibling);
        }
      }
    }

    // NEW: default ON if no prior preference stored
    let initial = true; // default to showing the setlist at the top
    try {
      const raw = localStorage.getItem(KEY);
      if (raw !== null) {
        initial = !!JSON.parse(raw);
      }
    } catch(_){ /* keep default */ }

    cb.checked = initial;
    apply(cb.checked);

    cb.addEventListener('change', () => {
      try { localStorage.setItem(KEY, JSON.stringify(cb.checked)); } catch(_){}
      apply(cb.checked);
      if (typeof showToast === 'function') showToast(cb.checked ? 'Setlist moved to top' : 'Setlist restored');
    });
  })();
</script>
<details class="section" id="sec-buildtools">
  <summary>
    ⚙️ Build tools
    <span class="muted summary-note">(presets & auto-build)</span>
  </summary>

  <style>
    .tabs { display:flex; gap:6px; border-bottom:1px solid var(--border,#ddd); margin:6px 0 10px; }
    .tabs .tab {
      padding:6px 10px; border:1px solid var(--border,#ddd); border-bottom:none;
      border-radius:8px 8px 0 0; cursor:pointer; background:var(--btn-bg,#fff);
    }
    .tabs .tab[aria-selected="true"] { font-weight:600; }
    .tabpanel { display:none; }
    .tabpanel.active { display:block; }
    [data-theme="dark"] .tabs .tab { background:#2a2b2e; border-color:#4c4f54; }
  </style>

  <div class="tabs" role="tablist" aria-label="Build tools">
    <button class="tab" id="tab-presets" role="tab" aria-controls="panel-presets" aria-selected="true">Event presets</button>
    <button class="tab" id="tab-autobuild" role="tab" aria-controls="panel-autobuild" aria-selected="false">Auto-build</button>
  </div>

  <!-- PANEL: Event Presets -->
  <div id="panel-presets" class="tabpanel active" role="tabpanel" aria-labelledby="tab-presets">
    <div style="margin-top:2px; display:flex; gap:8px; flex-wrap:wrap;">

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

      <!-- Coffeehouse preset -->
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

  <!-- PANEL: Auto-build -->
  <div id="panel-autobuild" class="tabpanel" role="tabpanel" aria-labelledby="tab-autobuild">
    <div style="margin-top:2px;">
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
        <p style="display:flex; gap:16px; align-items:center; flex-wrap:wrap;">
          <label style="display:inline-flex; gap:8px; align-items:center; margin:0;">
            <input type="checkbox" name="avoid_same_artist" value="1" checked />
            Avoid repeating the same artist
          </label>
          <label style="display:inline-flex; gap:8px; align-items:center; margin:0;">
            <input type="checkbox" name="clear_first" value="1" />
            Clear current setlist first
          </label>
        </p>
        <p><button class="btn" type="submit">Auto-build</button></p>
      </form>
    </div>
  </div>

  <script>
    (function(){
      const tabs = Array.from(document.querySelectorAll('#sec-buildtools .tab'));
      const panels = {
        presets: document.getElementById('panel-presets'),
        autobuild: document.getElementById('panel-autobuild')
      };
      function select(which){
        tabs.forEach(t => t.setAttribute('aria-selected', String(t.id==='tab-'+which)));
        panels.presets.classList.toggle('active', which==='presets');
        panels.autobuild.classList.toggle('active', which==='autobuild');
      }
      document.getElementById('tab-presets').addEventListener('click', () => select('presets'));
      document.getElementById('tab-autobuild').addEventListener('click', () => select('autobuild'));
    })();
  </script>
</details>
<!-- Compact actions toolbar -->
<div class="section" id="actionsBar" style="margin:8px 0; padding:8px; display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
  <!-- Primary quick actions -->
  <a class="btn" href="{{ url_for('live_mode', setlist_id=setlist.id) }}" target="_blank" title="Open Live Mode">🎛️ Live Mode</a>
  <a class="btn" href="{{ url_for('print_setlist', setlist_id=setlist.id) }}" target="_blank" title="Open print-friendly view">🖨️ Print</a>

  <!-- Export menu -->
  <div class="dropdown" id="exportMenu">
    <button class="btn" type="button" data-dd-toggle="export">⬇️ Export ▾</button>
    <div class="dd" data-dd="export" role="menu" style="display:none;">
      <a class="item" href="{{ url_for('export_setlist_pdf', setlist_id=setlist.id) }}" title="Download PDF">PDF</a>
      <a class="item" href="{{ url_for('export_setlist_csv', setlist_id=setlist.id) }}" title="Download CSV">CSV</a>
    </div>
  </div>

  <!-- Share menu -->
  <div class="dropdown" id="shareMenu">
    <button class="btn" type="button" data-dd-toggle="share">🔗 Share ▾</button>
    <div class="dd" data-dd="share" role="menu" style="display:none;">
      <a class="item" href="{{ url_for('view_setlist', setlist_id=setlist.id) }}" target="_blank">Share View (internal)</a>
      <a class="item" href="{{ url_for('create_share_link', setlist_id=setlist.id) }}">Create/Copy Secret Link</a>
      <a class="item" href="{{ url_for('rotate_share_link', setlist_id=setlist.id) }}" onclick="return confirm('Rotate (revoke) the existing secret link and create a new one?');">Rotate Secret Link</a>
      <a class="item" href="{{ url_for('share_qr', setlist_id=setlist.id) }}" target="_blank">Show QR</a>
    </div>
  </div>

  <span class="spacer" style="flex:1;"></span>

  <!-- Duplicates (kept as buttons) -->
  <form class="inline" method="post" action="{{ url_for('duplicate_setlist', setlist_id=setlist.id) }}">
    <button class="btn" type="submit" title="Duplicate this setlist">📄 Duplicate</button>
  </form>
  <form class="inline" method="post" action="{{ url_for('duplicate_setlist_newshow', setlist_id=setlist.id) }}">
    <button class="btn" type="submit" title="Copy setlist but clear per-song notes and section labels">🆕 Duplicate as New Show</button>
  </form>
</div>

<style>
  /* Dropdown styles (light/dark aware via existing theme vars) */
  .dropdown { position: relative; }
  .dropdown .dd{
    position:absolute; top:100%; left:0; min-width:180px;
    background: var(--btn-bg,#fff); color: var(--text,#111);
    border:1px solid var(--btn-border,#bbb); border-radius:10px;
    box-shadow: 0 8px 24px rgba(0,0,0,.12);
    padding:6px; z-index: 10;
  }
  .dropdown .dd .item{
    display:block; padding:6px 8px; border-radius:8px; text-decoration:none;
    color: inherit; border:1px solid transparent;
  }
  .dropdown .dd .item:hover{ background: var(--surface,#f7f7f8); border-color: var(--border,#ddd); }
</style>

<script>
  // Tiny dropdown controller; also closes on outside click / Esc
  (function(){
    function openWhich(name, open){
      document.querySelectorAll('.dropdown .dd').forEach(dd=>{
        const isTarget = dd.getAttribute('data-dd')===name;
        dd.style.display = (open && isTarget) ? 'block' : 'none';
      });
    }
    document.addEventListener('click', (e)=>{
      const btn = e.target.closest('[data-dd-toggle]');
      if (btn){
        const name = btn.getAttribute('data-dd-toggle');
        const dd   = document.querySelector('.dd[data-dd="'+name+'"]');
        const willOpen = dd && dd.style.display!=='block';
        openWhich(name, willOpen);
      } else {
        // Clicked elsewhere → close all
        openWhich('', false);
      }
    });
    document.addEventListener('keydown', (e)=>{
      if (e.key==='Escape') openWhich('', false);
    });
  })();
</script>
<p style="margin:6px 0; display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
  <a class="btn" href="#current-setlist">🎯 Jump to Current Setlist</a>
</p>
<script>
<style>
.viewopts .chip:has(input:focus-visible){
  outline:2px solid #6aa9ff; outline-offset:2px;
}
[data-theme="dark"] .viewopts .chip:has(input:focus-visible){
  outline-color:#8ec2ff;
}
</style>
(function(){
  const SETLIST_ID = {{ setlist.id }};
  const KEY = 'sg:hideNotesPrint:' + String(SETLIST_ID);

  const cb = document.getElementById('hideNotesToggle');
  const pdfLink = document.querySelector('a.btn[href$="/export.pdf"]');
  const printLink = document.querySelector('a.btn[href*="/print"]');

  if (!cb || !(pdfLink || printLink)) return;

  // helpers
  function setParam(url, key, val) {
    try {
      const u = new URL(url, window.location.origin);
      if (val === null) u.searchParams.delete(key);
      else u.searchParams.set(key, String(val));
      return u.pathname + (u.search ? u.search : '') + (u.hash || '');
    } catch (_) {
      // fallback simple appender
      if (val === null) return url.replace(new RegExp('[?&]'+key+'=1\\b'), '');
      return url + (url.includes('?') ? '&' : '?') + key + '=1';
    }
  }

  function applyToLinks(checked) {
    if (pdfLink)  pdfLink.href  = setParam(pdfLink.getAttribute('href'),  'hide_notes', checked ? 1 : null);
    if (printLink) printLink.href = setParam(printLink.getAttribute('href'), 'hide_notes', checked ? 1 : null);
  }

  // init from storage
  let saved = false;
  try { saved = JSON.parse(localStorage.getItem(KEY) || 'false'); } catch(_) {}
  cb.checked = !!saved;
  applyToLinks(cb.checked);

  // persist + update links on change
  cb.addEventListener('change', () => {
    try { localStorage.setItem(KEY, JSON.stringify(cb.checked)); } catch(_) {}
    applyToLinks(cb.checked);
    if (typeof showToast === 'function') showToast(cb.checked ? 'Notes hidden in Print/PDF' : 'Notes shown in Print/PDF');
  });
})();
</script>
<script>
(function(){
  const setlistId = {{ setlist.id }};
  const songCount = {{ setlist.songs|length }};
  let compact = false;
  try { compact = JSON.parse(localStorage.getItem('sg:compact:' + setlistId) || 'false'); } catch(_) {}

  // Smart rule: if there are already several songs OR compact mode is on,
  // keep these extras closed initially.
  const shouldCollapse = compact || (songCount >= 3);

  if (shouldCollapse) {
    document.querySelectorAll('details.section[data-smart="1"]').forEach(d => {
      d.removeAttribute('open');
    });
  }
})();
</script>
<style>
/* ===== Add songs panel (scoped to #sec-addsongs) ===== */

/* Row layout */
#sec-addsongs .row {
  border: none !important;
  padding: 8px 0;
  display: flex;
  align-items: flex-start;
  gap: 12px;
}
#sec-addsongs .row label {
  display: flex;
  align-items: flex-start;
  gap: 8px;
}
#sec-addsongs .row input[type="checkbox"] {
  margin-top: 2px;
  transform: translateY(1px);
}
#sec-addsongs .row .muted {
  font-size: 12px;
  color: var(--muted);
}

/* Right-side meta column on wide screens */
#sec-addsongs .row > .muted {
  margin-left: auto;
  white-space: nowrap;
  min-width: 220px;
  text-align: right;
}

/* Search controls in this panel only */
#sec-addsongs form[action*="/setlists/"][method="get"] { margin: 6px 0 8px; }
#sec-addsongs form[action*="/setlists/"][method="get"] .btn {
  padding: 4px 8px;
  font-size: 12px;
}
#sec-addsongs input[name="q"] { max-width: 420px; }

/* Keep labels tidy inside Add Selected form */
#sec-addsongs form[action$="/add_songs"] .row label { align-items: baseline; }
#sec-addsongs form[action$="/add_songs"] .row strong { font-weight: 600; }

/* “dupe artist” badge */
#sec-addsongs .badge {
  display: inline-block;
  margin-left: 8px;
  padding: 2px 8px;
  font-size: 11px;
  border-radius: 999px;
  border: 1px solid var(--field-border);
  background: var(--field-bg);
  color: var(--text);
  opacity: 0.8;
  vertical-align: middle;
}

/* Sticky Add Selected bar */
#sec-addsongs .addselbar {
  position: sticky;
  bottom: 0;
  display: flex;
  gap: 12px;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  margin-top: 8px;
  border: 1px solid var(--card-border);
  border-radius: 12px;
  background: var(--card-bg);
  box-shadow: var(--card-shadow);
  color: var(--text);
}
#sec-addsongs .addselbar .count { font-weight: 600; }
#sec-addsongs .addselbar .btn {
  padding: 8px 12px;
  background: var(--btn-bg);
  color: var(--btn-text);
  border-color: var(--btn-border);
}

/* Mobile */
@media (max-width: 720px) {
  #sec-addsongs .row { flex-direction: column; align-items: flex-start; }
  #sec-addsongs .row > .muted {
    text-align: left;
    margin: 2px 0 0 28px;
    min-width: 0;
    white-space: normal;
  }
}
</style>
<details class="section" id="sec-addsongs">
  <summary>
    Add songs to this setlist
  </summary>

  {% if setlist.no_repeat_artists %}
    <p class="muted" style="margin:6px 0 2px;">
      No-repeat artists is ON — songs by artists already in this show will be skipped.
    </p>
  {% endif %}

  <!-- Search -->
  <form method="get" action="{{ url_for('edit_setlist', setlist_id=setlist.id) }}">
    <input name="q" placeholder="Search your songs…" value="{{ q or '' }}" />
    <p style="margin-top:8px;">
      <button class="btn" type="submit">Search</button>
      <a class="btn" href="{{ url_for('edit_setlist', setlist_id=setlist.id) }}">Clear</a>
    </p>
  </form>

  <!-- List + add selected -->
  <form method="post" action="{{ url_for('add_songs_to_setlist', setlist_id=setlist.id) }}">
    {% if songs %}
      {% for s in songs %}
        {% set is_dup = (setlist.no_repeat_artists and s.artist and (s.artist.lower() in existing_artists)) %}
        <div class="row" data-dup="{{ '1' if is_dup else '0' }}">
          <label>
            <input type="checkbox" name="song_ids" value="{{ s.id }}" {% if is_dup %}disabled aria-disabled="true"{% endif %} />
            <span>
              <strong>{{ s.title }}</strong> — {{ s.artist }}
              {% if is_dup %}
                <span class="badge" title="Artist already in this setlist; skipped when adding">dupe artist</span>
              {% endif %}
            </span>
          </label>
          <div class="muted">
            {% if s.musical_key %}Key: {{ s.musical_key }} · {% endif %}
            {% if s.genre %}{{ s.genre }}{% endif %}
          </div>
        </div>
      {% endfor %}

      <!-- Sticky add/clear bar -->
      <div id="addSelBar" class="addselbar" hidden>
        <span class="count">0 selected</span>
        <div style="display:flex; gap:8px;">
          <button type="button" class="btn" id="addSelSubmit">Add Selected</button>
          <button type="button" class="btn" id="addSelClear">Clear</button>
        </div>
      </div>
    {% else %}
      <p class="muted">No songs match your search.</p>
    {% endif %}
  </form>
</details>
<script>
(function(){
  const panel = document.getElementById('sec-addsongs');
  if (!panel) return;

  const form = panel.querySelector('form[action$="/add_songs"]');
  if (!form) return;

  const bar = document.getElementById('addSelBar');
  const countEl = bar.querySelector('.count');
  const addBtn = document.getElementById('addSelSubmit');
  const clearBtn = document.getElementById('addSelClear');

  function boxes(){ return Array.from(form.querySelectorAll('input[type="checkbox"][name="song_ids"]')); }
  function update(){
    const n = boxes().filter(b => b.checked).length;
    countEl.textContent = n + ' selected';
    bar.hidden = n === 0;
  }
  form.addEventListener('change', (e) => {
    if (e.target && e.target.matches('input[type="checkbox"][name="song_ids"]')) update();
  });
  addBtn.addEventListener('click', () => {
    if (boxes().some(b => b.checked)) form.submit();
    else if (typeof showToast === 'function') showToast('Select at least one song');
  });
  clearBtn.addEventListener('click', () => { boxes().forEach(b => b.checked = false); update(); });
  update();
})();
</script>
<!-- Visual card polish for Current setlist (light/dark aware) -->
<style>
  /* Minimal, non-card styling for Current setlist */
  #current-setlist {
    background: transparent;
    border: 0;
    padding: 4px 0 6px;
    box-shadow: none;
    margin-top: 6px;
    position: relative;
  }
  /* slender accent bar on the left */
  #current-setlist::before {
    content: "";
    position: absolute;
    top: 0; bottom: 0; left: -8px;
    width: 3px;
    border-radius: 3px;
    background: rgba(0,0,0,.08);
  }
  /* gentle row separators */
  #current-setlist .row {
    border-bottom: 1px solid rgba(0,0,0,.06);
    padding: 8px 0;
  }
  #current-setlist .row:last-of-type { border-bottom: none; }
  #current-setlist > h3 { margin: 0 0 8px; }

  /* Dark theme overrides (explicit app theme) */
  [data-theme="dark"] #current-setlist::before { background: rgba(255,255,255,.10); }
  [data-theme="dark"] #current-setlist .row    { border-bottom-color: rgba(255,255,255,.12); }

  /* Fallback for system dark when no data-theme is set */
  @media (prefers-color-scheme: dark) {
    html:not([data-theme]) #current-setlist::before { background: rgba(255,255,255,.10); }
    html:not([data-theme]) #current-setlist .row    { border-bottom-color: rgba(255,255,255,.12); }
  }
</style>
<style>
  /* Compact buttons and chips ONLY within the Current setlist area */
  #current-setlist .btn {
    padding: 4px 8px;
    font-size: 12px;
    line-height: 1.1;
    border-radius: 8px;
  }
  #current-setlist .iconbtn.btn {
    padding: 3px 6px;
    font-size: 12px;
  }
  /* Give tool rows a little breathing room */
  #current-setlist .row + .row { margin-top: 2px; }

  /* Section overview chips on one tidy line with gentle wrap */
  #current-setlist .muted a.btn,
  #current-setlist a.btn[href^="#sec-"] {
    padding: 2px 8px;
    font-size: 12px;
    border-color: #ddd;
    color: inherit;
  }
  /* Reduce vertical gaps between toolbars at the very top of Current setlist */
  #current-setlist > .btn,
  #current-setlist form.inline {
    margin-bottom: 4px;
  }

  /* Dark theme subtle borders */
  [data-theme="dark"] #current-setlist a.btn[href^="#sec-"],
  [data-theme="dark"] #current-setlist .muted a.btn { border-color: #3a3a3a; }

  /* Small screens: keep things readable but compact */
  @media (max-width: 720px) {
    #current-setlist .btn { font-size: 11.5px; padding: 4px 7px; }
    #current-setlist .iconbtn.btn { padding: 3px 5px; }
  }
</style>
<style>
  /* Compact summary look */
  #tools-panel > summary { list-style: none; }
  #tools-panel > summary::-webkit-details-marker { display: none; }
  #tools-panel[open] > summary::after { content: ' ▾'; }
  #tools-panel:not([open]) > summary::after { content: ' ▸'; }
  #tools-panel .btn { margin-right: 4px; margin-bottom: 6px; }
</style>
<div id="current-setlist" class="section">
  <h3>Current order ({{ setlist.songs|length }} songs) · ~Total {{ total_str }}</h3>
  <details id="tools-panel" class="section" open>
  <summary>
    🧰 Tools
  </summary>
  <div style="margin-top:6px;">
  <form class="inline" method="post" action="{{ url_for('clear_all_sections_v2', setlist_id=setlist.id) }}" style="margin:6px 0 10px;" onsubmit="return confirm('Remove ALL section labels?');">
  <button class="btn btn-danger" type="submit" title="Remove every section header from this setlist">🗑️ Clear All Sections</button>
</form>
  <form class="inline" method="post" action="{{ url_for('apply_section_preset', setlist_id=setlist.id) }}" style="margin:6px 0 10px;">
  <button class="btn" type="submit" title="Insert Set 1 / Break / Set 2 / Encore at sensible positions">✨ Section Preset</button>
  <span class="muted">One click: Set 1 / Break / Set 2 / Encore</span>
</form>
<form class="inline" method="post" action="{{ url_for('preset_sections_three_sets', setlist_id=setlist.id) }}" style="margin:6px 0 10px;">
  <button class="btn" type="submit" title="Insert Set 1 / Break / Set 2 / Break / Set 3 / Encore at sensible positions">✨ Three-Set Preset</button>
  <span class="muted">Set 1 / Break / Set 2 / Break / Set 3 / Encore</span>
</form>
<form class="inline" method="post" action="{{ url_for('preset_sections_by_chunk', setlist_id=setlist.id) }}" style="margin:6px 0 10px; display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
  <label class="muted" style="display:flex; align-items:center; gap:6px;">
    Songs per set
    <input name="chunk" type="number" min="1" max="50" value="8" style="width:80px; padding:6px;">
  </label>
  <button class="btn" type="submit" title="Apply Set/Break blocks every N songs and add an Encore at the end">✨ Chunk Preset</button>
  <span class="muted">Sets of N songs → Break → next Set → … → Encore</span>
</form>
<form class="inline" method="post" action="{{ url_for('renumber_sets', setlist_id=setlist.id) }}" style="margin:6px 0 10px;">
  <button class="btn" type="submit" title="Rename visible 'Set …' headers sequentially after reordering">🔢 Renumber Set Labels</button>
  <span class="muted">Updates “Set …” headers to Set 1, Set 2, Set 3…</span>
</form>
<form class="inline" method="post" action="{{ url_for('fix_encore', setlist_id=setlist.id) }}" style="margin:6px 0 10px;">
  <button class="btn" type="submit" title="Remove stray Encores and ensure a single Encore on the last song">🎯 Fix Encore</button>
  <span class="muted">Ensures exactly one Encore at the end</span>
</form>
    {% if section_overview and section_overview|length > 0 %}
  <div class="muted" style="margin:6px 0 10px; display:flex; gap:8px; flex-wrap:wrap;">
    {% for sec in section_overview %}
      <a class="btn" href="#sec-{{ loop.index0 }}" style="padding:4px 8px; border-color:#ddd; text-decoration:none;">
        {{ sec.name }} — {{ sec.count }} song{{ '' if sec.count == 1 else 's' }} · {{ fmt_mmss(sec.sec) }}
      </a>
    {% endfor %}
  </div>
  <p style="margin-top:10px;">
  <a class="btn" href="#top">⬆️ Back to top</a>
</p>
  <form class="inline" method="post" action="{{ url_for('normalize_sections', setlist_id=setlist.id) }}" style="margin:4px 0 10px;">
  <button class="btn" type="submit" title="Keep only the first occurrence of a repeated section label">🧼 Clean Section Headers</button>
  <span class="muted">Removes duplicate consecutive headers with the same name.</span>
</form>
{% endif %}
  {% if setlist.songs %}
  {% set first_id = setlist.songs[0].song.id %}
  <div style="margin:8px 0; display:flex; gap:8px; flex-wrap:wrap;">
      <!-- One-click Section Preset -->
    <form class="inline" method="post" action="{{ url_for('apply_section_preset', setlist_id=setlist.id) }}">
      <button class="btn" type="submit"
              title="Clears current section labels and inserts: Set 1 (top), Break (middle), Set 2 (after break), Encore (last)">
        ⚡️ Section Preset: Set 1 / Break / Set 2 / Encore
      </button>
    </form>
        <!-- Clear all section labels -->
    <form class="inline" method="post" action="{{ url_for('clear_all_sections', setlist_id=setlist.id) }}"
          onsubmit="return confirm('Remove ALL section labels from this setlist?');">
      <button class="btn btn-danger" type="submit" title="Remove all section labels from every row">
        🧽 Clear All Section Labels
      </button>
    </form>
    <form class="inline" method="post" action="{{ url_for('update_setlist_section', setlist_id=setlist.id, song_id=first_id) }}">
      <input type="hidden" name="section_name" value="Set 1">
      <button class="btn" type="submit">🏁 Start “Set 1” at top</button>
    </form>
    <form class="inline" method="post" action="{{ url_for('update_setlist_section', setlist_id=setlist.id, song_id=first_id) }}">
      <input type="hidden" name="section_name" value="Encore">
      <button class="btn" type="submit">✨ Start “Encore” at top</button>
    </form>
  </div>
{% endif %}
    <p><a class="btn" href="{{ url_for('undo_order_route', setlist_id=setlist.id) }}">↩️ Undo Last Change</a></p>
    <p id="reorderStatus" class="muted" style="display:none;"></p>
    <p id="sectionCollapseToolbar" style="margin:6px 0;">
  <button class="btn" type="button" id="btnCollapseAll">➖ Collapse all sections</button>
  <button class="btn" type="button" id="btnExpandAll">➕ Expand all sections</button>
</p>
<script>
(function(){
  const collapseBtn = document.getElementById('btnCollapseAll');
  const expandBtn = document.getElementById('btnExpandAll');
  if (!collapseBtn || !expandBtn) return;

  function sectionHeaders(){
    // same detection as Step 101: section headers contain a .secmeta span
    return Array.from(document.querySelectorAll('.row')).filter(r => r.querySelector('.secmeta'));
  }

  function collapseAll(){
    sectionHeaders().forEach(h => { if (h.dataset.collapsed !== '1') h.click(); });
    if (typeof showToast === 'function') showToast('Collapsed all sections');
  }

  function expandAll(){
    sectionHeaders().forEach(h => { if (h.dataset.collapsed === '1') h.click(); });
    if (typeof showToast === 'function') showToast('Expanded all sections');
  }

  collapseBtn.addEventListener('click', collapseAll);
  expandBtn.addEventListener('click', expandAll);
})();
</script>
   <p><button class="btn" type="button" id="toggleReorder">🧲 Reorder mode: Off</button></p>
     </div>
</details>
<script>
(function(){
  const panel = document.getElementById('tools-panel');
  if (!panel) return;

  // Persisted open/closed per-setlist
  const KEY = 'sg:toolsOpen:' + String({{ setlist.id }});

  function save() {
    try { localStorage.setItem(KEY, JSON.stringify(panel.open)); } catch(e){}
  }
  function load() {
    try { return JSON.parse(localStorage.getItem(KEY) || 'null'); } catch(e){ return null; }
  }

  // First-time smart default: closed if many songs, open if tiny/empty
  const saved = load();
  if (saved === null) {
    panel.open = ({{ setlist.songs|length }} <= 2);
  } else {
    panel.open = !!saved;
  }

  panel.addEventListener('toggle', save);
})();
</script>
  {% if setlist.songs %}
    {% for ss in setlist.songs %}

  {# ---- SECTION DIVIDER (shows before the labeled song) ---- #}
  {% if ss.section_name %}
  {% set __sec_index = (__sec_index + 1) if __sec_index is defined else 0 %}
  {% set info = section_overview[__sec_index] if section_overview and __sec_index < section_overview|length else None %}
  <div id="sec-{{ __sec_index }}" class="row section-divider">
    <div class="section-divider__inner">
      <div class="section-divider__label">
        <span class="section-divider__name">— {{ ss.section_name }} —</span>
        {% if info %}
          <span class="secmeta">
            {{ info.count }} · {{ fmt_mmss(info.sec) }}
          </span>
        {% endif %}
      </div>
      <div class="section-divider__actions">
        <!-- Move section UP -->
        <form class="inline" method="post" action="{{ url_for('move_section_block', setlist_id=setlist.id) }}">
          <input type="hidden" name="idx" value="{{ __sec_index }}">
          <input type="hidden" name="dir" value="up">
          <button class="btn iconbtn" type="submit" title="Move section up">⬆️</button>
        </form>
        <!-- Move section DOWN -->
        <form class="inline" method="post" action="{{ url_for('move_section_block', setlist_id=setlist.id) }}">
          <input type="hidden" name="idx" value="{{ __sec_index }}">
          <input type="hidden" name="dir" value="down">
          <button class="btn iconbtn" type="submit" title="Move section down">⬇️</button>
        </form>
      </div>
    </div>
  </div>
{% endif %}

  {# ---- SONG ROW ---- #}
  <div class="row dnd-item" data-song-id="{{ ss.song.id }}" data-locked="{{ 1 if ss.locked else 0 }}">
    <div style="flex:1;">
      #{{ ss.position }} — <strong>{{ ss.song.title }}</strong> — {{ ss.song.artist }}
      <div class="muted">
        ~{{ estimates[ss.song.id] }}
        {% if ss.notes %} · <em>Notes:</em> {{ ss.notes }}{% endif %}
        {% if ss.song.release_year %} · Year: {{ ss.song.release_year }}{% endif %}
        {% if ss.song.files|length %}<span title="Has PDF">📄</span>{% endif %}
      </div>

      <!-- per-row NOTES form (already existed) -->
      <form class="inline" method="post" action="{{ url_for('update_setlist_song_notes', setlist_id=setlist.id, song_id=ss.song.id) }}" style="margin-top:6px; display:flex; gap:6px; align-items:center;">
        <input name="notes" placeholder="capo 2 · start on chorus · stop at 2:45" value="{{ ss.notes or '' }}" style="max-width:420px;" />
        <button class="btn" type="submit" title="Save notes for this song">💾 Save</button>
      </form>

           <!-- Per-row SECTION label form + quick chips -->
      <form class="inline"
      method="post"
      action="{{ url_for('update_setlist_section', setlist_id=setlist.id, song_id=ss.song.id) }}"
      style="margin-top:6px; display:flex; gap:6px; align-items:center; flex-wrap:wrap;">

  <input name="section_name"
         placeholder="Set 1 / Break / Encore"
         value="{{ ss.section_name or '' }}"
         style="max-width:220px;" />

  <button class="btn" type="submit" title="Set/rename section starting at this song">🏷️ Label here</button>

  {% if ss.section_name %}
    <button class="btn btn-danger" name="section_name" value="" type="submit" title="Remove section label">✖︎</button>
  {% endif %}

  <div class="muted" style="display:flex; gap:6px; flex-wrap:wrap; margin-left:6px;">
    <button class="btn iconbtn" type="submit" name="section_name" value="Set 1">Set 1</button>
    <button class="btn iconbtn" type="submit" name="section_name" value="Break">Break</button>
    <button class="btn iconbtn" type="submit" name="section_name" value="Set 2">Set 2</button>
    <button class="btn iconbtn" type="submit" name="section_name" value="Encore">Encore</button>
  </div>
</form>
    </div>
    <!-- Quick Section presets (collapsed) -->
<details class="rowmore" style="margin-top:4px;">
  <summary class="btn" style="display:inline-block; padding:4px 8px;">More…</summary>
  <div style="margin-top:6px; display:flex; gap:6px; flex-wrap:wrap;">
    <form class="inline" method="post" action="{{ url_for('update_setlist_section', setlist_id=setlist.id, song_id=ss.song.id) }}">
      <input type="hidden" name="section_name" value="Set 1" />
      <button class="btn iconbtn" type="submit" title="Start Set 1 here">🏁 Set 1</button>
    </form>
    <form class="inline" method="post" action="{{ url_for('update_setlist_section', setlist_id=setlist.id, song_id=ss.song.id) }}">
      <input type="hidden" name="section_name" value="Break" />
      <button class="btn iconbtn" type="submit" title="Insert Break here">☕ Break</button>
    </form>
    <form class="inline" method="post" action="{{ url_for('update_setlist_section', setlist_id=setlist.id, song_id=ss.song.id) }}">
      <input type="hidden" name="section_name" value="Set 2" />
      <button class="btn iconbtn" type="submit" title="Start Set 2 here">🎬 Set 2</button>
    </form>
    <form class="inline" method="post" action="{{ url_for('update_setlist_section', setlist_id=setlist.id, song_id=ss.song.id) }}">
      <input type="hidden" name="section_name" value="Encore" />
      <button class="btn iconbtn" type="submit" title="Start Encore here">✨ Encore</button>
    </form>
    <form class="inline" method="post" action="{{ url_for('update_setlist_section', setlist_id=setlist.id, song_id=ss.song.id) }}">
      <input type="hidden" name="section_name" value="" />
      <button class="btn iconbtn" type="submit" title="Remove section label at this row">🧹 Clear</button>
    </form>
  </div>
</details>
    <div class="right">
      <form class="inline" method="post" action="{{ url_for('setlist_toggle_lock', setlist_id=setlist.id, song_id=ss.song.id) }}">
        <button class="btn iconbtn" title="{{ 'Unlock to allow moves' if ss.locked else 'Lock to prevent moves' }}" type="submit">{{ '🔒' if ss.locked else '🔓' }}</button>
      </form>
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
  // Click-to-collapse/expand sections based on the presence of .secmeta in the header row
  const sectionHeaders = Array.from(document.querySelectorAll('.row')).filter(r => r.querySelector('.secmeta'));
  if (!sectionHeaders.length) return;

  sectionHeaders.forEach(header => {
    header.style.cursor = 'pointer';
    header.setAttribute('title', 'Click to collapse/expand this section');
    header.addEventListener('click', () => toggleSection(header));
  });

  function toggleSection(header) {
    const collapsed = header.dataset.collapsed === '1';
    let el = header.nextElementSibling;

    if (!collapsed) {
      // Hide rows until the next section header
      while (el && !el.querySelector('.secmeta')) {
        if (el.classList.contains('row')) el.style.display = 'none';
        el = el.nextElementSibling;
      }
      header.dataset.collapsed = '1';
    } else {
      // Show rows until the next section header
      while (el && !el.querySelector('.secmeta')) {
        if (el.classList.contains('row')) el.style.display = '';
        el = el.nextElementSibling;
      }
      header.dataset.collapsed = '0';
    }
  }

  // Keyboard shortcuts: 'c' = collapse all, 'e' = expand all (ignored while typing in fields)
  document.addEventListener('keydown', (e) => {
    const tag = (e.target && e.target.tagName) || '';
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)) return;

    if (e.key && e.key.toLowerCase() === 'c') {
      sectionHeaders.forEach(h => { if (h.dataset.collapsed !== '1') h.click(); });
      if (typeof showToast === 'function') showToast('Collapsed all sections');
    }
    if (e.key && e.key.toLowerCase() === 'e') {
      sectionHeaders.forEach(h => { if (h.dataset.collapsed === '1') h.click(); });
      if (typeof showToast === 'function') showToast('Expanded all sections');
    }
  });
})();
</script>
<script>
/* Persist collapsed/expanded state per setlist using localStorage */
(function () {
  // We key by setlist id so each setlist remembers its own collapsed sections
  const SETLIST_ID = {{ setlist.id }};
  const KEY = 'sg:collapsed:' + String(SETLIST_ID);

  function getSectionHeaders() {
    // A section header row is the .row that contains a .secmeta (added in Step 106)
    return Array.from(document.querySelectorAll('.row')).filter(r => r.querySelector('.secmeta'));
  }

  function loadCollapsed() {
    try { return JSON.parse(localStorage.getItem(KEY) || '[]'); }
    catch (_) { return []; }
  }

  function saveCollapsed(indices) {
    try { localStorage.setItem(KEY, JSON.stringify(indices)); } catch (_) {}
  }

  function refreshAndSave() {
    const headers = getSectionHeaders();
    const indices = [];
    headers.forEach((h, i) => {
      if (h.dataset.collapsed === '1') indices.push(i);
    });
    saveCollapsed(indices);
  }

  function applyInitialState() {
    const wantCollapsed = new Set(loadCollapsed());
    const headers = getSectionHeaders();

    headers.forEach((h, i) => {
      const shouldBeCollapsed = wantCollapsed.has(i);
      const isCollapsed = h.dataset.collapsed === '1';
      if (shouldBeCollapsed !== isCollapsed) {
        // We reuse the existing click handler from Step 101 to toggle the section
        h.click();
      }
    });
  }

  // After any section header click, save the new state.
  document.addEventListener('click', (e) => {
    const header = e.target.closest('.row');
    if (header && header.querySelector('.secmeta')) {
      // Let the original toggle logic run first, then snapshot
      setTimeout(refreshAndSave, 0);
    }
  });

  // Also refresh state after "Collapse all" / "Expand all" button presses
  ['btnCollapseAll', 'btnExpandAll'].forEach(id => {
    const btn = document.getElementById(id);
    if (btn) btn.addEventListener('click', () => setTimeout(refreshAndSave, 0));
  });

  // Apply remembered state on load
  applyInitialState();
})();
</script>
<script>
(function(){
  const toggleBtn = document.getElementById('toggleReorder');
  const container = document.getElementById('current-setlist');
  const statusEl = document.getElementById('reorderStatus');
  const reorderUrl = "{{ url_for('reorder_setlist', setlist_id=setlist.id) }}";
  if (!toggleBtn || !container || !reorderUrl) return;

  let active = false;
  let dragItem = null;
  let saveTimer = null;

  function songItems() {
    return Array.from(container.querySelectorAll('.dnd-item'));
  }

  function showStatus(message, kind = "") {
    if (!statusEl) return;
    statusEl.textContent = message || "";
    statusEl.style.display = message ? "block" : "none";
    statusEl.classList.remove("is-error", "is-success", "is-busy");
    if (kind) statusEl.classList.add(kind);
  }

  function enableDrag(on) {
    active = on;
    const body = document.body;
    if (!body) return;
    if (on) {
      body.classList.add('has-reorder');
      toggleBtn.textContent = '🧲 Reorder mode: On';
    } else {
      body.classList.remove('has-reorder');
      toggleBtn.textContent = '🧲 Reorder mode: Off';
      showStatus('');
    }
    songItems().forEach(item => {
      if (item.dataset.locked === "1") {
        item.setAttribute('draggable', 'false');
        return;
      }
      item.setAttribute('draggable', on ? 'true' : 'false');
      item.classList.toggle('is-draggable', on);
    });
  }

  function commitOrder() {
    if (!active) return;
    const ids = songItems().map(item => item.dataset.songId).filter(Boolean);
    if (!ids.length) return;
    showStatus('Saving order…', 'is-busy');
    fetch(reorderUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'X-Requested-With': 'XMLHttpRequest',
      },
      body: 'order=' + encodeURIComponent(ids.join(',')),
    }).then(resp => {
      if (!resp.ok) throw new Error('Save failed');
      showStatus('Order updated ✓', 'is-success');
      clearTimeout(saveTimer);
      saveTimer = setTimeout(() => showStatus(''), 2500);
    }).catch(err => {
      console.error(err);
      showStatus('Could not save order.', 'is-error');
    });
  }

  function onDragStart(e) {
    if (!active) {
      e.preventDefault();
      return;
    }
    dragItem = this;
    this.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', this.dataset.songId || '');
  }

  function onDragEnd() {
    this.classList.remove('dragging');
    dragItem = null;
    commitOrder();
  }

  function onDragOver(e) {
    if (!active || !dragItem) return;
    e.preventDefault();
    const target = e.target.closest('.dnd-item');
    if (!target || target === dragItem || target.dataset.locked === "1") return;

    const rect = target.getBoundingClientRect();
    const offset = e.clientY - rect.top;
    const shouldMoveAfter = offset > rect.height / 2;
    const parent = target.parentElement;
    if (!parent) return;
    if (shouldMoveAfter) {
      parent.insertBefore(dragItem, target.nextElementSibling);
    } else {
      parent.insertBefore(dragItem, target);
    }
  }

  function onKeydown(e) {
    if (!active || !dragItem) return;
    if (e.key === 'Escape') {
      dragItem.classList.remove('dragging');
      dragItem = null;
    showStatus('Reorder cancelled.');
    }
  }

  songItems().forEach(item => {
    if (item.dataset.locked === "1") {
      item.setAttribute('draggable', 'false');
      return;
    }
    item.addEventListener('dragstart', onDragStart);
    item.addEventListener('dragend', onDragEnd);
  });

  container.addEventListener('dragover', onDragOver);
  container.addEventListener('drop', e => {
    if (active) e.preventDefault();
  });
  document.addEventListener('keydown', onKeydown);

  toggleBtn.addEventListener('click', () => {
    enableDrag(!active);
  });
})();
</script>
<div id="slBottomAnchor"></div>
<script>
document.addEventListener('DOMContentLoaded', function(){
  // Move Current setlist to the very top of the page content
  const top = document.getElementById('slTopAnchor');
  const cur = document.getElementById('current-setlist');
  if (top && cur && top.parentNode) {
    top.parentNode.insertBefore(cur, top.nextSibling);
  }

  // Collapse Build tools on Edit page (they still exist, now lower-traffic)
  const bt = document.getElementById('sec-buildtools');
  if (bt) bt.removeAttribute('open');

  // Move the Edit Details form block to the bottom
  const bottom = document.getElementById('slBottomAnchor');
  const formBlock = document.getElementById('editFormBlock');
  if (bottom && formBlock && bottom.parentNode) {
    bottom.parentNode.appendChild(formBlock);
  }
});
</script>
"""
# Live Mode page template (Jinja)
LIVE_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Live Mode — {{ sl.name }}</title>
  <meta name="color-scheme" content="dark light">
  <style>
    :root { color-scheme: dark light; }
    * { box-sizing: border-box; }
    html, body { margin:0; padding:0; height:100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }
    body { display:flex; flex-direction:column; }

    header {
      padding: 10px 14px;
      display:flex;
      align-items:center;
      gap:12px;
      border-bottom: 1px solid #ddd;
    }
    header h1 { font-size: 16px; margin:0; font-weight:600; }
    header .spacer { flex:1; }
    header a { text-decoration:none; }
    header .btn {
      display:inline-block; padding:6px 10px; border:1px solid #ccc; border-radius:8px;
    }

    .wrap { flex:1; display:flex; min-height:0; }
    .sidebar {
      width: 320px; border-right:1px solid #ddd; overflow:auto;
    }
    .item {
      padding:10px 12px; border-bottom:1px solid #eee; cursor:pointer;
      display:flex; align-items:center; gap:8px;
    }
    .item.active { background: rgba(0,0,0,0.07); font-weight:600; }
    .item .pos { display:inline-block; min-width:2.2em; opacity:0.7; }
    .item .title { flex:1; }
    .item .tag { font-size:12px; opacity:0.7; }

    .stage { flex:1; display:flex; flex-direction:column; min-width:0; }
    .viewer {
      flex:1;
      min-height:0;
      display:flex;
      gap:12px;
      background:#0f1115;
      padding:12px;
    }
    .chart-wrapper {
      flex:1;
      display:none;
      flex-direction:column;
      gap:12px;
      background:#161b24;
      border-radius:12px;
      padding:16px 18px;
      color:#e4ebff;
      box-shadow:0 10px 28px rgba(0,0,0,0.35);
      overflow:auto;
      font-family: "Menlo", "Consolas", "SFMono-Regular", "Courier New", monospace;
      line-height:1.6;
      font-size:1.1rem;
    }
    .chart-header {
      display:flex;
      justify-content:space-between;
      align-items:flex-start;
      flex-wrap:wrap;
      gap:8px;
      font-size:0.9rem;
      color:rgba(228,235,255,0.7);
    }
    .chart-title {
      font-weight:600;
      font-size:1rem;
      color:#fff;
    }
    .chart-lines {
      flex:1;
      display:flex;
      flex-direction:column;
      gap:6px;
    }
    .chart-block {
      display:flex;
      flex-direction:column;
      gap:2px;
    }
    .chart-chords {
      font-family: Menlo, Consolas, "SFMono-Regular", "Courier New", monospace;
      white-space:pre;
      line-height:1.2;
    }
    .chart-chords .chord {
      margin-left:0;
      margin-right:0;
    }
    .chart-lyric {
      white-space:pre-wrap;
    }
    .chart-line {
      white-space:pre-wrap;
      word-break:break-word;
    }
    .chart-line .lyric { color:rgba(228,235,255,0.9); }
    .chart-line .chord {
      display:inline-flex;
      align-items:center;
      justify-content:center;
      min-width:1.2em;
      padding:0 0.18em;
      margin:0 1px;
      border-radius:6px;
      background:rgba(255,209,102,0.18);
      color:#ffd166;
      font-weight:600;
      cursor:pointer;
      transition:background .15s ease, color .15s ease, transform .1s ease;
    }
    .chart-line .chord:hover {
      background:rgba(255,209,102,0.28);
      color:#ffe08a;
    }
    .chart-line .chord.highlighted {
      background:#ffd166;
      color:#1b1f27;
    }
    .chart-directive {
      font-size:0.85rem;
      color:rgba(228,235,255,0.56);
      text-transform:uppercase;
      letter-spacing:0.12em;
    }
    .chart-comment {
      font-style:italic;
      color:rgba(228,235,255,0.7);
    }
    #pdfCanvas {
      flex:1;
      background:#fff;
      border-radius:8px;
      box-shadow:0 8px 24px rgba(0,0,0,0.35);
      max-width:100%;
    }
    #thumbs {
      width:140px;
      overflow-y:auto;
      display:flex;
      flex-direction:column;
      gap:8px;
    }
    .thumb {
      border:2px solid transparent;
      border-radius:6px;
      background:#1c1f2a;
      padding:4px;
      cursor:pointer;
      transition:border-color .15s ease;
    }
    .thumb.active { border-color:#4a90e2; }
    .thumb canvas { width:100%; display:block; border-radius:4px; }

    #noPdf {
      flex:1;
      display:none;
      align-items:center;
      justify-content:center;
      flex-direction:column;
      gap:12px;
      color:#e0e0e0;
      font-size:1.05rem;
      text-align:center;
    }
    #noPdf .btn { padding:8px 12px; border:1px solid #ccc; border-radius:8px; background:#fff; color:#111; }

    .controls {
      border-top:1px solid #ddd; padding:8px; display:flex; align-items:center; gap:12px; flex-wrap:wrap; background:rgba(15,17,21,0.8);
    }
    .controls button,
    .controls a.btn {
      padding:8px 12px;
      border:1px solid var(--btn-border);
      background:var(--btn-bg);
      color:var(--btn-text);
      border-radius:8px;
      cursor:pointer;
      text-decoration:none;
      transition:background .15s ease, color .15s ease;
    }
    .controls button:disabled { opacity:0.45; cursor:not-allowed; }
    .controls .btn-requests.alert {
      background:#ffda79;
      color:#111;
      border-color:#e3b341;
    }
    .controls .btn-requests {
      position:relative;
      display:inline-flex;
      align-items:center;
      gap:6px;
    }
    .controls .btn-requests .badge {
      display:inline-flex;
      align-items:center;
      justify-content:center;
      min-width:1.4em;
      padding:0 0.45em;
      border-radius:999px;
      font-size:0.72rem;
      background:#4a90e2;
      color:#fff;
    }
    .controls .btn-requests .badge[hidden] { display:none; }
    [data-theme="dark"] .controls {
      border-top-color:rgba(255,255,255,0.12);
      background:rgba(10,12,16,0.9);
    }
    [data-theme="dark"] .controls .btn-requests.alert {
      background:#ffd166;
      color:#111;
      border-color:#c79432;
    }
    [data-theme="dark"] .controls .btn-requests .badge {
      background:#6da8ff;
      color:#051534;
    }
    [data-theme="dark"] .chart-wrapper {
      background:#0f141e;
      color:#e9efff;
      box-shadow:0 10px 30px rgba(0,0,0,0.6);
    }
    [data-theme="dark"] .chart-line .lyric { color:#f2f6ff; }
    [data-theme="dark"] .chart-line .chord {
      background:rgba(255,209,102,0.2);
      color:#ffda79;
    }
    [data-theme="dark"] .chart-line .chord.highlighted {
      background:#ffd166;
      color:#0f141e;
    }
    .controls .now { flex:1; text-align:center; opacity:0.8; min-width:160px; }
    .page-controls { display:flex; align-items:center; gap:8px; }
    .page-controls button { padding:6px 10px; }
    .chart-tools {
      display:none;
      align-items:center;
      gap:6px;
      flex-wrap:wrap;
    }
    .chart-tools button,
    .chart-tools select {
      padding:6px 10px;
      border-radius:8px;
      border:1px solid var(--btn-border, #ccc);
      background:var(--btn-bg, #fff);
      color:var(--btn-text, #111);
      cursor:pointer;
      font-size:0.85rem;
    }
    .chart-tools button.active {
      border-color:#4a90e2;
      color:#fff;
      background:#4a90e2;
    }
    .chart-tools .label {
      font-size:0.8rem;
      opacity:0.75;
      margin-right:4px;
    }
    .chart-tools select {
      cursor:pointer;
    }
    .chart-tools .btn-small {
      padding:4px 8px;
      font-size:0.8rem;
    }
    .muted { opacity:0.7; }

    .requests-backdrop {
      position:fixed;
      inset:0;
      background:rgba(0,0,0,0.45);
      opacity:0;
      pointer-events:none;
      transition:opacity .2s ease;
      z-index:1990;
    }
    .requests-backdrop.visible {
      opacity:1;
      pointer-events:auto;
    }
    .requests-drawer {
      position:fixed;
      left:0;
      right:0;
      bottom:0;
      max-height:70vh;
      background:rgba(15,18,26,0.98);
      color:#f4f6ff;
      box-shadow:0 -14px 36px rgba(0,0,0,0.55);
      border-top-left-radius:18px;
      border-top-right-radius:18px;
      transform:translateY(100%);
      visibility:hidden;
      transition:transform .25s ease, visibility 0s linear .25s;
      display:flex;
      flex-direction:column;
      z-index:2000;
      padding-bottom:calc(env(safe-area-inset-bottom, 16px));
      backdrop-filter:blur(18px);
      color-scheme:dark;
    }
    .requests-drawer.open {
      transform:translateY(0);
      visibility:visible;
      transition:transform .25s ease;
    }
    .requests-handle {
      width:52px;
      height:5px;
      border-radius:999px;
      background:rgba(255,255,255,0.22);
      margin:10px auto 4px;
    }
    .requests-drawer header {
      display:flex;
      align-items:center;
      gap:12px;
      padding:12px 20px 6px;
    }
    .requests-drawer header h2 {
      margin:0;
      font-size:1rem;
      font-weight:600;
      letter-spacing:0.01em;
    }
    .requests-close {
      margin-left:auto;
      background:transparent;
      border:1px solid rgba(255,255,255,0.25);
      border-radius:8px;
      padding:4px 10px;
      color:inherit;
      cursor:pointer;
      font-size:0.85rem;
    }
    .requests-close:hover { background:rgba(255,255,255,0.08); }
    .requests-feedback {
      min-height:1.2em;
      padding:0 20px 6px;
      font-size:0.85rem;
      opacity:0.85;
    }
    .requests-feedback.success { color:#8be68b; }
    .requests-feedback.error { color:#ff8484; opacity:1; }
    .requests-content {
      overflow-y:auto;
      padding:6px 20px 12px;
      display:flex;
      flex-direction:column;
      gap:14px;
    }
    .requests-section h3 {
      margin:6px 0;
      font-size:0.9rem;
      text-transform:uppercase;
      letter-spacing:0.05em;
      opacity:0.75;
    }
    .request-card {
      border:1px solid rgba(255,255,255,0.12);
      border-radius:14px;
      padding:12px 14px;
      background:rgba(255,255,255,0.05);
      display:flex;
      flex-direction:column;
      gap:8px;
    }
    .request-card-header {
      display:flex;
      align-items:center;
      gap:8px;
    }
    .request-card-header h4 {
      flex:1;
    }
    .request-card h4 {
      margin:0;
      font-size:0.98rem;
      font-weight:600;
      color:#fff;
    }
    .request-note {
      font-size:0.85rem;
      color:#cbd4ff;
    }
    .request-meta {
      font-size:0.78rem;
      opacity:0.75;
    }
    .request-status {
      font-size:0.7rem;
      text-transform:uppercase;
      letter-spacing:0.08em;
      display:inline-block;
      padding:2px 8px;
      border-radius:999px;
      background:rgba(255,255,255,0.12);
    }
    .requests-actions {
      display:flex;
      flex-wrap:wrap;
      gap:8px;
    }
    .requests-actions button {
      padding:6px 10px;
      border-radius:8px;
      border:1px solid rgba(255,255,255,0.25);
      background:rgba(255,255,255,0.1);
      color:#f4f7ff;
      font-size:0.78rem;
      cursor:pointer;
      transition:background .15s ease, color .15s ease, border-color .15s ease;
    }
    .requests-actions button[data-variant="primary"] {
      background:#4a90e2;
      border-color:#4a90e2;
      color:#fff;
    }
    .requests-actions button[data-variant="accent"] {
      background:#ffd166;
      border-color:#f5b342;
      color:#121417;
    }
    .requests-actions button[data-variant="success"] {
      background:#2ecc71;
      border-color:#26a75c;
      color:#fff;
    }
    .requests-actions button[data-variant="danger"] {
      background:#ff6b6b;
      border-color:#e25555;
      color:#fff;
    }
    .requests-actions button:disabled {
      opacity:0.45;
      cursor:not-allowed;
    }
    .requests-actions button.loading {
      pointer-events:none;
      opacity:0.6;
    }
    .requests-empty {
      margin:20px 0;
      text-align:center;
      opacity:0.75;
    }
    .requests-drawer footer {
      padding:6px 20px 18px;
      display:flex;
      justify-content:flex-end;
    }
    .requests-drawer footer .btn {
      display:inline-flex;
      align-items:center;
      gap:6px;
      padding:8px 14px;
      border-radius:8px;
      border:1px solid rgba(255,255,255,0.25);
      color:#f5f6ff;
      text-decoration:none;
      background:rgba(255,255,255,0.08);
      font-size:0.82rem;
    }
    .requests-drawer footer .btn:hover {
      background:rgba(255,255,255,0.14);
    }
    @media (max-width: 600px) {
      .requests-drawer {
        max-height:72vh;
      }
      .requests-drawer header {
        padding:12px 16px 6px;
      }
      .requests-content {
        padding:6px 16px 12px;
      }
    }

    @media (max-width: 900px) {
      .sidebar { width: 260px; }
      header h1 { display:none; }
      .viewer { flex-direction:column; }
      #thumbs { flex-direction:row; width:100%; overflow-x:auto; overflow-y:hidden; }
      .thumb { min-width:100px; }
    }
  </style>
</head>
<body>
  <header>
    <a class="btn" href="{{ url_for('edit_setlist', setlist_id=sl.id) }}">← Back</a>
    <h1>Live Mode — {{ sl.name }}</h1>
    <div class="spacer"></div>
    <a class="btn" href="{{ toggle_url }}">{{ toggle_label }}</a>
  </header>

  <div class="wrap">
    <aside class="sidebar" id="list"></aside>

    <main class="stage">
      <div class="viewer">
        <div class="chart-wrapper" id="chartWrapper" aria-live="polite" aria-label="Chord chart">
          <div class="chart-header">
            <div class="chart-title" id="chartTitle"></div>
            <div class="chart-meta" id="chartMeta"></div>
          </div>
          <div class="chart-lines" id="chartLines"></div>
        </div>
        <canvas id="pdfCanvas"></canvas>
        <div id="thumbs"></div>
        <div id="noPdf"></div>
      </div>
      <div class="controls">
        <button id="prev">← Prev Song</button>
        <div class="page-controls">
          <button id="pagePrev">↑ Page</button>
          <span id="pageInfo">-- / --</span>
          <button id="pageNext">Page ↓</button>
        </div>
        <div class="chart-tools" id="chartTools">
          <span class="label">Key</span>
          <button type="button" class="btn-small" id="transposeDown">−</button>
          <span id="transposeDisplay">0</span>
          <button type="button" class="btn-small" id="transposeUp">+</button>
          <span class="label">Capo</span>
          <button type="button" class="btn-small" id="capoDown">−</button>
          <span id="capoDisplay">0</span>
          <button type="button" class="btn-small" id="capoUp">+</button>
          <button type="button" class="btn-small" id="transposeReset">Reset</button>
        </div>
        <div class="now" id="now">—</div>
        <button id="viewToggle" type="button" class="btn" hidden>View: Chart</button>
        <button id="requestsToggle" type="button" class="btn btn-requests{% if pending_requests %} alert{% endif %}" aria-expanded="false" aria-controls="requestsDrawer">
          Requests
          <span id="requestsBadge" class="badge"{% if not pending_requests %} hidden{% endif %}>{{ pending_requests }}</span>
        </button>
        <button id="next">Next Song →</button>
      </div>
    </main>
  </div>

  <div id="requestsBackdrop" class="requests-backdrop" hidden></div>
  <section id="requestsDrawer" class="requests-drawer" aria-hidden="true" tabindex="-1">
    <div class="requests-handle" aria-hidden="true"></div>
    <header>
      <h2>Audience Requests</h2>
      <button id="requestsClose" class="requests-close" type="button" aria-label="Close requests drawer">Close</button>
    </header>
    <div id="requestsFeedback" class="requests-feedback" role="status" aria-live="polite"></div>
    <div id="requestsContent" class="requests-content" role="list"></div>
    <footer>
      <a class="btn" href="{{ requests_url }}" target="_blank">Open full requests console</a>
    </footer>
  </section>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
  <script>
    const items = {{ items_json|safe }};
    console.log('Live Mode items', items);
    const listEl = document.getElementById('list');
    const canvas = document.getElementById('pdfCanvas');
    const ctx = canvas.getContext('2d');
    const thumbsEl = document.getElementById('thumbs');
    const noPdfEl = document.getElementById('noPdf');
    const nowEl  = document.getElementById('now');
    const prevBtn = document.getElementById('prev');
    const nextBtn = document.getElementById('next');
    const pagePrevBtn = document.getElementById('pagePrev');
    const pageNextBtn = document.getElementById('pageNext');
    const pageInfo = document.getElementById('pageInfo');
    const chartWrapper = document.getElementById('chartWrapper');
    const chartLines = document.getElementById('chartLines');
    const chartTitle = document.getElementById('chartTitle');
    const chartMeta = document.getElementById('chartMeta');
    const chartTools = document.getElementById('chartTools');
    const transposeDownBtn = document.getElementById('transposeDown');
    const transposeUpBtn = document.getElementById('transposeUp');
    const transposeResetBtn = document.getElementById('transposeReset');
    const transposeDisplay = document.getElementById('transposeDisplay');
    const capoDownBtn = document.getElementById('capoDown');
    const capoUpBtn = document.getElementById('capoUp');
    const capoDisplay = document.getElementById('capoDisplay');
    const viewToggleBtn = document.getElementById('viewToggle');
    const requestsToggleBtn = document.getElementById('requestsToggle');
    const requestsDrawer = document.getElementById('requestsDrawer');
    const requestsBackdrop = document.getElementById('requestsBackdrop');
    const requestsCloseBtn = document.getElementById('requestsClose');
    const requestsContent = document.getElementById('requestsContent');
    const requestsFeedback = document.getElementById('requestsFeedback');
    const requestsBadge = document.getElementById('requestsBadge');

    const SETLIST_ID = {{ sl.id }};
const REQUESTS_JSON_URL = "{{ requests_json_url }}";
const initialPendingRequests = {{ pending_requests }};
const REQUESTS_REFRESH_MS = 15000;
const REQUESTS_STALE_THRESHOLD = 5000;
const REQUESTS_BADGE_MAX = 99;

    const requestsState = {
      open: false,
      items: [],
      loading: false,
      lastFetched: 0,
      lastData: null,
      refreshTimer: null,
    };
let requestsFeedbackTimer = null;
const NOTE_INDEX = {
  'C':0,'B#':0,
      'C#':1,'Db':1,
      'D':2,
      'D#':3,'Eb':3,
      'E':4,'Fb':4,
      'F':5,'E#':5,
      'F#':6,'Gb':6,
      'G':7,
      'G#':8,'Ab':8,
      'A':9,
      'A#':10,'Bb':10,
      'B':11,'Cb':11
    };
    const INDEX_TO_SHARP = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const INDEX_TO_FLAT  = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B'];

let currentViewMode = 'pdf';
let currentTranspose = 0;
let currentCapo = 0;
let currentHighlightSet = new Set();

    function isLikelyChordSymbol(token) {
      if (!token) return false;
      const normalized = token.trim().replace(/\u266d/g, 'b').replace(/\u266f/g, '#');
      if (!normalized) return false;
      if (/^N\\.?C\\.?$/i.test(normalized)) return true;
      if (/\\s/.test(normalized)) return false;
      if (!/^[A-Ga-g][#b]?/.test(normalized)) return false;
      if (!/^[A-Ga-g][#b]?(?:add|sus|maj|min|aug|dim|m|M|[0-9]|[#b\/\+\-\(\)°ø])*$/i.test(normalized)) return false;
      return true;
    }

    function chartStateKey(songId) {
      return 'liveChart:' + SETLIST_ID + ':' + songId;
    }

    function loadChartPrefs(songId) {
      try {
        const raw = localStorage.getItem(chartStateKey(songId));
        if (!raw) return {};
        const parsed = JSON.parse(raw);
        if (parsed && Array.isArray(parsed.highlights)) {
          parsed.highlights = parsed.highlights.map(Number).filter(n => Number.isInteger(n));
        } else {
          parsed.highlights = [];
        }
        return parsed || {};
      } catch (err) {
        console.warn('Chart state parse error', err);
        return {};
      }
    }

    function persistChartPrefs() {
      if (!currentItem || !currentItem.song_id) return;
      const payload = {
        transpose: currentTranspose,
        capo: currentCapo,
        view: currentViewMode,
        highlights: Array.from(currentHighlightSet),
      };
      try {
        localStorage.setItem(chartStateKey(currentItem.song_id), JSON.stringify(payload));
      } catch (err) {
        console.warn('Unable to persist chart prefs', err);
      }
    }

    function formatSemitone(val) {
      if (!Number.isFinite(val)) return '0';
      if (val === 0) return '0';
      return (val > 0 ? '+' : '') + val;
    }

    function transposeNote(token, semitones, preferFlat, originalToken) {
      if (!token) return token;
      let key = token.toUpperCase();
      if (!NOTE_INDEX.hasOwnProperty(key)) return token;
      let idx = NOTE_INDEX[key];
      if (!Number.isInteger(idx)) return token;
      let shifted = (idx + semitones) % 12;
      if (shifted < 0) shifted += 12;
      let name = preferFlat ? INDEX_TO_FLAT[shifted] : INDEX_TO_SHARP[shifted];
      if (originalToken && originalToken[0] === originalToken[0].toLowerCase()) {
        name = name[0].toLowerCase() + name.slice(1);
      }
      return name;
    }

    function transposeChordPart(symbol, semitones) {
      if (!symbol) return symbol;
      const match = symbol.match(/^([A-Ga-g][#b]?)(.*)$/);
      if (!match) return symbol;
      const originalRoot = match[1];
      const suffix = match[2] || '';
      const preferFlat = originalRoot.includes('b');
      let normalized = originalRoot[0].toUpperCase();
      if (originalRoot.length > 1) {
        const accidental = originalRoot[1];
        if (accidental === '#' || accidental === '♯') {
          normalized += '#';
        } else if (accidental && (accidental.toLowerCase() === 'b' || accidental === '♭')) {
          normalized += 'b';
        }
      }
      const resultRoot = transposeNote(normalized, semitones, preferFlat, originalRoot);
      const adjustedRoot = originalRoot[0] === originalRoot[0].toLowerCase()
        ? resultRoot[0].toLowerCase() + resultRoot.slice(1)
        : resultRoot;
      return adjustedRoot + suffix;
    }

    function transposeChordSymbol(symbol, semitones) {
      if (!symbol) return symbol;
      if (!Number.isFinite(semitones) || semitones === 0) return symbol;
      const parts = symbol.split('/');
      const main = transposeChordPart(parts[0], semitones);
      if (parts.length > 1) {
        const bass = transposeChordPart(parts[1], semitones);
        return main + '/' + bass;
      }
      return main;
    }

    function transposeKeyLabel(label, semitones) {
      if (!label || !Number.isFinite(semitones) || semitones === 0) return label;
      const trimmed = label.trim();
      const match = trimmed.match(/^([A-Ga-g][#b]?)(.*)$/);
      if (!match) return label;
      const root = transposeChordPart(match[1], semitones);
      return root + match[2];
    }

    function escapeHtml(str) {
      return String(str).replace(/[&<>"']/g, (ch) => {
        switch (ch) {
          case '&': return '&amp;';
          case '<': return '&lt;';
          case '>': return '&gt;';
          case '"': return '&quot;';
          case "'": return '&#39;';
          default: return ch;
        }
      });
    }

    function updateRequestsBadge(count) {
      if (!requestsBadge || !requestsToggleBtn) return;
      if (count && count > 0) {
        const display = count > REQUESTS_BADGE_MAX ? REQUESTS_BADGE_MAX + '+' : String(count);
        requestsBadge.textContent = display;
        requestsBadge.hidden = false;
        requestsToggleBtn.classList.add('alert');
      } else {
        requestsBadge.textContent = '';
        requestsBadge.hidden = true;
        requestsToggleBtn.classList.remove('alert');
      }
    }

    function clearRequestsFeedback() {
      if (!requestsFeedback) return;
      if (requestsFeedbackTimer) {
        clearTimeout(requestsFeedbackTimer);
        requestsFeedbackTimer = null;
      }
      requestsFeedback.textContent = '';
      requestsFeedback.className = 'requests-feedback';
    }

    function showRequestsFeedback(message, variant = '') {
      if (!requestsFeedback) return;
      clearRequestsFeedback();
      if (!message) return;
      if (variant) {
        requestsFeedback.classList.add(variant);
      }
      requestsFeedback.textContent = message;
      requestsFeedbackTimer = setTimeout(() => {
        clearRequestsFeedback();
      }, 4000);
    }

    function setRequestsBusy(isBusy) {
      if (!requestsContent) return;
      if (isBusy) {
        requestsContent.setAttribute('aria-busy', 'true');
      } else {
        requestsContent.removeAttribute('aria-busy');
      }
    }

    function renderRequestsLoading() {
      if (!requestsContent) return;
      setRequestsBusy(true);
      requestsContent.innerHTML = '<p class="requests-empty">Loading requests…</p>';
    }

    function formatRequestMeta(req) {
      const parts = [];
      if (req.createdAt) {
        const dt = new Date(req.createdAt);
        if (!Number.isNaN(dt.getTime())) {
          parts.push(dt.toLocaleString([], { hour: 'numeric', minute: '2-digit' }));
        }
      }
      if (req.fromName) parts.push(req.fromName);
      if (req.fromContact) parts.push(req.fromContact);
      return parts.length ? 'Requested ' + parts.join(' · ') : '';
    }

    function createActionButton(label, variant, handler) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = label;
      if (variant) btn.dataset.variant = variant;
      btn.addEventListener('click', () => handler(btn));
      return btn;
    }

    function buildRequestCard(req) {
      const card = document.createElement('article');
      card.className = 'request-card';
      card.setAttribute('role', 'listitem');

      const header = document.createElement('div');
      header.className = 'request-card-header';
      const title = document.createElement('h4');
      title.textContent = req.label || 'Request';
      header.appendChild(title);
      const status = document.createElement('span');
      status.className = 'request-status';
      status.textContent = req.statusLabel || (req.status || '').toUpperCase();
      header.appendChild(status);
      card.appendChild(header);

      if (req.freeText && (!req.songTitle || req.freeText !== req.label)) {
        const note = document.createElement('div');
        note.className = 'request-note';
        note.textContent = req.freeText;
        card.appendChild(note);
      }

      const metaText = formatRequestMeta(req);
      if (metaText) {
        const meta = document.createElement('div');
        meta.className = 'request-meta';
        meta.textContent = metaText;
        card.appendChild(meta);
      }

      const actions = document.createElement('div');
      actions.className = 'requests-actions';

      if (req.status === 'new') {
        actions.appendChild(createActionButton('Queue', 'primary', (btn) => handleRequestsAction(btn, req, { type: 'status', status: 'queued' })));
        actions.appendChild(createActionButton('Decline', 'danger', (btn) => handleRequestsAction(btn, req, { type: 'status', status: 'declined' })));
        if (req.hasSong) {
          actions.appendChild(createActionButton('Add to setlist', 'accent', (btn) => handleRequestsAction(btn, req, { type: 'add' })));
        }
      } else if (req.status === 'queued') {
        actions.appendChild(createActionButton('Mark done', 'success', (btn) => handleRequestsAction(btn, req, { type: 'status', status: 'done' })));
        actions.appendChild(createActionButton('Decline', 'danger', (btn) => handleRequestsAction(btn, req, { type: 'status', status: 'declined' })));
        if (req.hasSong) {
          actions.appendChild(createActionButton('Add to setlist', 'accent', (btn) => handleRequestsAction(btn, req, { type: 'add' })));
        }
        actions.appendChild(createActionButton('Reopen', 'primary', (btn) => handleRequestsAction(btn, req, { type: 'status', status: 'new' })));
      }

      if (actions.childElementCount > 0) {
        card.appendChild(actions);
      }

      return card;
    }

    function createRequestSection(title, requestsList) {
      const section = document.createElement('section');
      section.className = 'requests-section';
      const heading = document.createElement('h3');
      heading.textContent = title;
      section.appendChild(heading);
      requestsList.forEach((req) => {
        section.appendChild(buildRequestCard(req));
      });
      return section;
    }

    function renderRequestsContent(items) {
      if (!requestsContent) return;
      requestsContent.innerHTML = '';
      const newItems = items.filter((req) => req.status === 'new');
      const queuedItems = items.filter((req) => req.status === 'queued');
      if (!newItems.length && !queuedItems.length) {
        const empty = document.createElement('p');
        empty.className = 'requests-empty';
        empty.textContent = 'No pending requests right now.';
        requestsContent.appendChild(empty);
        setRequestsBusy(false);
        return;
      }
      if (newItems.length) {
        requestsContent.appendChild(createRequestSection('New', newItems));
      }
      if (queuedItems.length) {
        requestsContent.appendChild(createRequestSection('Queued', queuedItems));
      }
      setRequestsBusy(false);
    }

    async function performRequestMutation(req, action) {
      if (!req || !action) throw new Error('Invalid request action');
      let endpoint = '';
      let payload = {};
      if (action.type === 'status') {
        endpoint = `/requests/${req.id}/status`;
        payload = { status: action.status };
      } else if (action.type === 'add') {
        endpoint = `/requests/${req.id}/add`;
        payload = {};
      } else {
        throw new Error('Unsupported action');
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'X-Requested-With': 'XMLHttpRequest',
        },
        body: JSON.stringify(payload),
      });

      const text = await res.text();
      let data = null;
      if (text) {
        try { data = JSON.parse(text); } catch (err) { data = null; }
      }
      if (!res.ok || !data || data.ok === false) {
        const errorMessage = (data && data.error) ? data.error : `Action failed (${res.status})`;
        throw new Error(errorMessage);
      }
      await loadRequests({ force: true, silent: true });
      return data;
    }

    async function handleRequestsAction(button, req, action) {
      if (!button) return;
      button.disabled = true;
      button.classList.add('loading');
      try {
        const result = await performRequestMutation(req, action);
        showRequestsFeedback(result.message || 'Request updated', 'success');
      } catch (err) {
        showRequestsFeedback(err.message || 'Unable to update request', 'error');
      } finally {
        button.disabled = false;
        button.classList.remove('loading');
      }
    }

    async function loadRequests({ force = false, silent = false } = {}) {
      if (!REQUESTS_JSON_URL) return null;
      const now = Date.now();
      if (!force && requestsState.lastFetched && (now - requestsState.lastFetched) < REQUESTS_STALE_THRESHOLD) {
        return requestsState.lastData;
      }
      if (requestsState.loading) {
        return requestsState.lastData;
      }
      requestsState.loading = true;
      try {
        if (requestsState.open && !silent) {
          renderRequestsLoading();
        }
        const res = await fetch(REQUESTS_JSON_URL, {
          headers: {
            'Accept': 'application/json',
            'X-Requested-With': 'XMLHttpRequest',
          },
          cache: 'no-store',
        });
        if (!res.ok) {
          throw new Error(`Failed to load requests (${res.status})`);
        }
        const data = await res.json();
        requestsState.items = data.requests || [];
        requestsState.lastFetched = now;
        requestsState.lastData = data;
        updateRequestsBadge(data.pendingCount || 0);
        if (requestsState.open) {
          renderRequestsContent(requestsState.items);
        }
        return data;
      } catch (err) {
        if (requestsState.open && requestsContent) {
          requestsContent.innerHTML = '<p class="requests-empty">Unable to load requests.</p>';
          setRequestsBusy(false);
        }
        if (!silent) {
          showRequestsFeedback(err.message || 'Unable to load requests', 'error');
        }
        return null;
      } finally {
        requestsState.loading = false;
      }
    }

    function ensureRequestsPolling() {
      if (!REQUESTS_JSON_URL || requestsState.refreshTimer) return;
      requestsState.refreshTimer = setInterval(() => {
        if (document.hidden) return;
        if (requestsState.open) {
          loadRequests({ force: true, silent: true });
        } else {
          loadRequests({ force: false, silent: true });
        }
      }, REQUESTS_REFRESH_MS);
    }

    function openRequestsDrawer() {
      if (!requestsDrawer || requestsState.open) return;
      requestsState.open = true;
      if (requestsBackdrop) {
        requestsBackdrop.hidden = false;
        requestsBackdrop.classList.add('visible');
      }
      requestsDrawer.classList.add('open');
      requestsDrawer.setAttribute('aria-hidden', 'false');
      if (requestsToggleBtn) {
        requestsToggleBtn.setAttribute('aria-expanded', 'true');
      }
      clearRequestsFeedback();
      if (requestsContent) {
        requestsContent.scrollTop = 0;
        renderRequestsContent(requestsState.items);
      }
      loadRequests({ force: true });
      if (requestsCloseBtn) {
        requestsCloseBtn.focus({ preventScroll: true });
      } else {
        requestsDrawer.focus({ preventScroll: true });
      }
    }

    function closeRequestsDrawer() {
      if (!requestsDrawer || !requestsState.open) return;
      requestsState.open = false;
      requestsDrawer.classList.remove('open');
      requestsDrawer.setAttribute('aria-hidden', 'true');
      if (requestsToggleBtn) {
        requestsToggleBtn.setAttribute('aria-expanded', 'false');
      }
      clearRequestsFeedback();
      if (requestsBackdrop) {
        requestsBackdrop.classList.remove('visible');
        setTimeout(() => {
          if (!requestsState.open && requestsBackdrop) {
            requestsBackdrop.hidden = true;
          }
        }, 220);
      }
      if (requestsToggleBtn) {
        requestsToggleBtn.focus({ preventScroll: true });
      }
    }

    function toggleRequestsDrawer() {
      if (requestsState.open) {
        closeRequestsDrawer();
      } else {
        openRequestsDrawer();
      }
    }

    function setupRequestsUI() {
      if (!requestsToggleBtn || !requestsDrawer) return;
      updateRequestsBadge(initialPendingRequests);
      requestsToggleBtn.addEventListener('click', toggleRequestsDrawer);
      if (requestsCloseBtn) {
        requestsCloseBtn.addEventListener('click', closeRequestsDrawer);
      }
      if (requestsBackdrop) {
        requestsBackdrop.addEventListener('click', closeRequestsDrawer);
      }
      ensureRequestsPolling();
      loadRequests({ force: false, silent: true });
    }

    let pdfDoc = null;
    let pageNum = 1;
    let pageCount = 0;
    let pageRendering = false;
    let pageNumPending = null;
    let currentItemIdx = 0;
    let currentItem = null;

    pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

    function saveIndex(i) {
      try { localStorage.setItem('liveIndex:' + SETLIST_ID, String(i)); } catch(e){}
    }
    function loadIndex() {
      try {
        const v = localStorage.getItem('liveIndex:' + SETLIST_ID);
        return v == null ? null : Math.max(0, Math.min(items.length-1, parseInt(v,10)||0));
      } catch(e){ return null; }
    }

    function pageStateKey(songId){
      return 'livePage:' + SETLIST_ID + ':' + songId;
    }
    function savePage(num){
      if (!currentItem || !currentItem.song_id) return;
      try { localStorage.setItem(pageStateKey(currentItem.song_id), String(num)); } catch(e){}
    }
    function loadSavedPage(songId){
      try {
        const v = localStorage.getItem(pageStateKey(songId));
        return v ? Math.max(1, parseInt(v,10)||1) : 1;
      } catch(e){ return 1; }
    }

    function renderList(activeIdx) {
      listEl.innerHTML = '';
      if (!items.length) {
        const empty = document.createElement('div');
        empty.className = 'item muted';
        empty.textContent = 'No songs loaded.';
        listEl.appendChild(empty);
        console.warn('[LiveMode] No items available for this setlist.');
        return;
      }
      items.forEach((it, idx) => {
        const div = document.createElement('div');
        div.className = 'item' + (idx === activeIdx ? ' active' : '');
        div.dataset.idx = idx;

        const pos = document.createElement('span');
        pos.className = 'pos';
        pos.textContent = '#' + it.pos;

        const title = document.createElement('span');
        title.className = 'title';
        title.textContent = it.title;

        const tag = document.createElement('span');
        tag.className = 'tag muted';
        let tagLabel = 'Fallback';
        if (it.has_chart && it.has_pdf) tagLabel = 'Chart + PDF';
        else if (it.has_chart) tagLabel = 'Chart';
        else if (it.has_pdf) tagLabel = 'PDF';
        tag.textContent = tagLabel;

        div.appendChild(pos);
        div.appendChild(title);
        div.appendChild(tag);
        div.addEventListener('click', () => show(idx));
        listEl.appendChild(div);
      });
    }

    function clearViewer(){
      canvas.style.display = 'none';
      canvas.width = 0;
      canvas.height = 0;
      canvas.style.width = '0px';
      canvas.style.height = '0px';
      if (thumbsEl) {
        thumbsEl.innerHTML = '';
        thumbsEl.style.display = 'none';
      }
      if (chartLines) chartLines.innerHTML = '';
      if (chartWrapper) chartWrapper.style.display = 'none';
      if (chartTitle) chartTitle.textContent = '';
      if (chartMeta) chartMeta.textContent = '';
      if (chartTools) chartTools.style.display = 'none';
      if (viewToggleBtn) viewToggleBtn.hidden = true;
      pagePrevBtn.disabled = pageNextBtn.disabled = true;
      pageInfo.textContent = '-- / --';
      noPdfEl.style.display = 'none';
      noPdfEl.innerHTML = '';
    }

    function show(idx) {
      if (items.length === 0) {
        nowEl.textContent = '—';
        noPdfEl.innerHTML = '<p>No songs are currently in this setlist.</p>';
        noPdfEl.style.display = 'flex';
        console.warn('[LiveMode] show() called with empty items.');
        return;
      }
      idx = Math.max(0, Math.min(items.length-1, idx));
      const it = items[idx];
      nowEl.textContent = (idx+1) + ' / ' + items.length + '  —  ' + it.title;
      saveIndex(idx);
      renderList(idx);
      history.replaceState({}, '', updateQueryString('i', idx));
      currentItemIdx = idx;
      currentItem = it;
      loadDocument(it);
    }

    function updateQueryString(key, value) {
      const url = new URL(location.href);
      url.searchParams.set(key, value);
      return url.toString();
    }

    function nextSong() { if (items.length) show((loadIndex() ?? 0) + 1); }
    function prevSong() { if (items.length) show((loadIndex() ?? 0) - 1); }

    function updatePageControls(){
      if (!pdfDoc || currentViewMode !== 'pdf') {
        pageInfo.textContent = '-- / --';
        pagePrevBtn.disabled = true;
        pageNextBtn.disabled = true;
        Array.from(thumbsEl.querySelectorAll('.thumb')).forEach((el) => {
          el.classList.remove('active');
        });
        return;
      }
      pageInfo.textContent = pageCount ? (pageNum + ' / ' + pageCount) : '-- / --';
      pagePrevBtn.disabled = pageNum <= 1;
      pageNextBtn.disabled = pageNum >= pageCount;
      Array.from(thumbsEl.querySelectorAll('.thumb')).forEach((el, idx) => {
        el.classList.toggle('active', idx + 1 === pageNum);
      });
    }

    function renderPage(num){
      if (!pdfDoc) return;
      pageRendering = true;
      pdfDoc.getPage(num).then(page => {
        const containerWidth = canvas.parentElement.clientWidth - 8;
        const viewport = page.getViewport({ scale: 1 });
        const scale = Math.max(0.1, containerWidth / viewport.width);
        const scaled = page.getViewport({ scale });
        const outputScale = window.devicePixelRatio || 1;

        canvas.width = scaled.width * outputScale;
        canvas.height = scaled.height * outputScale;
        canvas.style.width = scaled.width + 'px';
        canvas.style.height = scaled.height + 'px';

        const transform = outputScale !== 1 ? [outputScale, 0, 0, outputScale, 0, 0] : null;
        return page.render({ canvasContext: ctx, viewport: scaled, transform }).promise;
      }).then(() => {
        pageRendering = false;
        if (pageNumPending !== null){
          const numPending = pageNumPending;
          pageNumPending = null;
          renderPage(numPending);
        }
      }).catch(err => console.error('PDF render error', err))
      .finally(() => {
        updatePageControls();
      });
      savePage(num);
    }

    function queueRenderPage(num){
      if (pageRendering){
        pageNumPending = num;
      } else {
        renderPage(num);
      }
    }

    function buildThumbnails(){
      thumbsEl.innerHTML = '';
      if (!pdfDoc) return;
      for (let i = 1; i <= pdfDoc.numPages; i++){
        const wrapper = document.createElement('div');
        wrapper.className = 'thumb';
        const c = document.createElement('canvas');
        wrapper.appendChild(c);
        wrapper.addEventListener('click', () => {
          if (pageNum !== i){ pageNum = i; queueRenderPage(i); }
        });
        thumbsEl.appendChild(wrapper);

        pdfDoc.getPage(i).then(page => {
          const viewport = page.getViewport({ scale: 0.2 });
          const ctxThumb = c.getContext('2d');
          c.width = viewport.width;
          c.height = viewport.height;
          return page.render({ canvasContext: ctxThumb, viewport }).promise;
        }).catch(err => console.warn('Thumb render', err));
      }
    }

    let currentPdfItemId = null;

    function hidePdfViewer() {
      canvas.style.display = 'none';
      canvas.width = 0;
      canvas.height = 0;
      canvas.style.width = '0px';
      canvas.style.height = '0px';
      if (thumbsEl) {
        thumbsEl.style.display = 'none';
      }
      pagePrevBtn.disabled = true;
      pageNextBtn.disabled = true;
      pageInfo.textContent = '-- / --';
    }

    function loadPdfDocument(item, forceReload = true) {
      if (!item || !item.has_pdf) return;
      canvas.style.display = 'block';
      if (thumbsEl) {
        thumbsEl.style.display = 'flex';
      }
      noPdfEl.style.display = 'none';
      noPdfEl.innerHTML = '';

      if (!forceReload && currentPdfItemId === item.song_id && pdfDoc) {
        pageCount = pdfDoc.numPages;
        pageNum = Math.max(1, Math.min(pageCount, pageNum || 1));
        buildThumbnails();
        queueRenderPage(pageNum);
        return;
      }

      pdfDoc = null;
      pageNum = 1;
      pageCount = 0;
      updatePageControls();
      const loadingTask = pdfjsLib.getDocument({ url: item.url + '?t=' + Date.now() });
      loadingTask.promise.then(doc => {
        pdfDoc = doc;
        currentPdfItemId = item.song_id;
        pageCount = doc.numPages;
        pageNum = Math.min(doc.numPages, loadSavedPage(item.song_id) || 1);
        buildThumbnails();
        queueRenderPage(pageNum);
      }).catch(err => {
        console.error('PDF load error', err);
        hidePdfViewer();
        noPdfEl.innerHTML = '<p>Failed to load PDF.</p><p><a class="btn" target="_blank" href="' + item.url + '">Open fallback view</a></p>';
        noPdfEl.style.display = 'flex';
      });
    }

    function updateChartMeta(item) {
      if (!chartMeta || !item) return;
      const parts = [];
      const baseKey = (item.song_key || '').trim();
      if (baseKey) {
        parts.push('Base ' + baseKey);
        const concertKey = transposeKeyLabel(baseKey, currentTranspose);
        if (concertKey && concertKey !== baseKey) {
          parts.push('Concert ' + concertKey);
        }
        const displayKey = transposeKeyLabel(baseKey, currentTranspose - currentCapo);
        if (displayKey) {
          parts.push('Display ' + displayKey);
        }
      }
      if (currentTranspose !== 0) {
        parts.push('Transpose ' + formatSemitone(currentTranspose));
      }
      parts.push('Capo ' + currentCapo);
      chartMeta.textContent = parts.join(' • ');
    }

    function updateChartControls(item) {
      if (!chartTools || !item || !item.has_chart) {
        if (chartTools) chartTools.style.display = 'none';
        return;
      }
      transposeDisplay.textContent = formatSemitone(currentTranspose);
      capoDisplay.textContent = String(currentCapo);
      updateChartMeta(item);
    }

    function renderChart(item) {
      if (!chartLines || !item || !item.has_chart) return;
      chartLines.innerHTML = '';
      const lines = (item.chart || '').split(/\\r?\\n/);
      const effectiveSteps = currentTranspose - currentCapo;
      let chordIndex = 0;

      lines.forEach((rawLine) => {
        const line = rawLine ?? '';
        const trimmed = line.trim();
        if (!trimmed) {
          const spacerBlock = document.createElement('div');
          spacerBlock.className = 'chart-line';
          spacerBlock.innerHTML = '&nbsp;';
          chartLines.appendChild(spacerBlock);
          return;
        }

        if (/^\[[^\]]+\]$/.test(trimmed) && !isLikelyChordSymbol(trimmed.slice(1, -1))) {
          const directiveLine = document.createElement('div');
          directiveLine.className = 'chart-line chart-directive';
          directiveLine.textContent = trimmed.slice(1, -1).trim();
          chartLines.appendChild(directiveLine);
          return;
        }

        if (/^\{.*\}$/.test(trimmed)) {
          const directive = document.createElement('div');
          directive.className = 'chart-line chart-directive';
          const dirMatch = trimmed.match(/^\{([^:]+):(.*)\}$/);
          if (dirMatch) {
            directive.textContent = dirMatch[1].trim() + ': ' + dirMatch[2].trim();
          } else {
            directive.textContent = trimmed.slice(1, -1);
          }
          chartLines.appendChild(directive);
          return;
        }

        if (trimmed.startsWith('#')) {
          const comment = document.createElement('div');
          comment.className = 'chart-line chart-comment';
          comment.textContent = trimmed.replace(/^#\s*/, '');
          chartLines.appendChild(comment);
          return;
        }

        const block = document.createElement('div');
        block.className = 'chart-block';
        let lyricBuffer = '';
        const chordEntries = [];
        let cursor = 0;
        while (cursor < line.length) {
          const open = line.indexOf('[', cursor);
          if (open === -1) {
            const lyricText = line.slice(cursor);
            if (lyricText) {
              lyricBuffer += lyricText;
            }
            break;
          }
          if (open > cursor) {
            const lyricText = line.slice(cursor, open);
            if (lyricText) {
              lyricBuffer += lyricText;
            }
          }
          const close = line.indexOf(']', open + 1);
          if (close === -1) {
            lyricBuffer += line.slice(open);
            break;
          }
          const rawChord = line.slice(open + 1, close).trim();
          if (!isLikelyChordSymbol(rawChord)) {
            lyricBuffer += rawChord;
            cursor = close + 1;
            continue;
          }

          const displayChord = transposeChordSymbol(rawChord, effectiveSteps) || rawChord;
          const concertChord = transposeChordSymbol(rawChord, currentTranspose) || rawChord;
          chordEntries.push({
            original: rawChord,
            display: displayChord,
            concert: concertChord,
            pos: lyricBuffer.length,
            index: chordIndex,
          });
          chordIndex += 1;
          cursor = close + 1;
        }

        if (chordEntries.length) {
          const chordRow = document.createElement('div');
          chordRow.className = 'chart-line chart-chords';
          let currentColumn = 0;
          chordEntries.forEach(entry => {
            const spaces = Math.max(0, entry.pos - currentColumn);
            if (spaces > 0) {
              chordRow.appendChild(document.createTextNode(' '.repeat(spaces)));
              currentColumn += spaces;
            }
            const chordSpan = document.createElement('span');
            chordSpan.className = 'chord';
            chordSpan.textContent = entry.display || entry.original || ' ';
            chordSpan.title = entry.concert !== entry.display
              ? `Concert: ${entry.concert}\nDisplay: ${entry.display}`
              : entry.display;
            chordSpan.dataset.original = entry.original;
            chordSpan.dataset.display = entry.display;
            chordSpan.dataset.concert = entry.concert;
            chordSpan.dataset.index = String(entry.index);
            if (currentHighlightSet.has(entry.index)) {
              chordSpan.classList.add('highlighted');
            }
            chordRow.appendChild(chordSpan);
            currentColumn += entry.display.length;
          });
          block.appendChild(chordRow);
        }

        const lyricRow = document.createElement('div');
        lyricRow.className = 'chart-line chart-lyric';
        lyricRow.textContent = lyricBuffer || '\u00A0';
        block.appendChild(lyricRow);
        chartLines.appendChild(block);
      });
      updateChartControls(item);
    }

    function updateViewToggle(item) {
      if (!viewToggleBtn) return;
      if (!item || !(item.has_chart && item.has_pdf)) {
        viewToggleBtn.hidden = true;
        return;
      }
      viewToggleBtn.hidden = false;
      viewToggleBtn.textContent = currentViewMode === 'chart' ? 'Show PDF' : 'Show Chart';
    }

    function applyViewDisplay(item, options = {}) {
      if (!item) return;
      const hasChart = !!item.has_chart;
      const hasPdf = !!item.has_pdf;

      console.info('[LiveMode] Applying view', { song_id: item.song_id, hasChart, hasPdf, viewMode: currentViewMode, options });
      updateViewToggle(item);

      if (hasChart) {
        if (chartTitle) {
          const titlePieces = [];
          if (item.song_title) titlePieces.push(item.song_title);
          if (item.song_artist) titlePieces.push(item.song_artist);
          chartTitle.textContent = titlePieces.join(' — ');
        }
        renderChart(item);
      }

      if (currentViewMode === 'chart' && hasChart) {
        if (chartWrapper) chartWrapper.style.display = 'flex';
        if (chartTools) chartTools.style.display = 'flex';
        hidePdfViewer();
        noPdfEl.style.display = 'none';
        noPdfEl.innerHTML = '';
      } else if (currentViewMode === 'pdf' && hasPdf) {
        if (chartWrapper) chartWrapper.style.display = 'none';
        if (chartTools) chartTools.style.display = hasChart ? 'none' : 'none';
        loadPdfDocument(item, options.reload !== false);
      } else if (hasChart) {
        currentViewMode = 'chart';
        applyViewDisplay(item, options);
        return;
      } else if (hasPdf) {
        currentViewMode = 'pdf';
        applyViewDisplay(item, options);
        return;
      } else {
        hidePdfViewer();
        if (chartWrapper) chartWrapper.style.display = 'none';
        if (chartTools) chartTools.style.display = 'none';
        noPdfEl.innerHTML = '<p>No chart or PDF available.</p>';
        noPdfEl.style.display = 'flex';
      }

      updateChartControls(item);
      persistChartPrefs();
    }

    function loadDocument(item){
      clearViewer();
      if (!item) return;

      const prefs = loadChartPrefs(item.song_id);
      currentTranspose = Number.isFinite(prefs.transpose) ? prefs.transpose : 0;
      currentCapo = Number.isFinite(prefs.capo) ? Math.max(0, Math.min(12, prefs.capo)) : 0;
      currentHighlightSet = new Set(Array.isArray(prefs.highlights) ? prefs.highlights.map(Number) : []);
      const desiredView = (prefs.view === 'chart' || prefs.view === 'pdf') ? prefs.view : null;
      currentViewMode = desiredView || (item.has_chart ? 'chart' : 'pdf');
      if (currentViewMode === 'chart' && !item.has_chart && item.has_pdf) currentViewMode = 'pdf';
      if (currentViewMode === 'pdf' && !item.has_pdf && item.has_chart) currentViewMode = 'chart';

      console.info('[LiveMode] loadDocument', { song_id: item.song_id, currentViewMode, prefs });
      applyViewDisplay(item, { reload: true });
    }

    function changeTranspose(delta) {
      if (!currentItem || !currentItem.has_chart) return;
      const next = Math.max(-12, Math.min(12, currentTranspose + delta));
      if (next === currentTranspose) return;
      currentTranspose = next;
      renderChart(currentItem);
      updateChartControls(currentItem);
      persistChartPrefs();
    }

    function changeCapo(delta) {
      if (!currentItem || !currentItem.has_chart) return;
      const next = Math.max(0, Math.min(12, currentCapo + delta));
      if (next === currentCapo) return;
      currentCapo = next;
      renderChart(currentItem);
      updateChartControls(currentItem);
      persistChartPrefs();
    }

    function resetChartSettings() {
      if (!currentItem || !currentItem.has_chart) return;
      currentTranspose = 0;
      currentCapo = 0;
      renderChart(currentItem);
      updateChartControls(currentItem);
      persistChartPrefs();
    }

    if (transposeUpBtn) transposeUpBtn.addEventListener('click', () => changeTranspose(1));
    if (transposeDownBtn) transposeDownBtn.addEventListener('click', () => changeTranspose(-1));
    if (capoUpBtn) capoUpBtn.addEventListener('click', () => changeCapo(1));
    if (capoDownBtn) capoDownBtn.addEventListener('click', () => changeCapo(-1));
    if (transposeResetBtn) transposeResetBtn.addEventListener('click', resetChartSettings);

    if (chartLines) {
      chartLines.addEventListener('click', (event) => {
        const target = event.target.closest('.chord');
        if (!target || !currentItem || !currentItem.has_chart) return;
        const idx = parseInt(target.dataset.index, 10);
        if (!Number.isInteger(idx)) return;
        if (target.classList.contains('highlighted')) {
          target.classList.remove('highlighted');
          currentHighlightSet.delete(idx);
        } else {
          target.classList.add('highlighted');
          currentHighlightSet.add(idx);
        }
        persistChartPrefs();
      });
    }

    if (viewToggleBtn) {
      viewToggleBtn.addEventListener('click', () => {
        if (!currentItem) return;
        const next = currentViewMode === 'chart' ? 'pdf' : 'chart';
        if (next === 'chart' && !currentItem.has_chart) return;
        if (next === 'pdf' && !currentItem.has_pdf) return;
        currentViewMode = next;
        applyViewDisplay(currentItem, { reload: next === 'pdf' });
      });
    }

    function nextPage(){
      if (!pdfDoc || pageNum >= pageCount) return;
      pageNum += 1;
      queueRenderPage(pageNum);
    }
    function prevPage(){
      if (!pdfDoc || pageNum <= 1) return;
      pageNum -= 1;
      queueRenderPage(pageNum);
    }

    const PEDAL_NEXT_KEYS = new Set(['ArrowRight','PageDown',' ','Spacebar','Enter','N','MediaTrackNext']);
    const PEDAL_PREV_KEYS = new Set(['ArrowLeft','PageUp','Backspace','P','MediaTrackPrevious']);
    const PEDAL_SCROLL_DOWN_KEYS = new Set(['ArrowDown','PageDown',' ','Spacebar']);
    const PEDAL_SCROLL_UP_KEYS = new Set(['ArrowUp','PageUp']);

    function handleScroll(direction) {
      if (!chartWrapper || chartWrapper.style.display === 'none') {
        if (direction === 1) nextPage();
        else prevPage();
        return;
      }
      const amount = Math.round(chartWrapper.clientHeight * 0.8);
      chartWrapper.scrollBy({ top: direction * amount, behavior: 'smooth' });
    }

    // Keyboard / pedal navigation
    document.addEventListener('keydown', (e) => {
      if (e.repeat) return;
      const key = e.key || e.code;
      if (PEDAL_NEXT_KEYS.has(key)) {
        e.preventDefault();
        if (chartWrapper && chartWrapper.style.display !== 'none') {
          const nearBottom = chartWrapper.scrollTop + chartWrapper.clientHeight >= chartWrapper.scrollHeight - 10;
          if (nearBottom) {
            nextSong();
          } else {
            handleScroll(1);
          }
        } else {
          nextPage();
        }
      } else if (PEDAL_PREV_KEYS.has(key)) {
        e.preventDefault();
        if (chartWrapper && chartWrapper.style.display !== 'none') {
          const nearTop = chartWrapper.scrollTop <= 10;
          if (nearTop) {
            prevSong();
          } else {
            handleScroll(-1);
          }
        } else {
          prevPage();
        }
      } else if (PEDAL_SCROLL_DOWN_KEYS.has(key) && chartWrapper && chartWrapper.style.display !== 'none') {
        e.preventDefault();
        handleScroll(1);
      } else if (PEDAL_SCROLL_UP_KEYS.has(key) && chartWrapper && chartWrapper.style.display !== 'none') {
        e.preventDefault();
        handleScroll(-1);
      } else if (key === 'Home') {
        e.preventDefault();
        if (chartWrapper && chartWrapper.style.display !== 'none') {
          chartWrapper.scrollTo({ top: 0, behavior: 'smooth' });
        } else {
          pageNum = 1;
          queueRenderPage(pageNum);
        }
      } else if (key === 'End') {
        e.preventDefault();
        if (chartWrapper && chartWrapper.style.display !== 'none') {
          chartWrapper.scrollTo({ top: chartWrapper.scrollHeight, behavior: 'smooth' });
        } else if (pdfDoc) {
          pageNum = pageCount;
          queueRenderPage(pageNum);
        }
      } else if ((key === '+' || key === '=') && currentItem && currentItem.has_chart) {
        e.preventDefault();
        changeTranspose(1);
      } else if ((key === '-' || key === '_') && currentItem && currentItem.has_chart) {
        e.preventDefault();
        changeTranspose(-1);
      } else if ((key === '[') && currentItem && currentItem.has_chart) {
        e.preventDefault();
        changeCapo(-1);
      } else if ((key === ']') && currentItem && currentItem.has_chart) {
        e.preventDefault();
        changeCapo(1);
      } else if (key && key.toLowerCase() === 'r' && currentItem && currentItem.has_chart) {
        e.preventDefault();
        resetChartSettings();
      } else if (key === 'Escape' && requestsState.open) {
        e.preventDefault();
        closeRequestsDrawer();
      }
    });


    prevBtn.addEventListener('click', prevSong);
    nextBtn.addEventListener('click', nextSong);
    pagePrevBtn.addEventListener('click', prevPage);
    pageNextBtn.addEventListener('click', nextPage);

    // Initial index: URL ?i=, else last saved, else 0
    (function init(){
      setupRequestsUI();
      let urlIndex = null;
      try {
        const params = new URL(window.location.href).searchParams;
        if (params.has('i')) {
          const parsed = parseInt(params.get('i'), 10);
          if (!Number.isNaN(parsed)) {
            urlIndex = Math.max(0, parsed);
          }
        }
      } catch (err) {
        console.warn('[LiveMode] Failed to parse initial index from URL', err);
      }
      const start = (urlIndex !== null) ? urlIndex : (loadIndex() ?? 0);
      renderList(start);
      show(start);
    })();
  </script>
</body>
</html>
"""

SETLIST_SHARE_HTML = """
<h2>{{ sl.name }}</h2>
{% set details = [] %}
{% if sl.event_type %}{% set _ = details.append('Event: ' ~ sl.event_type) %}{% endif %}
{% if sl.venue_type %}{% set _ = details.append('Venue: ' ~ sl.venue_type) %}{% endif %}
{% if sl.target_minutes %}{% set _ = details.append('Target: ' ~ sl.target_minutes ~ ' min') %}{% endif %}
{% if details %}<p class="muted">{{ details|join(' · ') }}</p>{% endif %}
<p class="muted">You're viewing the setlist shared by the band — feel free to request a song!</p>

<div class="section">
  <h3>Request a song</h3>
  <form method="post" action="{{ url_for('submit_request_token', token=sl.share_token) }}" style="display:grid; gap:10px; max-width:520px;">
    <label>
      Choose from tonight's set
      <select name="song_id">
        <option value="">— Select a song —</option>
        {% for row in songs %}
          <option value="{{ row.song.id }}">#{{ row.position }} — {{ row.song.title }} — {{ row.song.artist }}</option>
        {% endfor %}
      </select>
    </label>
    <label>
      Or request something else
      <input name="free_text" placeholder="Song · Artist" />
    </label>
    <div class="grid" style="gap:10px;">
      <label>Your name (optional)<br/>
        <input name="from_name" />
      </label>
      <label>Contact (optional)<br/>
        <input name="from_contact" placeholder="Phone or email" />
      </label>
    </div>
    <p class="muted" style="margin-top:-6px;">We'll do our best — thanks for the request!</p>
    <button class="btn" type="submit">Send Request</button>
  </form>
</div>

<div class="section" style="margin-top:20px;">
  <h3>Tonight's songs</h3>
  {% for row in songs %}
    <div class="row">
      <div>
        <strong>#{{ row.position }} — {{ row.song.title }}</strong> — {{ row.song.artist }}
        <div class="muted">
          {% if row.song.musical_key %}Key: {{ row.song.musical_key }} · {% endif %}
          {% if row.song.tempo_bpm %}BPM: {{ row.song.tempo_bpm }} · {% endif %}
          {% if row.song.release_year %}Year: {{ row.song.release_year }} · {% endif %}
          {{ fmt_mmss(row.song.duration_override_sec or estimate(row.song.tempo_bpm)) }}
        </div>
        {% if row.notes %}<div class="muted"><em>Notes:</em> {{ row.notes }}</div>{% endif %}
      </div>
    </div>
  {% endfor %}
</div>
"""

SETLIST_REQUESTS_HTML = """
<h2>Audience Requests — {{ sl.name }}</h2>
<div class="section" style="display:flex; gap:20px; flex-wrap:wrap; align-items:center;">
  <div>
    <img src="{{ qr_url }}" alt="Requests QR" width="200" height="200" style="border:1px solid #ddd; border-radius:12px; padding:8px; background:#fff;"/>
  </div>
  <div style="flex:1; min-width:220px;">
    <p class="muted" style="margin-bottom:8px;">Share this QR so the crowd can scan and request songs.</p>
    <p><a class="btn" href="{{ share_url }}" target="_blank">Open public page</a></p>
    <p><a class="btn" href="{{ qr_url }}" download="setlist-{{ sl.id }}-requests.png">Download QR PNG</a></p>
  </div>
</div>

<div class="section" style="margin-top:20px;">
  <h3>New requests</h3>
  {% if new_requests %}
    {% for req in new_requests %}
      <div class="section" style="margin-top:10px;">
        <strong>{{ req.label() }}</strong>
        <div class="muted">
          Requested {{ req.created_at.strftime('%Y-%m-%d %H:%M') }}
          {% if req.from_name %} · by {{ req.from_name }}{% endif %}
          {% if req.from_contact %} · contact: {{ req.from_contact }}{% endif %}
        </div>
        {% if req.free_text_title and req.song %}<div class="muted"><em>Note:</em> {{ req.free_text_title }}</div>{% elif req.free_text_title %}<div class="muted"><em>Requested:</em> {{ req.free_text_title }}</div>{% endif %}
        <div style="margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;">
          <form method="post" action="{{ url_for('update_request_status', request_id=req.id) }}">
            <input type="hidden" name="status" value="queued" />
            <button class="btn" type="submit">Queue</button>
          </form>
          <form method="post" action="{{ url_for('update_request_status', request_id=req.id) }}">
            <input type="hidden" name="status" value="declined" />
            <button class="btn btn-danger" type="submit">Decline</button>
          </form>
          {% if req.song_id and req.setlist_id %}
          <form method="post" action="{{ url_for('add_request_to_setlist', request_id=req.id) }}">
            <button class="btn" type="submit" title="Add this song to the setlist">Add to setlist</button>
          </form>
          {% endif %}
        </div>
      </div>
    {% endfor %}
  {% else %}
    <p class="muted">No new requests right now.</p>
  {% endif %}
</div>

<div class="section" style="margin-top:20px;">
  <h3>Queued / handled</h3>
  {% if other_requests %}
    <table style="width:100%; border-collapse:collapse;">
      <thead>
        <tr>
          <th style="text-align:left; padding:6px; border-bottom:1px solid #ddd;">Request</th>
          <th style="text-align:left; padding:6px; border-bottom:1px solid #ddd;">Status</th>
          <th style="text-align:left; padding:6px; border-bottom:1px solid #ddd;">Updated</th>
          <th style="text-align:left; padding:6px; border-bottom:1px solid #ddd;">Actions</th>
        </tr>
      </thead>
      <tbody>
        {% for req in other_requests %}
        <tr>
          <td style="padding:6px;">
            <strong>{{ req.label() }}</strong>
            {% if req.free_text_title and req.song %}<div class="muted"><em>Note:</em> {{ req.free_text_title }}</div>{% elif req.free_text_title %}<div class="muted"><em>Requested:</em> {{ req.free_text_title }}</div>{% endif %}
            <div class="muted">
              {{ req.created_at.strftime('%Y-%m-%d %H:%M') }}{% if req.from_name %} · {{ req.from_name }}{% endif %}{% if req.from_contact %} · {{ req.from_contact }}{% endif %}
            </div>
          </td>
          <td style="padding:6px;">{{ req.status.title() }}</td>
          <td style="padding:6px;">{{ req.updated_at.strftime('%Y-%m-%d %H:%M') if req.updated_at else req.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
          <td style="padding:6px; display:flex; gap:6px; flex-wrap:wrap;">
            {% for status in statuses %}
              {% if status != req.status %}
              <form method="post" action="{{ url_for('update_request_status', request_id=req.id) }}">
                <input type="hidden" name="status" value="{{ status }}" />
                <button class="btn" type="submit">Mark {{ status }}</button>
              </form>
              {% endif %}
            {% endfor %}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  {% else %}
    <p class="muted">No queued or completed requests yet.</p>
  {% endif %}
</div>

<p style="margin-top:20px;"><a class="btn" href="{{ url_for('edit_setlist', setlist_id=sl.id) }}">← Back to setlist</a></p>
"""

# --- routes: Home ---
@app.get("/")
def home():
    inner = """
    <h2>Welcome</h2>
    <p>This is your local Setlist Genie dev app. Use the buttons above to manage songs and setlists.</p>
    """
    return render_template_string(BASE_HTML, content=inner)
    
def _admin_ok(req):
    """Allow initdb only if no ADMIN_TOKEN is set, or if it matches."""
    token = os.getenv("ADMIN_TOKEN")
    if not token:
        # No token configured → allow (dev default)
        return True
    return (req.args.get("admin") == token) or (req.headers.get("X-Admin-Token") == token)

# --- routes: Songs ---
@app.post("/songs/<int:song_id>/attachments")
def upload_attachment(song_id):
    return upload_song_file(song_id)


@app.post("/songs/<int:song_id>/files")
def upload_song_file(song_id):
    song = Song.query.get_or_404(song_id)
    f = request.files.get("file")
    if not f or (f.filename or "").strip() == "":
        flash("No file selected.")
        return redirect(url_for("edit_song", song_id=song.id))

    if not _allowed_file(f):
        flash("Only PDF files are allowed.")
        return redirect(url_for("edit_song", song_id=song.id))

    try:
        stored, original, size, mimetype = _store_file(f)
        sf = SongFile(
            song_id=song.id,
            original_name=original,
            orig_name=original,
            stored_name=stored,
            size_bytes=size,
            mimetype=mimetype,
            kind="pdf",
        )
        db.session.add(sf)
        db.session.flush()
        if song.default_file_id is None:
            song.default_file_id = sf.id
        db.session.commit()
        flash(f'Uploaded “{original}”.')
    except Exception as e:
        db.session.rollback()
        flash(f"Upload failed: {e}")

    return redirect(url_for("edit_song", song_id=song.id))


def _apply_song_default_file(song: Song, song_file: SongFile) -> None:
    song.default_file_id = song_file.id
    db.session.commit()


@app.post("/songs/<int:song_id>/attachments/<int:att_id>/default", endpoint="song_set_default_attachment")
def set_default_attachment_legacy(song_id, att_id):
    return set_default_file(song_id, att_id)


@app.post("/songs/<int:song_id>/files/<int:file_id>/default", endpoint="song_set_default_file")
def set_default_file(song_id, file_id):
    song = Song.query.get_or_404(song_id)
    sf = SongFile.query.get_or_404(file_id)
    if sf.song_id != song.id:
        flash("That file doesn’t belong to this song.")
        return redirect(url_for("edit_song", song_id=song.id))
    _apply_song_default_file(song, sf)
    flash("Default file set for this song.", "success")
    return redirect(url_for("edit_song", song_id=song.id))


@app.get("/attachments/<int:att_id>")
def view_attachment(att_id):
    sf = SongFile.query.get_or_404(att_id)
    abs_path = os.path.abspath(sf.path)

    if not os.path.exists(abs_path):
        flash("File not found on disk.")
        return redirect(url_for("edit_song", song_id=sf.song_id))

    base_dir = os.path.abspath(str(UPLOAD_DIR))
    if not abs_path.startswith(base_dir + os.sep) and abs_path != base_dir:
        abort(403)

    return send_file(
        abs_path,
        mimetype=sf.mimetype or "application/pdf",
        as_attachment=False,
        download_name=sf.filename or os.path.basename(abs_path),
        max_age=0,
        conditional=True,
        etag=True,
    )


@app.get("/attachments/<int:att_id>/download")
def download_attachment(att_id):
    sf = SongFile.query.get_or_404(att_id)
    return send_from_directory(
        UPLOAD_DIR,
        sf.stored_name,
        mimetype=sf.mimetype,
        as_attachment=True,
        download_name=sf.original_name or sf.filename,
        max_age=0,
    )


@app.post("/attachments/<int:att_id>/delete")
def delete_attachment(att_id):
    sf = SongFile.query.get_or_404(att_id)
    song_id = sf.song_id

    if getattr(sf.song, "default_file_id", None) == sf.id:
        sf.song.default_file_id = None
    SetlistSong.query.filter_by(preferred_file_id=sf.id).update(
        {"preferred_attachment_id": None}
    )

    try:
        path = _song_file_abs_path(sf)
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

    db.session.delete(sf)
    db.session.commit()
    flash("File removed.", "success")
    return redirect(url_for("edit_song", song_id=song_id))

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
    release_year_raw = (request.form.get("release_year") or "").strip()
    release_year = None
    if release_year_raw:
        try:
            release_year = int(release_year_raw[:4])
        except Exception:
            release_year = None
    chord_chart_raw = request.form.get("chord_chart", "")
    chord_chart = chord_chart_raw.strip() or None
    generate_ai_chart = request.form.get("generate_ai_chart") == "1"

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
        release_year=release_year,
        chord_chart=chord_chart,
    )
    db.session.add(song)
    db.session.flush()  # ensure song.id is available for relationships/logging

    ai_chart_generated = False
    try:
        meta = _fetch_song_metadata(song)
        if _apply_song_metadata(song, meta, force=False):
            src = "Spotify" if meta.get("_source") == "spotify" else "OpenAI"
            flash(f"Song details auto-filled via {src}.", "info")
    except Exception as e:
        print("Auto metadata enrichment failed:", e)

    if generate_ai_chart and not song.chord_chart:
        chart_text, error = openai_generate_chord_chart(song.title, song.artist)
        if chart_text:
            song.chord_chart = chart_text
            ai_chart_generated = True
        else:
            flash(f"AI chord chart unavailable: {error}", "warning")

    db.session.commit()
    if ai_chart_generated:
        flash("Draft chord chart generated with AI. Accuracy not guaranteed — please review before performing.", "warning")
    flash(f'Added “{song.title}” — {song.artist}.')
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
    release_year_raw = (request.form.get("release_year") or "").strip()
    release_year = None
    if release_year_raw:
        try:
            release_year = int(release_year_raw[:4])
        except Exception:
            release_year = None
    chord_chart_raw = request.form.get("chord_chart", "")
    chord_chart = chord_chart_raw.strip() or None

    if not title or not artist:
        return "Title and Artist are required.", 400

    song.title = title
    song.artist = artist
    song.tempo_bpm = int(tempo_bpm) if tempo_bpm else None
    song.musical_key = musical_key
    song.genre = genre
    song.tags = tags
    song.duration_override_sec = duration_override
    song.release_year = release_year
    song.chord_chart = chord_chart
    db.session.commit()
    flash(f'Updated “{song.title}” — {song.artist}.')
    return redirect(url_for("list_songs"))

@app.post("/songs/<int:song_id>/delete")
def delete_song(song_id):
    song = Song.query.get_or_404(song_id)
    # Remove references from setlists first to satisfy NOT NULL constraint on song_id
    SetlistSong.query.filter_by(song_id=song.id).delete(synchronize_session=False)
    db.session.delete(song)
    db.session.commit()
    flash(f'Deleted “{song.title}” — {song.artist}.')
    return redirect(url_for("list_songs"))

@app.post("/songs/<int:song_id>/ai-chart", endpoint="song_generate_ai_chart")
def generate_ai_chart(song_id):
    song = Song.query.get_or_404(song_id)
    chart, error = openai_generate_chord_chart(song.title, song.artist)
    if chart:
        song.chord_chart = chart
        db.session.commit()
        flash("Updated chord chart with a fresh AI draft. Please review for accuracy.", "warning")
    else:
        db.session.rollback()
        flash(f"AI chord chart unavailable: {error}", "warning")
    return redirect(request.referrer or url_for("edit_song", song_id=song.id))

@app.post("/songs/<int:song_id>/ai")
def ai_enrich_song(song_id):
    song = Song.query.get_or_404(song_id)
    force = request.form.get("force") == "1"
    meta = _fetch_song_metadata(song)
    if not meta:
        db.session.commit()
        flash("No metadata found via Spotify/OpenAI.", "warning")
        return redirect(url_for("edit_song", song_id=song.id))

    if _apply_song_metadata(song, meta, force=force):
        db.session.commit()
        src_label = "Spotify" if meta.get("_source") == "spotify" else "OpenAI"
        flash(f"Autofill complete ({src_label}).")
    else:
        db.session.commit()
        flash("Nothing to update — fields already filled.", "info")
    return redirect(url_for("edit_song", song_id=song.id))

@app.route("/songs/import", methods=["GET"])
def songs_import_form():
    # Simple upload form for CSV import
        return """
    <h1>Import Songs (CSV)</h1>
    <form action="/songs/import" method="post" enctype="multipart/form-data" style="margin-bottom:1rem;">
      <p><input type="file" name="file" accept=".csv" required></p>
      <p><button class="btn" type="submit">Upload CSV</button></p>
    </form>

    <div class="section">
      <h3>How this import works</h3>
      <ul>
        <li><strong>Required columns:</strong> <code>title</code>, <code>artist</code></li>
        <li><strong>Optional columns:</strong> <code>tempo_bpm</code>, <code>musical_key</code>, <code>genre</code>, <code>tags</code>, <code>release_year</code>, <code>duration</code></li>
        <li><strong>Duration format:</strong> <code>mm:ss</code> (e.g., <code>3:10</code>) or plain seconds (e.g., <code>190</code>)</li>
        <li><strong>Release year:</strong> four-digit year (e.g., <code>1997</code>)</li>
        <li><strong>Tags:</strong> comma or semicolon separated; we normalize to comma</li>
        <li><strong>Delimiters:</strong> comma, semicolon, or tab are auto-detected</li>
        <li><strong>Upsert key:</strong> rows are matched by <code>(title, artist)</code>; if a match exists, it’s updated</li>
      </ul>
      <p style="margin-top:.5rem;">
        Need a starting point? <a class="btn" href="/songs/template.csv">Download Template</a>
      </p>
      <p style="margin-top:.25rem;"><a class="btn" href="/songs">← Back to Songs</a></p>
    </div>
    """

@app.route("/songs/import", methods=["POST"])
def songs_import_post():
    file = request.files.get("file")
    if not file or file.filename == "":
        return "No file uploaded.", 400

    # --- Read text (handle BOM) ---
    try:
        raw = file.read()
        # utf-8-sig strips BOM if present
        text_data = raw.decode("utf-8-sig", errors="replace")
    except Exception as e:
        return f"Failed to read CSV: {e}", 400

    # --- Detect delimiter (comma default) ---
    try:
        sample = text_data[:4096]
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","  # fall back

    # --- Normalize headers (lower/trim) + friendly aliases ---
    def norm(s): return (s or "").strip().lower()

    # Read once to get headers
    sio = io.StringIO(text_data)
    reader0 = csv.reader(sio, delimiter=delimiter)
    try:
        raw_headers = next(reader0)
    except StopIteration:
        return "Empty CSV.", 400

    headers = [norm(h) for h in raw_headers]

    # Aliases map
    aliases = {
        "tempo": "tempo_bpm",
        "tempo (bpm)": "tempo_bpm",
        "bpm": "tempo_bpm",
        "key": "musical_key",
        "year": "release_year",
        "release year": "release_year",
        "duration (mm:ss)": "duration",
        "length": "duration",
        "time": "duration",
    }
    headers = [aliases.get(h, h) for h in headers]

    # Recreate a DictReader using normalized headers
    sio2 = io.StringIO(text_data)
    reader = csv.DictReader(sio2, fieldnames=headers, delimiter=delimiter)
    next(reader, None)  # skip header row (already read)

    created = updated = skipped = 0
    errors = []

    # Helper: parse row values (trim; unify separators)
    def get_val(row, key):
        val = row.get(key)
        if val is None:
            return ""
        return str(val).strip()

    line_num = 2  # data starts at line 2
    for row in reader:
        try:
            title = get_val(row, "title")
            artist = get_val(row, "artist")
            if not title or not artist:
                skipped += 1
                errors.append(f"Row {line_num}: missing title or artist")
                line_num += 1
                continue

            tempo_str = get_val(row, "tempo_bpm")
            try:
                tempo_bpm = int(tempo_str) if tempo_str else None
            except ValueError:
                tempo_bpm = None
                errors.append(f"Row {line_num}: invalid tempo_bpm '{tempo_str}'")

            musical_key = get_val(row, "musical_key") or None
            genre = get_val(row, "genre") or None

            # tags: accept comma or semicolon; normalize to comma+space
            raw_tags = get_val(row, "tags")
            tags = None
            if raw_tags:
                tags = ", ".join([t.strip() for t in raw_tags.replace(";", ",").split(",") if t.strip()])

            year_str = get_val(row, "release_year")
            release_year = None
            if year_str:
                try:
                    release_year = int(year_str[:4])
                except ValueError:
                    errors.append(f"Row {line_num}: invalid release_year '{year_str}'")

            duration = get_val(row, "duration")
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
                if release_year is not None:
                    existing.release_year = release_year
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
                    release_year=release_year,
                    duration_override_sec=duration_override_sec,
                )
                db.session.add(s)
                created += 1
        except Exception as e:
            skipped += 1
            errors.append(f"Row {line_num}: {e}")
        finally:
            line_num += 1

    db.session.commit()

        # Pretty result page (now rendered inside BASE_HTML so toasts work)
    notes = ""
    if errors:
        notes = "<h3>Notes</h3><ul>" + "".join(f"<li>{e}</li>" for e in errors[:200]) + "</ul>"
        if len(errors) > 200:
            notes += f"<p>…and {len(errors)-200} more.</p>"

    summary = f"Import complete — Created: {created} · Updated: {updated} · Skipped: {skipped}"
    flash(summary)

    inner = f"""
    <h1>Import Complete</h1>
    <div class="section" style="padding-top:0;">
      <div style="max-width: 720px; border: 1px solid #ddd; border-radius: 12px; padding: 1.0rem; box-shadow: 0 2px 10px rgba(0,0,0,0.04);">
        <div class="stats" style="font-size:1.05rem; margin: 0 0 0.75rem;">
          <strong>Created:</strong> {created} &nbsp;|&nbsp;
          <strong>Updated:</strong> {updated} &nbsp;|&nbsp;
          <strong>Skipped:</strong> {skipped}
        </div>
        {notes}
        <div class="btns" style="margin-top:.75rem; display:flex; gap:.5rem; flex-wrap:wrap;">
          <a class="btn" href="/songs">Back to Songs</a>
          <a class="btn" href="/songs/import">Import More</a>
          <a class="btn" href="/songs/template.csv">Download Template CSV</a>
        </div>
      </div>
    </div>
    """

    return render_template_string(BASE_HTML, content=inner)

@app.route("/songs/template.csv")
def songs_template_csv():
    """Downloadable blank template for imports."""
    sample = (
        "title,artist,tempo_bpm,musical_key,genre,tags,release_year,duration\n"
        "Imagine,John Lennon,75,C major,Pop,classic; mellow,1971,03:10\n"
        "Brown Eyed Girl,Van Morrison,148,G major,Classic Rock,upbeat; singalong,1967,03:05\n"
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
    w.writerow(["Title", "Artist", "Tempo (BPM)", "Key", "Genre", "Tags", "Release Year", "Duration (mm:ss)"])
    for s in songs:
        dur_sec = s.duration_override_sec if s.duration_override_sec else estimate_duration_seconds(s.tempo_bpm)
        w.writerow([
            s.title,
            s.artist,
            s.tempo_bpm or "",
            s.musical_key or "",
            s.genre or "",
            s.tags or "",
            s.release_year or "",
            fmt_mmss(dur_sec),
        ])

    resp = Response(out.getvalue(), mimetype="text/csv")
    resp.headers["Content-Disposition"] = 'attachment; filename="songs_export.csv"'
    return resp

@app.post("/songs/autofill_all")
def autofill_all_songs():
    # respect the same search filter as /songs
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

    updated = 0
    for s in songs:
        meta = _fetch_song_metadata(s)
        if _apply_song_metadata(s, meta, force=False):
            updated += 1

    if updated:
        db.session.commit()
        flash(f'Autofilled metadata for {updated} song{"s" if updated != 1 else ""}.')
    else:
        flash("No additional metadata updates found via Spotify/OpenAI.", "info")
    return redirect(url_for("list_songs", q=q))

# --- routes: Setlists ---
@app.post("/setlists/<int:setlist_id>/lock/<int:song_id>", endpoint="setlist_toggle_lock")
def toggle_lock_setlist_song(setlist_id, song_id):
    row = SetlistSong.query.filter_by(setlist_id=setlist_id, song_id=song_id).first_or_404()
    row.locked = not bool(row.locked)
    db.session.commit()
    flash(("Locked" if row.locked else "Unlocked") + f" #{row.position} — {row.song.title}.")
    return redirect(url_for("edit_setlist", setlist_id=setlist_id))

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

    # existing rows (fresh, for duplicate filtering)
    existing_rows = (SetlistSong.query
                     .filter_by(setlist_id=setlist.id)
                     .order_by(SetlistSong.position.asc())
                     .all())

    # maybe clear existing rows first
    if clear_first:
        (SetlistSong.query
        .filter_by(setlist_id=setlist.id)
        .filter(SetlistSong.locked != True)  # keep locked rows
        .delete(synchronize_session=False))
        db.session.commit()
        existing_rows = (SetlistSong.query
                         .filter_by(setlist_id=setlist.id)
                         .order_by(SetlistSong.position.asc())
                         .all())

    # Avoid repeating songs already in this setlist (locked rows remain even when clearing)
    existing_song_ids = {row.song_id for row in existing_rows}
    selected = select_songs_for_target(
        all_songs=all_songs,
        target_minutes=int(target_minutes) if target_minutes else (setlist.target_minutes or 45),
        vibe=vibe,
        required_tags=req_tags,
        required_genres=req_genres,
        avoid_same_artist=avoid_same_artist,
    )
    selected = [s for s in selected if s.id not in existing_song_ids]
    if avoid_same_artist:
        existing_artists = {row.song.artist.lower() for row in existing_rows}
        selected = [s for s in selected if s.artist.lower() not in existing_artists]

    # append to setlist
    current_max = db.session.query(db.func.max(SetlistSong.position)).filter_by(setlist_id=setlist.id).scalar() or 0
    pos = current_max
    for s in selected:
        pos += 1
        db.session.add(SetlistSong(setlist_id=setlist.id, song_id=s.id, position=pos))
    db.session.commit()
    normalize_positions(setlist)

    count = len(selected)
    if count:
        flash(f"Auto-build added {count} song{'s' if count != 1 else ''}.")
    else:
        flash("Auto-build found no additional songs to add.", "info")
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

@app.route("/setlists/<int:setlist_id>/duplicate_newshow", methods=["POST"], endpoint="duplicate_setlist_newshow")
def duplicate_setlist_newshow(setlist_id):
    orig = Setlist.query.get_or_404(setlist_id)
    new = Setlist(
        name=f"{orig.name} (New Show)",
        event_type=orig.event_type,
        venue_type=orig.venue_type,
        target_minutes=orig.target_minutes,
        notes=orig.notes,
        reset_numbering_per_section=orig.reset_numbering_per_section,
    )
    db.session.add(new)
    db.session.flush()
    for ss in orig.songs:
        db.session.add(SetlistSong(
            setlist_id=new.id,
            song_id=ss.song_id,
            position=ss.position,
            notes=None,
            section_name=None,
        ))
    db.session.commit()
    flash(f'Duplicated as new show “{new.name}”.')
    return redirect(url_for("edit_setlist", setlist_id=new.id))

@app.post("/setlists/<int:setlist_id>/update", endpoint="update_setlist")
def update_setlist(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)

    sl.name = (request.form.get("name") or sl.name).strip()
    tm = (request.form.get("target_minutes") or "").strip()
    sl.target_minutes = int(tm) if tm else None
    sl.event_type = (request.form.get("event_type") or "").strip() or None
    sl.venue_type = (request.form.get("venue_type") or "").strip() or None
    sl.notes = (request.form.get("notes") or "").strip() or None

    # checkboxes
    sl.no_repeat_artists = (request.form.get("no_repeat_artists") == "1")
    sl.reset_numbering_per_section = (request.form.get("reset_numbering_per_section") == "1")

    db.session.commit()
    flash('Setlist details saved.')
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.get("/setlists")
def list_setlists():
    setlists = Setlist.query.order_by(Setlist.created_at.desc()).all()
    inner = render_template_string(SETLISTS_LIST_HTML, setlists=setlists)
    return render_template_string(BASE_HTML, content=inner)

@app.get("/s/<token>")
def view_setlist_by_token(token):
    sl = Setlist.query.filter_by(share_token=token).first_or_404()
    songs = (SetlistSong.query
             .filter_by(setlist_id=sl.id)
             .order_by(SetlistSong.position.asc())
             .all())
    inner = render_template_string(
        SETLIST_SHARE_HTML,
        sl=sl,
        songs=songs,
        fmt_mmss=fmt_mmss,
        estimate=estimate_duration_seconds,
    )
    return render_template_string(BASE_HTML, content=inner)


@app.post("/s/<token>/request", endpoint="submit_request_token")
def submit_request_token(token):
    sl = Setlist.query.filter_by(share_token=token).first_or_404()
    song_id_raw = (request.form.get("song_id") or "").strip()
    free_text = (request.form.get("free_text") or "").strip()
    from_name = (request.form.get("from_name") or "").strip() or None
    from_contact = (request.form.get("from_contact") or "").strip() or None

    song_id = None
    if song_id_raw:
        try:
            candidate = int(song_id_raw)
            if SetlistSong.query.filter_by(setlist_id=sl.id, song_id=candidate).first():
                song_id = candidate
            else:
                flash("That song isn't part of this setlist.", "warning")
        except ValueError:
            flash("Invalid song selection.", "warning")

    if not song_id and not free_text:
        flash("Pick a song or tell us what you'd like to hear!", "warning")
        return redirect(url_for("view_setlist_by_token", token=token))

    req = PatronRequest(
        setlist_id=sl.id,
        song_id=song_id,
        free_text_title=free_text or None,
        from_name=from_name,
        from_contact=from_contact,
        status="new",
    )
    db.session.add(req)
    db.session.commit()
    flash("Thanks for your request!", "info")
    return redirect(url_for("view_setlist_by_token", token=token))

@app.get("/setlists/new")
def new_setlist():
    inner = render_template_string(SETLIST_NEW_HTML)
    return render_template_string(BASE_HTML, content=inner)

@app.post("/setlists")
def create_setlist():
    name = request.form.get("name", "").strip()
    if not name:
        return "Name is required.", 400

    # Base setlist fields
    target_minutes_raw = (request.form.get("target_minutes") or "").strip() or None
    event_type = (request.form.get("event_type") or "").strip() or None
    venue_type = (request.form.get("venue_type") or "").strip() or None
    notes = (request.form.get("notes") or "").strip() or None

    sl = Setlist(
        name=name,
        target_minutes=int(target_minutes_raw) if target_minutes_raw else None,
        event_type=event_type,
        venue_type=venue_type,
        notes=notes,
    )
    db.session.add(sl)
    db.session.flush()  # we need sl.id for potential auto-build

    # --- Optional: Auto-build immediately (from New Setlist form) ---
    do_ab = (request.form.get("ab_do") == "1")
    if do_ab:
        # Read options (with sane defaults)
        vibe = (request.form.get("ab_vibe") or "mixed").lower()
        req_tags = [t.strip() for t in (request.form.get("ab_tags") or "").split(",") if t.strip()]
        req_genres = [g.strip() for g in (request.form.get("ab_genres") or "").split(",") if g.strip()]
        avoid_same_artist = (request.form.get("ab_avoid_same_artist") == "1")

        # If this setlist’s toggle is on by default, force no repeats
        if sl.no_repeat_artists:
            avoid_same_artist = True

        # Pool: all songs (latest first for a pleasant default)
        all_songs = Song.query.order_by(Song.created_at.desc()).all()

        # Choose a target (form value beats setlist default; else 45)
        target_minutes = int(target_minutes_raw) if target_minutes_raw else (sl.target_minutes or 45)

        chosen = select_songs_for_target(
            all_songs=all_songs,
            target_minutes=target_minutes,
            vibe=vibe,
            required_tags=req_tags,
            required_genres=req_genres,
            avoid_same_artist=avoid_same_artist,
        )

        # Append in order
        pos = 0
        for s in chosen:
            pos += 1
            db.session.add(SetlistSong(setlist_id=sl.id, song_id=s.id, position=pos))

        # Persist songs + setlist
        db.session.commit()
        normalize_positions(sl)

        # Optional section preset after auto-build
        preset = (request.form.get("ab_preset") or "").strip().lower()
        if preset == "basic":
            apply_section_preset_basic(sl.id)
        elif preset == "three":
            apply_section_preset_three_sets(sl.id)
        elif preset == "chunk":
             apply_section_preset_by_chunk(sl.id)

        flash(f'Created setlist “{sl.name}” and auto-built {len(chosen)} song{"s" if len(chosen)!=1 else ""}.')
        return redirect(url_for("edit_setlist", setlist_id=sl.id))

    # --- No auto-build: just save the setlist normally ---
    db.session.commit()
    flash(f'Created setlist “{sl.name}”.')
    return redirect(url_for("edit_setlist", setlist_id=sl.id))


@app.get("/setlists/<int:setlist_id>/requests")
def setlist_requests(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    token = get_or_create_share_token(sl)
    new_requests = (PatronRequest.query
                    .filter_by(setlist_id=sl.id, status="new")
                    .order_by(PatronRequest.created_at.asc())
                    .all())
    other_requests = (PatronRequest.query
                      .filter(PatronRequest.setlist_id == sl.id, PatronRequest.status != "new")
                      .order_by(PatronRequest.updated_at.desc())
                      .all())
    inner = render_template_string(
        SETLIST_REQUESTS_HTML,
        sl=sl,
        new_requests=new_requests,
        other_requests=other_requests,
        statuses=REQUEST_STATUS_CHOICES,
        qr_url=url_for("setlist_requests_qr", setlist_id=sl.id),
        share_url=url_for("view_setlist_by_token", token=token, _external=True),
    )
    return render_template_string(BASE_HTML, content=inner)

@app.get("/setlists/<int:setlist_id>/requests/live.json")
def live_requests_json(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    requests_query = (PatronRequest.query
                      .filter(PatronRequest.setlist_id == sl.id, PatronRequest.status.in_(("new", "queued")))
                      .order_by(PatronRequest.created_at.asc()))
    requests_list = requests_query.all()
    pending_count = sum(1 for r in requests_list if r.status == "new")
    queued_count = sum(1 for r in requests_list if r.status == "queued")
    return jsonify({
        "setlistId": sl.id,
        "pendingCount": pending_count,
        "queuedCount": queued_count,
        "requests": [serialize_patron_request(r) for r in requests_list],
    })


@app.get("/setlists/<int:setlist_id>/requests/qr.png", endpoint="setlist_requests_qr")
def setlist_requests_qr(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    token = get_or_create_share_token(sl)
    share_url = url_for("view_setlist_by_token", token=token, _external=True)
    img = qrcode.make(share_url)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png", download_name=f"setlist-{setlist_id}-requests.png")


@app.post("/requests/<int:request_id>/status")
def update_request_status(request_id):
    req = PatronRequest.query.get_or_404(request_id)
    payload = request.get_json(silent=True) or {}
    status = (payload.get("status") or request.form.get("status") or "").lower()
    wants_json = wants_json_response()

    if status not in REQUEST_STATUS_CHOICES:
        msg = "Invalid status."
        if wants_json:
            return jsonify({"ok": False, "error": msg}), 400
        flash(msg, "warning")
        return redirect(request.referrer or url_for("setlist_requests", setlist_id=req.setlist_id))

    previous_status = req.status
    req.status = status
    req.updated_at = datetime.utcnow()
    db.session.commit()

    message_map = {
        "queued": "Queued request for review.",
        "done": "Marked request as done.",
        "declined": "Declined request.",
        "new": "Marked request as new.",
    }
    msg = message_map.get(status, f"Marked request as {status}.")

    if wants_json:
        return jsonify({
            "ok": True,
            "request": serialize_patron_request(req),
            "message": msg,
            "previousStatus": previous_status,
        })

    category_map = {
        "queued": "success",
        "done": "success",
        "declined": "info",
    }
    category = category_map.get(status)
    if category:
        flash(msg, category)
    else:
        flash(msg)
    return redirect(request.referrer or url_for("setlist_requests", setlist_id=req.setlist_id))


@app.post("/requests/<int:request_id>/add")
def add_request_to_setlist(request_id):
    req = PatronRequest.query.get_or_404(request_id)
    wants_json = wants_json_response()

    if not req.setlist_id:
        msg = "This request isn't linked to a setlist yet."
        if wants_json:
            return jsonify({"ok": False, "error": msg}), 400
        flash(msg, "warning")
        return redirect(request.referrer or url_for("list_setlists"))
    if not req.song_id:
        msg = "No specific song was selected for this request."
        if wants_json:
            return jsonify({"ok": False, "error": msg}), 400
        flash(msg, "warning")
        return redirect(request.referrer or url_for("setlist_requests", setlist_id=req.setlist_id))

    sl = Setlist.query.get_or_404(req.setlist_id)
    # Avoid duplicates
    existing = SetlistSong.query.filter_by(setlist_id=sl.id, song_id=req.song_id).first()
    added = False
    if existing:
        msg = "Song is already in the setlist."
        category = "info"
    else:
        current_max = db.session.query(db.func.max(SetlistSong.position)).filter_by(setlist_id=sl.id).scalar() or 0
        db.session.add(SetlistSong(setlist_id=sl.id, song_id=req.song_id, position=current_max + 1))
        normalize_positions(sl)
        msg = "Song added to setlist."
        category = "success"
        added = True
    req.status = "queued"
    req.updated_at = datetime.utcnow()
    db.session.commit()
    if wants_json:
        return jsonify({
            "ok": True,
            "request": serialize_patron_request(req),
            "added": added,
            "message": msg,
        })
    if category:
        flash(msg, category)
    else:
        flash(msg)
    return redirect(request.referrer or url_for("setlist_requests", setlist_id=sl.id))

@app.get("/setlists/<int:setlist_id>/edit")
def edit_setlist(setlist_id):
    setlist = Setlist.query.get_or_404(setlist_id)
    pending_requests = PatronRequest.query.filter_by(setlist_id=setlist.id, status="new").count()

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
        # --- Build Section Overview (name, count, total seconds) ---
    sections = []
    current = {"name": None, "sec": 0, "count": 0}

    for ss in setlist.songs:
        # Start a new bucket when a section label appears on this row
        if ss.section_name:
            if current["count"] > 0:
                sections.append(current)
            current = {"name": ss.section_name, "sec": 0, "count": 0}

        # Add this song's duration to the current bucket
        if ss.song.duration_override_sec and ss.song.duration_override_sec > 0:
            sec = ss.song.duration_override_sec
        else:
            sec = estimate_duration_seconds(ss.song.tempo_bpm)
        current["sec"] += sec
        current["count"] += 1

    # Flush last bucket
    if current["count"] > 0:
        sections.append(current)

        # Label unlabeled initial chunk (songs before the first section)
    for s in sections:
        if not s["name"]:
            s["name"] = "(No section)"

    # compute once, outside the loop (but still inside the function)
    existing_artists = {ss.song.artist.lower() for ss in setlist.songs if ss.song.artist}

    # render once, outside the loop
    inner = render_template_string(
        SETLIST_EDIT_HTML,
        setlist=setlist,
        songs=songs,
        q=q,
        estimates=estimates,
        total_str=fmt_mmss(total_sec),
        existing_artists=existing_artists,
        section_overview=sections,   # pass overview to template if you use it
        pending_requests=pending_requests,
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

@app.post("/setlists/<int:setlist_id>/attachment/<int:song_id>")
def update_setlist_song_attachment(setlist_id, song_id):
    return update_setlist_song_file(setlist_id, song_id)


@app.post("/setlists/<int:setlist_id>/file/<int:song_id>")
def update_setlist_song_file(setlist_id, song_id):
    sl = Setlist.query.get_or_404(setlist_id)
    row = SetlistSong.query.filter_by(setlist_id=sl.id, song_id=song_id).first_or_404()

    file_id_raw = (request.form.get("file_id") or request.form.get("attachment_id") or "").strip()
    file_id = int(file_id_raw) if file_id_raw.isdigit() else None

    if file_id:
        sf = SongFile.query.get(file_id)
        if sf and sf.song_id == song_id:
            row.preferred_file_id = file_id
        else:
            flash("File not found or not linked to this song.")
    else:
        row.preferred_file_id = None  # clear selection

    db.session.commit()
    flash("File preference saved.")
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.post("/setlists/<int:setlist_id>/section/<int:song_id>")
def update_setlist_section(setlist_id, song_id):
    setlist = Setlist.query.get_or_404(setlist_id)
    row = SetlistSong.query.filter_by(setlist_id=setlist.id, song_id=song_id).first_or_404()
    section = (request.form.get("section_name") or "").strip()
    row.section_name = section or None
    db.session.commit()
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

@app.post("/setlists/<int:setlist_id>/sections/preset")
def apply_section_preset(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    rows = (SetlistSong.query
            .filter_by(setlist_id=sl.id)
            .order_by(SetlistSong.position.asc())
            .all())
    n = len(rows)
    if n == 0:
        flash("Add songs first, then apply the section preset.")
        return redirect(url_for("edit_setlist", setlist_id=sl.id))

    # Clear any existing section labels
    for r in rows:
        r.section_name = None

    # Always start with Set 1 at the first song
    rows[0].section_name = "Set 1"

    if n == 1:
        db.session.commit()
        flash("Applied: Set 1.")
        return redirect(url_for("edit_setlist", setlist_id=sl.id))

    if n == 2:
        rows[1].section_name = "Encore"
        db.session.commit()
        flash("Applied: Set 1 / Encore.")
        return redirect(url_for("edit_setlist", setlist_id=sl.id))

    if n == 3:
        rows[1].section_name = "Break"
        rows[2].section_name = "Encore"
        db.session.commit()
        flash("Applied: Set 1 / Break / Encore.")
        return redirect(url_for("edit_setlist", setlist_id=sl.id))

    # n >= 4 → Set 1 (top), Break (middle), Set 2 (after Break), Encore (last)
    # choose indices that keep labels distinct
    if n % 2 == 0:
        break_idx = max(1, n // 2 - 1)   # even counts: bias Break slightly earlier
    else:
        break_idx = max(1, n // 2)       # odd counts: middle

    set2_idx = min(break_idx + 1, n - 2)  # ensure Set 2 isn't the last song
    encore_idx = n - 1

    # keep indices distinct
    if set2_idx == break_idx:
        set2_idx = min(break_idx + 1, n - 2)
    if set2_idx == encore_idx:
        set2_idx = max(1, encore_idx - 1)
        if set2_idx == break_idx:
            break_idx = max(1, set2_idx - 1)

    rows[break_idx].section_name = "Break"
    rows[set2_idx].section_name = "Set 2"
    rows[encore_idx].section_name = "Encore"

    db.session.commit()
    flash("Applied: Set 1 / Break / Set 2 / Encore.")
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.post("/setlists/<int:setlist_id>/sections/clear_all", endpoint="clear_all_sections_v2")
def clear_all_sections_v2(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    rows = (SetlistSong.query
            .filter_by(setlist_id=sl.id)
            .order_by(SetlistSong.position.asc())
            .all())
    cleared = 0
    for r in rows:
        if r.section_name:
            r.section_name = None
            cleared += 1
    db.session.commit()
    flash(f"Cleared {cleared} section label(s).")
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.post("/setlists/<int:setlist_id>/sections/move")
def move_section_block(setlist_id):
    """
    Move an entire labeled section (from its header until the next header) up or down
    among other labeled sections. Songs before the first header and after the last
    header remain in place.
    """
    sl = Setlist.query.get_or_404(setlist_id)
    direction = (request.form.get("dir") or "").lower()  # "up" or "down"
    try:
        idx = int(request.form.get("idx", "-1"))  # 0-based index among labeled sections
    except ValueError:
        idx = -1

    # Get full ordered rows
    rows = (SetlistSong.query
            .filter_by(setlist_id=sl.id)
            .order_by(SetlistSong.position.asc())
            .all())
    n = len(rows)
    if n == 0 or idx < 0:
        flash("Nothing to move.")
        return redirect(url_for("edit_setlist", setlist_id=sl.id))

    # Find starts of each labeled section
    header_starts = [i for i, r in enumerate(rows) if (r.section_name or "").strip()]
    if not header_starts or idx >= len(header_starts):
        flash("Invalid section index.")
        return redirect(url_for("edit_setlist", setlist_id=sl.id))

    # Build blocks: each labeled section goes from header_start to just before next header
    blocks = []
    for j, start in enumerate(header_starts):
        end = (header_starts[j + 1] - 1) if j + 1 < len(header_starts) else (n - 1)
        blocks.append((start, end))

    # Keep prefix (before the very first header) and suffix (after the last header)
    prefix_end = header_starts[0] if header_starts else 0
    suffix_start = (blocks[-1][1] + 1) if blocks else n

    prefix = rows[:prefix_end]
    suffix = rows[suffix_start:]

    # Slice each labeled block into a list
    block_slices = [rows[s:e + 1] for (s, e) in blocks]

    # Figure out the swap target
    target = None
    if direction == "up" and idx > 0:
        target = idx - 1
    elif direction == "down" and idx < (len(block_slices) - 1):
        target = idx + 1

    if target is None:
        flash("Section already at edge; no move.")
        return redirect(url_for("edit_setlist", setlist_id=sl.id))

    # Snapshot for undo and perform swap
    stash_order(sl.id)
    block_slices[idx], block_slices[target] = block_slices[target], block_slices[idx]

    # Rebuild final order
    new_rows = prefix + [r for block in block_slices for r in block] + suffix

    # Reassign positions 1..n
    for i, r in enumerate(new_rows, start=1):
        r.position = i
    db.session.commit()

    flash("Section moved " + direction + ".")
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.post("/setlists/<int:setlist_id>/move/<int:song_id>/up")
def move_song_up(setlist_id, song_id):
    setlist = Setlist.query.get_or_404(setlist_id) 
    stash_order(setlist.id)
    row = SetlistSong.query.filter_by(setlist_id=setlist.id, song_id=song_id).first_or_404()
        
    if getattr(row, "locked", False):
        flash("That song is locked and can’t be moved.")
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))

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
    if getattr(row, "locked", False):
        flash("That song is locked and can’t be moved.")
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))

    if below:
        row.position, below.position = below.position, row.position
        db.session.commit()
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

@app.post("/setlists/<int:setlist_id>/delete")
def delete_setlist(setlist_id):
    setlist = Setlist.query.get_or_404(setlist_id)
    db.session.delete(setlist)
    db.session.commit()
    flash('Setlist deleted.')
    return redirect(url_for("list_setlists"))

@app.get("/setlists/<int:setlist_id>/print")
def print_setlist(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    hide_notes = request.args.get("hide_notes") == "1"

    # Build rows with optional section headers + per-section numbering
    rows = []        # sequence of ("section", name) or ("song", dict)
    total_sec = 0
    song_count = 0

    per_section = sl.reset_numbering_per_section is True
    current_num = 0  # display number (resets at sections when per_section=True)

    for ss in sl.songs:
        # If a section label starts here, push a section row and (optionally) reset numbering
        if ss.section_name:
            rows.append(("section", ss.section_name))
            if per_section:
                current_num = 0  # reset at each section header

        # compute duration + display number
        dur_sec = ss.song.duration_override_sec or estimate_duration_seconds(ss.song.tempo_bpm)
        total_sec += dur_sec
        song_count += 1
        current_num = current_num + 1 if per_section else ss.position

        rows.append(("song", {
            "num": current_num,                 # what we show in the first column
            "pos": ss.position,                 # absolute position (kept for reference if needed)
            "title": ss.song.title,
            "artist": ss.song.artist,
            "key": ss.song.musical_key or "",
            "bpm": ss.song.tempo_bpm or "",
            "year": ss.song.release_year,
            "dur": fmt_mmss(dur_sec),
            "notes": getattr(ss, "notes", "") or "",
        }))

        # --- Section Overview chips (same logic as Step 82) ---
    sec_overview = []
    bucket = {"name": None, "sec": 0, "count": 0}
    for ss in sl.songs:
        if ss.section_name:
            if bucket["count"] > 0:
                sec_overview.append(bucket)
            bucket = {"name": ss.section_name, "sec": 0, "count": 0}
        dsec = ss.song.duration_override_sec or estimate_duration_seconds(ss.song.tempo_bpm)
        bucket["sec"] += dsec
        bucket["count"] += 1
    if bucket["count"] > 0:
        sec_overview.append(bucket)
    for b in sec_overview:
        if not b["name"]:
            b["name"] = "(No section)"

    chips = ""
    if sec_overview:
        parts = []
        for i, b in enumerate(sec_overview):
            label = f'{b["name"]} — {b["count"]} song{"s" if b["count"] != 1 else ""} · {fmt_mmss(b["sec"])}'
            parts.append(f'<a class="chip" href="#sec-{i}">{label}</a>')
        chips = '<div class="chips">' + " ".join(parts) + "</div>"

    #    # Build tbody HTML pieces safely
    body_parts = []
    sec_idx = -1  # track which section we're in (aligned with sec_overview order)
    for kind, val in rows:
        if kind == "section":
            sec_idx += 1
            info = sec_overview[sec_idx] if 0 <= sec_idx < len(sec_overview) else {"name": val, "sec": 0, "count": 0}
            label_right = f'<span class="secmeta">{info["count"]} · {fmt_mmss(info["sec"])}</span>'
            section_name = info.get("name", val)
            body_parts.append(
                f'<tr id="sec-{sec_idx}" class="section"><td colspan="7">— {section_name} — {label_right}</td></tr>'
            )
        else:
            r = val
            notes_html = "" if hide_notes else (f'<div class="notes"><em>Notes:</em> {r["notes"]}</div>' if r["notes"] else "")
            body_parts.append(
                "<tr>"
                f"<td>{r['num']}</td>"
                f"<td>{r['title']}{notes_html}</td>"
                f"<td>{r['artist']}</td>"
                f"<td>{r['key']}</td>"
                f"<td>{r['bpm']}</td>"
                f"<td>{r['year'] or ''}</td>"
                f"<td>{r['dur']}</td>"
                "</tr>"
            )

    tbody_html = "".join(body_parts) or "<tr><td colspan=\"7\">No songs yet.</td></tr>"

    # Page HTML
    toggle_label = "Show notes" if hide_notes else "Hide notes"
    toggle_url = url_for("print_setlist", setlist_id=sl.id, hide_notes=(0 if hide_notes else 1))
    html = f"""
    <!doctype html>
    <html>
    <head>
          <meta name="color-scheme" content="light" />
      <meta charset="utf-8">
      <title>{sl.name} — Print View</title>
            <style>
        /* Force a light theme in print view (even when embedded in dark Live Mode) */
        :root {{ color-scheme: light; }}
        html, body {{ background:#ffffff !important; color:#0b0b0b !important; }}

        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 24px; }}
        h1 {{ margin: 0 0 6px; color:#0b0b0b; }}
        a {{ color:#0a58ca; }}

        .meta {{ color:#333; margin-bottom:8px; }}

        .chips {{ margin: 6px 0 12px; display:flex; gap:8px; flex-wrap:wrap; }}
        .chip {{ display:inline-block; padding:4px 8px; border:1px solid #ddd; border-radius:8px; color:#222; background:#fafafa; text-decoration:none; }}
        .chip:focus, .chip:hover {{ background:#f0f0f0; }}

        table {{ border-collapse: collapse; width: 100%; color:#0b0b0b; }}
        th, td {{ border-bottom: 1px solid #e8e8e8; padding: 8px 6px; text-align:left; vertical-align:top; }}
        th {{ background:#f7f7f7; color:#111; border-bottom-color:#dcdcdc; }}
        .secmeta {{ font-weight:400; color:#555; font-size: 0.9em; margin-left: 6px; }}
        tr.section td {{ border-top: 2px solid #bdbdbd; background:#f7f7f7; font-weight:700; color:#111; }}
        .notes {{ color:#444; font-style: italic; margin-top: 4px; }}

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
        &nbsp;·&nbsp;
        <a href="{toggle_url}">{toggle_label}</a>
    </div>
      <h1>{sl.name}</h1>
      <div class="meta">
        {" · ".join([p for p in [
          f"Event: {sl.event_type}" if sl.event_type else "",
          f"Venue: {sl.venue_type}" if sl.venue_type else "",
          f"Target: {sl.target_minutes} min" if sl.target_minutes else "",
          ("Numbering resets per section" if per_section else "Continuous numbering"),
          f"Songs: {song_count}",
          f"Approx Total: {fmt_mmss(total_sec)}"
        ] if p])}
      </div>

      {chips}

      <table>
        <thead>
          <tr><th>#</th><th>Song</th><th>Artist</th><th>Key</th><th>BPM</th><th>Year</th><th>Dur</th></tr>
        </thead>
        <tbody>
          {tbody_html}
        </tbody>
      </table>
    </body>
    </html>
    """
    return html

@app.get("/setlists/<int:setlist_id>")
def view_setlist(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)

    # --- Build ordered section buckets (for counts + duration) ---
    section_infos = []  # list of {"name": str|None, "sec": int, "count": int}
    current = {"name": None, "sec": 0, "count": 0}

    for ss in sl.songs:
        if ss.section_name:
            # flush prior bucket
            if current["count"] > 0:
                section_infos.append(current)
            current = {"name": ss.section_name, "sec": 0, "count": 0}
        dsec = ss.song.duration_override_sec or estimate_duration_seconds(ss.song.tempo_bpm)
        current["sec"] += dsec
        current["count"] += 1
    if current["count"] > 0:
        section_infos.append(current)

    # name the opening unlabeled bucket (if any)
    for b in section_infos:
        if not b["name"]:
            b["name"] = "(No section)"

    # totals
    total_sec = sum(b["sec"] for b in section_infos)
    song_total = sum(b["count"] for b in section_infos)

    # --- Chips line (summary) ---
    chips = ""
    if section_infos:
        parts = []
        for i, b in enumerate(section_infos):
            label = f'{b["name"]} — {b["count"]} song{"s" if b["count"] != 1 else ""} · {fmt_mmss(b["sec"])}'
            parts.append(f'<a class="chip" href="#sec-{i}">{label}</a>')
        chips = '<div class="chips">' + " ".join(parts) + "</div>"

    # --- Build blocks (per-section header includes duration) ---
    blocks = []
    sec_idx = -1  # which section_infos entry we are in
    for ss in sl.songs:
        if ss.section_name:
            # advance section index
            sec_idx += 1
            info = section_infos[sec_idx] if 0 <= sec_idx < len(section_infos) else {"name": ss.section_name, "sec": 0, "count": 0}
            header = f'— {info["name"]} — <span class="secmeta">{info["count"]} · {fmt_mmss(info["sec"])}</span>'
            blocks.append(("section_html", {"html": header, "idx": sec_idx}))

        dur_sec = ss.song.duration_override_sec or estimate_duration_seconds(ss.song.tempo_bpm)

        # first attachment id if present
        att_id = None
        try:
            if getattr(ss, "preferred_file_id", None):
                att_id = ss.preferred_file_id
            elif getattr(ss.song, "default_file_id", None):
                att_id = ss.song.default_file_id
            elif ss.song.files and len(ss.song.files) > 0:
                att_id = ss.song.files[0].id
        except Exception:
            att_id = None

        blocks.append(("song", {
    "song_id": ss.song_id,
    "pos": ss.position,
    "title": ss.song.title,
    "artist": ss.song.artist,
    "key": ss.song.musical_key or "",
    "bpm": ss.song.tempo_bpm or "",
    "year": ss.song.release_year,
    "dur": fmt_mmss(dur_sec),
    "notes": getattr(ss, "notes", "") or "",
    "att_id": att_id,
    "locked": bool(getattr(ss, "locked", False)),
}))

    # Render body
    parts = []
    for kind, data in blocks:
        if kind == "section_html":
            parts.append(f"""
            <div id="sec-{data['idx']}" class="row" style="border-top:2px solid #ddd; background:#fafafa;">
              <div style="font-weight:700; display:flex; gap:8px; align-items:center;">
                <span>{data['html']}</span>
              </div>
            </div>
            """)
        else:
            r = data
            # Build optional attachment link safely in Python to avoid f-string nesting issues
            att_html = ""
            if r.get("att_id"):
                att_url = url_for("view_attachment", att_id=r["att_id"])
                att_html = f' · <a class="btn" href="{att_url}" target="_blank" title="Open attachment">📄 Open</a>'

            info_bits = []
            if r['key']:
                info_bits.append(f"Key: {r['key']}")
            if r['bpm']:
                info_bits.append(f"BPM: {r['bpm']}")
            if r.get('year'):
                info_bits.append(f"Year: {r['year']}")
            if r.get('locked'):
                info_bits.append("🔒 locked")
            info_bits.append(f"Dur: {r['dur']}")
            info_line = " · ".join(info_bits) + (att_html or "")

            toggle_url = url_for("setlist_toggle_lock", setlist_id=sl.id, song_id=r["song_id"])
            toggle_label = "Unlock" if r.get("locked") else "Lock"
            toggle_form = (
                f'<form method="post" action="{toggle_url}" '
                'style="display:inline;margin-left:6px;">'
                f'<button class="btn" type="submit" title="Toggle lock">{toggle_label}</button>'
                "</form>"
            )
            notes_html = f"<div class=\"muted\"><em>Notes:</em> {r['notes']}</div>" if r['notes'] else ""

            parts.append(f"""
            <div class="row">
              <div>
                <strong>#{r['pos']} — {r['title']}</strong> — {r['artist']}
                <div class="muted">
                  {info_line}
                  {toggle_form}
                </div>
                {notes_html}
              </div>
            </div>
            """)

    # meta line
    meta_bits = [p for p in [
        f"Event: {sl.event_type}" if sl.event_type else "",
        f"Venue: {sl.venue_type}" if sl.venue_type else "",
        f"Target: {sl.target_minutes} min" if sl.target_minutes else "",
        f"Songs: {song_total}",
        f"Approx Total: {fmt_mmss(total_sec)}"
    ] if p]

    # page styles for chips + small section meta (safe: plain string, not an f-string)
    extra_css = """
    <style>
      .chips { margin: 6px 0 12px; display:flex; gap:8px; flex-wrap:wrap; }
      .chip { display:inline-block; padding:4px 8px; border:1px solid #ddd; border-radius:8px; color:#444; background:#fafafa; text-decoration:none; }
      .chip:focus, .chip:hover { background:#f0f0f0; }
      .secmeta { font-weight:400; color:#666; font-size: 0.9em; }
    </style>
    """

    inner = f"""
    {extra_css}
    <h2>{sl.name}</h2>
    <p class="muted">{' · '.join(meta_bits)}</p>

    {chips}

    <div class="section" style="padding-top:0;">
      {''.join(parts) or "<p class='muted'>This setlist is empty.</p>"}
    </div>

    <p>
      <a class="btn" href="{url_for('print_setlist', setlist_id=sl.id)}" target="_blank">🖨️ Print View</a>
      <a class="btn" href="{url_for('export_setlist_csv', setlist_id=sl.id)}">⬇️ Download CSV</a>
      <a class="btn" href="{url_for('export_setlist_pdf', setlist_id=sl.id)}">⬇️ Download PDF</a>
    </p>
    """

    return render_template_string(BASE_HTML, content=inner)
def _choose_file_for_row(ss) -> "SongFile | None":
    """
    Order of preference:
      1) SetlistSong.preferred_file_id (if valid PDF for this song)
      2) Song.default_file_id (if valid PDF for this song)
      3) First PDF file on the song
    """
    # 1) row-level preferred
    try:
        if getattr(ss, "preferred_file_id", None):
            sf = SongFile.query.get(ss.preferred_file_id)
            if sf and sf.song_id == ss.song_id and sf.is_pdf:
                return sf
    except Exception:
        pass

    # 2) song-level default
    try:
        if getattr(ss.song, "default_file_id", None):
            sf = SongFile.query.get(ss.song.default_file_id)
            if sf and sf.song_id == ss.song_id and sf.is_pdf:
                return sf
    except Exception:
        pass

    # 3) first PDF on the song
    try:
        for sf in (ss.song.files or []):
            if sf.is_pdf:
                return sf
    except Exception:
        pass

    return None
    
@app.get("/setlists/<int:setlist_id>/live")
def live_mode(setlist_id):
    from flask import render_template_string
    import json

    sl = Setlist.query.get_or_404(setlist_id)
    pending_requests = PatronRequest.query.filter_by(setlist_id=sl.id, status="new").count()

    try:
        i = int(request.args.get("i", "0"))
    except ValueError:
        i = 0
    skip_no_pdf = request.args.get("skip_no_pdf") == "1"

    rows = (SetlistSong.query
            .filter_by(setlist_id=sl.id)
            .order_by(SetlistSong.position.asc())
            .all())

    pages = []
    for ss in rows:
        sf = _choose_file_for_row(ss)
        title_text = f"#{ss.position} — {ss.song.title} — {ss.song.artist}"
        chart_text = ss.song.chord_chart or ""
        has_chart = bool(chart_text.strip())
        base = {
            "title": title_text,
            "pos": ss.position,
            "song_id": ss.song_id,
            "song_title": ss.song.title,
            "song_artist": ss.song.artist,
            "song_key": ss.song.musical_key,
            "chart": chart_text,
            "has_chart": has_chart,
        }
        if sf:
            pages.append({
                **base,
                "url": url_for("view_attachment", att_id=sf.id),
                "has_pdf": True,
            })
        else:
            if skip_no_pdf:
                continue
            pages.append({
                **base,
                "title": title_text + " (no PDF)",
                "url": url_for("print_setlist", setlist_id=sl.id),
                "has_pdf": False,
            })

    # If filtering removed everything, fall back to full list (disable skip)
    if not pages:
        skip_no_pdf = False
        for ss in rows:
            sf = _choose_file_for_row(ss)
            title_text = f"#{ss.position} — {ss.song.title} — {ss.song.artist}"
            chart_text = ss.song.chord_chart or ""
            has_chart = bool(chart_text.strip())
            base = {
                "title": title_text,
                "pos": ss.position,
                "song_id": ss.song_id,
                "song_title": ss.song.title,
                "song_artist": ss.song.artist,
                "song_key": ss.song.musical_key,
                "chart": chart_text,
                "has_chart": has_chart,
            }
            if sf:
                pages.append({
                    **base,
                    "url": url_for("view_attachment", att_id=sf.id),
                    "has_pdf": True,
                })
            else:
                pages.append({
                    **base,
                    "title": title_text + " (no PDF)",
                    "url": url_for("print_setlist", setlist_id=sl.id),
                    "has_pdf": False,
                })

    items_json = json.dumps(pages)
    toggle_url = url_for("live_mode", setlist_id=sl.id, i=i, skip_no_pdf=(0 if skip_no_pdf else 1))
    toggle_label = "Show all songs" if skip_no_pdf else "Skip songs without PDFs"

    inner = render_template_string(
    LIVE_HTML,
    sl=sl,
    items_json=items_json,
    toggle_url=toggle_url,
    toggle_label=toggle_label,
    pending_requests=pending_requests,
    requests_url=url_for("setlist_requests", setlist_id=sl.id),
    requests_json_url=url_for("live_requests_json", setlist_id=sl.id),
)
    return inner

@app.route("/setlists/<int:setlist_id>/duplicate", methods=["POST"], endpoint="duplicate_setlist")
def duplicate_setlist(setlist_id):
    orig = Setlist.query.get_or_404(setlist_id)
    new = Setlist(
        name=f"{orig.name} (Copy)",
        event_type=orig.event_type,
        venue_type=orig.venue_type,
        target_minutes=orig.target_minutes,
        notes=orig.notes,
        reset_numbering_per_section=orig.reset_numbering_per_section,
    )
    db.session.add(new)
    db.session.flush()
    for ss in orig.songs:
        db.session.add(SetlistSong(setlist_id=new.id, song_id=ss.song_id, position=ss.position))
    db.session.commit()
    flash(f'Duplicated to “{new.name}”.')
    return redirect(url_for("edit_setlist", setlist_id=new.id))

    # --- Clear entire setlist (remove all songs) ---
@app.post("/setlists/<int:setlist_id>/clear", endpoint="clear_setlist")
def clear_setlist(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    removed = len(sl.songs)
    SetlistSong.query.filter_by(setlist_id=sl.id).delete()
    db.session.commit()
    flash(f'Cleared setlist ({removed} song{"s" if removed != 1 else ""} removed).')
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.post("/setlists/<int:setlist_id>/sections/normalize")
def normalize_sections(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    last = None
    cleaned = 0

    # Walk in display order; remove repeated consecutive headers with the same name
    for ss in sl.songs:
        name = (ss.section_name or "").strip() or None
        if name and last == name:
            ss.section_name = None
            cleaned += 1
        elif name and last != name:
            last = name
        # if name is None, keep last as-is

    db.session.commit()
    flash(f"Cleaned {cleaned} duplicate section header(s).")
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.post("/setlists/<int:setlist_id>/sections/preset_basic", endpoint="preset_sections_basic")
def preset_sections_basic(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    rows = (SetlistSong.query
            .filter_by(setlist_id=sl.id)
            .order_by(SetlistSong.position.asc())
            .all())
    n = len(rows)
    if n == 0:
        flash("No songs to label.")
        return redirect(url_for("edit_setlist", setlist_id=sl.id))

    # Clear all existing section labels first
    for r in rows:
        r.section_name = None

    # Always label first and last if possible
    def put(idx, name):
        if 0 <= idx < n:
            rows[idx].section_name = name

    if n == 1:
        put(0, "Set 1")
    elif n == 2:
        put(0, "Set 1")
        put(1, "Encore")
    else:
        # 3+ songs: Set 1 at top, Encore at last, Break near middle, Set 2 after Break
        first = 0
        last = n - 1
        mid = max(1, min(n - 2, n // 2))              # somewhere in the middle, not first/last
        set2 = max(mid + 1, min(n - 2, (3 * n) // 4)) # safely after Break, before last

        put(first, "Set 1")
        put(mid, "Break")
        put(set2, "Set 2")
        put(last, "Encore")

    db.session.commit()
    flash("Section preset applied.")
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.post("/setlists/<int:setlist_id>/sections/preset_three_sets", endpoint="preset_sections_three_sets")
def preset_sections_three_sets(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    rows = (SetlistSong.query
            .filter_by(setlist_id=sl.id)
            .order_by(SetlistSong.position.asc())
            .all())
    n = len(rows)
    if n == 0:
        flash("No songs to label.")
        return redirect(url_for("edit_setlist", setlist_id=sl.id))

    # Clear existing section labels
    for r in rows:
        r.section_name = None

    def put(idx, name):
        if 0 <= idx < n:
            rows[idx].section_name = name

    if n == 1:
        put(0, "Set 1")
    elif n == 2:
        put(0, "Set 1")
        put(1, "Encore")
    elif n == 3:
        put(0, "Set 1"); put(2, "Encore")
    elif n == 4:
        put(0, "Set 1"); put(2, "Set 2"); put(3, "Encore")
    else:
        # General case: 5+ songs → Set1 / Break / Set2 / Break / Set3 / Encore
        first = 0
        last  = n - 1

        # Breaks near 1/3 and 2/3, clamped away from edges
        b1 = max(1, min(n - 2, n // 3))
        b2 = max(b1 + 1, min(n - 2, (2 * n) // 3))

        # Set starts immediately after breaks, but not at last
        s2 = min(n - 2, b1 + 1)
        s3 = min(n - 2, b2 + 1)

        # Ensure strict ordering and non-overlap
        if s2 <= first:    s2 = min(n - 2, first + 1)
        if b2 <= s2:       b2 = min(n - 2, s2 + 1)
        if s3 <= b2:       s3 = min(n - 2, b2 + 1)

        put(first, "Set 1")
        put(b1,   "Break")
        put(s2,   "Set 2")
        put(b2,   "Break")
        put(s3,   "Set 3")
        put(last, "Encore")

    db.session.commit()
    flash("Three-set section preset applied.")
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.post("/setlists/<int:setlist_id>/sections/preset_by_chunk", endpoint="preset_sections_by_chunk")
def preset_sections_by_chunk(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    rows = (SetlistSong.query
            .filter_by(setlist_id=sl.id)
            .order_by(SetlistSong.position.asc())
            .all())
    n = len(rows)
    if n == 0:
        flash("No songs to label.")
        return redirect(url_for("edit_setlist", setlist_id=sl.id))

    # read chunk size
    try:
        chunk = int(request.form.get("chunk", "8"))
    except ValueError:
        chunk = 8
    chunk = max(1, min(50, chunk))  # sanity clamp

    # clear existing labels
    for r in rows:
        r.section_name = None

    def put(idx, name):
        if 0 <= idx < n:
            rows[idx].section_name = name

    # always label first as Set 1
    set_no = 1
    put(0, f"Set {set_no}")

    # walk through in chunks; drop Break at chunk boundary, start next set after it
    i = chunk
    while i < n - 1:  # leave room for Encore at last
        put(i, "Break")
        if i + 1 < n - 1:
            set_no += 1
            put(i + 1, f"Set {set_no}")
            i += chunk  # next boundary relative to this new set start
        else:
            break

    # Encore at the very last song
    put(n - 1, "Encore")

    db.session.commit()
    flash(f"Chunk preset applied ({chunk} per set).")
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.post("/setlists/<int:setlist_id>/sections/fix_encore")
def fix_encore(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    rows = (SetlistSong.query
            .filter_by(setlist_id=sl.id)
            .order_by(SetlistSong.position.asc())
            .all())
    n = len(rows)
    if n == 0:
        flash("No songs in this setlist.")
        return redirect(url_for("edit_setlist", setlist_id=sl.id))

    # Clear all existing "Encore" labels (case-insensitive)
    cleared = 0
    for r in rows:
        if (r.section_name or "").strip().lower() == "encore":
            r.section_name = None
            cleared += 1

    # Set Encore on the last song
    rows[-1].section_name = "Encore"
    db.session.commit()

    msg = f"Encore fixed: removed {cleared} and set Encore on the last song."
    flash(msg)
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.post("/setlists/<int:setlist_id>/sections/renumber_sets")
def renumber_sets(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    rows = (SetlistSong.query
            .filter_by(setlist_id=sl.id)
            .order_by(SetlistSong.position.asc())
            .all())
    if not rows:
        flash("No songs in this setlist.")
        return redirect(url_for("edit_setlist", setlist_id=sl.id))

    import re
    is_set = re.compile(r"^\s*set\b", re.IGNORECASE)

    # Walk rows; when you hit a section header that looks like "Set ...",
    # rename it to Set {counter} and increment.
    counter = 1
    changed = 0
    for r in rows:
        name = (r.section_name or "").strip()
        if not name:
            continue
        if name.lower() in ("break", "encore"):
            continue
        if is_set.match(name):
            new_name = f"Set {counter}"
            if name != new_name:
                r.section_name = new_name
                changed += 1
            counter += 1

    db.session.commit()
    flash(f"Renumbered sets ({changed} change{'s' if changed != 1 else ''}).")
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.post("/setlists/<int:setlist_id>/sections/clear_all")
def clear_all_sections(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    rows = (SetlistSong.query
            .filter_by(setlist_id=sl.id)
            .order_by(SetlistSong.position.asc())
            .all())
    cleared = 0
    for r in rows:
        if r.section_name:
            r.section_name = None
            cleared += 1
    db.session.commit()
    flash(f"Cleared {cleared} section label(s).")
    return redirect(url_for("edit_setlist", setlist_id=sl.id))

@app.post("/setlists/<int:setlist_id>/move/<int:song_id>/top")
def move_song_top(setlist_id, song_id):
    setlist = Setlist.query.get_or_404(setlist_id)
    stash_order(setlist.id)

    row = SetlistSong.query.filter_by(setlist_id=setlist.id, song_id=song_id).first_or_404()
    if getattr(row, "locked", False):
        flash("That song is locked and can’t be moved.")
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))

    row.position = 0  # temporarily smallest
    db.session.commit()
    normalize_positions(setlist)
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

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
    if getattr(row, "locked", False):
        flash("That song is locked and can’t be moved.")
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))

    row.position = maxpos + 1
    db.session.commit()
    normalize_positions(setlist)
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

@app.route("/setlists/<int:setlist_id>/reorder", methods=["POST", "GET"])
def reorder_setlist(setlist_id):
    setlist = Setlist.query.get_or_404(setlist_id)
    order_str = (request.form.get("order") or request.args.get("order") or "").strip()
    if not order_str:
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))

    try:
        new_ids = [int(x) for x in order_str.split(",") if x.strip()]
    except ValueError:
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))

    rows = (SetlistSong.query
            .filter_by(setlist_id=setlist.id)
            .order_by(SetlistSong.position.asc())
            .all())
    by_song_id = {r.song_id: r for r in rows}

    # Don’t move locked rows
    locked_ids = {r.song_id for r in rows if getattr(r, "locked", False)}
    new_ids = [sid for sid in new_ids if sid in by_song_id and sid not in locked_ids]

    if not new_ids:
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))

    # Snapshot for Undo
    stash_order(setlist.id)

    # Apply new positions in given order
    pos = 1
    for sid in new_ids:
        by_song_id[sid].position = pos
        pos += 1

    # Append leftovers preserving their previous relative order
    for r in rows:
        if r.song_id not in new_ids:
            r.position = pos
            pos += 1

    db.session.commit()
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

    # Build a map of current rows in this setlist
    rows = (SetlistSong.query
            .filter_by(setlist_id=setlist.id)
            .order_by(SetlistSong.position.asc())
            .all())
    by_song_id = {r.song_id: r for r in rows}

    # Do not reorder locked rows; keep their relative positions fixed
    locked_ids = {r.song_id for r in rows if r.locked}
    # Filter new_ids to NOT move locked ones
    new_ids = [sid for sid in new_ids if sid not in locked_ids]

    # Filter to valid song_ids
    new_ids = [sid for sid in new_ids if sid in by_song_id]

    if not new_ids:
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))
    
        # --- KEEP LOCKED SONGS IN PLACE ---
    locked_ids = {r.song_id for r in rows if getattr(r, "locked", False)}
    new_ids = [sid for sid in new_ids if sid not in locked_ids]

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

@app.get("/setlists/<int:setlist_id>/share/qr")
def share_qr(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    # ensure we have a token
    token = get_or_create_share_token(sl)
    url = url_for("view_setlist_by_token", token=token, _external=True)

    # generate QR as PNG in-memory
    import qrcode
    from io import BytesIO
    buf = BytesIO()
    img = qrcode.make(url)
    img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=False,
        download_name=f"{sl.name.replace(' ', '_')}_qr.png",
        max_age=0,
    )

# --- export CSV ---
@app.get("/setlists/<int:setlist_id>/export.csv")
def export_setlist_csv(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    output = StringIO()
    writer = csv.writer(output)

    # New: include Section column
    writer.writerow(["Position", "Section", "Title", "Artist", "Tempo (BPM)", "Key", "Genre", "Tags", "Release Year", "Duration (mm:ss)", "Notes"])

    for ss in sl.songs:
        dur = ss.song.duration_override_sec if ss.song.duration_override_sec else estimate_duration_seconds(ss.song.tempo_bpm)
        writer.writerow([
            ss.position,
            ss.section_name or "",           # ← section label (if any)
            ss.song.title,
            ss.song.artist,
            ss.song.tempo_bpm or "",
            ss.song.musical_key or "",
            ss.song.genre or "",
            ss.song.tags or "",
            ss.song.release_year or "",
            fmt_mmss(dur),
            ss.notes or "",
        ])

    csv_data = output.getvalue()
    output.close()
    filename = f'{sl.name.replace(" ", "_")}.csv'
    resp = Response(csv_data, mimetype="text/csv")
    resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp

@app.get("/setlists/<int:setlist_id>/export.chopro")
def export_setlist_chopro(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    songs = (SetlistSong.query
             .filter_by(setlist_id=sl.id)
             .order_by(SetlistSong.position.asc())
             .all())
    lines = _build_chopro_lines(sl, songs)
    content = "\n".join(lines).encode("utf-8")
    filename = f"{_slugify(sl.name, 'setlist')}.chopro.txt"
    return Response(
        content,
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.get("/setlists/<int:setlist_id>/export.chopro.zip")
def export_setlist_chopro_zip(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    songs = (SetlistSong.query
             .filter_by(setlist_id=sl.id)
             .order_by(SetlistSong.position.asc())
             .all())

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{_slugify(sl.name, 'setlist')}.chopro.txt", "\n".join(_build_chopro_lines(sl, songs)))
        for ss in songs:
            block = _build_chopro_song_block(sl, ss)
            song_slug = _slugify(f"{ss.position:02d}-{ss.song.title}", "song")
            zf.writestr(f"{song_slug}.cho", block)

    buf.seek(0)
    filename = f"{_slugify(sl.name, 'setlist')}_chordpro.zip"
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=filename)


def _build_chopro_lines(sl: Setlist, songs: list[SetlistSong]) -> list[str]:
    lines = [f"# Setlist: {sl.name}"]
    meta = []
    if sl.event_type:
        meta.append(f"Event: {sl.event_type}")
    if sl.venue_type:
        meta.append(f"Venue: {sl.venue_type}")
    if sl.target_minutes:
        meta.append(f"Target: {sl.target_minutes} min")
    if meta:
        lines.append("# " + " · ".join(meta))
    lines.append("")

    current_section = None
    for ss in songs:
        if ss.section_name and current_section != ss.section_name:
            current_section = ss.section_name
            lines.append(f"## {current_section}")

        key = ss.song.musical_key or ""
        bpm = ss.song.tempo_bpm or ""
        dur_sec = ss.song.duration_override_sec or estimate_duration_seconds(ss.song.tempo_bpm)
        dur = fmt_mmss(dur_sec)
        year = ss.song.release_year or ""

        lines.append(f"# {ss.position}. {ss.song.title} — {ss.song.artist}")
        meta_bits = [
            f"Key: {key}" if key else "",
            f"BPM: {bpm}" if bpm else "",
            f"Dur: {dur}",
        ]
        if year:
            meta_bits.append(f"Year: {year}")
        lines.append("# " + " · ".join([b for b in meta_bits if b]))
        if ss.notes:
            lines.append("# Notes: " + ss.notes)
        lines.append("")

    return lines


def _build_chopro_song_block(sl: Setlist, ss: SetlistSong) -> str:
    dur_sec = ss.song.duration_override_sec or estimate_duration_seconds(ss.song.tempo_bpm)
    lines = [
        f"# {ss.position}. {ss.song.title}",
        f"# Artist: {ss.song.artist}",
    ]
    meta_bits = []
    if ss.song.musical_key:
        meta_bits.append(f"Key: {ss.song.musical_key}")
    if ss.song.tempo_bpm:
        meta_bits.append(f"BPM: {ss.song.tempo_bpm}")
    meta_bits.append(f"Dur: {fmt_mmss(dur_sec)}")
    if ss.song.release_year:
        meta_bits.append(f"Year: {ss.song.release_year}")
    lines.append("# " + " · ".join(meta_bits))
    if ss.section_name:
        lines.append(f"# Section: {ss.section_name}")
    if ss.notes:
        lines.append("# Notes: " + ss.notes)
    lines.extend([
        "",
        "# Add chords/lyrics below",
        "# Example: [G]Amazing [D]grace",
        "",
    ])
    return "\n".join(lines)

@app.get("/healthz")
def healthz():
    # quick DB ping; never crash health
    try:
        db.session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return (f"ok | db={ 'up' if db_ok else 'down' }", 200)

# --- Helper: ReportLab canvas that writes "Page N of M" footers ---
from reportlab.pdfgen import canvas as _rl_canvas

class NumberedCanvas(_rl_canvas.Canvas):
    # Canvas that draws a footer with 'Printed' on the left and 'Page N of M' on the right of every page.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []
        self._footer_left = ""     # e.g., "Printed Oct 11, 2025"
        self._margin = 0.75 * inch
        self._page_width = letter[0]

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(total)
            super().showPage()
        super().save()

    def _draw_footer(self, total_pages: int):
        self.setFont("Helvetica", 9)
        y = self._margin - 0.45 * inch
        if y < 0.3 * inch:
            y = 0.3 * inch
        if self._footer_left:
            self.drawString(self._margin, y, self._footer_left)
        self.drawRightString(self._page_width - self._margin, y, f"Page {self.getPageNumber()} of {total_pages}")

from PyPDF2 import PdfReader, PdfWriter
@app.get("/setlists/<int:setlist_id>/export.pdf")
def export_setlist_pdf(setlist_id):
    sl = Setlist.query.get_or_404(setlist_id)
    hide_notes = request.args.get("hide_notes") == "1"
    include_charts_param = request.args.get("include_charts")
    include_charts = True if include_charts_param is None else (include_charts_param == "1")
    include_requests = request.args.get("include_requests") == "1"

    requests_list = []
    if include_requests:
        requests_list = (PatronRequest.query
                         .filter_by(setlist_id=sl.id)
                         .order_by(PatronRequest.created_at.asc())
                         .all())

    # Build rows with optional section headers + (optional) per-section numbering
    rows = []        # sequence of ("section", name) or ("song", dict)
    total_sec = 0
    song_count = 0

    per_section = sl.reset_numbering_per_section is True
    current_num = 0  # display number (resets at sections when per_section=True)

    for ss in sl.songs:
        if ss.section_name:
            rows.append(("section", ss.section_name))
            if per_section:
                current_num = 0

        dur_sec = ss.song.duration_override_sec or estimate_duration_seconds(ss.song.tempo_bpm)
        total_sec += dur_sec
        song_count += 1
        current_num = current_num + 1 if per_section else ss.position

        rows.append(("song", {
            "num": current_num,                 # display number
            "pos": ss.position,                 # absolute position (not shown)
            "title": ss.song.title,
            "artist": ss.song.artist,
            "key": ss.song.musical_key or "",
            "bpm": str(ss.song.tempo_bpm) if ss.song.tempo_bpm else "",
            "dur": fmt_mmss(dur_sec),
            "notes": getattr(ss, "notes", "") or "",
        }))

    # Section overview (for header + per-section labels)
    sec_overview = []
    bucket = {"name": None, "sec": 0, "count": 0}
    for ss in sl.songs:
        if ss.section_name:
            if bucket["count"] > 0:
                sec_overview.append(bucket)
            bucket = {"name": ss.section_name, "sec": 0, "count": 0}
        dsec = ss.song.duration_override_sec or estimate_duration_seconds(ss.song.tempo_bpm)
        bucket["sec"] += dsec
        bucket["count"] += 1
    if bucket["count"] > 0:
        sec_overview.append(bucket)
    for b in sec_overview:
        if not b["name"]:
            b["name"] = "(No section)"

    # Render overview as one wrapped line
    sec_text = ""
    if sec_overview:
        parts = []
        for b in sec_overview:
            parts.append(f'{b["name"]} — {b["count"]} song{"s" if b["count"] != 1 else ""} · {fmt_mmss(b["sec"])}')
        sec_text = "   |   ".join(parts)

    # PDF setup
    buf = BytesIO()
    c = NumberedCanvas(buf, pagesize=letter)
    c._footer_left = "Printed " + datetime.now().strftime("%b %d, %Y")

    width, height = letter
    margin = 0.75 * inch

    if include_requests and requests_list:
        draw_requests_summary()
        c.showPage()

    # columns
    col_dur_w = 0.8 * inch
    col_bpm_w = 0.7 * inch
    col_key_w = 1.0 * inch
    x_dur = width - margin - col_dur_w
    x_bpm = x_dur - col_bpm_w
    x_key = x_bpm - col_key_w
    x_title = margin
    title_w = x_key - margin - 6

    # metrics
    title_line_h = 14  # Helvetica 11
    notes_line_h = 12  # Helvetica-Oblique 9
    row_gap = 2
    section_h = 18

    def header_height():
        # title (18) + meta (14) + count line (18) + sec_text wrapped + gap lines
        from reportlab.lib.utils import simpleSplit as _split
        wrap_lines = 0
        if sec_text:
            wrap_lines = len(_split(sec_text, "Helvetica", 9, width - 2 * margin))
        return 18 + 14 + 18 + (12 * max(1, wrap_lines) if sec_text else 0) + 4 + 10 + 1 + 8

    def draw_requests_summary():
        y = height - margin
        c.setFont("Helvetica-Bold", 18)
        c.drawString(margin, y, f"Requests — {sl.name}")
        y -= 22

        c.setFont("Helvetica", 11)
        if not requests_list:
            c.drawString(margin, y, "No audience requests yet.")
            return

        # Group by status
        grouped = {status: [] for status in REQUEST_STATUS_CHOICES}
        for req in requests_list:
            grouped.setdefault(req.status, []).append(req)

        for status in REQUEST_STATUS_CHOICES:
            items = grouped.get(status) or []
            if not items:
                continue
            c.setFont("Helvetica-Bold", 13)
            c.drawString(margin, y, status.title())
            y -= 16
            c.setFont("Helvetica", 10)
            for req in items:
                line = req.label()
                if req.from_name:
                    line += f"  ·  {req.from_name}"
                if req.from_contact:
                    line += f" ({req.from_contact})"
                c.drawString(margin + 8, y, line)
                y -= 14
                if req.free_text_title and not req.song:
                    c.setFont("Helvetica-Oblique", 9)
                    c.drawString(margin + 14, y, f"Requested: {req.free_text_title}")
                    c.setFont("Helvetica", 10)
                    y -= 12
                if y < margin + 40:
                    c.showPage()
                    y = height - margin
                    c.setFont("Helvetica", 10)
            y -= 8

    def draw_page_header(title_suffix=""):
        from reportlab.lib.utils import simpleSplit as _split
        y = height - margin
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y, f"Setlist: {sl.name}{title_suffix}")
        y -= 18

        c.setFont("Helvetica", 10)
        meta_bits = []
        if sl.event_type: meta_bits.append(f"Event: {sl.event_type}")
        if sl.venue_type: meta_bits.append(f"Venue: {sl.venue_type}")
        if sl.target_minutes: meta_bits.append(f"Target: {sl.target_minutes} min")
        mode = "Numbering resets per section" if per_section else "Continuous numbering"
        if hide_notes:
            mode += " · Notes hidden"
        meta_line = "  ·  ".join([*meta_bits, mode]) if meta_bits else mode
        c.drawString(margin, y, meta_line)
        y -= 14
        c.drawString(margin, y, f"Songs: {song_count}    Approx Total: {fmt_mmss(total_sec)}")
        y -= 18

        if sec_text:
            c.setFont("Helvetica", 9)
            for line in _split(sec_text, "Helvetica", 9, width - 2 * margin):
                c.drawString(margin, y, line)
                y -= 12
            y -= 4

        # Column headers
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

    def row_needed_height(kind, val):
        from reportlab.lib.utils import simpleSplit as _split
        if kind == "section":
            return section_h
        title_text = f"#{val['num']} — {val['title']} — {val['artist']}"
        title_wrapped = _split(title_text, "Helvetica", 11, title_w)
        if hide_notes or not val["notes"]:
            notes_h = 0
        else:
            notes_wrapped = _split(f"Notes: {val['notes']}", "Helvetica-Oblique", 9, title_w)
            notes_h = len(notes_wrapped) * notes_line_h
        return len(title_wrapped) * title_line_h + notes_h + row_gap

    # PASS 1: figure out total pages
    usable_min_y = margin + 20
    y = height - margin - header_height()
    total_pages = 1
    for kind, val in rows:
        needed = row_needed_height(kind, val)
        if y - needed < usable_min_y:
            total_pages += 1
            y = height - margin - header_height()
        y -= needed

    # PASS 2: draw pages
    page_suffix = ""
    y = draw_page_header(page_suffix)
    sec_idx = -1
    for kind, val in rows:
        needed = row_needed_height(kind, val)
        if y - needed < usable_min_y:
            c.showPage()
            page_suffix = " (cont.)"
            y = draw_page_header(page_suffix)

        if kind == "section":
            sec_idx += 1
            info = sec_overview[sec_idx] if 0 <= sec_idx < len(sec_overview) else {"name": val, "sec": 0, "count": 0}
            right_meta = f'{info["count"]} · {fmt_mmss(info["sec"])}'

            c.setLineWidth(1.0)
            c.setStrokeColorRGB(0.75, 0.75, 0.75)
            c.line(margin, y + 12, width - margin, y + 12)
            c.setStrokeColorRGB(0, 0, 0)

            c.setFont("Helvetica-Bold", 11)
            c.drawString(x_title, y, f"— {info.get('name', val)} —")
            c.setFont("Helvetica", 10)
            c.drawRightString(width - margin, y, right_meta)

            y -= section_h
            continue

        # song row
        r = val
        from reportlab.lib.utils import simpleSplit as _split
        c.setFont("Helvetica", 11)
        title_text = f"#{r['num']} — {r['title']} — {r['artist']}"
        title_wrapped = _split(title_text, "Helvetica", 11, title_w)
        for i, line in enumerate(title_wrapped):
            c.drawString(x_title, y, line)
            if i == 0:
                c.drawRightString(x_key + col_key_w - 2, y, r["key"])
                c.drawRightString(x_bpm + col_bpm_w - 2, y, r["bpm"])
                c.drawRightString(x_dur + col_dur_w - 2, y, r["dur"])
            y -= title_line_h

        if (not hide_notes) and r["notes"]:
            c.setFont("Helvetica-Oblique", 9)
            for line in _split(f"Notes: {r['notes']}", "Helvetica-Oblique", 9, title_w):
                c.drawString(x_title, y, line)
                y -= notes_line_h

        y -= row_gap

        # Finalize the last page so NumberedCanvas.save() can render footers
    c.showPage()
    c.save()
    setlist_pdf_bytes = buf.getvalue()
    buf.close()

        # --- NEW: append attached PDFs in setlist order (robust PdfWriter approach) ---
    writer = PdfWriter()

    # 1) Add the generated setlist pages first
    try:
        main_reader = PdfReader(BytesIO(setlist_pdf_bytes), strict=False)
        if not getattr(main_reader, "pages", None) or len(main_reader.pages) == 0:
            # If somehow empty, just return the setlist-only PDF bytes
            resp = Response(setlist_pdf_bytes, mimetype="application/pdf")
            filename = f'{sl.name.replace(" ", "_")}.pdf'
            resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
            return resp
        for page in main_reader.pages:
            writer.add_page(page)
    except Exception:
        # Fallback: at least return the setlist-only PDF
        resp = Response(setlist_pdf_bytes, mimetype="application/pdf")
        filename = f'{sl.name.replace(" ", "_")}.pdf'
        resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return resp

    # 2) Append each song's preferred chart (if enabled)
    if include_charts:
        song_rows = (SetlistSong.query
                     .filter_by(setlist_id=sl.id)
                     .order_by(SetlistSong.position.asc())
                     .all())

        for ss in song_rows:
            try:
                sf = _choose_file_for_row(ss)
                if not sf or not getattr(sf, "is_pdf", False):
                    continue
                abs_path = os.path.abspath(sf.path)
                if os.path.exists(abs_path):
                    try:
                        reader = PdfReader(abs_path, strict=False)
                        for page in reader.pages:
                            writer.add_page(page)
                    except Exception:
                        continue
            except Exception:
                continue

    # 3) Write out the combined document
    out_buf = BytesIO()
    writer.write(out_buf)
    final_bytes = out_buf.getvalue()
    out_buf.close()

    filename = f'{sl.name.replace(" ", "_")}.pdf'
    resp = Response(final_bytes, mimetype="application/pdf")
    resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp

def ensure_schema():
    """Add any missing columns/indexes used by this app, idempotently."""
    with app.app_context():
        db.create_all()
        _ensure_schema_columns()
        print("✅ ensure_schema completed successfully.")

# single init call
def _init_schema_on_import():
    try:
        ensure_schema()
    except Exception as e:
        print("ensure_schema on import warning:", e)

_init_schema_on_import()

if __name__ == "__main__":
    app.run(debug=True, port=5055)
