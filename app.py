import json
import mimetypes
import os
import re
import secrets
import unicodedata
import zipfile
from collections import deque
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Callable
import csv
import hashlib
import io
import uuid
import math

def parse_duration(duration_str):
    if not duration_str:
        return None
    parts = duration_str.split(':')
    if len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    elif len(parts) == 1:
        return int(parts[0])
    else:
        raise ValueError("Invalid duration format")

try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    PdfReader = None
    PdfWriter = None
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
    jsonify,
)
# Removed duplicate from flask import render_template
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from markupsafe import Markup
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas
from sqlalchemy import text, or_, inspect, func
from werkzeug.utils import secure_filename
from werkzeug.routing import BuildError
from werkzeug.security import generate_password_hash, check_password_hash

try:
    import requests
except ImportError:
    requests = None

load_dotenv()

# Move HTML templates to separate files for better organization
# Create templates directory and move HTML strings there

# Removed duplicate HTML template strings (e.g., BASE_HTML, REGISTER_FORM_HTML, etc.) as they are moved to separate template files

from config import *

app = Flask(__name__)

app.config["SECRET_KEY"] = FLASK_SECRET_KEY

UPLOAD_DIR = Path(app.instance_path) / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_MIME = {"application/pdf"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Share UPLOAD_DIR with models before importing ORM symbols
import models
models.UPLOAD_DIR = UPLOAD_DIR

from models import db, User, Song, SongFile, Setlist, SetlistSong, PatronRequest

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "info"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

from routes.auth import auth_bp
app.register_blueprint(auth_bp)

# --- Helpers for applying section presets programmatically ---
def apply_section_preset_basic(setlist_id: int) -> None:
    _run_section_preset(setlist_id, preset_sections_basic)


def apply_section_preset_three_sets(setlist_id: int) -> None:
    _run_section_preset(setlist_id, preset_sections_three_sets)


def apply_section_preset_by_chunk(setlist_id: int) -> None:
    _run_section_preset(setlist_id, preset_sections_by_chunk)

# --- CSV helpers ---
def parse_mmss_to_seconds(text):
    """Return total seconds from `mm:ss` strings or plain second values."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        if ":" in s:
            mm, ss = s.split(":", 1)
            return int(mm) * 60 + int(ss)
        # allow plain seconds like "185" or "185.0"
        return int(float(s))
    except Exception:
        return None

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
app.url_map.strict_slashes = False

def _assign_section_label(rows, idx, name):
    """Set a section label on rows[idx] when the index is valid."""
    if 0 <= idx < len(rows):
        rows[idx].section_name = name

def _run_section_preset(setlist_id: int, preset_func: Callable[[int], object]) -> None:
    """Invoke preset routines safely outside a real request."""
    with app.app_context():
        if not Setlist.query.get(setlist_id):
            return
        with app.test_request_context():
            preset_func(setlist_id)

# --- Database config (uses DATABASE_URL if set; else SQLite) ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_SQLITE = "sqlite:///" + os.path.join(BASE_DIR, "musician.db")
db_url = os.getenv("DATABASE_URL", DEFAULT_SQLITE)

# Render/Railway sometimes prefix with postgres:// – SQLAlchemy accepts postgresql://
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)
elif db_url.startswith("postgresql://") and "+psycopg" not in db_url:
    db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

# --- File uploads (PDF attachments) utilities ---

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


def _current_user_id() -> int | None:
    if not current_user.is_authenticated:
        return None
    try:
        return int(current_user.get_id())
    except (TypeError, ValueError):
        return None


def _is_admin() -> bool:
    return current_user.is_authenticated and bool(getattr(current_user, "is_admin", False))


def _song_editable(song: "Song") -> bool:
    if not current_user.is_authenticated:
        return False
    if song.user_id is None:
        return _is_admin()
    return song.user_id == _current_user_id() or _is_admin()


def _setlist_editable(sl: "Setlist") -> bool:
    if not current_user.is_authenticated:
        return False
    if sl.user_id is None:
        return _is_admin()
    return sl.user_id == _current_user_id() or _is_admin()


def _require_song_owner(song: "Song") -> None:
    if not _song_editable(song):
        abort(403)


def _require_setlist_owner(sl: "Setlist") -> None:
    if not _setlist_editable(sl):
        abort(403)


def _song_or_404(song_id: int, *, include_deleted: bool = False) -> "Song":
    song = Song.query.get_or_404(song_id)
    if song.deleted_at and not include_deleted:
        abort(404)
    return song

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
    pass
    # setlist table retrofits
    table_names = [row[0] for row in db.session.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()]
    if "setlist" in table_names:
        info3 = db.session.execute(text("PRAGMA table_info(setlist);")).fetchall()
        cols3 = [row[1] for row in info3]
        if "no_repeat_artists" not in cols3:
            db.session.execute(text("ALTER TABLE setlist ADD COLUMN no_repeat_artists BOOLEAN DEFAULT 0"))
            db.session.commit()
        if "share_token" not in cols3:
            db.session.execute(text("ALTER TABLE setlist ADD COLUMN share_token VARCHAR(64)"))
            db.session.commit()
        if "reset_numbering_per_section" not in cols3:
            db.session.execute(text("ALTER TABLE setlist ADD COLUMN reset_numbering_per_section BOOLEAN DEFAULT 0"))
            db.session.commit()
        if "user_id" not in cols3:
            db.session.execute(text("ALTER TABLE setlist ADD COLUMN user_id INTEGER"))
            db.session.commit()
    # Ensure columns exist in setlist table before altering
    info3 = db.session.execute(text("PRAGMA table_info(setlist);")).fetchall()
    cols3 = [row[1] for row in info3]
    if "no_repeat_artists" not in cols3:
        db.session.execute(text("ALTER TABLE setlist ADD COLUMN no_repeat_artists BOOLEAN DEFAULT 0"))
        db.session.commit()
    if "share_token" not in cols3:
        db.session.execute(text("ALTER TABLE setlist ADD COLUMN share_token VARCHAR(64)"))
        db.session.commit()
    if "reset_numbering_per_section" not in cols3:
        db.session.execute(text("ALTER TABLE setlist ADD COLUMN reset_numbering_per_section BOOLEAN DEFAULT 0"))
        db.session.commit()
    if "user_id" not in cols3:
        db.session.execute(text("ALTER TABLE setlist ADD COLUMN user_id INTEGER"))
        db.session.commit()
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
        if "song" in table_names:
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
            if "user_id" not in song_cols:
                db.session.execute(text("ALTER TABLE song ADD COLUMN user_id INTEGER"))
                db.session.commit()
        if "is_public" not in song_cols:
            db.session.execute(text("ALTER TABLE song ADD COLUMN is_public BOOLEAN DEFAULT 1"))
            db.session.commit()
            db.session.execute(text("UPDATE song SET is_public = 1 WHERE is_public IS NULL"))
            db.session.commit()
        if "deleted_at" not in song_cols:
            db.session.execute(text("ALTER TABLE song ADD COLUMN deleted_at DATETIME"))
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
            db.session.execute(text("ALTER TABLE setlist ADD COLUMN no_repeat_artists BOOLEAN DEFAULT 0"))
            db.session.commit()
        if "share_token" not in cols3:
            db.session.execute(text("ALTER TABLE setlist ADD COLUMN share_token VARCHAR(64)"))
            db.session.commit()
        if "reset_numbering_per_section" not in cols3:
            db.session.execute(text("ALTER TABLE setlist ADD COLUMN reset_numbering_per_section BOOLEAN DEFAULT 0"))
            db.session.commit()
        if "user_id" not in cols3:
            db.session.execute(text("ALTER TABLE setlist ADD COLUMN user_id INTEGER"))
            db.session.commit()


def _ensure_schema_columns_postgres() -> None:
    with app.app_context():
        bind = db.session.get_bind()
        if not bind or not bind.dialect.name.startswith("postgres"):
            return

        def exec_sql(sql: str) -> None:
            db.session.execute(text(sql))
            db.session.commit()

        exec_sql("ALTER TABLE song ADD COLUMN IF NOT EXISTS duration_override_sec INTEGER")
        exec_sql("ALTER TABLE song ADD COLUMN IF NOT EXISTS default_attachment_id INTEGER")
        exec_sql("ALTER TABLE song ADD COLUMN IF NOT EXISTS release_year INTEGER")
        exec_sql("ALTER TABLE song ADD COLUMN IF NOT EXISTS chord_chart TEXT")
        exec_sql("ALTER TABLE song ADD COLUMN IF NOT EXISTS user_id INTEGER")
        exec_sql("ALTER TABLE song ADD COLUMN IF NOT EXISTS is_public BOOLEAN DEFAULT TRUE")
        exec_sql("UPDATE song SET is_public = TRUE WHERE is_public IS NULL")
        exec_sql("ALTER TABLE song ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP")

        exec_sql("ALTER TABLE setlist_song ADD COLUMN IF NOT EXISTS notes VARCHAR(500)")
        exec_sql("ALTER TABLE setlist_song ADD COLUMN IF NOT EXISTS section_name VARCHAR(100)")
        exec_sql("ALTER TABLE setlist_song ADD COLUMN IF NOT EXISTS preferred_attachment_id INTEGER")
        exec_sql("CREATE INDEX IF NOT EXISTS ix_setlist_song_preferred_attachment_id ON setlist_song(preferred_attachment_id)")
        exec_sql("ALTER TABLE setlist_song ADD COLUMN IF NOT EXISTS locked BOOLEAN DEFAULT FALSE")

        exec_sql("ALTER TABLE setlist ADD COLUMN IF NOT EXISTS no_repeat_artists BOOLEAN DEFAULT FALSE")
        exec_sql("ALTER TABLE setlist ADD COLUMN IF NOT EXISTS share_token VARCHAR(64)")
        exec_sql("ALTER TABLE setlist ADD COLUMN IF NOT EXISTS reset_numbering_per_section BOOLEAN DEFAULT FALSE")
        exec_sql("ALTER TABLE setlist ADD COLUMN IF NOT EXISTS user_id INTEGER")

        exec_sql("ALTER TABLE attachment ADD COLUMN IF NOT EXISTS original_name VARCHAR(255)")
        exec_sql("ALTER TABLE attachment ADD COLUMN IF NOT EXISTS stored_name VARCHAR(255)")
        exec_sql("ALTER TABLE attachment ADD COLUMN IF NOT EXISTS mimetype VARCHAR(120)")
        exec_sql("ALTER TABLE attachment ADD COLUMN IF NOT EXISTS size_bytes INTEGER")
        exec_sql("ALTER TABLE attachment ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        exec_sql("ALTER TABLE attachment ADD COLUMN IF NOT EXISTS kind VARCHAR(20) DEFAULT 'pdf'")
        exec_sql("ALTER TABLE attachment ADD COLUMN IF NOT EXISTS pages INTEGER")

        exec_sql("ALTER TABLE patron_request ADD COLUMN IF NOT EXISTS setlist_id INTEGER")
        exec_sql("ALTER TABLE patron_request ADD COLUMN IF NOT EXISTS song_id INTEGER")
        exec_sql("ALTER TABLE patron_request ADD COLUMN IF NOT EXISTS free_text_title VARCHAR(255)")
        exec_sql("ALTER TABLE patron_request ADD COLUMN IF NOT EXISTS from_name VARCHAR(120)")
        exec_sql("ALTER TABLE patron_request ADD COLUMN IF NOT EXISTS from_contact VARCHAR(120)")
        exec_sql("ALTER TABLE patron_request ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'new'")
        exec_sql("ALTER TABLE patron_request ADD COLUMN IF NOT EXISTS created_at TIMESTAMP")
        exec_sql("ALTER TABLE patron_request ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP")

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

@app.context_processor
def inject_utils():
    return dict(
        fmt_mmss=fmt_mmss,
        estimate=estimate_duration_seconds,
        song_has_pdf=song_has_pdf,
    )

# Register fmt_mmss as a Jinja2 filter
app.jinja_env.filters["fmt_mmss"] = fmt_mmss

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


def _normalize_tags(text: str | None) -> set[str]:
    if not text:
        return set()
    return {t.strip().lower() for t in text.split(",") if t.strip()}


def _ai_expected_song_count(target_minutes: int | float | str | None, preset: str | None) -> int:
    try:
        minutes_float = float(target_minutes)
    except (TypeError, ValueError):
        minutes_float = 45.0
    minutes_float = max(minutes_float, 15.0)
    base = max(8, math.ceil(minutes_float / 3.5))
    preset_key = (preset or "").strip().lower()
    if preset_key == "three":
        base = max(base, 18)
    elif preset_key == "chunk":
        base = max(base, 16)
    elif preset_key == "basic":
        base = max(base, 12)
    return int(base)


def _derive_ai_context_preferences(context: dict | None,
                                   required_tags: list[str] | None,
                                   required_genres: list[str] | None) -> dict:
    ctx = context or {}
    pref_tags = {t.strip().lower() for t in (required_tags or []) if t.strip()}
    pref_genres = {g.strip().lower() for g in (required_genres or []) if g.strip()}
    hard_tags: set[str] = set()
    hard_genres: set[str] = set()
    vibe = (ctx.get("vibe") or "").strip().lower()
    event_type = (ctx.get("event_type") or "").strip().lower()
    venue_type = (ctx.get("venue_type") or "").strip().lower()
    notes = (ctx.get("notes") or "").strip().lower()
    preset = (ctx.get("preset") or "").strip().lower()

    avoid_terms: set[str] = set()
    tempo_focus: tuple[int, int] | None = None

    def requires_only(keyword: str) -> bool:
        if not notes:
            return False
        kw = re.escape(keyword)
        patterns = [
            rf"\\bonly\\s+{kw}\\b",
            rf"\\bonly\\s+{kw}\\s+songs?\\b",
            rf"\\bonly\\s+{kw}\\s+music\\b",
            rf"\\bstrictly\\s+{kw}\\b",
            rf"\\bnothing\\s+but\\s+{kw}\\b",
            rf"\\bmust\\s+be\\s+{kw}\\b",
            rf"\\b{kw}\\s+only\\b",
            rf"\\b{kw}\\s+songs?\\s+only\\b",
            rf"\\b{kw}\\s+music\\s+only\\b",
        ]
        return any(re.search(pattern, notes) for pattern in patterns)

    def add_tags(*tags: str):
        for tag in tags:
            if tag:
                pref_tags.add(tag)

    def add_genres(*genres: str):
        for genre in genres:
            if genre:
                pref_genres.add(genre.lower())

    if "wedding" in event_type:
        add_tags("love", "dance", "general")
        add_genres("pop", "r&b", "soul")
        tempo_focus = (80, 120)
    if "ceremony" in event_type or "church" in event_type:
        add_tags("love", "general")
        tempo_focus = tempo_focus or (65, 100)
    if any(word in event_type for word in ("reception", "banquet", "corporate", "gala")):
        add_tags("dance", "general")
        add_genres("pop", "funk")
        tempo_focus = tempo_focus or (95, 130)
    if any(word in event_type for word in ("festival", "party", "celebration")):
        add_tags("dance")
        tempo_focus = tempo_focus or (110, 140)
    if "cocktail" in event_type or "acoustic" in event_type:
        add_tags("acoustic")
        tempo_focus = tempo_focus or (68, 105)

    if "outdoor" in venue_type or "stage" in venue_type or "festival" in venue_type:
        add_tags("dance")
        tempo_focus = tempo_focus or (105, 140)
    if any(word in venue_type for word in ("cafe", "coffee", "lounge")):
        add_tags("acoustic")
        tempo_focus = tempo_focus or (68, 105)
    if any(word in venue_type for word in ("bar", "club")):
        add_tags("dance")
        tempo_focus = tempo_focus or (110, 140)

    if notes:
        for match in re.findall(r"\bno\s+([a-z0-9/&\-\']+)", notes):
            avoid_terms.add(match.strip())
        keyword_map = {
            "acoustic": ("acoustic", None, (65, 105)),
            "stripped": ("acoustic", None, (65, 105)),
            "ballad": ("love", None, (60, 95)),
            "slow": ("love", None, (60, 90)),
            "downtempo": ("acoustic", None, (60, 95)),
            "upbeat": ("dance", None, (105, 140)),
            "energetic": ("dance", None, (110, 145)),
            "party": ("dance", None, (110, 140)),
            "singalong": ("general", None, None),
            "motown": ("love", "soul", (90, 125)),
            "funk": ("dance", "funk", (105, 135)),
            "country": ("general", "country", None),
            "jazz": ("general", "jazz", None),
            "latin": ("dance", "latin", (100, 140)),
            "reggae": ("general", "reggae", (80, 110)),
            "holiday": ("love", "holiday", (70, 110)),
            "christmas": ("love", "holiday", (70, 110)),
        }
        for key, (tag, genre, tempo_tuple) in keyword_map.items():
            if key in notes:
                if tag:
                    add_tags(tag)
                if genre:
                    add_genres(genre)
                if tempo_tuple and tempo_focus is None:
                    tempo_focus = tempo_tuple
                if genre and requires_only(key):
                    hard_genres.add(genre.lower())
                elif tag and requires_only(key):
                    hard_tags.add(tag.lower())
        for decade in ("60s", "70s", "80s", "90s", "2000s", "2010s", "2020s"):
            if decade in notes:
                add_tags(decade)
                if requires_only(decade):
                    hard_tags.add(decade)

    for decade in ("60s", "70s", "80s", "90s", "2000s", "2010s", "2020s"):
        if decade in event_type or decade in venue_type:
            add_tags(decade)

    if tempo_focus is None:
        if vibe == "chill":
            tempo_focus = (70, 105)
        elif vibe == "energetic":
            tempo_focus = (110, 140)
        else:
            tempo_focus = (90, 130)

    expected_count = ctx.get("expected_song_count")
    if expected_count is None:
        expected_count = _ai_expected_song_count(ctx.get("target_minutes"), preset)

    return {
        "preferred_tags": pref_tags,
        "preferred_genres": pref_genres,
        "tempo_focus": tempo_focus,
        "avoid_terms": avoid_terms,
        "expected_song_count": int(expected_count or 0),
        "required_tags": hard_tags,
        "required_genres": hard_genres,
    }


def _metadata_contains_terms(meta: dict, terms: set[str]) -> bool:
    if not terms:
        return False
    tags = _normalize_tags(meta.get("tags"))
    genre = (meta.get("genre") or "").lower()
    blob = " ".join(tags) + " " + genre
    for term in terms:
        if term and term in blob:
            return True
    return False


def _score_ai_candidate(meta: dict, preferences: dict) -> float:
    tags = _normalize_tags(meta.get("tags"))
    tempo = meta.get("tempo_bpm") or 0
    genre = (meta.get("genre") or "").lower()

    score = 0.0
    pref_tags = preferences.get("preferred_tags", set())
    pref_genres = preferences.get("preferred_genres", set())
    tempo_focus = preferences.get("tempo_focus")
    avoid_terms = preferences.get("avoid_terms", set())

    if pref_tags:
        score += 2.5 * len(tags & pref_tags)
    if pref_genres and genre:
        for g in pref_genres:
            if g and g in genre:
                score += 2.0
                break
    if tempo_focus:
        low, high = tempo_focus
        if tempo and low <= tempo <= high:
            score += 1.8
        elif tempo:
            distance = min(abs(tempo - low), abs(tempo - high))
            score += max(0.2, 1.8 - distance / 35.0)
    for decade in ("60s", "70s", "80s", "90s", "2000s", "2010s", "2020s"):
        if decade in tags:
            score += 1.2 if decade in pref_tags else 0.3
    if avoid_terms:
        blob = " ".join(tags) + " " + genre
        for term in avoid_terms:
            if term and term in blob:
                score -= 3.5
    return score

def select_songs_for_target(all_songs,
                            target_minutes: int | None,
                            vibe: str = "mixed",
                            required_tags: list[str] | None = None,
                            required_genres: list[str] | None = None,
                            avoid_same_artist: bool = True,
                            context: dict | None = None):
    """Greedy fill to (about) target minutes with simple vibe/filters."""
    required_tags = [t.strip().lower() for t in (required_tags or []) if t.strip()]
    required_genres = [g.strip().lower() for g in (required_genres or []) if g.strip()]

    def tags_set(txt: str | None) -> set[str]:
        return _normalize_tags(txt)

    preferences = None
    if context:
        preferences = context.get("_derived") if isinstance(context, dict) else None
        if not preferences:
            preferences = _derive_ai_context_preferences(context, required_tags, required_genres)
            if isinstance(context, dict):
                context["_derived"] = preferences
    avoid_terms = preferences.get("avoid_terms") if preferences else set()
    tempo_focus = preferences.get("tempo_focus") if preferences else None
    if preferences:
        extra_tags = preferences.get("required_tags") or set()
        extra_genres = preferences.get("required_genres") or set()
        if extra_tags:
            required_tags = list(dict.fromkeys(required_tags + [t for t in extra_tags if t]))
        if extra_genres:
            required_genres = list(dict.fromkeys(required_genres + [g for g in extra_genres if g]))

    def base_passes_filters(s: "Song") -> bool:
        if required_tags and not (tags_set(s.tags) & set(required_tags)):
            return False
        if required_genres:
            sg = (s.genre or "").lower()
            if not any(g in sg for g in required_genres):
                return False
        return True

    raw_candidates = [s for s in all_songs if base_passes_filters(s)]

    if preferences and avoid_terms:
        preferred, fallback = [], []
        for s in raw_candidates:
            meta = {"tags": s.tags, "genre": s.genre}
            if _metadata_contains_terms(meta, avoid_terms):
                fallback.append(s)
            else:
                preferred.append(s)
        candidates = preferred + fallback if preferred else raw_candidates
    else:
        candidates = raw_candidates

    def tempo_of(s): return s.tempo_bpm or 120
    if vibe == "chill":
        candidates.sort(key=tempo_of)                      # low → high
    elif vibe == "energetic":
        candidates.sort(key=tempo_of, reverse=True)        # high → low
    else:  # mixed: bias toward preferred tempo band, default mid-tempo
        anchor = 120
        if tempo_focus:
            anchor = sum(tempo_focus) / 2
        candidates.sort(key=lambda s: abs(tempo_of(s) - anchor))

    target_seconds = (target_minutes or 0) * 60 if target_minutes else 0
    chosen, total_sec, used_artists, chosen_ids = [], 0, set(), set()
    skipped_by_artist: list[Song] = []

    def consider_song(song: "Song", enforce_artist: bool = True) -> bool:
        nonlocal total_sec
        if song.id in chosen_ids:
            return False
        artist_key = (song.artist or "").lower()
        if enforce_artist and avoid_same_artist and artist_key in used_artists:
            return False
        chosen.append(song)
        chosen_ids.add(song.id)
        total_sec += song_duration(song)
        if enforce_artist:
            used_artists.add(artist_key)
        return True

    for s in candidates:
        artist_key = (s.artist or "").lower()
        if avoid_same_artist and artist_key in used_artists:
            skipped_by_artist.append(s)
            continue
        consider_song(s, enforce_artist=True)
        if target_seconds and total_sec >= target_seconds:
            break
    if target_seconds and total_sec < target_seconds and skipped_by_artist:
        for s in skipped_by_artist:
            if consider_song(s, enforce_artist=False) and total_sec >= target_seconds:
                break
    if target_seconds and total_sec < target_seconds:
        remaining = []
        seen_remaining = set()
        for s in all_songs:
            if s.id in chosen_ids or s.id in seen_remaining:
                continue
            if not base_passes_filters(s):
                continue
            remaining.append(s)
            seen_remaining.add(s.id)
        if remaining:
            def tempo_gap(song: "Song"):
                anchor = sum(tempo_focus) / 2 if tempo_focus else 120
                tempo_value = song.tempo_bpm or anchor
                return abs((tempo_value or anchor) - anchor)
            remaining.sort(key=tempo_gap)
            for s in remaining:
                consider_song(s, enforce_artist=False)
                if target_seconds and total_sec >= target_seconds:
                    break
    return chosen

# Curated AI suggestion pool (static for offline environments)
AI_RECOMMENDATION_POOL = {
    "mixed": [
        ("Shut Up and Dance", "WALK THE MOON"),
        ("Mr. Brightside", "The Killers"),
        ("September", "Earth, Wind & Fire"),
        ("Valerie", "Mark Ronson ft. Amy Winehouse"),
        ("Ain't No Mountain High Enough", "Marvin Gaye & Tammi Terrell"),
        ("Thinking Out Loud", "Ed Sheeran"),
        ("Sweet Caroline", "Neil Diamond"),
        ("Treasure", "Bruno Mars"),
        ("Dancing in the Moonlight", "Toploader"),
        ("Uptown Funk", "Mark Ronson ft. Bruno Mars"),
        ("I Want It That Way", "Backstreet Boys"),
        ("You Make My Dreams", "Daryl Hall & John Oates"),
        ("Signed, Sealed, Delivered (I'm Yours)", "Stevie Wonder"),
        ("Just the Way You Are", "Billy Joel"),
        ("Shallow", "Lady Gaga & Bradley Cooper"),
        ("Hey Ya!", "OutKast"),
        ("Happy", "Pharrell Williams"),
        ("Walking on Sunshine", "Katrina & The Waves"),
        ("Budapest", "George Ezra"),
        ("Love on Top", "Beyoncé"),
    ],
    "chill": [
        ("Slow Dancing in a Burning Room", "John Mayer"),
        ("Landslide", "Fleetwood Mac"),
        ("Better Together", "Jack Johnson"),
        ("Someone Like You", "Adele"),
        ("Yellow", "Coldplay"),
        ("Skinny Love", "Bon Iver"),
        ("Bubbly", "Colbie Caillat"),
        ("Stay", "Rihanna"),
        ("Wonderful Tonight", "Eric Clapton"),
        ("Gravity", "Sara Bareilles"),
        ("Come Away With Me", "Norah Jones"),
        ("Tennessee Whiskey", "Chris Stapleton"),
        ("Heartbeats", "José González"),
        ("All of Me", "John Legend"),
        ("Everybody Wants to Rule the World", "Lorde"),
        ("Lost in Japan", "Shawn Mendes"),
        ("Cherry Wine", "Hozier"),
        ("Like Real People Do", "Hozier"),
        ("Ocean Eyes", "Billie Eilish"),
    ],
    "energetic": [
        ("Don't Stop Me Now", "Queen"),
        ("Can't Stop the Feeling!", "Justin Timberlake"),
        ("Levitating", "Dua Lipa"),
        ("Blinding Lights", "The Weeknd"),
        ("I Gotta Feeling", "Black Eyed Peas"),
        ("Crazy in Love", "Beyoncé ft. JAY-Z"),
        ("Shut Up and Drive", "Rihanna"),
        ("Good as Hell", "Lizzo"),
        ("Feel It Still", "Portugal. The Man"),
        ("Shake It Off", "Taylor Swift"),
        ("Cake by the Ocean", "DNCE"),
        ("Raise Your Glass", "P!nk"),
        ("Don't Start Now", "Dua Lipa"),
        ("Moves Like Jagger", "Maroon 5"),
        ("Levels", "Avicii"),
        ("Can't Hold Us", "Macklemore & Ryan Lewis"),
        ("Firework", "Katy Perry"),
        ("Stronger", "Kelly Clarkson"),
        ("Born This Way", "Lady Gaga"),
        ("24K Magic", "Bruno Mars"),
    ],
}


def ai_recommend_songs(vibe: str,
                       required_tags: list[str],
                       required_genres: list[str],
                       limit: int = 10,
                       seed: int | None = None,
                       context: dict | None = None):
    """
    Provide deterministic “AI” song suggestions without external API calls.
    Creates placeholder Song rows when necessary so the auto-build flow can attach them.
    """
    ctx = dict(context or {})
    ctx.setdefault("vibe", vibe)
    preferences = _derive_ai_context_preferences(ctx, required_tags, required_genres)
    hard_tags = preferences.get("required_tags") if preferences else set()
    hard_genres = preferences.get("required_genres") if preferences else set()
    if hard_tags:
        required_tags = list(dict.fromkeys((required_tags or []) + [t for t in hard_tags if t]))
    if hard_genres:
        required_genres = list(dict.fromkeys((required_genres or []) + [g for g in hard_genres if g]))
    expected_count = preferences.get("expected_song_count") or 0
    effective_limit = max(limit or 0, expected_count + 5 if expected_count else 0, 20)

    vibe_key = (vibe or "mixed").lower()
    ordered_sources = []
    if vibe_key in AI_RECOMMENDATION_POOL:
        ordered_sources.append(AI_RECOMMENDATION_POOL[vibe_key])
    ordered_sources.append(AI_RECOMMENDATION_POOL["mixed"])
    for key, songs in AI_RECOMMENDATION_POOL.items():
        if key in (vibe_key, "mixed"):
            continue
        ordered_sources.append(songs)
    pool: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for src in ordered_sources:
        for title, artist in src:
            pair = (title.strip(), artist.strip())
            norm_pair = (pair[0].lower(), pair[1].lower())
            if norm_pair in seen_pairs:
                continue
            seen_pairs.add(norm_pair)
            pool.append(pair)
    if seed is not None and pool:
        offset = seed % len(pool)
        pool = pool[offset:] + pool[:offset]

    tag_set = {t.strip().lower() for t in required_tags or [] if t.strip()}
    genre_set = {g.strip().lower() for g in required_genres or [] if g.strip()}

    seen: set[tuple[str, str]] = set()
    candidates: list[dict] = []

    for idx, (title, artist) in enumerate(pool):
        key = (title.strip().lower(), artist.strip().lower())
        if key in seen:
            continue
        seen.add(key)

        song = Song.query.filter(
            func.lower(Song.title) == key[0],
            func.lower(Song.artist) == key[1],
            Song.deleted_at.is_(None),
        ).first()

        if song:
            meta = {
                "tempo_bpm": song.tempo_bpm,
                "musical_key": song.musical_key,
                "genre": song.genre,
                "tags": song.tags,
            }
        else:
            meta = ai_guess_metadata(title, artist)

        tags_norm = _normalize_tags(meta.get("tags"))
        if tag_set and not (tags_norm & tag_set):
            continue
        genre_val = (meta.get("genre") or "").lower()
        if genre_set and genre_val and not any(g in genre_val for g in genre_set):
            continue

        score = _score_ai_candidate(meta, preferences)
        candidates.append({
            "score": score,
            "order": idx,
            "meta": meta,
            "song": song,
            "title": title,
            "artist": artist,
            "tags": tags_norm,
        })

    candidates.sort(key=lambda row: (-row["score"], row["order"]))

    results: list[Song] = []
    for row in candidates:
        song = row["song"]
        meta = row["meta"]
        if song is None:
            tags_norm = set(row["tags"])
            tags_norm |= preferences.get("preferred_tags", set())
            for term in preferences.get("avoid_terms", set()):
                tags_norm.discard(term)
            tag_text = ", ".join(sorted(tags_norm)) if tags_norm else "ai"
            song = Song(
                title=row["title"],
                artist=row["artist"],
                tempo_bpm=meta.get("tempo_bpm"),
                musical_key=meta.get("musical_key"),
                genre=meta.get("genre"),
                tags=tag_text,
                user_id=None,
                is_public=True,
            )
            db.session.add(song)
            db.session.flush()
        results.append(song)
        if len(results) >= effective_limit:
            break
    return results


def build_autobuild_song_pool(scope: str,
                              vibe: str,
                              required_tags: list[str],
                              required_genres: list[str],
                              *,
                              seed: int | None = None,
                              context: dict | None = None):
    """Return candidate songs for auto-build based on scope preference."""
    scope = (scope or "mine").lower()
    user_id = _current_user_id()
    query = Song.query.filter(Song.deleted_at.is_(None))
    ai_songs: list[Song] = []
    ctx = dict(context or {})

    if scope == "mine":
        if current_user.is_authenticated:
            query = query.filter(Song.user_id == current_user.id)
        else:
            query = query.filter(False)
    elif scope == "shared":
        query = query.filter(or_(Song.is_public.is_(True), Song.user_id.is_(None)))
    elif scope == "ai":
        ctx.setdefault("vibe", vibe)
        expected = ctx.get("expected_song_count")
        if expected is None:
            expected = _ai_expected_song_count(ctx.get("target_minutes"), ctx.get("preset"))
            ctx["expected_song_count"] = expected
        dynamic_limit = max(30, (expected or 0) + 10)
        ai_songs = ai_recommend_songs(vibe, required_tags, required_genres, limit=dynamic_limit, seed=seed, context=ctx)
    # scope == "all" falls through to include everything

    songs = query.order_by(Song.created_at.desc()).all()

    if scope == "all" and user_id:
        def norm_pair(song: Song) -> tuple[str, str]:
            return (
                (song.title or "").strip().lower(),
                (song.artist or "").strip().lower(),
            )

        owned_pairs = {norm_pair(s) for s in songs if s.user_id == user_id}

        filtered: list[Song] = []
        seen_catalog_pairs: set[tuple[str, str]] = set()
        for song in songs:
            pair = norm_pair(song)
            if song.user_id != user_id:
                if pair in owned_pairs:
                    continue
                if pair in seen_catalog_pairs:
                    continue
                seen_catalog_pairs.add(pair)
            filtered.append(song)
        songs = filtered


    if scope == "mine" and not songs:
        # If the user has no personal songs, fall back to shared catalog.
        songs = (Song.query
                 .filter(or_(Song.is_public.is_(True), Song.user_id.is_(None)), Song.deleted_at.is_(None))
                 .order_by(Song.created_at.desc())
                 .all())

    if ai_songs:
        # Prepend AI suggestions while avoiding duplicates.
        seen_ids = {s.id for s in songs}
        merged = []
        for s in ai_songs:
            if s.id not in seen_ids:
                merged.append(s)
                seen_ids.add(s.id)
        merged.extend(songs)
        songs = merged

    # Deduplicate while preserving order
    deduped = []
    seen = set()
    for s in songs:
        if s.id not in seen:
            deduped.append(s)
            seen.add(s.id)
    if ctx:
        preferences = ctx.get("_derived")
        if not preferences:
            preferences = _derive_ai_context_preferences(ctx, required_tags, required_genres)
            ctx["_derived"] = preferences
        enumerated = list(enumerate(deduped))
        def key_fn(item):
            idx, song = item
            meta = {"tempo_bpm": song.tempo_bpm, "genre": song.genre, "tags": song.tags}
            return (-_score_ai_candidate(meta, preferences), idx)
        enumerated.sort(key=key_fn)
        deduped = [song for _, song in enumerated]
    return deduped

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
        project_id = os.getenv("OPENAI_PROJECT_ID")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "allow_project_keys=true",
        }
        if project_id:
            headers["OpenAI-Project"] = project_id
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={
            "model": "gpt-4o",
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
            try:
                err_payload = resp.json()
                err_message = err_payload.get("error", {}).get("message") or resp.text
            except Exception:
                err_message = resp.text
            print("OpenAI metadata error:", err_message)
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
        f"Draft a performance-ready ChordPro chart for '{safe_title}' by '{safe_artist}'. "
        "Return only the chart. Use bracketed chord symbols inline with the original lyrics (e.g., [C]Line). "
        "Preserve lyric accuracy, section order, and repeat structure from well-known versions of the song. "
        "Label sections with directives like {{title: Verse 1}}, {{title: Chorus}}, {{title: Bridge}} when appropriate. "
        "If the song has modulations or key changes, reflect them explicitly. "
        "Never invent fictional sections, commentary, or chord names. "
        "Do not wrap the response in Markdown fences or provide explanations."
    )

    try:
        project_id = os.getenv("OPENAI_PROJECT_ID")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "allow_project_keys=true",
        }
        if project_id:
            headers["OpenAI-Project"] = project_id
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You craft live-performance chord charts using ChordPro formatting."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            try:
                err_payload = resp.json()
                err_message = err_payload.get("error", {}).get("message")
            except Exception:
                err_message = None
            message = err_message or resp.text
            # Avoid leaking credentials if the message contains them
            import re

            message = message.replace(api_key, "[secure]")
            message = re.sub(r"sk-[A-Za-z0-9]{4,}", "[secure]", message)
            if resp.status_code == 401:
                return None, (
                    "OpenAI rejected the request (401 Unauthorized). "
                    "Double-check the OPENAI_API_KEY value and, if you are using a project key, "
                    "set OPENAI_PROJECT_ID to the matching project ID or create a standard API key."
                )
            return None, f"OpenAI error {resp.status_code}: {message[:200]}"
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
        return None, "OpenAI chord chart request failed."

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

    # ...existing code...
    return render_template(
        "list_songs.html",
        songs=songs,
        q=q,
        scope=scope,
        fmt_mmss=fmt_mmss,
        estimate=estimate_duration_seconds,
    )

CHORD_CHART_SCRIPT = r"""
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
"""

# ---------- SETLIST PAGES ----------


# Live Mode page template (Jinja)



# --- routes: Home ---
@app.get("/")
def home():
    inner = """
    <h2>Welcome</h2>
    <p>This is your local Setlist Genie dev app. Use the buttons above to manage songs and setlists.</p>
    """
    return render_template("base.html", content=inner)
    
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
    song = _song_or_404(song_id)
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
    song = _song_or_404(song_id)
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

@app.route("/songs")
@login_required
def list_songs():
    scope = (request.args.get("scope", "all") or "all").lower()
    if scope not in {"all", "mine", "catalog"}:
        scope = "all"

    user_id = _current_user_id()
    q = request.args.get("q", "").strip()
    like = f"%{q}%" if q else None

    def apply_search_filters(queryset):
        if not like:
            return queryset
        return queryset.filter(
            or_(
                Song.title.ilike(like),
                Song.artist.ilike(like),
                Song.genre.ilike(like),
                Song.musical_key.ilike(like),
                Song.tags.ilike(like),
            )
        )

    if scope == "mine":
        songs = (
            apply_search_filters(Song.query.filter(Song.user_id == user_id, Song.deleted_at.is_(None)))
            .order_by(Song.created_at.desc())
            .all()
        )
    elif scope == "catalog":
        songs = (
            apply_search_filters(
                Song.query.filter(
                    or_(Song.is_public.is_(True), Song.user_id.is_(None)),
                    Song.deleted_at.is_(None),
                )
            )
            .order_by(Song.created_at.desc())
            .all()
        )
    else:
        user_songs = []
        if user_id:
            user_songs = (
                apply_search_filters(Song.query.filter(Song.user_id == user_id, Song.deleted_at.is_(None)))
                .order_by(Song.created_at.desc())
                .all()
            )
        catalog_query = Song.query.filter(
            or_(Song.is_public.is_(True), Song.user_id.is_(None)),
            Song.deleted_at.is_(None),
        )
        catalog_songs = (
            apply_search_filters(catalog_query)
            .order_by(Song.created_at.desc())
            .all()
        )

        def norm_pair(song: Song) -> tuple[str, str]:
            return (
                (song.title or "").strip().lower(),
                (song.artist or "").strip().lower(),
            )

        owned_pairs = {norm_pair(s) for s in user_songs}
        unique_catalog = []
        seen_catalog_pairs: set[tuple[str, str]] = set()
        for s in catalog_songs:
            pair = norm_pair(s)
            if pair in owned_pairs or pair in seen_catalog_pairs:
                continue
            seen_catalog_pairs.add(pair)
            unique_catalog.append(s)

        songs = sorted(
            user_songs + unique_catalog,
            key=lambda s: s.created_at or datetime.min,
            reverse=True,
        )
    return render_template(
        "songs/list.html",
        songs=songs,
        q=q,
        scope=scope,
        fmt_mmss=fmt_mmss,
        estimate=estimate_duration_seconds,
    )

@app.route("/songs/new", methods=["GET", "POST"])
@login_required
def new_song():
    if request.method == "POST":
        form = request.form
        title = form.get("title", "").strip()
        artist = form.get("artist", "").strip()
        if not title or not artist:
            flash("Title and Artist are required.", "danger")
            return render_template("new_song.html")

        tempo_raw = (form.get("tempo_bpm") or "").strip()
        tempo_bpm = int(tempo_raw) if tempo_raw else None
        musical_key = (form.get("musical_key") or "").strip() or None
        genre = (form.get("genre") or "").strip() or None
        tags = (form.get("tags") or "").strip() or None
        duration_override = parse_mmss(form.get("duration_override"))

        release_year = None
        release_year_raw = (form.get("release_year") or "").strip()
        if release_year_raw:
            try:
                release_year = int(release_year_raw[:4])
            except Exception:
                release_year = None

        chord_chart_raw = form.get("chord_chart", "")
        chord_chart = chord_chart_raw.strip() or None
        generate_ai_chart = form.get("generate_ai_chart") == "1"

        song = Song(
            title=title,
            artist=artist,
            tempo_bpm=tempo_bpm,
            musical_key=musical_key,
            genre=genre,
            tags=tags,
            duration_override_sec=duration_override,
            release_year=release_year,
            chord_chart=chord_chart,
            user_id=current_user.id if current_user.is_authenticated else None,
        )
        db.session.add(song)
        db.session.flush()  # ensure song.id exists for downstream helpers

        try:
            meta = _fetch_song_metadata(song)
            if _apply_song_metadata(song, meta, force=False):
                src = "Spotify" if meta.get("_source") == "spotify" else "OpenAI"
                flash(f"Song details auto-filled via {src}.", "info")
        except Exception as exc:
            print("Auto metadata enrichment failed:", exc)

        ai_chart_generated = False
        if generate_ai_chart and not song.chord_chart:
            try:
                chart_text, error = openai_generate_chord_chart(song.title, song.artist)
                if chart_text:
                    song.chord_chart = sanitize_ai_chart(chart_text) or song.chord_chart
                    ai_chart_generated = True
                elif error:
                    flash(f"AI chart generation unavailable: {error}", "warning")
            except Exception as exc:
                print("AI chord chart generation failed:", exc)
                flash("AI chord chart generation failed.", "warning")

        db.session.commit()
        if ai_chart_generated:
            flash(
                "Draft chord chart generated with AI. Accuracy not guaranteed — please review before performing.",
                "warning",
            )
        flash(f'Added “{song.title}” — {song.artist}.')
        return redirect(url_for("list_songs"))

    return render_template("new_song.html")

@app.route("/songs/<int:song_id>/edit")
@login_required
def edit_song(song_id: int):
    song = _song_or_404(song_id)
    _require_song_owner(song)
    return render_template("songs/edit.html", song=song)


@app.post("/songs/<int:song_id>")
@login_required
def update_song(song_id: int):
    song = _song_or_404(song_id)
    _require_song_owner(song)

    form = request.form
    title = form.get("title", "").strip()
    artist = form.get("artist", "").strip()
    if not title or not artist:
        flash("Title and Artist are required.", "danger")
        return redirect(url_for("edit_song", song_id=song.id))

    tempo_raw = (form.get("tempo_bpm") or "").strip()
    song.tempo_bpm = int(tempo_raw) if tempo_raw else None
    song.musical_key = (form.get("musical_key") or "").strip() or None
    song.genre = (form.get("genre") or "").strip() or None
    song.tags = (form.get("tags") or "").strip() or None
    song.duration_override_sec = parse_mmss(form.get("duration_override"))

    release_year = None
    release_year_raw = (form.get("release_year") or "").strip()
    if release_year_raw:
        try:
            release_year = int(release_year_raw[:4])
        except Exception:
            release_year = None
    song.release_year = release_year

    chord_chart_raw = form.get("chord_chart", "")
    song.chord_chart = chord_chart_raw.strip() or None

    song.title = title
    song.artist = artist
    db.session.commit()
    flash(f'Updated “{song.title}” — {song.artist}.')
    return redirect(url_for("list_songs"))


@app.post("/songs/<int:song_id>/delete")
@login_required
def delete_song(song_id: int):
    song = _song_or_404(song_id, include_deleted=True)
    _require_song_owner(song)
    if song.deleted_at:
        flash(f'“{song.title}” — {song.artist} is already in the trash.', "info")
        scope = request.args.get("scope") or request.form.get("scope")
        q = request.args.get("q") or request.form.get("q")
        try:
            if scope:
                return redirect(url_for("list_songs", scope=scope, q=q))
        except Exception:
            pass
        return redirect(url_for("list_songs"))

    SetlistSong.query.filter_by(song_id=song.id).delete(synchronize_session=False)
    song.deleted_at = datetime.utcnow()
    db.session.commit()

    scope = request.args.get("scope") or request.form.get("scope")
    q = request.args.get("q") or request.form.get("q")
    undo_params = {"song_id": song.id}
    if scope:
        undo_params["scope"] = scope
    if q:
        undo_params["q"] = q
    undo_url = url_for("restore_song", **undo_params)
    message = Markup(f'Deleted “{song.title}” — {song.artist}. <a href="{undo_url}">Undo</a>')
    flash(message, "info")
    try:
        if scope:
            return redirect(url_for("list_songs", scope=scope, q=q))
    except Exception:
        pass
    return redirect(url_for("list_songs"))


@app.post("/songs/<int:song_id>/ai-chart", endpoint="song_generate_ai_chart")
@login_required
def song_generate_ai_chart(song_id: int):
    song = _song_or_404(song_id)
    _require_song_owner(song)
    chart_text, error = openai_generate_chord_chart(song.title, song.artist)
    if chart_text:
        song.chord_chart = sanitize_ai_chart(chart_text) or chart_text
        db.session.commit()
        flash("Updated chord chart with a fresh AI draft. Please review for accuracy.", "warning")
    else:
        if error:
            flash(f"AI chord chart unavailable: {error}", "warning")
        else:
            flash("AI chord chart unavailable.", "warning")
    return redirect(url_for("edit_song", song_id=song.id))


@app.get("/songs/<int:song_id>/restore")
@login_required
def restore_song(song_id: int):
    song = _song_or_404(song_id, include_deleted=True)
    if song.user_id != _current_user_id():
        abort(403)

    if song.deleted_at:
        song.deleted_at = None
        db.session.commit()
        flash(f'Restored “{song.title}” — {song.artist}.', "success")
    else:
        flash(f'“{song.title}” — {song.artist} is already active.', "info")

    scope = request.args.get("scope") or request.form.get("scope") or "mine"
    q = request.args.get("q") or request.form.get("q")
    return redirect(url_for("list_songs", scope=scope, q=q))


@app.post("/songs/<int:song_id>/clone", endpoint="songs_clone_to_user")
@login_required
def clone_song_to_user(song_id: int):
    song = _song_or_404(song_id)
    if song.user_id == current_user.id:
        flash("Song is already in your library.", "info")
        return redirect(request.form.get("next") or request.referrer or url_for("edit_song", song_id=song.id))

    normalized_title = (song.title or "").strip().lower()
    normalized_artist = (song.artist or "").strip().lower()

    existing = (
        Song.query.filter(
            func.lower(Song.title) == normalized_title,
            func.lower(Song.artist) == normalized_artist,
            Song.user_id == current_user.id,
        )
        .order_by(Song.created_at.desc())
        .first()
    )

    created_new = False
    updated_existing = False
    destination = existing

    if destination is None:
        destination = Song(
            title=song.title,
            artist=song.artist,
            tempo_bpm=song.tempo_bpm,
            musical_key=song.musical_key,
            genre=song.genre,
            tags=song.tags,
            duration_override_sec=song.duration_override_sec,
            release_year=song.release_year,
            chord_chart=song.chord_chart,
            user_id=current_user.id,
            is_public=False,
        )
        db.session.add(destination)
        db.session.flush()
        created_new = True
    else:
        if destination.deleted_at:
            destination.deleted_at = None
            updated_existing = True
        # Fill in missing fields from the shared copy without overwriting user edits.
        fields_to_sync = (
            "tempo_bpm",
            "musical_key",
            "genre",
            "tags",
            "duration_override_sec",
            "release_year",
        )
        for attr in fields_to_sync:
            src_val = getattr(song, attr)
            dest_val = getattr(destination, attr)
            if src_val and (dest_val is None or dest_val == ""):
                setattr(destination, attr, src_val)
                updated_existing = True
        if song.chord_chart and not destination.chord_chart:
            destination.chord_chart = song.chord_chart
            updated_existing = True
        if destination.is_public:
            destination.is_public = False
            updated_existing = True

    swapped = False
    row_id = request.form.get("row_id")
    if row_id:
        try:
            row = SetlistSong.query.get(int(row_id))
        except Exception:
            row = None
        if row and row.song_id == song.id and row.setlist:
            owner_id = getattr(row.setlist, "user_id", None)
            if owner_id is None or owner_id == current_user.id:
                row.song_id = destination.id
                swapped = True

    db.session.commit()

    if swapped:
        if created_new:
            flash(f'Copied “{destination.title}” and updated your setlist.')
        elif updated_existing:
            flash(f'“{destination.title}” was already in your songs — details refreshed and setlist updated.')
        else:
            flash(f'“{destination.title}” was already in your songs — setlist updated.')
    else:
        if created_new:
            flash(f'Copied “{destination.title}” to your songs. You can edit it now.')
        elif updated_existing:
            flash(f'“{destination.title}” was already in your songs — details refreshed.', "info")
        else:
            flash(f'“{destination.title}” is already in your songs.', "info")
    next_url = request.form.get("next") or request.referrer or url_for("edit_song", song_id=destination.id)
    return redirect(next_url)

@app.post("/songs/<int:song_id>/ai")
def ai_enrich_song(song_id):
    song = _song_or_404(song_id)
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
            existing = (Song.query
                        .filter_by(title=title, artist=artist)
                        .filter(Song.deleted_at.is_(None))
                        .first())
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

    return render_template("base.html", content=inner)

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

    target_minutes_raw = request.form.get("target_minutes", "").strip()
    vibe = (request.form.get("vibe") or "mixed").lower()
    req_tags = [t.strip() for t in (request.form.get("tags") or "").split(",") if t.strip()]
    req_genres = [g.strip() for g in (request.form.get("genres") or "").split(",") if g.strip()]
    avoid_same_artist = request.form.get("avoid_same_artist") == "1"
    clear_first = request.form.get("clear_first") == "1"
    scope = (request.form.get("scope") or "mine").lower()

    try:
        target_minutes_val = int(target_minutes_raw)
    except (TypeError, ValueError):
        target_minutes_val = setlist.target_minutes or 45
    if target_minutes_val <= 0:
        target_minutes_val = setlist.target_minutes or 45

    context = {
        "target_minutes": target_minutes_val,
        "event_type": setlist.event_type,
        "venue_type": setlist.venue_type,
        "notes": setlist.notes,
        "vibe": vibe,
        "avoid_same_artist": avoid_same_artist,
        "preset": None,
    }

    derived = _derive_ai_context_preferences(context, req_tags, req_genres)
    extra_tags = derived.get("required_tags") if derived else None
    extra_genres = derived.get("required_genres") if derived else None
    if extra_tags:
        req_tags = list(dict.fromkeys(req_tags + [t for t in extra_tags if t]))
    if extra_genres:
        req_genres = list(dict.fromkeys(req_genres + [g for g in extra_genres if g]))
    context["_derived"] = derived

    # pool: songs based on scope (optionally includes AI suggestions)
    all_songs = build_autobuild_song_pool(scope, vibe, req_tags, req_genres, seed=setlist.id, context=context)

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
        target_minutes=target_minutes_val,
        vibe=vibe,
        required_tags=req_tags,
        required_genres=req_genres,
        avoid_same_artist=avoid_same_artist,
        context=context,
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
        if scope == "ai":
            msg = "AI couldn't find new recommendations for this set. Try 'Clear current setlist first', switch vibes, or mix in shared songs."
            flash(msg, "info")
        else:
            flash("Auto-build found no additional songs to add.", "info")
    query_args = {"setlist_id": setlist.id, "autobuilt": 1, "scope": scope}
    if avoid_same_artist:
        query_args["avoid_repeat"] = 1
    return redirect(url_for("edit_setlist", **query_args))

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

@app.route("/setlists")
@login_required
def list_setlists():
    setlists = Setlist.query.order_by(Setlist.created_at.desc()).all()

    def live_mode_url(setlist_id: int) -> str:
        try:
            return url_for("live_mode", setlist_id=setlist_id)
        except BuildError:
            return f"/setlists/{setlist_id}/live"

    return render_template(
        "setlists/list.html",
        setlists=setlists,
        live_mode_url=live_mode_url,
    )

@app.get("/s/<token>")
def view_setlist_by_token(token):
    sl = Setlist.query.filter_by(share_token=token).first_or_404()
    songs = (SetlistSong.query
             .filter_by(setlist_id=sl.id)
             .order_by(SetlistSong.position.asc())
             .all())
    return render_template(
        "setlists/share.html",
        sl=sl,
        songs=songs,
        fmt_mmss=fmt_mmss,
        estimate=estimate_duration_seconds,
    )


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

@app.route("/setlists/new")
@login_required
def new_setlist():
    return render_template("new_setlist.html")

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
    sl.no_repeat_artists = False
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
        scope = (request.form.get("ab_scope") or "mine").lower()
        preset_choice = (request.form.get("ab_preset") or "").strip().lower()

        try:
            target_minutes_val = int(target_minutes_raw)
        except (TypeError, ValueError):
            target_minutes_val = sl.target_minutes or 45
        if target_minutes_val is None or target_minutes_val <= 0:
            target_minutes_val = sl.target_minutes or 45 or 45

        context = {
            "target_minutes": target_minutes_val,
            "event_type": event_type,
            "venue_type": venue_type,
            "notes": notes,
            "preset": preset_choice,
            "vibe": vibe,
            "avoid_same_artist": avoid_same_artist,
        }

        derived = _derive_ai_context_preferences(context, req_tags, req_genres)
        extra_tags = derived.get("required_tags") if derived else None
        extra_genres = derived.get("required_genres") if derived else None
        if extra_tags:
            req_tags = list(dict.fromkeys(req_tags + [t for t in extra_tags if t]))
        if extra_genres:
            req_genres = list(dict.fromkeys(req_genres + [g for g in extra_genres if g]))
        context["_derived"] = derived

        # Pool: songs based on scope preference (optionally includes AI suggestions)
        all_songs = build_autobuild_song_pool(scope, vibe, req_tags, req_genres, seed=sl.id, context=context)

        chosen = select_songs_for_target(
            all_songs=all_songs,
            target_minutes=target_minutes_val,
            vibe=vibe,
            required_tags=req_tags,
            required_genres=req_genres,
            avoid_same_artist=avoid_same_artist,
            context=context,
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
        if preset_choice == "basic":
            apply_section_preset_basic(sl.id)
        elif preset_choice == "three":
            apply_section_preset_three_sets(sl.id)
        elif preset_choice == "chunk":
            apply_section_preset_by_chunk(sl.id)

        count = len(chosen)
        if count:
            flash(f'Created setlist “{sl.name}” and auto-built {count} song{"s" if count != 1 else ""}.')
        else:
            if scope == "ai":
                flash("AI couldn't find fresh recommendations for this set. Try adjusting the vibe or add songs manually.", "info")
            else:
                flash(f'Created setlist “{sl.name}”. You can now add songs or adjust auto-build filters.', "info")
        query_args = {"setlist_id": sl.id, "autobuilt": 1, "scope": scope}
        if avoid_same_artist:
            query_args["avoid_repeat"] = 1
        return redirect(url_for("edit_setlist", **query_args))

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
    return render_template(
        "setlists/requests.html",
        sl=sl,
        new_requests=new_requests,
        other_requests=other_requests,
        statuses=REQUEST_STATUS_CHOICES,
        qr_url=url_for("setlist_requests_qr", setlist_id=sl.id),
        share_url=url_for("view_setlist_by_token", token=token, _external=True),
    )

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
    try:
        export_pdf_url = url_for("export_setlist_pdf", setlist_id=setlist.id)
    except BuildError:
        export_pdf_url = f"/setlists/{setlist.id}/export.pdf"

    try:
        export_csv_url = url_for("export_setlist_csv", setlist_id=setlist.id)
    except BuildError:
        export_csv_url = f"/setlists/{setlist.id}/export.csv"

    try:
        export_chopro_url = url_for("export_setlist_chopro", setlist_id=setlist.id)
    except BuildError:
        export_chopro_url = f"/setlists/{setlist.id}/export.chopro"

    try:
        export_chopro_zip_url = url_for("export_setlist_chopro_zip", setlist_id=setlist.id)
    except BuildError:
        export_chopro_zip_url = f"/setlists/{setlist.id}/export.chopro.zip"

    try:
        share_internal_url = url_for("view_setlist", setlist_id=setlist.id)
    except BuildError:
        share_internal_url = f"/setlists/{setlist.id}"

    try:
        share_create_url = url_for("create_share_link", setlist_id=setlist.id)
    except BuildError:
        share_create_url = f"/setlists/{setlist.id}/share/create"

    try:
        share_rotate_url = url_for("rotate_share_link", setlist_id=setlist.id)
    except BuildError:
        share_rotate_url = f"/setlists/{setlist.id}/share/rotate"

    try:
        share_qr_url = url_for("share_qr", setlist_id=setlist.id)
    except BuildError:
        share_qr_url = f"/setlists/{setlist.id}/share/qr"

    try:
        live_mode_url = url_for("live_mode", setlist_id=setlist.id)
    except BuildError:
        live_mode_url = f"/setlists/{setlist.id}/live"

    try:
        reorder_url = url_for("reorder_setlist", setlist_id=setlist.id)
    except BuildError:
        reorder_url = f"/setlists/{setlist.id}/reorder"

    scope_arg = (request.args.get("scope") or "mine").lower()
    if scope_arg not in {"mine", "shared", "all", "ai"}:
        scope_arg = "mine"
    autobuild_scope = scope_arg
    autobuild_avoid_repeat = request.args.get("avoid_repeat") == "1"
    autobuilt_recent = request.args.get("autobuilt") == "1"

    return render_template(
        "setlists/edit.html",
        setlist=setlist,
        songs=songs,
        q=q,
        estimates=estimates,
        total_str=fmt_mmss(total_sec),
        existing_artists=existing_artists,
        section_overview=sections,   # pass overview to template if you use it
        pending_requests=pending_requests,
        export_pdf_url=export_pdf_url,
        export_csv_url=export_csv_url,
        export_chopro_url=export_chopro_url,
        export_chopro_zip_url=export_chopro_zip_url,
        share_internal_url=share_internal_url,
        share_create_url=share_create_url,
        share_rotate_url=share_rotate_url,
        share_qr_url=share_qr_url,
        reorder_url=reorder_url,
        live_mode_url=live_mode_url,
        autobuild_scope=autobuild_scope,
        autobuild_avoid_repeat=autobuild_avoid_repeat,
        autobuilt_recent=autobuilt_recent,
    )

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
        if not song or song.deleted_at:
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

@app.post("/setlists/<int:setlist_id>/songs/<int:song_id>/move-up", endpoint="move_song_up")
@login_required
def move_song_up(setlist_id: int, song_id: int):
    setlist = Setlist.query.get_or_404(setlist_id)
    _require_setlist_owner(setlist)
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


@app.post("/setlists/<int:setlist_id>/songs/<int:song_id>/move-down", endpoint="move_song_down")
@login_required
def move_song_down(setlist_id: int, song_id: int):
    setlist = Setlist.query.get_or_404(setlist_id)
    _require_setlist_owner(setlist)
    stash_order(setlist.id)

    row = SetlistSong.query.filter_by(setlist_id=setlist.id, song_id=song_id).first_or_404()
    if getattr(row, "locked", False):
        flash("That song is locked and can’t be moved.")
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))

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
    try:
        download_pdf_url = url_for("export_setlist_pdf", setlist_id=sl.id)
    except BuildError:
        download_pdf_url = f"/setlists/{sl.id}/export.pdf"
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
        <a href="{download_pdf_url}">Download PDF</a>
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

    try:
        live_pdf_url = url_for("export_setlist_pdf", setlist_id=sl.id)
    except BuildError:
        live_pdf_url = f"/setlists/{sl.id}/export.pdf"

    try:
        live_csv_url = url_for("export_setlist_csv", setlist_id=sl.id)
    except BuildError:
        live_csv_url = f"/setlists/{sl.id}/export.csv"

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
      <a class="btn" href="{live_csv_url}">⬇️ Download CSV</a>
      <a class="btn" href="{live_pdf_url}">⬇️ Download PDF</a>
    </p>
    """

    return render_template("base.html", content=inner)
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
    
@app.get("/setlists/<int:setlist_id>/live", endpoint="live_mode")
@login_required
def _live_mode_route(setlist_id):
    return live_mode(setlist_id)

def live_mode(setlist_id):
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

    pages: list[dict] = []
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
        elif not skip_no_pdf:
            pages.append({
                **base,
                "title": title_text + " (no PDF)",
                "url": url_for("print_setlist", setlist_id=sl.id),
                "has_pdf": False,
            })

    if skip_no_pdf and not pages:
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

    return render_template(
        "live_mode.html",
        sl=sl,
        items_json=items_json,
        toggle_url=toggle_url,
        toggle_label=toggle_label,
        pending_requests=pending_requests,
        requests_url=url_for("setlist_requests", setlist_id=sl.id),
        requests_json_url=url_for("live_requests_json", setlist_id=sl.id),
    )
    
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

    if n == 1:
        _assign_section_label(rows, 0, "Set 1")
    elif n == 2:
        _assign_section_label(rows, 0, "Set 1")
        _assign_section_label(rows, 1, "Encore")
    else:
        # 3+ songs: Set 1 at top, Encore at last, Break near middle, Set 2 after Break
        first = 0
        last = n - 1
        mid = max(1, min(n - 2, n // 2))              # somewhere in the middle, not first/last
        set2 = max(mid + 1, min(n - 2, (3 * n) // 4)) # safely after Break, before last

        _assign_section_label(rows, first, "Set 1")
        _assign_section_label(rows, mid, "Break")
        _assign_section_label(rows, set2, "Set 2")
        _assign_section_label(rows, last, "Encore")

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

    if n == 1:
        _assign_section_label(rows, 0, "Set 1")
    elif n == 2:
        _assign_section_label(rows, 0, "Set 1")
        _assign_section_label(rows, 1, "Encore")
    elif n == 3:
        _assign_section_label(rows, 0, "Set 1")
        _assign_section_label(rows, 2, "Encore")
    elif n == 4:
        _assign_section_label(rows, 0, "Set 1")
        _assign_section_label(rows, 2, "Set 2")
        _assign_section_label(rows, 3, "Encore")
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

        _assign_section_label(rows, first, "Set 1")
        _assign_section_label(rows, b1,   "Break")
        _assign_section_label(rows, s2,   "Set 2")
        _assign_section_label(rows, b2,   "Break")
        _assign_section_label(rows, s3,   "Set 3")
        _assign_section_label(rows, last, "Encore")

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

    # always label first as Set 1
    set_no = 1
    _assign_section_label(rows, 0, f"Set {set_no}")

    # walk through in chunks; drop Break at chunk boundary, start next set after it
    i = chunk
    while i < n - 1:  # leave room for Encore at last
        _assign_section_label(rows, i, "Break")
        if i + 1 < n - 1:
            set_no += 1
            _assign_section_label(rows, i + 1, f"Set {set_no}")
            i += chunk  # next boundary relative to this new set start
        else:
            break

    # Encore at the very last song
    _assign_section_label(rows, n - 1, "Encore")

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

@app.post("/setlists/<int:setlist_id>/songs/<int:song_id>/move-top", endpoint="move_song_top")
@login_required
def move_song_top(setlist_id: int, song_id: int):
    setlist = Setlist.query.get_or_404(setlist_id)
    _require_setlist_owner(setlist)
    stash_order(setlist.id)

    row = SetlistSong.query.filter_by(setlist_id=setlist.id, song_id=song_id).first_or_404()
    if getattr(row, "locked", False):
        flash("That song is locked and can’t be moved.")
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))

    row.position = 0  # temporarily smallest so normalize_positions places it first
    db.session.commit()
    normalize_positions(setlist)
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))


@app.post("/setlists/<int:setlist_id>/songs/<int:song_id>/move-bottom", endpoint="move_song_bottom")
@login_required
def move_song_bottom(setlist_id: int, song_id: int):
    setlist = Setlist.query.get_or_404(setlist_id)
    _require_setlist_owner(setlist)
    stash_order(setlist.id)

    row = SetlistSong.query.filter_by(setlist_id=setlist.id, song_id=song_id).first_or_404()
    if getattr(row, "locked", False):
        flash("That song is locked and can’t be moved.")
        return redirect(url_for("edit_setlist", setlist_id=setlist.id))

    maxpos = db.session.query(db.func.max(SetlistSong.position)).filter_by(setlist_id=setlist.id).scalar() or 0
    row.position = maxpos + 1
    db.session.commit()
    normalize_positions(setlist)
    return redirect(url_for("edit_setlist", setlist_id=setlist.id))

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
        _ensure_schema_columns_postgres()
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
