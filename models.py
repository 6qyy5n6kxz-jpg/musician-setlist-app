from datetime import datetime
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path

UPLOAD_DIR = Path("instance/uploads")  # placeholder, set in app.py

db = SQLAlchemy()

# --- User model ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    songs = db.relationship("Song", backref="owner", lazy="dynamic")
    setlists = db.relationship("Setlist", backref="owner", lazy="dynamic")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

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
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    is_public = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    deleted_at = db.Column(db.DateTime, nullable=True)
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
    no_repeat_artists = db.Column(db.Boolean, default=False)  # avoid duplicate artists in this show
    share_token = db.Column(db.String(64), unique=True, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    reset_numbering_per_section = db.Column(db.Boolean, default=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)

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
