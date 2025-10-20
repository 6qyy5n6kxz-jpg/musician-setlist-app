# filepath: /Users/devinfrank/setlist-genie/config.py
import os
from pathlib import Path

# --- File uploads (PDF attachments) ---

ALLOWED_MIME = {"application/pdf"}
# Max upload: 16 MB (Flask will 413 if exceeded)
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

REQUEST_STATUS_CHOICES = ("new", "queued", "done", "declined")

# Database config
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_SQLITE = "sqlite:///" + os.path.join(BASE_DIR, "musician.db")
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', DEFAULT_SQLITE)

# Other configs
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-secret")