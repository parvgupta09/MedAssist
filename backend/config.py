import os
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

# ─── BASE PATHS ───────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
METADATA_PATH   = BASE_DIR / "data" / "metadata" / "medassist_metadata.json"


def _resolve_chroma_path() -> str:
    """Resolve Chroma path across local and deployed folder layouts."""
    env_path = os.getenv("CHROMA_PATH")
    if env_path and Path(env_path).exists():
        return str(Path(env_path))

    embeddings_root = BASE_DIR / "data" / "embeddings"
    candidates = [
        embeddings_root / "chroma_db",
        embeddings_root / "embeddings" / "chroma_db",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Last resort: discover any nested chroma_db directory.
    for candidate in embeddings_root.rglob("chroma_db") if embeddings_root.exists() else []:
        if candidate.is_dir():
            return str(candidate)

    # Keep default path so startup can still proceed and report readiness failure clearly.
    return str(embeddings_root / "chroma_db")


CHROMA_PATH     = _resolve_chroma_path()

# ─── GROQ — only change: single key → list ────────────────
GROQ_API_KEYS = [
    key.strip() for key in [
        os.getenv("GROQ_API_KEY_1", ""),
        os.getenv("GROQ_API_KEY_2", ""),
        os.getenv("GROQ_API_KEY_3", ""),
    ] if key.strip()
]

if not GROQ_API_KEYS:
    raise ValueError("No Groq API keys found. Set GROQ_API_KEY_1 in .env")

GROQ_MODEL      = "llama-3.3-70b-versatile"

# ─── POSTGRESQL ───────────────────────────────────────────
DB_HOST         = os.getenv("DB_HOST",     "localhost")
DB_PORT         = os.getenv("DB_PORT",     "5432")
DB_NAME         = os.getenv("DB_NAME",     "medassist")
DB_USER         = os.getenv("DB_USER",     "postgres")
DB_PASSWORD     = os.getenv("DB_PASSWORD")

safe_password   = quote_plus(DB_PASSWORD)
DATABASE_URL    = f"postgresql://{DB_USER}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ─── EMBEDDING MODEL ──────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─── CHATBOT SETTINGS ─────────────────────────────────────
MAX_FOLLOWUP_QUESTIONS = 5
TOP_K_DISEASES         = 5
EARLY_EXIT_CONFIDENCE  = 0.75
EARLY_EXIT_GAP         = 0.25

# ─── NAME NORMALIZATION ───────────────────────────────────
def normalize_disease_name(name: str) -> str:
    """
    Normalize disease names for consistent lookup.
    Handles variations like:
    - "Diabetes Type 2" vs "Diabetes (Type 2)" vs "diabetes type 2"
    - "Common Cold" vs "cold, common"
    
    Returns lowercase, spaces normalized, special chars removed.
    """
    import re
    if not name or not isinstance(name, str):
        return ""
    # Convert to lowercase
    normalized = name.lower()
    # Remove extra spaces and normalize spacing
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    # Remove common punctuation variations (keep spaces and alphanumeric)
    normalized = re.sub(r'[(),-]', ' ', normalized)
    # Remove extra spaces again after punctuation removal
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

# File Summary:
# Central configuration for paths, API keys, database URL, and model settings.
# Includes disease-name normalization used by search and follow-up services.