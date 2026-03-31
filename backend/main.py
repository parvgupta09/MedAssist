# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.routers.chat import router as chat_router
from backend.database import engine, Base
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from sqlalchemy import text

# Download embeddings from HF Dataset on startup
try:
    snapshot_download(
        repo_id="Parv09/medassist-embeddings",
        repo_type="dataset",
        local_dir="data/embeddings",
        token=os.environ.get("HF_TOKEN")
    )
except Exception as exc:
    # Do not fail app startup if remote sync is unavailable.
    print(f"[WARN] Unable to download embeddings from Hugging Face: {exc}")

# Create tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title       = "MedAssist API",
    description = "AI-powered medical symptom checker",
    version     = "1.0.0"
)

# CORS — allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],  # restrict to frontend URL in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Include routers
app.include_router(chat_router)

def _check_database() -> tuple[bool, str]:
    """Run a lightweight DB probe for readiness checks."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "ok"
    except Exception as exc:
        return False, str(exc)

def _check_embeddings() -> tuple[bool, str]:
    """Check if embeddings directory and Chroma collection are available."""
    try:
        embeddings_dir = Path("data/embeddings")
        chroma_dir = Path("data/embeddings/chroma_db")

        if not embeddings_dir.exists():
            return False, "embeddings directory not found"
        if not chroma_dir.exists():
            return False, "chroma_db directory not found"

        # Lazy import so app startup is not affected if this check isn't called.
        import chromadb  # noqa: PLC0415

        client = chromadb.PersistentClient(path=str(chroma_dir))
        client.get_collection("diseases")
        return True, "ok"
    except Exception as exc:
        return False, str(exc)

@app.get("/")
def root():
    return {"status": "MedAssist API is running"}

@app.get("/favicon.ico")
def favicon():
    # Avoid repeated 404 noise from browser favicon requests.
    return JSONResponse(status_code=204, content=None)

@app.get("/health/live")
def health_live():
    return {"status": "alive"}

@app.get("/health/ready")
def health_ready():
    db_ok, db_detail = _check_database()
    embeddings_ok, embeddings_detail = _check_embeddings()

    checks = {
        "database": {"ok": db_ok, "detail": db_detail},
        "embeddings": {"ok": embeddings_ok, "detail": embeddings_detail},
    }
    ready = db_ok and embeddings_ok

    payload = {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)

@app.get("/health")
def health():
    # Keep legacy endpoint for compatibility with existing clients.
    return {"status": "healthy"}