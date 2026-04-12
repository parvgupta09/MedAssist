# backend/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from backend.routers.chat import router as chat_router
from backend.database import engine, Base
from backend.config import CHROMA_PATH
import os
from pathlib import Path
import time
from sqlalchemy import text


def _init_database(max_attempts: int = 15, delay_seconds: int = 1) -> bool:
    """Create DB tables with retries to survive DB cold start on deploy.
    Returns True if successful, False otherwise. Does NOT crash startup.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            Base.metadata.create_all(bind=engine)
            print(f"[INFO] Database initialized on attempt {attempt}")
            return True
        except Exception as exc:
            last_exc = exc
            print(f"[WARN] Database init attempt {attempt}/{max_attempts} failed: {str(exc)[:200]}")
            if attempt < max_attempts:
                time.sleep(delay_seconds)
    
    # Log the failure but don't crash - routes will still be available
    print(f"[WARN] Database initialization failed after {max_attempts} attempts. "
          f"API routes will be available but may fail without DB. Last error: {str(last_exc)[:200]}")
    return False


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Create tables as startup side-effect; embeddings are expected locally.
    print("[INFO] Starting application lifespan: initializing database...")
    success = _init_database()
    if success:
        print("[INFO] ✓ Database initialization successful - all routes available")
    else:
        print("[WARN] ⚠ Database initialization failed - routes may error on DB operations")
    print("[INFO] Application startup complete")
    yield
    print("[INFO] Application shutdown")


def _cors_settings() -> tuple[list[str], bool]:
    """Build CORS origins from env; keep credentials off for wildcard origins."""
    raw = os.getenv("ALLOWED_ORIGINS", "*")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if not origins:
        origins = ["*"]
    allow_credentials = origins != ["*"]
    return origins, allow_credentials

app = FastAPI(
    title       = "MedAssist API",
    description = "AI-powered medical symptom checker",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS — allow frontend to call the API
cors_origins, cors_allow_credentials = _cors_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins     = cors_origins,
    allow_credentials = cors_allow_credentials,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Include routers
app.include_router(chat_router)

def _check_database() -> tuple[bool, str]:
    """Run a lightweight DB probe for readiness checks."""
    try:
        # Use engine.begin() context for better connection handling
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
        return True, "ok"
    except Exception as exc:
        error_msg = str(exc)[:200]  # Truncate long error messages
        print(f"[WARN] Database check failed: {error_msg}")
        return False, error_msg

def _check_embeddings() -> tuple[bool, str]:
    """Check if embeddings directory and Chroma collection are available."""
    try:
        embeddings_dir = Path("data/embeddings")
        chroma_dir = Path(CHROMA_PATH)

        if not embeddings_dir.exists():
            return False, "embeddings directory not found"
        if not chroma_dir.exists():
            return False, f"chroma_db directory not found at {chroma_dir}"

        # Lazy import so app startup is not affected if this check isn't called.
        import chromadb  # noqa: PLC0415

        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection_names = [
            getattr(c, "name", c) for c in client.list_collections()
        ]
        if "diseases" not in collection_names:
            return False, f"Collection diseases does not exist. found={collection_names} path={chroma_dir}"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)

@app.get("/")
def root():
    return {"status": "MedAssist API is running"}

@app.get("/favicon.ico")
def favicon():
    # Avoid repeated 404 noise from browser favicon requests.
    # A 204 response must not include a body.
    return Response(status_code=204)

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

# File Summary:
# FastAPI application entrypoint with startup setup, CORS, and health endpoints.
# Registers chat router and readiness checks for database plus embeddings.