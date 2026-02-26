"""
api/app.py — FastAPI server for MedAssist
Run: uvicorn api.app:app --reload --port 8000
"""

import os
import uuid
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from chatbot.vector_store_setup import load_faiss_store
from chatbot.chatbot import MedicalChatbot

load_dotenv()

logger = logging.getLogger("medassist")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAX_SESSIONS = 1000
SESSION_TTL_SECONDS = 60 * 60  # 1 hour — auto-evict idle sessions

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

vectorstore = None

# sessions stores: { session_id: {"bot": MedicalChatbot, "last_active": float} }
sessions: dict[str, dict] = {}


def _evict_old_sessions():
    """Remove sessions idle for more than SESSION_TTL_SECONDS."""
    now = time.time()
    to_delete = [
        sid for sid, data in sessions.items()
        if now - data["last_active"] > SESSION_TTL_SECONDS
    ]
    for sid in to_delete:
        del sessions[sid]
    if to_delete:
        logger.info(f"Evicted {len(to_delete)} idle sessions.")


# ─────────────────────────────────────────────
# Lifespan (replaces deprecated @app.on_event)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore

    # ── Validate API key early ──
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Add it in HF Space Secrets or your .env file."
        )

    # ── Load FAISS ──
    try:
        vectorstore = load_faiss_store("disease_vector_db")
        logger.info("✅ FAISS vector store loaded successfully.")
    except Exception as e:
        raise RuntimeError(
            f"Could not load FAISS vector store: {e}\n"
            "Run: python scripts/index_diseases.py"
        )

    yield  # app runs here

    # ── Shutdown cleanup ──
    sessions.clear()
    logger.info("Sessions cleared on shutdown.")


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(title="MedAssist API", version="1.0.0", lifespan=lifespan)

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
allow_origins = (
    ["*"]
    if allowed_origins_env in {"*", ""}
    else [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class NewSessionResponse(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    phase: str
    top_candidates: list = []


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "MedAssist API",
        "active_sessions": len(sessions),
    }


@app.post("/session/new", response_model=NewSessionResponse)
async def new_session():
    _evict_old_sessions()

    if len(sessions) >= MAX_SESSIONS:
        raise HTTPException(
            503,
            "Server is at capacity. Please try again in a few minutes."
        )

    session_id = str(uuid.uuid4())
    bot = MedicalChatbot(
        vectorstore=vectorstore,
        groq_api_key=GROQ_API_KEY,
        top_k=25,
        final_top_n=4,
    )
    sessions[session_id] = {"bot": bot, "last_active": time.time()}

    return NewSessionResponse(
        session_id=session_id,
        message=(
            "Hello! I'm MedAssist, your medical triage assistant. "
            "I'm here to help identify possible conditions based on your symptoms. "
            "Please remember I'm not a doctor — always consult a licensed professional.\n\n"
            "To get started: what symptoms are you experiencing?"
        ),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.session_id not in sessions:
        raise HTTPException(
            404,
            "Session not found or expired. Create a new one with POST /session/new"
        )

    entry = sessions[req.session_id]
    bot: MedicalChatbot = entry["bot"]
    entry["last_active"] = time.time()

    raw_response = bot.chat(req.message)

    # bot.chat() sometimes returns a dict — extract the string safely
    if isinstance(raw_response, dict):
        response = (
            raw_response.get("message")
            or raw_response.get("response")
            or raw_response.get("text")
            or raw_response.get("content")
            or str(raw_response)
        )
    else:
        response = str(raw_response)

    # Auto-evict completed sessions after response is sent
    phase = bot.state.phase.value
    if phase == "result":
        sessions.pop(req.session_id, None)

    return ChatResponse(
        session_id=req.session_id,
        response=response,
        phase=phase,
        top_candidates=bot.get_rankings(),
    )


@app.get("/session/{session_id}/state")
async def get_state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found or expired.")

    bot: MedicalChatbot = sessions[session_id]["bot"]
    sessions[session_id]["last_active"] = time.time()

    return {
        "phase": bot.state.phase.value,
        "followup_rounds": bot.state.followup_rounds,
        "patient_profile": bot.state.patient.dict(),
        "top_candidates": bot.get_rankings(),
    }


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared"}


@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found or expired.")

    bot: MedicalChatbot = sessions[session_id]["bot"]
    sessions[session_id]["last_active"] = time.time()

    # Serialize chat_history safely — LangChain message objects need manual conversion
    raw_history = bot.state.chat_history
    serialized = []
    for msg in raw_history:
        if hasattr(msg, "type") and hasattr(msg, "content"):
            # LangChain BaseMessage objects (HumanMessage, AIMessage, etc.)
            serialized.append({"role": msg.type, "content": msg.content})
        elif isinstance(msg, dict):
            serialized.append(msg)
        else:
            serialized.append({"role": "unknown", "content": str(msg)})

    return {
        "session_id": session_id,
        "phase": bot.state.phase.value,
        "chat_history": serialized,
    }