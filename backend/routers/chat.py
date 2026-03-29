# backend/routers/chat.py
import uuid
import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from backend.database import get_db
from backend.models.db_models import ChatSession, ChatMessage
from backend.services.summarizer import generate_user_summary
from backend.services.semantic_search import search_diseases
from backend.services.followup_engine import process_followup_answer, generate_next_stage2_question

router = APIRouter(prefix="/chat", tags=["chat"])

# ─── REQUEST / RESPONSE MODELS ────────────────────────────
class StartSessionRequest(BaseModel):
    user_id:   str
    message:   str          # Stage 1 — initial symptom description
    
    def validate(self):
        """Validate input fields"""
        if not self.user_id or not isinstance(self.user_id, str) or len(self.user_id.strip()) == 0:
            raise ValueError("user_id cannot be empty")
        if not self.message or not isinstance(self.message, str) or len(self.message.strip()) == 0:
            raise ValueError("message cannot be empty")
        self.message = self.message.strip()
        self.user_id = self.user_id.strip()

class ContinueSessionRequest(BaseModel):
    session_id: str
    user_id:    str
    message:    str         # Stage 2 answers or Stage 5 follow-up answers
    
    def validate(self):
        """Validate input fields"""
        if not self.session_id or len(self.session_id.strip()) == 0:
            raise ValueError("session_id cannot be empty")
        if not self.user_id or len(self.user_id.strip()) == 0:
            raise ValueError("user_id cannot be empty")
        if not self.message or len(self.message.strip()) == 0:
            raise ValueError("message cannot be empty")
        self.message = self.message.strip()
        self.session_id = self.session_id.strip()
        self.user_id = self.user_id.strip()

class SessionResponse(BaseModel):
    session_id:   str
    status:       str       # questioning / complete
    message:      str       # assistant message to show user
    stage:        str
    data:         Optional[dict] = None  # top5, final result etc

# ─── HELPER: SAVE MESSAGE ─────────────────────────────────
def save_message(db: Session, session_id: str, role: str, content: str, stage: str):
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        message=content,
        stage=stage,
        timestamp=datetime.utcnow()
    )
    db.add(msg)
    db.commit()

# ─── HELPER: GET CONVERSATION HISTORY ────────────────────
def get_conversation(db: Session, session_id: str) -> list[dict]:
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.timestamp)
        .all()
    )
    return [{"role": m.role, "content": m.message} for m in messages]

# ─── STAGE 1 + 2: START SESSION ──────────────────────────
@router.post("/start", response_model=SessionResponse)
def start_session(req: StartSessionRequest, db: Session = Depends(get_db)):
    """
    Stage 1 — User describes symptoms.
    Creates a new session, saves the message, asks Stage 2 profiling questions.
    """
    try:
        req.validate()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    session_id = str(uuid.uuid4())

    # Create session
    session = ChatSession(
        id         = session_id,
        user_id    = req.user_id,
        started_at = datetime.utcnow(),
        status     = "active"
    )
    db.add(session)
    db.commit()

    # Save user message
    save_message(db, session_id, "user", req.message, "stage_1")

    # Stage 2 — ask first profiling question (one at a time)
    summary = {"initial_symptoms": req.message}
    conversation = get_conversation(db, session_id)
    
    first_stage2_question = generate_next_stage2_question(summary, conversation, question_index=1)
    
    if first_stage2_question:
        save_message(db, session_id, "assistant", first_stage2_question, "stage_2")
        return SessionResponse(
            session_id = session_id,
            status     = "questioning",
            message    = first_stage2_question,
            stage      = "stage_2"
        )
    else:
        # Fallback if generation fails
        return SessionResponse(
            session_id = session_id,
            status     = "error",
            message    = "Failed to generate profiling question",
            stage      = "stage_2"
        )

# ─── STAGE 3 + 4 + 5: CONTINUE SESSION ───────────────────
@router.post("/continue", response_model=SessionResponse)
def continue_session(req: ContinueSessionRequest, db: Session = Depends(get_db)):
    """
    Handles all messages after the first one.
    Routes: Stage 2 → Stage 3/4/5 → Stage 5 loop
    """
    try:
        req.validate()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Load session - validate BEFORE saving any data
    session = db.query(ChatSession).filter_by(id=req.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status == "completed":
        raise HTTPException(status_code=400, detail="Session already completed")

    # Get full conversation
    conversation = get_conversation(db, req.session_id)
    
    # Save user message with detected stage
    current_stage = _detect_stage(session)
    save_message(db, req.session_id, "user", req.message, current_stage)
    
    # Refresh conversation after saving user message
    conversation = get_conversation(db, req.session_id)

    # ── STAGE 2: Ask next profiling question or move to disease ranking ──
    if not session.top_5_diseases:
        # Count how many Stage 2 questions have been asked
        stage2_questions = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.session_id == req.session_id,
                ChatMessage.stage == "stage_2",
                ChatMessage.role == "assistant"
            )
            .all()
        )
        num_stage2_asked = len(stage2_questions)
        max_stage2_questions = 3
        next_question_index = num_stage2_asked + 1

        # Check if we should ask another Stage 2 question
        if next_question_index <= max_stage2_questions:
            # Generate and ask next Stage 2 question
            summary = {"initial_symptoms": conversation[0]["content"] if conversation else ""}
            next_question = generate_next_stage2_question(
                summary, 
                conversation, 
                question_index=next_question_index,
                max_stage2_questions=max_stage2_questions
            )
            
            if next_question:
                save_message(db, req.session_id, "assistant", next_question, "stage_2")
                return SessionResponse(
                    session_id = req.session_id,
                    status     = "questioning",
                    message    = next_question,
                    stage      = "stage_2"
                )
        
        # All Stage 2 questions answered → Move to Stage 3 + 4 + 5
        # Stage 3 — generate clinical summary
        summary = generate_user_summary(conversation)
        if not summary:
            raise HTTPException(status_code=500, detail="Failed to generate clinical summary")
        session.user_summary = json.dumps(summary)

        # Stage 4 — semantic search
        top5 = search_diseases(summary)
        if not top5 or len(top5) == 0:
            raise HTTPException(status_code=500, detail="No diseases found matching patient profile. Please try again.")
        session.top_5_diseases = top5
        db.commit()

        # Stage 5 — ask first follow-up question
        # Include all Stage 2 questions in asked_questions to avoid repetition
        stage2_questions = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.session_id == req.session_id,
                ChatMessage.stage == "stage_2",
                ChatMessage.role == "assistant"
            )
            .all()
        )
        asked_questions = [m.message for m in stage2_questions if m.message.strip()]
        
        try:
            result = process_followup_answer(
                top5           = top5,
                conversation   = conversation,
                summary        = summary,
                asked_questions= asked_questions,
                question_count = 0
            )
        except Exception as e:
            print(f"[Error processing followup] {e}")
            raise HTTPException(status_code=500, detail="Error processing diagnosis")

        if result["status"] == "complete":
            return _complete_session(db, session, result, req.session_id)

        # Save and return first follow-up question
        assistant_msg = result["question"]
        save_message(db, req.session_id, "assistant", assistant_msg, "stage_5")

        # Save top5 to session
        session.top_5_diseases = result["top5"]
        db.commit()

        return SessionResponse(
            session_id = req.session_id,
            status     = "questioning",
            message    = assistant_msg,
            stage      = "stage_5",
            data       = {"top5": result["top5"]}
        )

    # ── STAGE 5: Subsequent follow-up answers ──
    else:
        summary = json.loads(session.user_summary) if session.user_summary else {}
        top5    = session.top_5_diseases
        
        # Validate top5 exists and is not empty
        if not top5 or not isinstance(top5, list) or len(top5) == 0:
            raise HTTPException(status_code=400, detail="Session state corrupted: no diseases ranked")

        # Get ALL questions asked (both Stage 2 and Stage 5) to prevent repetition
        all_stage2_messages = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.session_id == req.session_id,
                ChatMessage.stage == "stage_2",
                ChatMessage.role == "assistant"
            )
            .all()
        )
        
        all_stage5_messages = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.session_id == req.session_id,
                ChatMessage.stage == "stage_5",
                ChatMessage.role == "assistant"
            )
            .order_by(ChatMessage.timestamp)
            .all()
        )
        
        question_count  = len(all_stage5_messages)
        
        # Combine questions from BOTH Stage 2 and Stage 5 to avoid repetition
        asked_questions = [m.message for m in all_stage2_messages if m.message.strip()]
        asked_questions.extend([m.message for m in all_stage5_messages if m.message.strip()])

        try:
            result = process_followup_answer(
                top5            = top5,
                conversation    = conversation,
                summary         = summary,
                asked_questions = asked_questions,
                question_count  = question_count
            )
        except Exception as e:
            print(f"[Error processing followup] {e}")
            raise HTTPException(status_code=500, detail="Error processing diagnosis")

        if result["status"] == "complete":
            return _complete_session(db, session, result, req.session_id)

        # Another follow-up question
        assistant_msg = result["question"]
        save_message(db, req.session_id, "assistant", assistant_msg, "stage_5")
        session.top_5_diseases = result["top5"]
        db.commit()

        return SessionResponse(
            session_id = req.session_id,
            status     = "questioning",
            message    = assistant_msg,
            stage      = "stage_5",
            data       = {"top5": result["top5"]}
        )

# ─── GET CHAT HISTORY ─────────────────────────────────────
@router.get("/history/{session_id}")
def get_history(session_id: str, db: Session = Depends(get_db)):
    """Returns full conversation history for a session."""
    session = db.query(ChatSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.timestamp)
        .all()
    )

    return {
        "session_id":       session_id,
        "status":           session.status,
        "started_at":       session.started_at,
        "ended_at":         session.ended_at,
        "final_diagnosis":  session.final_diagnosis,
        "messages": [
            {
                "role":      m.role,
                "message":   m.message,
                "stage":     m.stage,
                "timestamp": m.timestamp
            }
            for m in messages
        ]
    }

# ─── GET ALL USER SESSIONS ────────────────────────────────
@router.get("/sessions/{user_id}")
def get_user_sessions(user_id: str, db: Session = Depends(get_db)):
    """Returns all sessions for a user — for history page."""
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .order_by(ChatSession.started_at.desc())
        .all()
    )
    return [
        {
            "session_id":       s.id,
            "status":           s.status,
            "started_at":       s.started_at,
            "ended_at":         s.ended_at,
            "final_diagnosis":  s.final_diagnosis,
        }
        for s in sessions
    ]

# ─── HELPERS ──────────────────────────────────────────────
def _detect_stage(session: ChatSession) -> str:
    if not session.top_5_diseases:
        return "stage_2"
    return "stage_5"

def _complete_session(
    db: Session,
    session: ChatSession,
    result: dict,
    session_id: str
) -> SessionResponse:
    """Mark session complete, save final diagnosis, return result."""
    final = result["result"]
    session.final_diagnosis = final
    session.status          = "completed"
    session.ended_at        = datetime.utcnow()
    db.commit()

    # Build user-friendly message
    top1      = final["top3"][0]
    message   = (
        f"Based on your symptoms, the most likely condition is **{top1['name']}** "
        f"({round(top1['confidence'] * 100)}% match).\n\n"
        f"{top1['overview']}\n\n"
        f"Typical duration: {top1['typical_duration']}\n"
        f"When to see a doctor: {top1['when_to_see_doctor']}\n\n"
        f"⚠️ {final['disclaimer']}"
    )

    save_message(db, session_id, "assistant", message, "result")

    return SessionResponse(
        session_id = session_id,
        status     = "complete",
        message    = message,
        stage      = "result",
        data       = final
    )