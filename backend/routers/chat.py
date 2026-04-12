# backend/routers/chat.py
import uuid
import json
import textwrap
import importlib
from io import BytesIO
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from backend.database import get_db
from backend.models.db_models import ChatSession, ChatMessage, UserProfile
from backend.services.summarizer import generate_user_summary
from backend.services.semantic_search import search_diseases
from backend.services.followup_engine import process_followup_answer, generate_next_stage2_question

router = APIRouter(tags=["chat"])


def _require_user_id(x_user_id: str | None = Header(default=None)) -> str:
    """Trust user identity forwarded by upstream backend after verification."""
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(status_code=401, detail="Unauthorized: missing x-user-id header")
    return x_user_id.strip()

# ─── REQUEST / RESPONSE MODELS ────────────────────────────
class StartSessionRequest(BaseModel):
    message:   str          # Stage 1 — initial symptom description
    user_id:   Optional[str] = None  # Deprecated: identity now comes from auth
    patient_name: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    weight: Optional[str] = None
    
    def validate(self):
        """Validate input fields"""
        if not self.message or not isinstance(self.message, str) or len(self.message.strip()) == 0:
            raise ValueError("message cannot be empty")
        self.message = self.message.strip()
        if self.user_id and isinstance(self.user_id, str):
            self.user_id = self.user_id.strip()
        if self.patient_name and isinstance(self.patient_name, str):
            self.patient_name = self.patient_name.strip()
        if self.sex and isinstance(self.sex, str):
            self.sex = self.sex.strip().lower()
            if self.sex not in {"male", "female", "other"}:
                raise ValueError("sex must be one of: male, female, other")
        if self.age is not None and (self.age < 0 or self.age > 120):
            raise ValueError("age must be between 0 and 120")
        if self.weight and isinstance(self.weight, str):
            self.weight = self.weight.strip()

class ContinueSessionRequest(BaseModel):
    session_id: str
    user_id:    Optional[str] = None  # Deprecated: identity now comes from auth
    message:    str         # Stage 2 answers or Stage 5 follow-up answers
    
    def validate(self):
        """Validate input fields"""
        if not self.session_id or len(self.session_id.strip()) == 0:
            raise ValueError("session_id cannot be empty")
        if not self.message or len(self.message.strip()) == 0:
            raise ValueError("message cannot be empty")
        self.message = self.message.strip()
        self.session_id = self.session_id.strip()
        if self.user_id and isinstance(self.user_id, str):
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


def _get_history_payload(db: Session, session_id: str) -> list[dict]:
    """Build full ordered message payload for API responses."""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.timestamp)
        .all()
    )
    return [
        {
            "role": m.role,
            "message": m.message,
            "stage": m.stage,
            "timestamp": m.timestamp,
        }
        for m in messages
    ]


def _history_to_conversation(history: list[dict]) -> list[dict]:
    """Convert detailed history to a simple role/content stream."""
    return [
        {"role": m["role"], "content": m["message"]}
        for m in history
    ]


def _merge_response_data(
    db: Session,
    session_id: str,
    base_data: Optional[dict],
    include_history: bool,
) -> Optional[dict]:
    """Optionally append full conversation history to response payload."""
    if not include_history:
        return base_data

    merged = dict(base_data) if isinstance(base_data, dict) else {}
    history = _get_history_payload(db, session_id)
    # conversation: simple GPT/Claude-like role/content stream.
    merged["conversation"] = _history_to_conversation(history)
    # history: detailed stream with stage/timestamp for debugging.
    merged["history"] = history
    merged["message_count"] = len(history)
    merged["last_message"] = history[-1] if history else None
    return merged


def _extract_profile_from_request(req: StartSessionRequest) -> dict:
    return {
        "name": req.patient_name,
        "age": req.age,
        "sex": req.sex,
        "weight": req.weight,
    }


def _get_or_create_user_profile(db: Session, user_id: str, incoming_profile: dict) -> dict:
    """
    Store profile on first chat and reuse later.
    If profile exists, keep it and only fill missing fields from incoming data.
    """
    profile = db.query(UserProfile).filter_by(user_id=user_id).first()

    if not profile:
        profile = UserProfile(
            user_id=user_id,
            name=incoming_profile.get("name"),
            age=incoming_profile.get("age"),
            sex=incoming_profile.get("sex"),
            weight=incoming_profile.get("weight"),
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
    else:
        changed = False
        if not profile.name and incoming_profile.get("name"):
            profile.name = incoming_profile.get("name")
            changed = True
        if profile.age is None and incoming_profile.get("age") is not None:
            profile.age = incoming_profile.get("age")
            changed = True
        if not profile.sex and incoming_profile.get("sex"):
            profile.sex = incoming_profile.get("sex")
            changed = True
        if not profile.weight and incoming_profile.get("weight"):
            profile.weight = incoming_profile.get("weight")
            changed = True
        if changed:
            db.commit()
            db.refresh(profile)

    return {
        "name": profile.name,
        "age": profile.age,
        "sex": profile.sex,
        "weight": profile.weight,
    }

# ─── STAGE 1 + 2: START SESSION ──────────────────────────
@router.post("/chat/start", response_model=SessionResponse)
def start_session(
    req: StartSessionRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(_require_user_id),
    include_history: bool = Query(default=True),
):
    """
    Stage 1 — User describes symptoms.
    Creates a new session, saves the message, asks Stage 2 profiling questions.
    """
    try:
        req.validate()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    session_id = str(uuid.uuid4())
    incoming_profile = _extract_profile_from_request(req)
    user_profile = _get_or_create_user_profile(db, user_id, incoming_profile)

    # Create session
    session = ChatSession(
        id         = session_id,
        user_id    = user_id,
        started_at = datetime.utcnow(),
        status     = "active"
    )
    db.add(session)
    db.commit()

    # Save user message
    save_message(db, session_id, "user", req.message, "stage_1")

    # Stage 2 — ask first profiling question (one at a time)
    summary = {
        "initial_symptoms": req.message,
        "patient_profile": {
            "name": user_profile.get("name"),
            "age": user_profile.get("age"),
            "sex": user_profile.get("sex"),
            "weight": user_profile.get("weight"),
        },
    }
    conversation = get_conversation(db, session_id)
    
    first_stage2_question = generate_next_stage2_question(summary, conversation, question_index=1)
    
    if first_stage2_question:
        save_message(db, session_id, "assistant", first_stage2_question, "stage_2")
        return SessionResponse(
            session_id = session_id,
            status     = "questioning",
            message    = first_stage2_question,
            stage      = "stage_2",
            data       = _merge_response_data(db, session_id, None, include_history)
        )
    else:
        # Fallback if generation fails
        return SessionResponse(
            session_id = session_id,
            status     = "error",
            message    = "Failed to generate profiling question",
            stage      = "stage_2",
            data       = _merge_response_data(db, session_id, None, include_history)
        )

# ─── STAGE 3 + 4 + 5: CONTINUE SESSION ───────────────────
@router.post("/chat/continue", response_model=SessionResponse)
def continue_session(
    req: ContinueSessionRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(_require_user_id),
    include_history: bool = Query(default=True),
):
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
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden: session does not belong to current user")
    if session.status == "completed":
        raise HTTPException(status_code=400, detail="Session already completed")

    # Get full conversation
    conversation = get_conversation(db, req.session_id)
    profile_row = db.query(UserProfile).filter_by(user_id=user_id).first()
    user_profile = {
        "name": profile_row.name if profile_row else None,
        "age": profile_row.age if profile_row else None,
        "sex": profile_row.sex if profile_row else None,
        "weight": profile_row.weight if profile_row else None,
    }
    
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
            summary = {
                "initial_symptoms": conversation[0]["content"] if conversation else "",
                "patient_profile": {
                    "name": user_profile.get("name"),
                    "age": user_profile.get("age"),
                    "sex": user_profile.get("sex"),
                    "weight": user_profile.get("weight"),
                },
            }
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
                    stage      = "stage_2",
                    data       = _merge_response_data(db, req.session_id, None, include_history)
                )
        
        # All Stage 2 questions answered → Move to Stage 3 + 4 + 5
        # Stage 3 — generate clinical summary
        summary = generate_user_summary(conversation, user_profile=user_profile)
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
            return _complete_session(
                db,
                session,
                result,
                req.session_id,
                user_profile=user_profile,
                include_history=include_history,
            )

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
            data       = _merge_response_data(
                db,
                req.session_id,
                {"top5": result["top5"]},
                include_history,
            )
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
            return _complete_session(
                db,
                session,
                result,
                req.session_id,
                user_profile=user_profile,
                include_history=include_history,
            )

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
            data       = _merge_response_data(
                db,
                req.session_id,
                {"top5": result["top5"]},
                include_history,
            )
        )

# ─── GET CHAT HISTORY ─────────────────────────────────────
@router.get("/chat/history/{session_id}")
def get_history(
    session_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(_require_user_id),
):
    """Returns full conversation history for a session."""
    session = db.query(ChatSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden: session does not belong to current user")

    history = _get_history_payload(db, session_id)
    conversation = _history_to_conversation(history)

    return {
        "session_id":       session_id,
        "status":           session.status,
        "started_at":       session.started_at,
        "ended_at":         session.ended_at,
        "final_diagnosis":  session.final_diagnosis,
        "message_count":    len(history),
        "last_message":     history[-1] if history else None,
        "conversation":     conversation,
        "messages":         history,
    }

@router.delete("/chat/delete/{session_id}")
def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(_require_user_id),
):
    """Delete a session and all related messages for the current user."""
    session = db.query(ChatSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden: session does not belong to current user")

    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete(synchronize_session=False)
    db.delete(session)
    db.commit()

    return {
        "success": True,
        "message": "Session deleted",
        "session_id": session_id,
    }

# ─── HELPERS ──────────────────────────────────────────────
def _detect_stage(session: ChatSession) -> str:
    if not session.top_5_diseases:
        return "stage_2"
    return "stage_5"


def _build_probable_reasons(summary: dict, top1: dict) -> list[str]:
    """Build short, patient-friendly reasons for why top diagnosis is probable."""
    reasons: list[str] = []

    symptoms = summary.get("reported_symptoms", []) if isinstance(summary, dict) else []
    if isinstance(symptoms, list) and symptoms:
        reasons.append(f"Your symptoms are consistent with {top1.get('name', 'the likely condition')}.")

    duration = summary.get("symptom_duration") if isinstance(summary, dict) else None
    if duration:
        reasons.append(f"Your symptom duration ({duration}) matches the usual clinical pattern.")

    regions = summary.get("body_regions_affected", []) if isinstance(summary, dict) else []
    if isinstance(regions, list) and regions:
        reasons.append("The affected body region pattern supports this possibility.")

    reasoning = top1.get("reasoning") if isinstance(top1, dict) else None
    if reasoning:
        reasons.append(str(reasoning))

    if not reasons:
        reasons.append("Overall symptom matching is stronger for this condition than other candidates.")

    return reasons[:4]


def _build_report_payload(session: ChatSession, user_profile: dict) -> dict:
    """Create a concise report payload from final diagnosis and summary."""
    final = session.final_diagnosis or {}
    top3 = final.get("top3", []) if isinstance(final, dict) else []
    if not top3:
        raise HTTPException(status_code=400, detail="Report is available only after diagnosis completion")

    top1 = top3[0]
    summary = {}
    if session.user_summary:
        try:
            summary = json.loads(session.user_summary)
        except Exception:
            summary = {}

    reasons = _build_probable_reasons(summary, top1)

    return {
        "session_id": session.id,
        "patient": {
            "name": user_profile.get("name"),
            "age": user_profile.get("age"),
            "sex": user_profile.get("sex"),
            "weight": user_profile.get("weight"),
        },
        "top_diagnosis": {
            "name": top1.get("name"),
            "confidence": top1.get("confidence"),
            "overview": top1.get("overview"),
            "typical_duration": top1.get("typical_duration"),
            "when_to_see_doctor": top1.get("when_to_see_doctor"),
        },
        "other_possible_conditions": [d.get("name") for d in top3[1:] if isinstance(d, dict)],
        "probable_reasons": reasons,
        "disclaimer": final.get(
            "disclaimer",
            "This is AI-assisted guidance, not a medical diagnosis. Please consult a doctor.",
        ),
    }


def _render_pdf_report(report_data: dict) -> bytes:
    """Render a professional medical report PDF without doctor details."""
    try:
        pagesizes = importlib.import_module("reportlab.lib.pagesizes")
        canvas_module = importlib.import_module("reportlab.pdfgen.canvas")
        A4 = pagesizes.A4
        canvas = canvas_module
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF dependency unavailable: {exc}")

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    y = height - 30

    def section_divider():
        nonlocal y
        y -= 6
        pdf.setLineWidth(1.5)
        pdf.line(40, y, 570, y)
        y -= 8

    def section_header(text: str):
        nonlocal y
        y -= 4
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(40, y, text)
        y -= 14
        pdf.setLineWidth(0.5)
        pdf.line(40, y, 570, y)
        y -= 8

    def write_line(text: str, size: int = 10, indent: int = 0, gap: int = 12, bold: bool = False):
        nonlocal y
        font = "Helvetica-Bold" if bold else "Helvetica"
        pdf.setFont(font, size)
        wrapped = textwrap.wrap(text, width=95)
        for line in wrapped:
            pdf.drawString(40 + indent, y, line)
            y -= gap

    # Header
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(40, y, "MEDASSIST")
    pdf.setFont("Helvetica", 9)
    pdf.drawString(40, y - 14, "AI-Assisted Health Assessment Report")
    y -= 32

    section_divider()

    # Patient Information
    patient = report_data.get("patient", {})
    top = report_data.get("top_diagnosis", {})
    reasons = report_data.get("probable_reasons", [])
    others = report_data.get("other_possible_conditions", [])

    section_header("PATIENT INFORMATION")
    write_line(f"Name: {patient.get('name') or 'N/A'}", bold=True, size=11)
    write_line(f"Age: {patient.get('age') or 'N/A'} years")
    write_line(f"Gender: {patient.get('sex') or 'N/A'}")
    write_line(f"Weight: {patient.get('weight') or 'N/A'} kg")
    y -= 2

    section_divider()

    # Assessment Summary
    section_header("CLINICAL ASSESSMENT")
    write_line(f"Report Status: Assessment Complete")
    write_line(f"Assessment Type: AI-Based Symptom Analysis")
    y -= 2

    section_divider()

    # Primary Diagnosis
    section_header("PRIMARY DIAGNOSIS")
    write_line(f"{top.get('name') or 'N/A'}", size=12, bold=True)
    confidence = top.get("confidence")
    confidence_pct = "N/A" if confidence is None else f"{round(float(confidence) * 100)}%"
    write_line(f"Confidence Level: {confidence_pct}", size=10)
    write_line(f"Typical Duration: {top.get('typical_duration') or 'N/A'}", size=10)
    y -= 2

    # Overview
    section_header("OVERVIEW & FINDINGS")
    write_line(f"{top.get('overview') or 'No additional information available.'}", size=10, gap=11)
    y -= 2

    # Reasoning
    if reasons:
        section_header("REASONING & KEY INDICATORS")
        for i, reason in enumerate(reasons[:4], 1):
            write_line(f"{i}. {reason}", indent=8, size=9, gap=10)
        y -= 2

    # Recommendations
    section_header("PROGNOSIS & RECOMMENDATIONS")
    write_line("When to Seek Medical Attention:", bold=True, size=10)
    write_line(f"{top.get('when_to_see_doctor') or 'Consult a healthcare provider if symptoms persist or worsen.'}", indent=8, size=9, gap=10)
    y -= 2
    write_line("General Care Recommendations:", bold=True, size=10)
    write_line("• Rest and follow medical advice if symptoms persist", indent=8, size=9, gap=10)
    write_line("• Monitor symptoms closely in the coming days", indent=8, size=9, gap=10)
    write_line("• Stay hydrated and maintain proper nutrition", indent=8, size=9, gap=10)
    y -= 2

    # Other Possible Conditions
    if others and len(others) > 0:
        section_header("OTHER POSSIBLE CONDITIONS")
        other_list = [str(name) for name in others if name]
        for i, condition in enumerate(other_list[:3], 1):
            write_line(f"{i}. {condition}", indent=8, size=9, gap=10)
        y -= 2

    section_divider()

    # Disclaimer
    section_header("IMPORTANT DISCLAIMER")
    disclaimer_text = (
        "This report is generated by artificial intelligence and is provided for informational purposes only. "
        "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult with a qualified healthcare provider before making any medical decisions."
    )
    write_line(disclaimer_text, size=8, gap=10)

    # Footer
    y -= 8
    pdf.setFont("Helvetica", 7)
    pdf.drawString(40, y, "MedAssist v1.0 | AI-Assisted Health Assessment")
    pdf.drawRightString(570, y, f"Generated: {datetime.utcnow().strftime('%d-%m-%Y %H:%M UTC')}")

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

def _complete_session(
    db: Session,
    session: ChatSession,
    result: dict,
    session_id: str,
    user_profile: Optional[dict] = None,
    include_history: bool = False,
) -> SessionResponse:
    """Mark session complete, save final diagnosis, return result."""
    final = result["result"]
    session.final_diagnosis = final
    session.status          = "completed"
    session.ended_at        = datetime.utcnow()
    db.commit()

    # Build user-friendly message
    top1      = final["top3"][0]
    patient_name = (user_profile or {}).get("name")
    intro_line = f"{patient_name}, based on your symptoms" if patient_name else "Based on your symptoms"
    message   = (
        f"{intro_line}, the most likely condition is **{top1['name']}** "
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
        data       = _merge_response_data(db, session_id, final, include_history)
    )


@router.get("/chat/report/{session_id}")
def get_report_data(
    session_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(_require_user_id),
):
    """Return structured report data for a completed session."""
    session = db.query(ChatSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden: session does not belong to current user")
    if session.status != "completed":
        raise HTTPException(status_code=400, detail="Report available only for completed sessions")

    profile_row = db.query(UserProfile).filter_by(user_id=user_id).first()
    user_profile = {
        "name": profile_row.name if profile_row else None,
        "age": profile_row.age if profile_row else None,
        "sex": profile_row.sex if profile_row else None,
        "weight": profile_row.weight if profile_row else None,
    }
    return _build_report_payload(session, user_profile)


@router.get("/chat/report/{session_id}/pdf")
def download_report_pdf(
    session_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(_require_user_id),
):
    """Generate and return a one-page PDF report for a completed session."""
    session = db.query(ChatSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden: session does not belong to current user")
    if session.status != "completed":
        raise HTTPException(status_code=400, detail="Report available only for completed sessions")

    profile_row = db.query(UserProfile).filter_by(user_id=user_id).first()
    user_profile = {
        "name": profile_row.name if profile_row else None,
        "age": profile_row.age if profile_row else None,
        "sex": profile_row.sex if profile_row else None,
        "weight": profile_row.weight if profile_row else None,
    }

    report_data = _build_report_payload(session, user_profile)
    pdf_bytes = _render_pdf_report(report_data)
    file_name = f"medassist_report_{session_id}.pdf"

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename={file_name}"},
    )

# File Summary:
# Implements chat API endpoints for session lifecycle, diagnosis flow, and reports.
# Coordinates stage handling, persistence, authorization checks, and PDF export.