"""
chatbot.py — Medical Diagnosis Chatbot (v3 — Full Fix)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROOT CAUSE FIXES:
  1. Intake is now STRICTLY limited to 5 mandatory fields only.
     Conditional fields (contact, travel, lifestyle etc.) are collected
     as a single optional pass ONLY if symptoms clearly warrant them.
     → Intake ends in 5-7 turns max, not 10+.

  2. Disease schema questions (question_flow) are ALWAYS asked in followup.
     The LLM can no longer "steal" schema questions during intake.

  3. _intake_options() now correctly detects gender questions and returns
     Male/Female/Other (not Yes/No/Not sure).

  4. Duplicate question detection is tightened — intake history is NOT
     passed to followup rephrasing, preventing "again?" style repeats.

  5. Options generation always produces meaningful hints based on the
     EXACT schema question text + expected_answer_type.
"""

import json
import re
import logging
from typing import List, Optional, Dict, Tuple, Set
from enum import Enum

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

INTAKE_SYSTEM_PROMPT = """
You are a warm, empathetic medical intake assistant. Collect ONLY what is listed below.
Do NOT ask clinical questions — those come later. Keep intake SHORT.

═══════════════════════════════════════════
MANDATORY — collect ALL 5 before finishing:
═══════════════════════════════════════════
1. symptoms_list    → every symptom the patient mentions
2. age              → must be a real integer (e.g. 25). Ask if not given.
3. gender           → male / female / other. Ask if not given.
4. symptom_duration → how long they've had the symptoms (e.g. "3 days", "a week")
5. onset            → sudden or gradual

═══════════════════════════════════════════════════════════════════════
OPTIONAL — collect ONLY if the symptom type makes it obviously relevant:
═══════════════════════════════════════════════════════════════════════
• existing_conditions  → only for chest pain, breathlessness, fainting, chronic fatigue
• location_recent      → only for fever, unusual rash, diarrhoea (travel-related)
• contact_with_sick    → only for fever, sore throat, cough, flu-like illness
• lifestyle_factors    → only for chest pain, liver/GI symptoms, significant weight change
• family_history       → only for suspected cancer, heart disease, diabetes
• reproductive_context → females only, only for pelvic/abdominal/hormonal symptoms
• distinguishing_detail → ask ONE targeted follow-up if clinically essential

═══════════════════
CONVERSATION RULES:
═══════════════════
• ONE question per turn. Never two.
• Acknowledge what patient said (1 sentence), blank line, then ask.
• NEVER ask about cold, sinus, allergies, air pressure, water in ear, flight history,
  or ANY disease-specific questions — those are handled AFTER intake.
• NEVER re-ask something already answered.
• Do NOT complete intake until age is a real integer.
• Do NOT complete intake until all 5 mandatory fields have real values.
• Complete intake as soon as mandatory fields are done — do NOT pad with extra questions.

═══════════════════════════════════════════════════════════════════
COMPLETION — write ONE warm sentence, then IMMEDIATELY on next line:
═══════════════════════════════════════════════════════════════════
[INTAKE_COMPLETE:{"symptoms_list":["symptom1"],"age":25,"gender":"male","symptom_duration":"3 days","onset":"gradual","existing_conditions":null,"location_recent":null,"contact_with_sick":null,"lifestyle_factors":null,"family_history":null,"reproductive_context":null,"distinguishing_detail":null}]

RULES FOR THE JSON:
• age MUST be a real integer like 25, NOT "<int|null>" or null
• gender MUST be "male", "female", or "other" — NOT "<str>"
• symptoms_list must contain at least one real symptom string
• symptom_duration must be a real string like "3 days" — NOT null
• onset must be "sudden", "gradual", or "unknown" — NOT null
• The [INTAKE_COMPLETE:...] tag must be on ONE line with no line breaks inside it
• NEVER emit [INTAKE_COMPLETE:...] until all 5 mandatory fields have real values
""".strip()


QUERY_GENERATION_PROMPT = """
Medical search specialist. Generate 3 concise FAISS vector search queries.

Query 1 — SYMPTOM CLUSTER: the 2-3 most prominent symptoms together.
Query 2 — CLINICAL PATTERN: onset speed + duration + age group + key distinguishing detail.
Query 3 — RISK CONTEXT: conditions, travel, contact, lifestyle, family history.
           If no context, use a differential diagnostic category angle.

Patient Profile:
{patient_profile}

Return ONLY valid JSON, no other text:
{{"query_1":"<symptom cluster query>","query_2":"<clinical pattern query>","query_3":"<risk context query>"}}
""".strip()


FOLLOWUP_QUESTION_PROMPT = """
You are a warm, empathetic medical assistant in a diagnostic interview.

Write EXACTLY this structure:
  Line 1: Acknowledge the patient's previous answer (1 sentence — empathetic, varies each turn)
  Line 2: [BLANK LINE — REQUIRED]
  Line 3: Ask the clinical question in plain conversational language (1 sentence only)

Tone guidance:
  Painful/serious → show empathy: "That sounds really uncomfortable."
  Neutral/factual → keep it light: "Got it, that's helpful."
  First question   → warm opener: "Thanks for being patient with me."

Variety rules — rotate acknowledgment style each turn:
  • Reflect back what they said
  • Validate their experience
  • Normalise ("That's actually quite common with...")
  • Simple confirmation ("Understood, thank you.")
  • Empathise ("I'm sorry you're dealing with this.")

Hard rules:
  • NEVER use: "Sure!", "Of course!", "Certainly!", "Absolutely!", "Great!"
  • NO medical jargon — plain patient-friendly language
  • ONE question only — never combine two questions
  • Do NOT repeat information already collected in intake

Previous patient answer: {previous_answer}
Clinical question to ask: {raw_question}
Clinical purpose of question: {purpose}
Disease being investigated: {disease_context}

Your response (acknowledgment + blank line + question):
""".strip()


SMART_OPTIONS_PROMPT = """
Medical UI designer. Generate 2–3 quick-reply hint options for a patient.

Purpose: Help the patient see what KIND of answer we need. Cover the most common real responses.

Rules:
  • First-person patient voice: "Yes, I do" not "Affirmative"
  • Max 10 words per option
  • Options must be MEANINGFULLY different (different severity / type / yes-no)
  • yes/no questions → "Yes", "No", (optionally) "Not sure"
  • pain/severity → describe HOW it actually FEELS, not just labels:
      ✓ "Very strong and constant, hard to ignore"
      ✓ "Moderate — uncomfortable but manageable"
      ✓ "Mild, barely noticeable"
      ✗ "Severe" / "Mild" / "Moderate" (too short, not descriptive enough)
  • timing → "During takeoff", "During landing", "After the flight"
  • location → "Left ear only", "Both ears", "Right ear only"
  • hearing → "Muffled sound, like underwater", "Ringing/buzzing", "No change"
  • NEVER use medical jargon

Question text   : {question}
Expected type   : {answer_type}
Related symptoms: {related_symptoms}
Disease context : {disease_name}
Question purpose: {purpose}

Return ONLY valid JSON, nothing else:
{{"options":["<option1>","<option2>","<option3 if needed>"]}}
""".strip()


RERANKING_PROMPT = """
Senior medical AI. Update disease candidate probabilities based on new Q&A.

Patient Profile:
{patient_profile}

New Q&A:
  Question : {question}
  Answer   : {answer}
  Purpose  : {purpose}
  Symptoms probed: {related_symptoms}
  Source disease : {source_disease}

Candidate diseases to update:
{candidates_json}

For EACH candidate update ALL fields:
  • probability_score (0.0–1.0)
  • confidence: "low" | "medium" | "high"
  • status: "rising" | "falling" | "stable" | "ruled_out"
  • evidence_for: specific patient observations supporting this disease
  • evidence_against: specific patient observations against this disease

Scoring rules:
  • Hallmark/pathognomonic feature confirmed → large increase (0.15–0.30)
  • Core feature directly contradicted → drop to below 0.10
  • Neutral/ambiguous answer → ±0.05 max change
  • Update the SOURCE DISEASE candidate most precisely first
  • Max score 0.95 unless ALL major criteria are confirmed
  • "ruled_out" + score ≤ 0.05 only when clinically implausible
  • Evidence must cite ACTUAL patient details, not generic statements

Return ONLY valid JSON:
{{"updated_candidates":[{{"disease_id":"<id>","disease_name":"<name>","probability_score":<float>,"confidence":"<low|medium|high>","status":"<rising|falling|stable|ruled_out>","evidence_for":["<specific observation>"],"evidence_against":["<specific observation>"]}}]}}
""".strip()


FINAL_DIAGNOSIS_PROMPT = """
Senior medical AI. Generate a COMPREHENSIVE personalised diagnostic report.

PATIENT PROFILE:
{patient_profile}

ALL FOLLOW-UP Q&A (these are from disease-specific questions):
{qa_summary}

TOP {n} DISEASE CANDIDATES:
{top_candidates_json}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write EVERY section below. Be DETAILED and SPECIFIC. Do NOT be brief.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🩺 What Your Symptoms Most Likely Point To
Name the most likely condition clearly. Then 3-4 detailed paragraphs:
  • What this condition IS — plain language explanation
  • WHY it fits THIS patient specifically — cite their age, gender, duration, onset, specific Q&A answers
  • Confidence level and reasoning
  • If top probability < 0.55: "No single condition stands out clearly — the most likely is..."

## 📋 What You Told Us — And What It Means
For EVERY data point collected, write: "When you said [X], that pointed toward [Y] because [Z]."
Cover ALL follow-up answers AND intake info. Write minimum 7-8 data points.
Also mention what REDUCED likelihood of other conditions.

## 🔍 Other Conditions We Considered
For EACH of the top 3-4 candidates:
  • Name it and describe it in plain language (1-2 sentences)
  • Why it initially appeared (which symptoms matched)
  • What specific answer raised or lowered its probability
  • Current likelihood estimate and reasoning

## 🌡️ Severity Assessment
Choose EXACTLY ONE: 🟢 MILD / 🟡 MODERATE / 🔴 SEVERE / 🚨 POTENTIALLY URGENT
Write 4-5 sentences:
  • Justify the choice using THIS patient's specific symptoms, duration, age
  • List any red flags found (or confirm none)
  • State what would change this rating (e.g. "If you develop X, this becomes urgent")

## ✅ What You Should Do Right Now — Step by Step

1. 🏥 Where to go and HOW urgently (ER / urgent care / GP appointment)
   Explain WHY based on their specific case — not generic advice.

2. 🏠 Home care steps (specific and practical):
   List 3-5 concrete actions they can take today.

3. 🚫 What to AVOID:
   Condition-specific things that could make it worse.

4. 🚨 Go to ER IMMEDIATELY if you notice:
   List 4-5 specific warning signs relevant to their condition.

5. 🔬 Tests to ask your doctor about:
   Name each test and explain in plain language what it checks for.

6. 💊 Medications or remedies that may help:
   Name them with dosage guidance if applicable. Note any contraindications.

7. 📅 Recovery timeline and follow-up:
   Realistic expectations. When to return to normal activities.

## ⚠️ Important Disclaimer
4-5 warm, reassuring sentences:
  • This is AI triage only — not a diagnosis
  • A doctor must confirm this
  • When to seek care urgently
  • Encouragement to take action

━━━━━━━━━━━━━━━━━━
Style guide:
  • Address patient as "you" throughout
  • Define every medical term in plain language
  • Reference their SPECIFIC answers to personalise everything
  • Be thorough — detailed enough to bring to a doctor's appointment
━━━━━━━━━━━━━━━━━━
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
#  STATE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class Phase(str, Enum):
    INTAKE   = "intake"
    SEARCH   = "search"
    FOLLOWUP = "followup"
    RESULT   = "result"


class PatientProfile(BaseModel):
    symptoms:              List[str]            = Field(default_factory=list)
    age:                   Optional[int]         = None
    gender:                Optional[str]         = None
    symptom_duration:      Optional[str]         = None
    onset:                 Optional[str]         = None
    location_recent:       Optional[str]         = None
    existing_conditions:   Optional[str]         = None
    contact_with_sick:     Optional[str]         = None
    lifestyle_factors:     Optional[str]         = None
    family_history:        Optional[str]         = None
    reproductive_context:  Optional[str]         = None
    distinguishing_detail: Optional[str]         = None
    qa_log:                List[Tuple[str, str]] = Field(default_factory=list)


class QuestionItem(BaseModel):
    question:         str
    purpose:          str        = ""
    q_type:           str        = "initial_questions"
    expected_type:    str        = "text"
    related_symptoms: List[str]  = Field(default_factory=list)
    disease_id:       str        = ""
    disease_name:     str        = ""
    schema_options:   List[str]  = Field(default_factory=list)


class DiseaseCandidate(BaseModel):
    disease_id:        str
    disease_name:      str
    probability:       float              = 0.0
    confidence:        str               = "low"
    status:            str               = "stable"
    matched_symptoms:  List[str]         = Field(default_factory=list)
    evidence_for:      List[str]         = Field(default_factory=list)
    evidence_against:  List[str]         = Field(default_factory=list)
    pending_questions: List[QuestionItem] = Field(default_factory=list)
    asked_questions:   List[str]          = Field(default_factory=list)
    red_flags:         List[str]          = Field(default_factory=list)
    all_symptoms:      List[str]          = Field(default_factory=list)


class ChatState(BaseModel):
    phase:                Phase            = Phase.INTAKE
    intake_completed:     bool             = False
    patient:              PatientProfile   = Field(default_factory=PatientProfile)
    candidates:           List[DiseaseCandidate] = Field(default_factory=list)

    asked_question_texts: List[str]        = Field(default_factory=list)
    asked_question_words: List[Set[str]]   = Field(default_factory=list)

    current_question_item: Optional[QuestionItem] = None
    current_options:        List[str]              = Field(default_factory=list)

    last_user_answer:     str  = ""
    followup_rounds:      int  = 0
    max_followup_rounds:  int  = 12
    chat_history:         List[Dict] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_full_schema(doc) -> Dict:
    """
    Extract the full disease schema from a FAISS Document.

    Your vectorstore (vector_store_setup.py) stores the full JSON in
    doc.metadata["full_schema"] as a string. This function reads that.

    Fallback chain:
      1. metadata["full_schema"]  — string or dict  ← YOUR FORMAT (primary)
      2. page_content as JSON with question_flow    ← backup
      3. Minimal dict with id/name only             ← last resort (logs warning)
    """
    meta = getattr(doc, "metadata", {}) or {}

    # ── PRIMARY: metadata["full_schema"] — exactly how your vectorstore saves it ──
    raw = meta.get("full_schema")
    if raw:
        if isinstance(raw, dict):
            # Already parsed (shouldn't happen but handle it)
            if raw.get("question_flow"):
                return raw
        elif isinstance(raw, str) and raw.strip():
            try:
                schema = json.loads(raw)
                if isinstance(schema, dict):
                    return schema   # includes question_flow
            except Exception as e:
                logger.warning(f"get_full_schema: failed to parse metadata.full_schema for "
                                f"'{meta.get('name', '?')}': {e}")

    # ── BACKUP: page_content as JSON ──────────────────────────────────────────
    try:
        schema = json.loads(doc.page_content)
        if isinstance(schema, dict) and schema.get("question_flow"):
            return schema
    except Exception:
        pass

    # ── LAST RESORT: minimal dict (no question_flow → no disease Qs will be asked) ──
    logger.warning(
        f"get_full_schema: Could not find full_schema for "
        f"'{meta.get('name', 'unknown')}'. "
        f"Disease-specific questions will be skipped for this disease. "
        f"Ensure vector_store_setup.py stores full_schema in metadata."
    )
    return {
        "id":   meta.get("id", ""),
        "name": meta.get("name", "Unknown"),
    }


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower().strip())


_STOPWORDS = {
    "the","and","for","are","was","you","any","this","that","have","has","with",
    "been","did","does","your","from","they","when","how","what","can","not","its",
    "do","is","in","or","of","a","an","to","at","by","on","up","if","as","be",
}

def _word_set(text: str) -> Set[str]:
    return {w for w in _normalise(text).split() if len(w) > 2 and w not in _STOPWORDS}


def _is_duplicate_question(new_q: str, asked_word_sets: List[Set[str]]) -> bool:
    new_words = _word_set(new_q)
    if not new_words:
        return True
    for asked_words in asked_word_sets:
        if not asked_words:
            continue
        overlap = len(new_words & asked_words) / min(len(new_words), len(asked_words))
        if overlap >= 0.60:
            return True
    return False


def _ensure_blank_line_format(text: str) -> str:
    """Ensure acknowledgment and question are separated by exactly one blank line."""
    lines = text.strip().splitlines()
    non_empty = [l.strip() for l in lines if l.strip()]
    if len(non_empty) < 2:
        return text.strip()
    # If already has blank line separator, return as-is
    for i, line in enumerate(lines):
        if line.strip() == "" and i > 0 and i < len(lines) - 1:
            return text.strip()
    # Find sentence boundary and insert blank line
    joined = " ".join(non_empty)
    match = re.search(r'([.!?])\s+([A-Z])', joined)
    if match:
        ack  = joined[:match.start() + 1].strip()
        rest = joined[match.start() + 2:].strip()
        if ack and rest:
            return f"{ack}\n\n{rest}"
    return text.strip()


def _strip_json_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def _age_group(age: Optional[int]) -> str:
    if age is None: return "adult"
    if age <= 12:   return "child"
    if age <= 17:   return "teen"
    if age <= 35:   return "young adult"
    if age <= 60:   return "adult"
    return "older adult"


def _extract_all_symptom_names(schema: Dict) -> List[str]:
    names = []
    for cat in ["primary", "secondary", "distinguishing"]:
        for s in (schema.get("symptoms") or {}).get(cat) or []:
            if isinstance(s, dict):
                n = (s.get("name") or "").strip()
                if n:
                    names.append(n)
    return names


def _extract_red_flags(schema: Dict) -> List[str]:
    flags = []
    for cat in ["primary", "secondary"]:
        for s in (schema.get("symptoms") or {}).get(cat) or []:
            if isinstance(s, dict) and s.get("warning_sign"):
                n = (s.get("name") or "").strip()
                if n:
                    flags.append(n)
    for f in (schema.get("diagnosis_clues") or {}).get("red_flags") or []:
        if isinstance(f, str) and f.strip():
            flags.append(f.strip())
    return list(dict.fromkeys(flags))


# ══════════════════════════════════════════════════════════════════════════════
#  OPTION SETS — descriptive, first-person, no jargon
# ══════════════════════════════════════════════════════════════════════════════

_YES_NO          = ["Yes", "No", "Not sure"]
_GENDER          = ["Male", "Female", "Other / Prefer not to say"]
_ONSET           = ["Came on suddenly (within hours)", "Developed gradually over a few days", "Built up slowly over weeks"]
_DURATION        = ["Less than 24 hours", "2–3 days", "About a week", "Longer than a week"]
_PAIN_SEVERITY   = ["Very strong and constant — hard to ignore", "Moderate — uncomfortable but manageable", "Mild — barely noticeable"]
_PAIN_TYPE       = ["Sharp, stabbing or piercing pain", "Dull, throbbing ache", "Burning or stinging sensation", "Pressure or fullness feeling"]
_FREQUENCY       = ["Constant — doesn't go away", "Comes and goes every few hours", "Occasional — only sometimes"]
_EAR_SIDE        = ["Only in one ear", "Both ears equally", "Mostly one ear"]
_EAR_TIMING      = ["During takeoff", "During landing", "After the flight ended", "Not related to flying"]
_HEARING         = ["Muffled, like I'm underwater", "Ringing or buzzing in the ear", "Noticeable hearing reduction", "No hearing changes"]
_DIZZINESS       = ["Yes — room feels like it's spinning", "Mild lightheadedness only", "No dizziness at all"]
_DISCHARGE       = ["Clear or watery", "Yellow or green (pus-like)", "Brown or bloody", "No discharge"]
_CONTACT         = ["Yes — close contact with someone sick", "No contact with sick people", "Not sure"]
_IMPROVEMENT     = ["Yes — gradually getting better", "No change at all", "Getting worse", "Better then worse again"]
_COLD_SINUS      = ["Yes, I have a cold right now", "I had a cold recently (resolved)", "No cold or sinus issues", "I have allergies/hay fever"]
_BODY_LOCATION   = ["Left side only", "Right side only", "Both sides", "Central / hard to locate"]
_SKIN_LOCATION   = ["Neck", "Armpits", "Groin", "Multiple areas"]
_SKIN_TEXTURE    = ["Yes — thick and velvety", "No — feels like normal skin", "Not sure, just looks darker"]
_WEIGHT_CHANGE   = ["Yes, lost weight without trying", "Yes, gained weight recently", "No weight change"]
_MEDICATION      = ["Yes, I take regular medications", "No medications currently", "Unsure / need to check"]
_COUGH_TYPE      = ["Dry, tickling cough", "Wet cough with mucus", "Barking or wheezing cough"]


def _build_static_options(qi: QuestionItem) -> List[str]:
    """
    Return smart static options matching question content + expected_type.
    Returns [] if no pattern matches → caller uses LLM.
    """
    q = qi.question.lower()
    t = qi.expected_type.lower()

    # ── YES/NO type ──────────────────────────────────────────────────────────
    if t in ("yes/no", "boolean"):
        if any(w in q for w in ["dizzy", "dizziness", "spinning", "vertigo"]):
            return _DIZZINESS
        if any(w in q for w in ["blood", "bleeding", "discharge"]):
            return _DISCHARGE
        if any(w in q for w in ["thick", "velvety"]):
            return _SKIN_TEXTURE
        if any(w in q for w in ["weight loss", "losing weight", "lost weight", "weight"]):
            return _WEIGHT_CHANGE
        if any(w in q for w in ["medication", "medicine", "drug", "pill", "taking any"]):
            return _MEDICATION
        if any(w in q for w in ["hearing loss", "ringing", "tinnitus", "muffled"]):
            return _HEARING
        if any(w in q for w in ["cold", "sinus", "allerg", "infection", "ill", "sick"]):
            return _COLD_SINUS
        if any(w in q for w in ["improve", "better", "worse", "resolv"]):
            return _IMPROVEMENT
        return _YES_NO

    # ── CHOICE type ──────────────────────────────────────────────────────────
    if t in ("choice", "multiple_choice", "multiple choice", "select"):
        # Ear side
        if any(w in q for w in ["one ear", "both ear", "which ear", "ear or both"]):
            return _EAR_SIDE
        # Ear timing / flight
        if any(w in q for w in ["takeoff", "landing", "flight", "plane", "ascent", "descent", "begin during"]):
            return _EAR_TIMING
        # Hearing
        if any(w in q for w in ["hearing", "ringing", "tinnitus", "muffled", "dizziness"]):
            return _HEARING
        # Pain severity
        if any(w in q for w in ["severity", "severe", "how bad", "how much pain", "pain level", "rate"]):
            return _PAIN_SEVERITY
        # Skin itch/odor/tags
        if any(w in q for w in ["itch", "itching", "odor", "smell", "tag", "skin tag"]):
            return ["Itching only", "Skin odor only", "Skin tags only", "Multiple of these", "None"]
        # Discharge color
        if any(w in q for w in ["discharge", "fluid", "color", "colour", "drainage"]):
            return _DISCHARGE
        # Cold/sinus/allergy
        if any(w in q for w in ["cold", "sinus", "allerg", "upper respiratory"]):
            return _COLD_SINUS
        # Cough
        if any(w in q for w in ["cough"]):
            return _COUGH_TYPE
        # Location (skin)
        if any(w in q for w in ["where", "location", "area", "body", "which part"]):
            if any(w in q for w in ["ear"]):
                return _EAR_SIDE
            if any(w in q for w in ["skin", "patch", "dark"]):
                return _SKIN_LOCATION
            return _BODY_LOCATION

    # ── TEXT type — detect by question content ───────────────────────────────
    # Gender
    if any(p in q for p in ["gender", "male or female", "male, female", "are you male",
                              "boy or girl", "your sex", "identify as"]):
        return _GENDER

    # Onset
    if any(p in q for p in ["sudden", "gradually", "how did it start", "come on",
                              "onset", "begin", "start suddenly"]):
        return _ONSET

    # Duration
    if any(p in q for p in ["how long", "long have", "many days", "many weeks",
                              "since when", "duration", "how many"]):
        return _DURATION

    # Pain severity
    if any(p in q for p in ["how severe", "how bad", "pain level", "rate your",
                              "how intense", "how much pain", "severity of"]):
        return _PAIN_SEVERITY

    # Pain character
    if any(p in q for p in ["describe", "type of pain", "feel like", "kind of pain",
                              "nature of", "what does"]):
        if any(w in q for w in ["pain", "ache", "hurt", "discomfort", "feeling"]):
            return _PAIN_TYPE

    # Frequency/pattern
    if any(p in q for p in ["constant", "come and go", "intermittent", "how often",
                              "all the time", "frequency", "always"]):
        return _FREQUENCY

    # Ear side
    if any(p in q for p in ["which ear", "one ear", "both ear", "left ear", "right ear"]):
        return _EAR_SIDE

    # Ear timing
    if any(p in q for p in ["takeoff", "landing", "flight", "plane", "ascent", "descent"]):
        return _EAR_TIMING

    # Hearing
    if any(p in q for p in ["hearing", "ringing", "tinnitus", "muffled"]):
        return _HEARING

    # Improvement
    if any(p in q for p in ["improved", "better", "worse", "progress", "resolving"]):
        return _IMPROVEMENT

    # Generic yes/no-type phrasing
    if re.match(r"^(do you|have you|are you|is there|does|did you|is your|has the|can you)", q.strip()):
        return _YES_NO

    return []


# ══════════════════════════════════════════════════════════════════════════════
#  INTAKE OPTIONS — for questions asked during the intake phase
# ══════════════════════════════════════════════════════════════════════════════

def _intake_options(question_text: str) -> List[str]:
    """Return smart quick-reply hints for intake-phase questions."""
    q = question_text.lower()

    # Gender — must check BEFORE generic yes/no patterns
    if any(p in q for p in ["gender", "male or female", "male, female", "are you male",
                              "boy or girl", "identify as", "your sex", "male, female, or"]):
        return _GENDER

    # Onset
    if any(p in q for p in ["sudden", "gradually", "how did it start", "come on",
                              "onset", "did it start"]):
        return _ONSET

    # Duration
    if any(p in q for p in ["how long", "long have", "many days", "many weeks", "duration"]):
        return _DURATION

    # Pain severity
    if any(p in q for p in ["how severe", "how bad", "rate your", "pain level", "intensity"]):
        return _PAIN_SEVERITY

    # Pain type
    if any(p in q for p in ["describe", "type of pain", "feel like", "kind of pain"]):
        return _PAIN_TYPE

    # Explicit yes/no questions — only AFTER gender check
    if re.match(r"^(do you|have you|are you|is there|does|did you|is your|has)", q.strip()):
        return _YES_NO

    return []


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CHATBOT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class MedicalChatbot:
    """
    Medical triage chatbot.

    chat() returns:
      {
        "message":         str        — clean text, no JSON leakage
        "options":         List[str]  — 2-3 quick-reply hints
        "phase":           str        — intake | search | followup | result
        "is_red_flag":     bool       — True if current question is red-flag type
        "disease_context": str        — name of disease being investigated (followup only)
      }
    """

    def __init__(
        self,
        vectorstore,
        groq_api_key:         str,
        top_k:                int   = 20,
        final_top_n:          int   = 4,
        max_followup_rounds:  int   = 12,
        similarity_threshold: float = 0.0,
    ):
        self.vectorstore          = vectorstore
        self.top_k                = top_k
        self.final_top_n          = final_top_n
        self.similarity_threshold = similarity_threshold
        self.state                = ChatState(max_followup_rounds=max_followup_rounds)

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b", temperature=0.25, api_key=groq_api_key
        )
        self.llm_ranker = ChatGroq(
            model="openai/gpt-oss-120b", temperature=0.0, api_key=groq_api_key
        )
        self.llm_final = ChatGroq(
            model="openai/gpt-oss-120b", temperature=0.3, api_key=groq_api_key
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> Dict:
        user_message = user_message.strip()
        if not user_message:
            return self._response(
                "I didn't catch that — could you share how you're feeling?", []
            )

        self.state.chat_history.append({"role": "user", "content": user_message})
        self.state.last_user_answer = user_message
        self.state.current_options  = []

        try:
            if self.state.phase == Phase.INTAKE:
                message, options = self._handle_intake(user_message)
            elif self.state.phase == Phase.SEARCH:
                message, options = self._run_search_and_start_followup()
            elif self.state.phase == Phase.FOLLOWUP:
                message, options = self._handle_followup(user_message)
            elif self.state.phase == Phase.RESULT:
                message = (
                    "Your health assessment is shown above. "
                    "Please take it to a licensed doctor to confirm and get proper treatment. 🙏"
                )
                options = []
            else:
                message = "Something went wrong. Please refresh and try again."
                options = []
        except Exception as e:
            logger.error(f"chat() error: {e}", exc_info=True)
            message = "I ran into a small issue. Could you repeat that?"
            options = []

        self.state.current_options = options
        self.state.chat_history.append({"role": "assistant", "content": message})

        qi            = self.state.current_question_item
        is_red_flag   = qi is not None and qi.q_type == "red_flag_screening_questions"
        disease_ctx   = qi.disease_name if qi else ""

        return self._response(message, options, is_red_flag=is_red_flag, disease_context=disease_ctx)

    def _response(
        self, message: str, options: List[str],
        is_red_flag: bool = False, disease_context: str = ""
    ) -> Dict:
        return {
            "message":         message,
            "options":         options,
            "phase":           self.state.phase.value,
            "is_red_flag":     is_red_flag,
            "disease_context": disease_context,
        }

    def get_rankings(self) -> List[Dict]:
        return [
            {
                "rank":             i + 1,
                "disease":          c.disease_name,
                "probability":      f"{round(c.probability * 100, 1)}%",
                "confidence":       c.confidence,
                "status":           c.status,
                "evidence_for":     c.evidence_for[:3],
                "evidence_against": c.evidence_against[:2],
            }
            for i, c in enumerate(self.state.candidates[:10])
        ]

    def get_phase(self) -> str:
        return self.state.phase.value

    def reset(self):
        self.state = ChatState(max_followup_rounds=self.state.max_followup_rounds)
        logger.info("Chatbot state reset.")

    # ── INTAKE ────────────────────────────────────────────────────────────────

    def _handle_intake(self, user_message: str) -> Tuple[str, List[str]]:
        # Build message history for the intake LLM
        # IMPORTANT: pass the FULL conversation so the LLM knows what's been collected
        messages = [SystemMessage(content=INTAKE_SYSTEM_PROMPT)]
        for msg in self.state.chat_history[:-1]:
            cls = HumanMessage if msg["role"] == "user" else AIMessage
            messages.append(cls(content=msg["content"]))
        messages.append(HumanMessage(content=user_message))

        try:
            response_text = self.llm.invoke(messages).content
        except Exception as e:
            logger.error(f"Intake LLM error: {e}")
            return "I'm having a little trouble. Could you repeat that?", []

        # ── Check for completion marker ──────────────────────────────────────
        if "[INTAKE_COMPLETE:" in response_text:
            data = self._parse_intake_json(response_text)
            if data and self._intake_data_valid(data):
                self._populate_patient(data)
                self.state.intake_completed = True
                self.state.phase = Phase.SEARCH

                # Strip the marker — only show text before it
                marker_pos = response_text.index("[INTAKE_COMPLETE:")
                visible = response_text[:marker_pos].strip()
                # Clean any residual JSON/tag artifacts
                visible = re.sub(r'\[INTAKE_COMPLETE:.*', '', visible, flags=re.DOTALL).strip()
                if not visible:
                    visible = "Thank you for all of that — let me look into this for you now."

                search_msg, search_opts = self._run_search_and_start_followup()
                return visible + "\n\n" + search_msg, search_opts

            else:
                # Invalid JSON (age placeholder etc.) — strip and continue intake
                marker_pos = response_text.index("[INTAKE_COMPLETE:")
                visible = response_text[:marker_pos].strip()
                if not visible:
                    visible = "Could you please tell me your age as a number?"
                lines  = [l.strip() for l in visible.splitlines() if l.strip()]
                q_text = lines[-1] if lines else visible
                return visible, _intake_options(q_text)

        # ── Normal intake turn — generate appropriate options ────────────────
        lines  = [l.strip() for l in response_text.strip().splitlines() if l.strip()]
        q_text = lines[-1] if lines else response_text
        return response_text, _intake_options(q_text)

    def _intake_data_valid(self, data: Dict) -> bool:
        """Reject if any mandatory field is missing or contains a placeholder."""
        # Age
        age = data.get("age")
        if age is None:
            return False
        if isinstance(age, str):
            if any(c in age for c in ["<", ">", "int", "null"]):
                return False
            try:
                data["age"] = int(float(age))
            except (ValueError, TypeError):
                return False
        elif not isinstance(age, (int, float)):
            return False

        # Gender
        gender = str(data.get("gender") or "").strip().lower()
        if not gender or gender in ("", "null", "none", "<str>", "str") or "<" in gender:
            return False

        # Symptoms
        symptoms = data.get("symptoms_list", [])
        if not isinstance(symptoms, list) or len(symptoms) == 0:
            return False
        if not any(isinstance(s, str) and s.strip() for s in symptoms):
            return False

        # Duration
        duration = str(data.get("symptom_duration") or "").strip()
        if not duration or duration.lower() in ("null", "none", "", "str"):
            return False

        # Onset
        onset = str(data.get("onset") or "").strip().lower()
        if not onset or onset in ("null", "none", "", "unknown_placeholder"):
            return False

        return True

    def _parse_intake_json(self, text: str) -> Optional[Dict]:
        decoder = json.JSONDecoder()
        pos = 0
        while True:
            try:
                tag_pos = text.index("[INTAKE_COMPLETE:", pos)
            except ValueError:
                return None
            after = tag_pos + len("[INTAKE_COMPLETE:")
            brace = text.find("{", after)
            if brace == -1:
                pos = after
                continue
            try:
                obj, _ = decoder.raw_decode(text[brace:])
                if isinstance(obj, dict):
                    return obj
            except Exception as e:
                logger.warning(f"Intake JSON parse error: {e}")
            pos = after

    def _populate_patient(self, data: Dict):
        p = self.state.patient
        age_val = data.get("age")
        if isinstance(age_val, str):
            try:
                age_val = int(float(age_val))
            except Exception:
                age_val = None
        elif isinstance(age_val, float):
            age_val = int(age_val)
        p.age                   = age_val
        p.gender                = data.get("gender")
        p.symptom_duration      = data.get("symptom_duration")
        p.onset                 = data.get("onset")
        p.location_recent       = data.get("location_recent")
        p.existing_conditions   = data.get("existing_conditions")
        p.contact_with_sick     = data.get("contact_with_sick")
        p.lifestyle_factors     = data.get("lifestyle_factors")
        p.family_history        = data.get("family_history")
        p.reproductive_context  = data.get("reproductive_context")
        p.distinguishing_detail = data.get("distinguishing_detail")
        raw = data.get("symptoms_list", [])
        p.symptoms = [s.strip() for s in raw if isinstance(s, str) and s.strip()]
        logger.info(f"Patient complete: symptoms={p.symptoms}, age={p.age}, gender={p.gender}, "
                    f"duration={p.symptom_duration}, onset={p.onset}")

    # ── SEARCH ────────────────────────────────────────────────────────────────

    def _run_search_and_start_followup(self) -> Tuple[str, List[str]]:
        queries  = self._build_search_queries()
        combined: Dict[str, DiseaseCandidate] = {}

        for query in queries:
            try:
                results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
            except Exception as e:
                logger.error(f"FAISS error for query '{query}': {e}")
                continue

            for doc, score in results:
                similarity = 1.0 / (1.0 + max(float(score), 0.0))
                if similarity < self.similarity_threshold:
                    continue

                schema     = get_full_schema(doc)
                disease_id = doc.metadata.get("id", "") or schema.get("id", "")
                key        = disease_id or doc.metadata.get("name", "Unknown")

                has_qflow = bool(schema.get("question_flow"))
                logger.info(
                    f"  Doc: '{schema.get('name', key)}' | "
                    f"has_question_flow={has_qflow} | "
                    f"page_content_type={type(doc.page_content).__name__} | "
                    f"page_content_len={len(str(doc.page_content))} | "
                    f"similarity={round(similarity,3)}"
                )

                if key in combined:
                    if similarity > combined[key].probability:
                        combined[key].probability = round(similarity, 4)
                    continue

                questions = self._extract_questions(schema)
                logger.info(
                    f"  → '{schema.get('name', key)}': extracted {len(questions)} questions "
                    f"(initial={sum(1 for q in questions if q.q_type=='initial_questions')}, "
                    f"refinement={sum(1 for q in questions if q.q_type=='refinement_questions')}, "
                    f"red_flag={sum(1 for q in questions if q.q_type=='red_flag_screening_questions')})"
                )

                combined[key] = DiseaseCandidate(
                    disease_id        = disease_id,
                    disease_name      = doc.metadata.get("name", schema.get("name", "Unknown")),
                    probability       = round(similarity, 4),
                    matched_symptoms  = self._find_symptom_overlap(schema),
                    pending_questions = questions,
                    red_flags         = _extract_red_flags(schema),
                    all_symptoms      = _extract_all_symptom_names(schema),
                )

        if not combined:
            logger.warning("No candidates from FAISS.")
            self.state.phase = Phase.RESULT
            return self._generate_final_result(), []

        candidates = sorted(
            combined.values(),
            key=lambda c: (len(c.matched_symptoms) > 0, c.probability),
            reverse=True,
        )
        self.state.candidates = candidates[:self.top_k]
        self.state.phase      = Phase.FOLLOWUP

        top = self.state.candidates[0]
        total_q = sum(len(c.pending_questions) for c in self.state.candidates)
        logger.info(
            f"Search complete: {len(self.state.candidates)} candidates, "
            f"{total_q} total questions available. "
            f"Top: '{top.disease_name}' ({round(top.probability*100,1)}%) "
            f"with {len(top.pending_questions)} questions."
        )

        if total_q == 0:
            logger.error(
                "CRITICAL: No questions extracted from ANY candidate. "
                "This means metadata['full_schema'] is missing or unparseable. "
                "Check that disease_to_chunks() in your vectorstore code sets "
                "metadata={'full_schema': json.dumps(disease)} for each chunk."
            )

        first_q = self._pick_next_question()
        if first_q:
            logger.info(f"First Q from '{first_q.disease_name}' [{first_q.q_type}]: {first_q.question}")
            rephrased = self._rephrase_question(first_q, previous_answer="")
            options   = self._generate_options(first_q)
            self._commit_question(first_q)
            msg = (
                f"I've matched your symptoms to **{len(self.state.candidates)} possible conditions**. "
                f"I'll now ask you some specific questions to narrow things down.\n\n"
                f"{rephrased}"
            )
            return msg, options
        else:
            logger.warning(
                "No first question found. Either all questions are duplicates or "
                "no questions were extracted. Going to final result."
            )
            self.state.phase = Phase.RESULT
            return self._generate_final_result(), []

    def _build_search_queries(self) -> List[str]:
        profile = self._profile_to_text(include_qa=False)
        prompt  = QUERY_GENERATION_PROMPT.format(patient_profile=profile)
        try:
            raw  = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            data = json.loads(_strip_json_fences(raw))
            qs   = [str(data.get(k, "")).strip() for k in ("query_1", "query_2", "query_3")]
            qs   = [q for q in qs if q]
            if qs:
                logger.info(f"Search queries: {qs}")
                return qs
        except Exception as e:
            logger.warning(f"Query gen failed: {e}")
        fallback = ", ".join(self.state.patient.symptoms) or "general illness"
        return [fallback]

    def _find_symptom_overlap(self, schema: Dict) -> List[str]:
        if not schema:
            return []
        reported = {s.lower() for s in self.state.patient.symptoms}
        matched  = []
        for cat in ["primary", "secondary", "distinguishing"]:
            for s in (schema.get("symptoms") or {}).get(cat) or []:
                if not isinstance(s, dict):
                    continue
                name = (s.get("name") or "").lower()
                if name and any(r in name or name in r for r in reported):
                    matched.append(name)
        return list(dict.fromkeys(matched))

    def _extract_questions(self, schema: Dict) -> List[QuestionItem]:
        """
        Extract ALL questions from disease schema question_flow.
        Order: initial_questions → refinement_questions → red_flag_screening_questions
        """
        if not schema:
            return []

        questions    = []
        disease_id   = schema.get("id") or ""
        disease_name = schema.get("name") or ""
        qflow        = schema.get("question_flow") or {}

        if not qflow:
            logger.warning(f"No question_flow found for disease '{disease_name}'")
            return []

        for q_type in ["initial_questions", "refinement_questions", "red_flag_screening_questions"]:
            section = qflow.get(q_type) or []
            for q in section:
                if not isinstance(q, dict):
                    continue
                text = (q.get("question") or "").strip()
                if not text:
                    continue

                raw_type = (q.get("expected_answer_type") or "text").lower().strip()
                if raw_type in ("yes/no", "boolean", "yn"):
                    ans_type = "yes/no"
                elif raw_type in ("choice", "multiple_choice", "multiple choice", "select"):
                    ans_type = "choice"
                else:
                    ans_type = "text"

                raw_related = q.get("related_symptoms") or []
                related = []
                for r in raw_related:
                    if isinstance(r, str) and r.strip():
                        related.append(r.strip())
                    elif isinstance(r, dict):
                        n = r.get("name") or r.get("symptom") or ""
                        if n.strip():
                            related.append(n.strip())

                schema_opts = [
                    opt.strip() for opt in (q.get("options") or q.get("answer_options") or [])
                    if isinstance(opt, str) and opt.strip()
                ]

                questions.append(QuestionItem(
                    question         = text,
                    purpose          = (q.get("purpose") or "").strip(),
                    q_type           = q_type,
                    expected_type    = ans_type,
                    related_symptoms = related,
                    disease_id       = disease_id,
                    disease_name     = disease_name,
                    schema_options   = schema_opts,
                ))
                logger.debug(f"  [{q_type}] {disease_name}: '{text}' (type={ans_type})")

        logger.info(f"Extracted {len(questions)} questions from '{disease_name}'")
        return questions

    # ── FOLLOWUP ──────────────────────────────────────────────────────────────

    def _handle_followup(self, user_answer: str) -> Tuple[str, List[str]]:
        if not self.state.current_question_item:
            self.state.phase = Phase.RESULT
            return self._generate_final_result(), []

        qi = self.state.current_question_item
        self.state.patient.qa_log.append((qi.question, user_answer))
        self._rerank(qi=qi, answer=user_answer)
        self.state.followup_rounds += 1

        if self._should_stop():
            self.state.phase = Phase.RESULT
            return self._generate_final_result(), []

        next_q = self._pick_next_question()
        if next_q:
            rephrased = self._rephrase_question(next_q, previous_answer=user_answer)
            options   = self._generate_options(next_q)
            self._commit_question(next_q)
            return rephrased, options
        else:
            self.state.phase = Phase.RESULT
            return self._generate_final_result(), []

    def _generate_options(self, qi: QuestionItem) -> List[str]:
        """
        Priority: schema_options → static keywords → LLM → yes/no fallback
        """
        # 1. Schema-provided options (most authoritative)
        if qi.schema_options and len(qi.schema_options) >= 2:
            return qi.schema_options[:3]

        # 2. Static keyword matching (instant, no LLM)
        static = _build_static_options(qi)
        if static:
            return static

        # 3. LLM generation with full disease context
        prompt = SMART_OPTIONS_PROMPT.format(
            question         = qi.question,
            answer_type      = qi.expected_type,
            related_symptoms = ", ".join(qi.related_symptoms) if qi.related_symptoms else "not specified",
            disease_name     = qi.disease_name,
            purpose          = qi.purpose,
        )
        try:
            raw  = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            data = json.loads(_strip_json_fences(raw))
            opts = data.get("options", [])
            if isinstance(opts, list):
                cleaned = [str(o).strip() for o in opts if str(o).strip()]
                if len(cleaned) >= 2:
                    return cleaned[:3]
        except Exception as e:
            logger.warning(f"LLM options failed: {e}")

        # 4. Final fallback
        if qi.expected_type == "yes/no":
            return _YES_NO
        return []

    def _rephrase_question(self, qi: QuestionItem, previous_answer: str) -> str:
        """Rephrase a schema question into warm conversational language."""
        disease_context = f"{qi.disease_name}"
        if qi.related_symptoms:
            disease_context += f" (probing: {', '.join(qi.related_symptoms)})"

        prompt = FOLLOWUP_QUESTION_PROMPT.format(
            raw_question    = qi.question,
            purpose         = qi.purpose,
            previous_answer = previous_answer if previous_answer else "(starting — no previous answer)",
            disease_context = disease_context,
        )
        try:
            raw = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            return _ensure_blank_line_format(raw)
        except Exception as e:
            logger.error(f"Rephrase error: {e}")
            return f"Thanks for that.\n\n{qi.question}"

    def _rerank(self, qi: QuestionItem, answer: str):
        """Update disease probabilities based on new Q&A."""
        top15 = [
            {
                "disease_id":        c.disease_id,
                "disease_name":      c.disease_name,
                "probability_score": round(c.probability, 4),
                "confidence":        c.confidence,
                "status":            c.status,
                "evidence_for":      c.evidence_for,
                "evidence_against":  c.evidence_against,
            }
            for c in self.state.candidates[:15]
        ]
        prompt = RERANKING_PROMPT.format(
            patient_profile  = self._profile_to_text(include_qa=True),
            question         = qi.question,
            answer           = answer,
            purpose          = qi.purpose,
            related_symptoms = ", ".join(qi.related_symptoms) if qi.related_symptoms else "general",
            source_disease   = qi.disease_name,
            candidates_json  = json.dumps(top15, indent=2),
        )
        try:
            raw    = self.llm_ranker.invoke([HumanMessage(content=prompt)]).content.strip()
            result = json.loads(_strip_json_fences(raw))
            score_map = {
                item["disease_id"]: item
                for item in result.get("updated_candidates", [])
                if isinstance(item, dict) and "disease_id" in item
            }
            for c in self.state.candidates:
                if c.disease_id in score_map:
                    upd = score_map[c.disease_id]
                    c.probability      = round(max(0.0, min(1.0, float(upd.get("probability_score", c.probability)))), 4)
                    c.confidence       = upd.get("confidence", c.confidence)
                    c.status           = upd.get("status", c.status)
                    c.evidence_for     = upd.get("evidence_for", c.evidence_for)
                    c.evidence_against = upd.get("evidence_against", c.evidence_against)
            self.state.candidates.sort(key=lambda x: x.probability, reverse=True)
            top = self.state.candidates[0]
            logger.info(f"Rerank: '{top.disease_name}' {round(top.probability*100,1)}% [{top.status}]")
        except Exception as e:
            logger.warning(f"Rerank failed (non-fatal): {e}")

    def _pick_next_question(self) -> Optional[QuestionItem]:
        """
        Strict order:
        1. ALL initial questions from top-1 candidate (confirms or rules it out fast)
        2. ALL initial questions from top-3
        3. Refinement questions from top-3
        4. Initial questions from top-10
        5. Refinement questions from top-10
        6. Red-flag questions from top-3
        7. Red-flag questions from top-10
        Skip: ruled-out candidates, duplicate questions
        """
        asked = self.state.asked_question_words
        top1  = [c for c in self.state.candidates[:1]  if c.status != "ruled_out"]
        top3  = [c for c in self.state.candidates[:3]  if c.status != "ruled_out"]
        top10 = [c for c in self.state.candidates[:10] if c.status != "ruled_out"]

        plan = [
            (top1,  "initial_questions"),
            (top3,  "initial_questions"),
            (top3,  "refinement_questions"),
            (top10, "initial_questions"),
            (top10, "refinement_questions"),
            (top3,  "red_flag_screening_questions"),
            (top10, "red_flag_screening_questions"),
        ]

        for tier, q_type in plan:
            for c in tier:
                for qi in c.pending_questions:
                    if qi.q_type != q_type:
                        continue
                    if _is_duplicate_question(qi.question, asked):
                        continue
                    if qi.question in c.asked_questions:
                        continue
                    return qi
        return None

    def _commit_question(self, qi: QuestionItem):
        """Mark question as asked and remove from pending list."""
        self.state.current_question_item = qi
        if qi.question not in self.state.asked_question_texts:
            self.state.asked_question_texts.append(qi.question)
            self.state.asked_question_words.append(_word_set(qi.question))
        for c in self.state.candidates:
            if c.disease_id != qi.disease_id:
                continue
            for pending in list(c.pending_questions):
                if pending.question == qi.question:
                    c.asked_questions.append(qi.question)
                    c.pending_questions.remove(pending)
                    break

    def _should_stop(self) -> bool:
        if self.state.followup_rounds >= self.state.max_followup_rounds:
            logger.info("Stop: max rounds.")
            return True
        active = [c for c in self.state.candidates if c.status != "ruled_out"]
        if self.state.followup_rounds >= 4 and len(active) >= 2:
            if active[0].probability > 0.88 and (active[0].probability - active[1].probability) > 0.30:
                logger.info(f"Early stop: '{active[0].disease_name}' decisive.")
                return True
        asked = self.state.asked_question_words
        for c in active[:10]:
            for qi in c.pending_questions:
                if not _is_duplicate_question(qi.question, asked):
                    return False
        logger.info("Stop: no more unique questions.")
        return True

    # ── FINAL RESULT ──────────────────────────────────────────────────────────

    def _generate_final_result(self) -> str:
        top_n = self.state.candidates[:self.final_top_n]
        if not top_n:
            return (
                "I wasn't able to find a strong match in the database. "
                "Please consult a licensed doctor. 🙏"
            )

        top_candidates_json = json.dumps(
            [
                {
                    "rank":             i + 1,
                    "disease_name":     c.disease_name,
                    "probability":      f"{round(c.probability * 100, 1)}%",
                    "confidence":       c.confidence,
                    "status":           c.status,
                    "evidence_for":     c.evidence_for,
                    "evidence_against": c.evidence_against,
                    "matched_symptoms": c.matched_symptoms,
                }
                for i, c in enumerate(top_n)
            ],
            indent=2,
        )

        qa_summary = ""
        if self.state.patient.qa_log:
            parts = []
            for i, (q, a) in enumerate(self.state.patient.qa_log, 1):
                parts.append(f"  Q{i}: {q}\n  A{i}: {a}")
            qa_summary = "\n".join(parts)
        else:
            qa_summary = "No follow-up Q&A recorded."

        prompt = FINAL_DIAGNOSIS_PROMPT.format(
            n                   = self.final_top_n,
            patient_profile     = self._profile_to_text(include_qa=True),
            qa_summary          = qa_summary,
            top_candidates_json = top_candidates_json,
        )

        try:
            return self.llm_final.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            logger.error(f"Final report error: {e}")
            top = top_n[0]
            return (
                f"## 🩺 Assessment Summary\n\n"
                f"Most likely condition: **{top.disease_name}** "
                f"({round(top.probability * 100, 1)}% likelihood).\n\n"
                f"**Please see a doctor to confirm.** If symptoms worsen, seek care today. 🙏"
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _profile_to_text(self, include_qa: bool = True) -> str:
        p = self.state.patient
        age_label = f"{p.age} ({_age_group(p.age)})" if p.age else "Not specified"
        lines = [
            f"Symptoms              : {', '.join(p.symptoms) if p.symptoms else 'Not specified'}",
            f"Age                   : {age_label}",
            f"Gender                : {p.gender or 'Not specified'}",
            f"Duration              : {p.symptom_duration or 'Not specified'}",
            f"Onset                 : {p.onset or 'Not specified'}",
            f"Recent location/travel: {p.location_recent or 'None reported'}",
            f"Existing conditions   : {p.existing_conditions or 'None reported'}",
            f"Contact with sick     : {p.contact_with_sick or 'Unknown'}",
            f"Lifestyle factors     : {p.lifestyle_factors or 'Not collected'}",
            f"Family history        : {p.family_history or 'Not collected'}",
            f"Reproductive context  : {p.reproductive_context or 'Not applicable'}",
            f"Distinguishing detail : {p.distinguishing_detail or 'Not collected'}",
        ]
        if include_qa and p.qa_log:
            lines.append("\nDisease-specific Q&A:")
            for i, (q, a) in enumerate(p.qa_log, 1):
                lines.append(f"  Q{i}: {q}")
                lines.append(f"  A{i}: {a}")
        return "\n".join(lines)