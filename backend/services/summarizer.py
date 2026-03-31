import json
import re
from backend.services.groq_client import groq

# DELETE these two lines from original:
# from backend.config import GROQ_API_KEY, GROQ_MODEL
# client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a clinical summarization expert.
Given a conversation between a patient and a medical chatbot, extract a structured clinical profile.
Return ONLY a valid JSON object — no explanation, no markdown, no backticks.
"""

def build_summary_prompt(conversation: list[dict], user_profile: dict | None = None) -> str:
    convo_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in conversation
    ])

    schema = {
        "reported_symptoms": ["list of all symptoms mentioned by patient, normalized to medical terms"],
        "symptom_duration": "e.g. 3 days / 1 week / since yesterday",
        "onset_pattern": "sudden or gradual, any triggers mentioned",
        "body_regions_affected": ["e.g. chest", "throat", "abdomen"],
        "patient_profile": {
            "age": "extracted age or null",
            "sex": "male/female/other or null",
            "weight": "extracted weight or null"
        },
        "risk_signals": ["travel history", "contact with sick person", "food consumed", "any triggers"],
        "severity_impression": "mild / moderate / severe — based on how patient describes symptoms",
        "additional_context": "any other clinically relevant details mentioned"
    }

    profile_block = json.dumps(user_profile or {}, indent=2)

    return f"""Here is the patient-chatbot conversation:

{convo_text}

Trusted user profile from system records (prefer this over uncertain extraction):
{profile_block}

Extract a structured clinical profile from this conversation.
Fill this exact JSON schema:
{json.dumps(schema, indent=2)}

Rules:
1. Return ONLY the JSON object
2. reported_symptoms must use proper medical terms where possible
3. If information is not mentioned set it to null
4. body_regions_affected must be a list of body parts mentioned or implied
5. severity_impression must be exactly: mild / moderate / severe
6. If age/sex/weight are present in trusted user profile, use those values in patient_profile
"""

def generate_user_summary(conversation: list[dict], user_profile: dict | None = None) -> dict:
    try:
        # ── only change: client.chat.completions.create → groq.chat ──
        raw = groq.chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_summary_prompt(conversation, user_profile=user_profile)}
            ],
            temperature = 0.1,
            max_tokens  = 1500,
        )

        if raw.startswith("```"):
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

        summary = json.loads(raw)
        
        # Validate required fields exist
        required_fields = {
            "reported_symptoms": [],
            "symptom_duration": None,
            "onset_pattern": None,
            "body_regions_affected": [],
            "patient_profile": {"age": None, "sex": None, "weight": None},
            "risk_signals": [],
            "severity_impression": "moderate",
            "additional_context": None
        }
        
        # Merge with defaults and validate types
        for key, default_val in required_fields.items():
            if key not in summary:
                summary[key] = default_val
            if key == "reported_symptoms" and not isinstance(summary[key], list):
                summary[key] = []
            if key == "body_regions_affected" and not isinstance(summary[key], list):
                summary[key] = []
            if key == "risk_signals" and not isinstance(summary[key], list):
                summary[key] = []
            if key == "severity_impression" and summary[key] not in ["mild", "moderate", "severe"]:
                summary[key] = "moderate"
        
        return summary

    except Exception as e:
        print(f"[Summarizer Error] {e}")
        return {
            "reported_symptoms": [],
            "symptom_duration": None,
            "onset_pattern": None,
            "body_regions_affected": [],
            "patient_profile": {"age": None, "sex": None, "weight": None},
            "risk_signals": [],
            "severity_impression": "moderate",
            "additional_context": None
        }