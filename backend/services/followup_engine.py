
import json
import re
from backend.config import (
    METADATA_PATH,
    MAX_FOLLOWUP_QUESTIONS, EARLY_EXIT_CONFIDENCE, EARLY_EXIT_GAP,
    normalize_disease_name
)
from backend.services.groq_client import groq

# Load full metadata with error handling
ALL_DISEASES = {}
ALL_DISEASES_NORMALIZED = {}  # Normalized name → original disease data
try:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        diseases_list = json.load(f)
        ALL_DISEASES = {d["name"]: d for d in diseases_list}
        # Create normalized mapping for robust lookups
        ALL_DISEASES_NORMALIZED = {normalize_disease_name(d["name"]): d for d in diseases_list}
    print(f"[FollowupEngine] Loaded {len(ALL_DISEASES)} diseases from metadata")
except FileNotFoundError:
    print(f"[ERROR] Metadata file not found at {METADATA_PATH}")
    print("[ERROR] Disease metadata is required. Please run generate_metadata.py and seed_db.py")
except json.JSONDecodeError as e:
    print(f"[ERROR] Invalid JSON in metadata file: {e}")
except Exception as e:
    print(f"[ERROR] Failed to load metadata: {e}")

# ─── SELECT BEST QUESTION ─────────────────────────────────
def select_next_question(
    top5: list[dict],
    asked_questions: list[str]
) -> dict | None:
    """
    From the question_flow of the top 5 diseases,
    select the most discriminating question not yet asked.
    A question is discriminating if it appears in multiple
    diseases' question flows — meaning the answer helps
    differentiate between them.
    """
    question_scores = {}  # question_text -> {count, disease, question_obj}

    for disease_entry in top5:
        disease_name = disease_entry["name"]
        # Normalize name for lookup to handle name variations
        normalized_name = normalize_disease_name(disease_name)
        disease_data = ALL_DISEASES_NORMALIZED.get(normalized_name) or ALL_DISEASES.get(disease_name, {})
        qflow        = disease_data.get("question_flow", {})

        all_questions = (
            qflow.get("initial_questions", []) +
            qflow.get("refinement_questions", []) +
            qflow.get("red_flag_screening_questions", [])
        )

        for q in all_questions:
            if not isinstance(q, dict):
                continue
            q_text = q.get("question", "")
            if not q_text or q_text in asked_questions:
                continue

            if q_text not in question_scores:
                question_scores[q_text] = {
                    "count":    0,
                    "disease":  disease_name,
                    "question": q
                }
            question_scores[q_text]["count"] += 1

    if not question_scores:
        return None

    # Pick question that appears across most diseases (most discriminating)
    best = max(question_scores.values(), key=lambda x: x["count"])
    return best["question"]

# ─── GENERATE INITIAL STAGE 2 QUESTIONS ───────────────────
def generate_next_stage2_question(
    summary: dict, 
    conversation: list[dict],
    question_index: int = 1,
    max_stage2_questions: int = 3
) -> str | None:
    """
    Dynamically generate ONE Stage 2 profiling question at a time.
    question_index: 1, 2, 3, etc. (which question number to generate)
    max_stage2_questions: total number of Stage 2 questions to ask (default 3)
    
    Returns the question or None if max questions reached.
    """
    if question_index > max_stage2_questions:
        return None

    # Build conversation context
    convo_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in conversation
    ])

    prompt = f"""You are a clinical triage expert. Based on the patient's initial symptom description, 
generate ONE targeted profiling question (question #{question_index} of {max_stage2_questions}) to understand their condition better.

Patient's stated symptoms/concerns:
{json.dumps(summary, indent=2)}

Conversation so far:
{convo_text}

Important Rules:
1. Generate ONLY ONE question that is NOT already answered in the conversation
2. Focus on timeline, severity, and context (duration, onset, related factors)
3. Avoid repeating information the patient already provided
4. Keep the question concise and natural
5. Return ONLY the question text, nothing else

Generate question #{question_index}:
"""

    try:
        raw = groq.chat(
            messages=[
                {"role": "system", "content": "You are a clinical triage expert. Generate one clear, concise question. Return only the question text."},
                {"role": "user",   "content": prompt}
            ],
            temperature = 0.1,
            max_tokens  = 150,
        )
        
        return raw.strip()

    except Exception as e:
        print(f"[Stage 2 Generation Error] {e}")
        # Fallback questions if generation fails
        fallback_questions = [
            "How many days have you been experiencing these symptoms?",
            "Did the symptoms start suddenly or gradually?",
            "Have you tried any treatments or noticed anything that makes it better or worse?"
        ]
        # Validate index bounds
        if 0 <= question_index - 1 < len(fallback_questions):
            return fallback_questions[question_index - 1]
        print(f"[WARNING] Fallback question index {question_index} out of bounds")
        return None

# ─── RE-RANK AFTER ANSWER ─────────────────────────────────
def rerank_after_answer(
    top5: list[dict],
    conversation: list[dict],
    summary: dict
) -> list[dict]:
    """
    After each follow-up answer, pass the full conversation
    to Groq to re-score the top 5 diseases.
    """
    # Build disease profiles for prompt
    disease_profiles = []
    for entry in top5:
        name    = entry["name"]
        # Normalize name for lookup to handle name variations
        normalized_name = normalize_disease_name(name)
        disease = ALL_DISEASES_NORMALIZED.get(normalized_name) or ALL_DISEASES.get(name, {})
        qflow   = disease.get("question_flow", {})

        all_questions = (
            qflow.get("initial_questions", []) +
            qflow.get("refinement_questions", []) +
            qflow.get("red_flag_screening_questions", [])
        )

        symptoms       = disease.get("symptoms", {})
        primary        = symptoms.get("primary", [])        if isinstance(symptoms, dict) else []
        primary_names  = [s["name"] for s in primary        if isinstance(s, dict)]
        distinguishing = symptoms.get("distinguishing", []) if isinstance(symptoms, dict) else []
        dist_names     = [s["name"] for s in distinguishing if isinstance(s, dict)]

        disease_profiles.append({
            "name":                 name,
            "current_confidence":   entry.get("confidence", 0.5),
            "primary_symptoms":     primary_names,
            "distinguishing":       dist_names,
            "expected_qa_patterns": [
                {
                    "question":        q.get("question", ""),
                    "positive_answer": q.get("expected_positive_answer", ""),
                    "purpose":         q.get("purpose", "")
                }
                for q in all_questions if isinstance(q, dict)
            ]
        })

    # Build conversation text
    convo_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in conversation
    ])

    prompt = f"""You are a clinical decision support expert performing differential diagnosis.

Patient clinical summary:
{json.dumps(summary, indent=2)}

Full conversation so far:
{convo_text}

Current top 5 candidate diseases with their expected symptom patterns:
{json.dumps(disease_profiles, indent=2)}

Based on ALL the information above — the initial summary AND every follow-up answer —
re-score each disease's probability.

Rules:
1. Increase confidence for diseases whose expected patterns match the patient's answers
2. Decrease confidence for diseases whose patterns contradict the patient's answers
3. Consider the full clinical picture — not just the last answer
4. Do NOT suggest dangerous diseases unless clear red flags are present
5. Return ONLY a JSON array of 5 objects in this exact format:

[
  {{
    "rank": 1,
    "name": "Disease Name",
    "confidence": 0.82,
    "reasoning": "specific clinical reasoning based on patient answers"
  }},
  ...
]

Confidence must be float 0.0-1.0. Rank by confidence descending.
"""

    try:
        raw = groq.chat(
            messages=[
                {"role": "system", "content": "You are a clinical decision support expert. Return only valid JSON."},
                {"role": "user",   "content": prompt}
            ],
            temperature = 0.1,
            max_tokens  = 1000,
        )

        if raw.startswith("```"):
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

        reranked = json.loads(raw)
        
        # Validate confidence bounds
        for item in reranked:
            conf = item.get("confidence", 0.5)
            if not isinstance(conf, (int, float)) or conf < 0.0 or conf > 1.0:
                item["confidence"] = 0.5
                print(f"[WARNING] Invalid confidence {conf} for {item.get('name')}, reset to 0.5")
        
        return sorted(reranked, key=lambda x: x.get("confidence", 0), reverse=True)

    except Exception as e:
        print(f"[Rerank Error] {e}")
        return top5

# ─── CHECK EARLY EXIT ─────────────────────────────────────
def check_early_exit(top5: list[dict], question_count: int) -> bool:
    """
    Exit if:
    1. Top disease has high confidence AND big gap over second place, OR
    2. Max follow-up questions reached
    
    This prevents repeating questions unnecessarily if diagnosis is already clear.
    """
    if question_count >= MAX_FOLLOWUP_QUESTIONS:
        return True
    
    if len(top5) < 2:
        return True

    top1_conf = top5[0].get("confidence", 0)
    top2_conf = top5[1].get("confidence", 0)

    return (
        top1_conf >= EARLY_EXIT_CONFIDENCE and
        (top1_conf - top2_conf) >= EARLY_EXIT_GAP
    )

# ─── BUILD FINAL RESULT ───────────────────────────────────
def build_final_result(top5: list[dict]) -> dict:
    """
    Build the final output shown to the user.
    
    Determines "see_doctor" based on diagnosis confidence:
    - If confidence >= EARLY_EXIT_CONFIDENCE (75%): System is confident → see_doctor = False
    - If confidence < EARLY_EXIT_CONFIDENCE (75%): System is uncertain → see_doctor = True
    
    This ensures the warning is shown when system doesn't have high confidence,
    aligning with the exit threshold (system stops asking at 75% confidence).
    """
    top3    = top5[:3]
    results = []
    
    top1_confidence = top5[0].get("confidence", 0) if top5 else 0

    for entry in top3:
        name    = entry["name"]
        # Normalize name for lookup to handle name variations
        normalized_name = normalize_disease_name(name)
        disease = ALL_DISEASES_NORMALIZED.get(normalized_name) or ALL_DISEASES.get(name, {})

        results.append({
            "rank":               entry.get("rank", 1),
            "name":               name,
            "confidence":         entry.get("confidence", 0),
            "reasoning":          entry.get("reasoning", ""),
            "overview":           disease.get("overview", ""),
            "typical_duration":   disease.get("typical_duration", ""),
            "when_to_see_doctor": disease.get("when_to_see_doctor", ""),
            "prevention":         [
                p.get("method", "") for p in disease.get("prevention", [])
                if isinstance(p, dict)
            ][:3],
            "red_flags": disease.get("diagnosis_clues", {}).get("red_flags", [])[:3],
        })

    # Show "see_doctor" warning if diagnosis confidence is below our exit confidence threshold
    # This ensures users get professional confirmation when system is uncertain
    see_doctor_needed = top1_confidence < EARLY_EXIT_CONFIDENCE

    return {
        "top3":       results,
        "disclaimer": "This is not a medical diagnosis. Please consult a doctor for confirmation.",
        "see_doctor": see_doctor_needed,
        "confidence_note": f"Top diagnosis confidence: {round(top1_confidence * 100)}%"
    }

# ─── MAIN FOLLOWUP FUNCTION ───────────────────────────────
def process_followup_answer(
    top5:            list[dict],
    conversation:    list[dict],
    summary:         dict,
    asked_questions: list[str],
    question_count:  int
) -> dict:
    """
    Called after each follow-up answer.
    Returns next question OR final result if done.
    """
    # Re-rank based on latest answer
    updated_top5 = rerank_after_answer(top5, conversation, summary)

    # Check if we should stop (confidence-based OR question limit)
    should_exit = check_early_exit(updated_top5, question_count)

    if should_exit:
        return {
            "status": "complete",
            "result": build_final_result(updated_top5),
            "top5":   updated_top5
        }

    # Select next question
    next_q = select_next_question(updated_top5, asked_questions)

    if not next_q:
        return {
            "status": "complete",
            "result": build_final_result(updated_top5),
            "top5":   updated_top5
        }

    return {
        "status":   "question",
        "question": next_q.get("question", ""),
        "purpose":  next_q.get("purpose", ""),
        "top5":     updated_top5
    }