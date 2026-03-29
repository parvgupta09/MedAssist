import re
import json
import time
import os
from groq import Groq
from pydantic import ValidationError

# ─── CONFIG ───────────────────────────────────────────────
API_KEYS = [
    os.getenv("GROQ_API_KEY_1", ""),
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", ""),
]
API_KEYS = [k for k in API_KEYS if k]  # Filter out empty keys

if not API_KEYS:
    raise ValueError("No Groq API keys found. Set GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3 in environment variables.")

current_key_index = 0
client = Groq(api_key=API_KEYS[0])

OUTPUT_PATH   = "data/metadata/medassist_metadata.json"
BATCH_SIZE    = 5
DELAY_BETWEEN = 10

# ─── DISEASE LIST ─────────────────────────────────────────
DISEASES = [
    "Common Cold", "Influenza (Flu)", "COVID-19", "Pneumonia",
    "Bronchitis", "Asthma", "Tuberculosis", "Sinusitis",
    "Strep Throat", "Tonsillitis", "Otitis Media", "Allergic Rhinitis",
    "Gastroenteritis", "Acid Reflux / GERD", "Irritable Bowel Syndrome",
    "Appendicitis", "Food Poisoning", "Peptic Ulcer", "Hepatitis A",
    "Dengue Fever", "Malaria", "Typhoid", "Chickenpox", "Measles",
    "Urinary Tract Infection", "Migraine", "Tension Headache", "Vertigo",
    "Eczema / Dermatitis", "Psoriasis", "Fungal Skin Infection", "Scabies",
    "Lower Back Pain", "Osteoarthritis", "Type 2 Diabetes",
]

# ─── SYSTEM PROMPT ────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior medical expert and clinical knowledge engineer.
Your job is to generate a complete, accurate, structured JSON object for a given disease.
You must return ONLY a valid JSON object — no explanation, no markdown, no backticks.
Every field must be filled with clinically accurate information from your medical knowledge.
For follow-up questions, generate questions that are clinically specific to THIS disease
and would help differentiate it from the 3-5 most similar diseases.
Questions must not be generic — they must be specific enough that the answer
would increase or decrease the probability of THIS disease vs similar ones.
"""

# ─── PROMPT BUILDER ───────────────────────────────────────
def build_prompt(disease_name: str) -> str:
    schema_example = {
        "id": "snake_case_disease_name",
        "name": "Full Disease Name",
        "category": "e.g. infectious / respiratory / gastrointestinal",
        "body_system": "respiratory / ENT / gastrointestinal / neurological / skin / musculoskeletal / infectious / urinary / metabolic",
        "icd_code": "e.g. J06.9 or null",
        "overview": "2-3 sentence plain English summary of what this disease is",
        "when_to_see_doctor": "low / medium / high / emergency",
        "typical_duration": "e.g. 7-10 days / lifelong / 2-4 weeks",
        "causes": [{"pathogen_type": "virus/bacteria/fungus/parasite/other", "pathogen_name": "e.g. Influenza A virus", "description": "brief description"}],
        "transmission": {
            "modes": ["e.g. airborne", "droplet"],
            "human_to_human": True,
            "contagious": True,
            "incubation_period": "e.g. 1-4 days",
            "seasonality": "e.g. winter months or year-round"
        },
        "symptoms": {
            "primary": [{"name": "symptom name", "severity": "mild/moderate/severe", "frequency": "common/less common/rare", "onset_timing": "e.g. sudden or gradual", "duration": "e.g. 3-5 days", "warning_sign": False}],
            "secondary": [{"name": "symptom name", "severity": "mild", "frequency": "less common", "onset_timing": None, "duration": None, "warning_sign": False}],
            "distinguishing": [{"name": "what makes this disease unique symptom-wise", "severity": "moderate", "frequency": "common", "onset_timing": None, "duration": None, "warning_sign": True}],
            "asymptomatic_possible": False,
            "notes": "any extra notes or null"
        },
        "affected_population": {
            "age_groups": ["children", "adults", "elderly"],
            "genders": ["all"],
            "high_risk_groups": ["immunocompromised", "pregnant women"]
        },
        "risk_factors": [{"factor": "risk factor name", "description": "why it increases risk", "increases_severity": False}],
        "diagnosis_clues": {
            "key_differentiators": ["what makes this disease stand out clinically"],
            "commonly_confused_with": ["Disease A", "Disease B"],
            "red_flags": ["symptom that means go to hospital immediately"],
            "typical_tests": ["blood test", "PCR", "X-ray"]
        },
        "complications": [{"name": "complication name", "life_threatening": False}],
        "prognosis": {
            "typical_recovery_time": "e.g. 1-2 weeks",
            "mortality_risk": "low/moderate/high",
            "long_term_effects": ["e.g. chronic fatigue"],
            "recurrence_possible": False
        },
        "prevention": [{"method": "prevention method", "description": "how it helps"}],
        "question_flow": {
            "initial_questions": [
                {"question": "Specific first question for this disease", "purpose": "What disease feature this identifies", "related_symptoms": ["fever", "cough"], "expected_answer_type": "yes/no", "expected_positive_answer": "yes"}
            ],
            "refinement_questions": [
                {"question": "More specific follow-up", "purpose": "Differentiates from Disease X", "related_symptoms": ["rash"], "expected_answer_type": "multiple choice", "expected_positive_answer": "started on face then spread"}
            ],
            "red_flag_screening_questions": [
                {"question": "Is there any difficulty breathing or chest pain?", "purpose": "Identifies need for emergency care", "related_symptoms": ["breathlessness", "chest pain"], "expected_answer_type": "yes/no", "expected_positive_answer": "yes"}
            ]
        },
        "differential_diagnosis": ["Disease A", "Disease B", "Disease C"]
    }

    return f"""Generate complete and clinically accurate medical metadata for the disease: {disease_name}

Use your full medical knowledge about {disease_name} to fill every field accurately.

Fill this JSON schema completely:
{json.dumps(schema_example, indent=2)}

Rules:
1. Return ONLY the JSON object — no explanation, no markdown backticks
2. Fill every field with accurate medical knowledge about {disease_name}
3. Generate exactly 3 initial questions, 3 refinement questions, 2 red flag questions
4. Questions must be clinically specific to {disease_name} — not generic
5. differential_diagnosis must list 3-5 real diseases commonly confused with {disease_name}
6. distinguishing symptoms must be unique to {disease_name} vs similar diseases
7. when_to_see_doctor must be exactly one of: low / medium / high / emergency
"""

# ─── KEY ROTATION ─────────────────────────────────────────
def rotate_api_key():
    global current_key_index, client
    current_key_index += 1
    if current_key_index >= len(API_KEYS):
        raise Exception("All API keys exhausted for today. Resume tomorrow.")
    client = Groq(api_key=API_KEYS[current_key_index])

# ─── CORE GENERATOR ───────────────────────────────────────
def generate_metadata_for_disease(disease_name: str, attempt: int = 1):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(disease_name)}
            ],
            temperature=0.2,
            max_tokens=4000,
        )

        raw_response = response.choices[0].message.content.strip()

        if raw_response.startswith("```"):
            raw_response = re.sub(r"```(?:json)?", "", raw_response).strip().rstrip("```").strip()

        parsed = json.loads(raw_response)

        from backend.models.pydantic_schema import DiseaseSchema
        validated = DiseaseSchema(**parsed)
        return validated.model_dump()

    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        if attempt < 3:
            time.sleep(5)
            return generate_metadata_for_disease(disease_name, attempt + 1)
        return None

    except ValidationError as e:
        print(f"    Pydantic validation error: {e}")
        if attempt < 3:
            time.sleep(5)
            return generate_metadata_for_disease(disease_name, attempt + 1)
        return None

    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            print(f"    Rate limit hit on key {current_key_index + 1}")
            try:
                rotate_api_key()
                print(f"    Retrying with new key...")
                time.sleep(3)
                return generate_metadata_for_disease(disease_name, attempt)
            except Exception as rotate_error:
                print(f"    {rotate_error}")
                return None
        print(f"    Unexpected error: {e}")
        return None

# ─── BATCH RUNNER ─────────────────────────────────────────
def run():
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
        already_done = {d["name"] for d in all_metadata}
    else:
        all_metadata = []
        already_done = set()

    failed = []
    batches = [DISEASES[i:i+BATCH_SIZE] for i in range(0, len(DISEASES), BATCH_SIZE)]

    for batch_num, batch in enumerate(batches, 1):
        for disease_name in batch:
            if disease_name in already_done:
                continue

            result = generate_metadata_for_disease(disease_name)

            if result:
                all_metadata.append(result)
                already_done.add(disease_name)
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(all_metadata, f, indent=2, ensure_ascii=False)
            else:
                failed.append(disease_name)

            time.sleep(3)

        if batch_num < len(batches):
            time.sleep(DELAY_BETWEEN)

run()