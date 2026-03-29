import json
import re
import chromadb
from sentence_transformers import SentenceTransformer
from backend.config import (
    CHROMA_PATH, EMBEDDING_MODEL,
    METADATA_PATH, normalize_disease_name
)
from backend.services.groq_client import groq  # ← only addition

# DELETE these two lines from original:
# from groq import Groq
# from backend.config import GROQ_API_KEY
# groq_client = Groq(api_key=GROQ_API_KEY)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
chroma_client   = chromadb.PersistentClient(path=CHROMA_PATH)

# Initialize ChromaDB collection with error handling
collection = None
try:
    collection = chroma_client.get_collection("diseases")
    print("[SemanticSearch] ChromaDB collection 'diseases' loaded successfully")
except Exception as e:
    print(f"[ERROR] ChromaDB collection 'diseases' not found: {e}")
    print("[ERROR] Please run generate_embeddings.py to create embeddings")

# Load full metadata with error handling
ALL_DISEASES = {}
ALL_DISEASES_NORMALIZED = {}  # Normalized name → original disease data
try:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        diseases_list = json.load(f)
        ALL_DISEASES = {d["name"]: d for d in diseases_list}
        # Create normalized mapping for robust lookups
        ALL_DISEASES_NORMALIZED = {normalize_disease_name(d["name"]): d for d in diseases_list}
    print(f"[SemanticSearch] Loaded {len(ALL_DISEASES)} diseases")
except FileNotFoundError:
    print(f"[ERROR] Metadata file not found at {METADATA_PATH}")
except json.JSONDecodeError as e:
    print(f"[ERROR] Invalid JSON in metadata: {e}")
except Exception as e:
    print(f"[ERROR] Failed to load metadata: {e}")

def build_query_text(summary: dict) -> str:
    parts = []
    symptoms = summary.get("reported_symptoms", [])
    if symptoms:
        parts.append(f"Symptoms: {', '.join(symptoms)}")
    regions = summary.get("body_regions_affected", [])
    if regions:
        parts.append(f"Body regions: {', '.join(regions)}")
    duration = summary.get("symptom_duration")
    if duration:
        parts.append(f"Duration: {duration}")
    onset = summary.get("onset_pattern")
    if onset:
        parts.append(f"Onset: {onset}")
    severity = summary.get("severity_impression")
    if severity:
        parts.append(f"Severity: {severity}")
    risk = summary.get("risk_signals", [])
    if risk:
        parts.append(f"Risk factors: {', '.join(risk)}")
    return " | ".join([p for p in parts if p])

def vector_search(query_text: str, top_k: int = 10) -> list[str]:
    try:
        # Validate input
        if not query_text or not isinstance(query_text, str):
            print("[WARNING] vector_search called with empty/invalid query_text")
            return []
        
        embedding = embedding_model.encode(query_text).tolist()
        results   = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, 35),
            include=["metadatas", "distances"]
        )
        return [meta["name"] for meta in results["metadatas"][0]]
    except Exception as e:
        print(f"[ERROR] vector_search failed: {e}")
        return []

def rule_filter(candidates: list[str], summary: dict) -> list[str]:
    regions = [r.lower() for r in summary.get("body_regions_affected", [])]
    if not regions:
        return candidates

    REGION_SYSTEM_MAP = {
        "chest":    ["respiratory", "infectious"],
        "throat":   ["ENT", "infectious", "respiratory"],
        "head":     ["neurological", "ENT"],
        "stomach":  ["gastrointestinal"],
        "abdomen":  ["gastrointestinal"],
        "skin":     ["skin", "infectious"],
        "back":     ["musculoskeletal"],
        "joint":    ["musculoskeletal", "metabolic"],
        "urinary":  ["urinary"],
        "eye":      ["ENT"],
        "ear":      ["ENT"],
        "nose":     ["ENT", "respiratory"],
    }

    relevant_systems = set()
    for region in regions:
        for key, systems in REGION_SYSTEM_MAP.items():
            if key in region:
                relevant_systems.update(systems)

    if not relevant_systems:
        return candidates

    filtered = [
        name for name in candidates
        if (ALL_DISEASES_NORMALIZED.get(normalize_disease_name(name), {}) or ALL_DISEASES.get(name, {})).get("body_system", "") in relevant_systems
    ]
    return filtered if len(filtered) >= 8 else candidates

def llm_rerank(candidates: list[str], summary: dict) -> list[dict]:
    candidate_info = []
    for name in candidates[:10]:
        # Normalize name for lookup to handle name variations
        normalized_name = normalize_disease_name(name)
        disease = ALL_DISEASES_NORMALIZED.get(normalized_name) or ALL_DISEASES.get(name, {})
        symptoms       = disease.get("symptoms", {})
        primary        = symptoms.get("primary", []) if isinstance(symptoms, dict) else []
        distinguishing = symptoms.get("distinguishing", []) if isinstance(symptoms, dict) else []
        candidate_info.append({
            "name":                   name,
            "overview":               disease.get("overview", ""),
            "primary_symptoms":       [s["name"] for s in primary        if isinstance(s, dict)],
            "distinguishing":         [s["name"] for s in distinguishing if isinstance(s, dict)],
            "commonly_confused_with": disease.get("diagnosis_clues", {}).get("commonly_confused_with", []),
            "body_system":            disease.get("body_system", ""),
        })

    prompt = f"""You are a clinical decision support expert.

Patient clinical summary:
{json.dumps(summary, indent=2)}

Candidate diseases to rank:
{json.dumps(candidate_info, indent=2)}

Rank the top 5 most probable diseases for this patient based on the full clinical picture.
Consider: symptom match, duration, onset pattern, risk factors, body region, age/sex if available.
Do NOT suggest rare or dangerous diseases unless red flag symptoms are clearly present.

Return ONLY a JSON array of exactly 5 objects:
[
  {{
    "rank": 1,
    "name": "Disease Name",
    "confidence": 0.75,
    "reasoning": "brief clinical reasoning in 1-2 sentences"
  }},
  ...
]

Rules:
1. confidence must be a float between 0.0 and 1.0
2. confidence values must decrease from rank 1 to rank 5
3. reasoning must be clinically specific — not generic
4. Return ONLY the JSON array
"""

    try:
        # ── only change: groq_client.chat.completions.create → groq.chat ──
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
        
        result = json.loads(raw)[:5]
        
        # Validate confidence bounds
        for item in result:
            conf = item.get("confidence", 0.5)
            if not isinstance(conf, (int, float)) or conf < 0.0 or conf > 1.0:
                item["confidence"] = 0.5
                print(f"[WARNING] Invalid confidence {conf} for {item.get('name')}, reset to 0.5")
        
        return result

    except Exception as e:
        print(f"[LLM Rerank Error] {e}")
        return [
            {"rank": i+1, "name": name, "confidence": 0.5 - (i * 0.05), "reasoning": "Based on symptom similarity"}
            for i, name in enumerate(candidates[:5])
        ]

def search_diseases(summary: dict) -> list[dict]:
    query_text = build_query_text(summary)
    candidates = vector_search(query_text, top_k=10)
    filtered   = rule_filter(candidates, summary)
    top5       = llm_rerank(filtered, summary)
    return top5