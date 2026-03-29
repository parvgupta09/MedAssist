import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

# ─── CONFIG ───────────────────────────────────────────────
METADATA_PATH = "data/metadata/medassist_metadata.json"
CHROMA_PATH   = "data/embeddings/chroma_db"

os.makedirs(CHROMA_PATH, exist_ok=True)

# ─── LOAD METADATA ────────────────────────────────────────
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    diseases = json.load(f)

# ─── LOAD EMBEDDING MODEL ─────────────────────────────────
# all-MiniLM-L6-v2 is lightweight, fast, and works perfectly on Colab free tier
model = SentenceTransformer("all-MiniLM-L6-v2")

# ─── INIT CHROMADB ────────────────────────────────────────
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Delete collection if exists to avoid duplicates on re-run
try:
    client.delete_collection("diseases")
except:
    pass

collection = client.create_collection(
    name="diseases",
    metadata={"hnsw:space": "cosine"}  # cosine similarity for medical text
)

# ─── BUILD EMBEDDING TEXT ─────────────────────────────────
def build_embedding_text(disease: dict) -> str:
    parts = []

    # Overview
    parts.append(str(disease.get("overview", "")))

    # Body system and category
    parts.append(f"Body system: {disease.get('body_system', '')}")
    parts.append(f"Category: {disease.get('category', '')}")

    # Primary symptoms — safe check for list of dicts
    primary = disease.get("symptoms", {}).get("primary", [])
    if primary and isinstance(primary, list):
        sym_text = ", ".join([
            s["name"] if isinstance(s, dict) else str(s)
            for s in primary
        ])
        parts.append(f"Primary symptoms: {sym_text}")

    # Secondary symptoms
    secondary = disease.get("symptoms", {}).get("secondary", [])
    if secondary and isinstance(secondary, list):
        sym_text = ", ".join([
            s["name"] if isinstance(s, dict) else str(s)
            for s in secondary
        ])
        parts.append(f"Secondary symptoms: {sym_text}")

    # Distinguishing symptoms
    distinguishing = disease.get("symptoms", {}).get("distinguishing", [])
    if distinguishing and isinstance(distinguishing, list):
        sym_text = ", ".join([
            s["name"] if isinstance(s, dict) else str(s)
            for s in distinguishing
        ])
        parts.append(f"Distinguishing symptoms: {sym_text}")

    # Risk factors
    risk_factors = disease.get("risk_factors", [])
    if risk_factors and isinstance(risk_factors, list):
        rf_text = ", ".join([
            r["factor"] if isinstance(r, dict) else str(r)
            for r in risk_factors
        ])
        parts.append(f"Risk factors: {rf_text}")

    # Affected population
    pop = disease.get("affected_population", {})
    if pop and isinstance(pop, dict):
        age_groups = pop.get("age_groups", [])
        high_risk  = pop.get("high_risk_groups", [])
        if age_groups:
            parts.append(f"Affected age groups: {', '.join(age_groups)}")
        if high_risk:
            parts.append(f"High risk groups: {', '.join(high_risk)}")

    # Differential diagnosis
    diff = disease.get("differential_diagnosis", [])
    if diff and isinstance(diff, list):
        parts.append(f"Commonly confused with: {', '.join(diff)}")

    # Key differentiators
    clues = disease.get("diagnosis_clues", {})
    if clues and isinstance(clues, dict):
        key_diff = clues.get("key_differentiators", [])
        if key_diff:
            parts.append(f"Key differentiators: {', '.join(key_diff)}")

    # Transmission
    transmission = disease.get("transmission", {})
    if transmission and isinstance(transmission, dict):
        modes = transmission.get("modes", [])
        if modes:
            parts.append(f"Transmission: {', '.join(modes)}")

    return " | ".join([p for p in parts if p.strip()])

# ─── GENERATE AND STORE EMBEDDINGS ────────────────────────

documents = []
embeddings = []
metadatas = []
ids = []

for disease in diseases:
    disease_id   = disease.get("id", disease["name"].lower().replace(" ", "_"))
    disease_name = disease["name"]

    # Build rich text for embedding
    embed_text = build_embedding_text(disease)

    # Generate embedding
    embedding = model.encode(embed_text).tolist()

    # Store minimal metadata in ChromaDB for retrieval
    metadata = {
        "name":              disease_name,
        "body_system":       disease.get("body_system", ""),
        "category":          disease.get("category", ""),
        "when_to_see_doctor": disease.get("when_to_see_doctor", ""),
        "typical_duration":  disease.get("typical_duration", ""),
    }

    documents.append(embed_text)
    embeddings.append(embedding)
    metadatas.append(metadata)
    ids.append(disease_id)

# ─── ADD TO COLLECTION ────────────────────────────────────
collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)