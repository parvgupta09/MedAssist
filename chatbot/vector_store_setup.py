import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import Optional, Dict

# ─────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ─────────────────────────────────────────────
# Safe helpers
# ─────────────────────────────────────────────

def safe_list(value) -> list:
    return value if isinstance(value, list) else []

def safe_dict(value) -> dict:
    return value if isinstance(value, dict) else {}

def safe_str(value) -> str:
    return value if isinstance(value, str) else ""


# ─────────────────────────────────────────────
# Build section-level chunks from DiseaseSchema
# ─────────────────────────────────────────────

def disease_to_chunks(disease: Dict) -> List[Document]:
    docs: List[Document] = []

    disease_id = safe_str(disease.get("id")) or safe_str(disease.get("name"))
    name = safe_str(disease.get("name"))
    category = safe_str(disease.get("category"))
    icd_code = safe_str(disease.get("icd_code"))
    overview = safe_str(disease.get("overview"))

    base_meta = {
        "id": disease_id,
        "name": name,
        "category": category,
        "icd_code": icd_code,
    }

    full_schema_str = json.dumps(disease, ensure_ascii=False)

    # ---- Overview ----
    if overview:
        docs.append(Document(
            page_content=f"Disease name: {name}\nOverview: {overview}",
            metadata={**base_meta, "section": "overview", "full_schema": full_schema_str}
        ))

    # ---- Symptoms ----
    symptoms = safe_dict(disease.get("symptoms"))
    symptom_lines = []

    for group_name in ["primary", "secondary", "distinguishing"]:
        for s in safe_list(symptoms.get(group_name)):
            s = safe_dict(s)
            parts = []
            n = safe_str(s.get("name"))
            if not n:
                continue
            parts.append(n)

            d = safe_str(s.get("description"))
            if d:
                parts.append(f"desc: {d}")

            sev = safe_str(s.get("severity"))
            if sev:
                parts.append(f"severity: {sev}")

            freq = safe_str(s.get("frequency"))
            if freq:
                parts.append(f"frequency: {freq}")

            onset = safe_str(s.get("onset_timing"))
            if onset:
                parts.append(f"onset: {onset}")

            dur = safe_str(s.get("duration"))
            if dur:
                parts.append(f"duration: {dur}")

            if s.get("warning_sign"):
                parts.append("WARNING SIGN")

            symptom_lines.append(f"{group_name.upper()}: " + " | ".join(parts))

    if symptoms.get("asymptomatic_possible"):
        symptom_lines.append("Asymptomatic cases possible")

    notes = safe_str(symptoms.get("notes"))
    if notes:
        symptom_lines.append(f"Notes: {notes}")

    if symptom_lines:
        docs.append(Document(
            page_content=f"Disease name: {name}\nSymptoms:\n" + "\n".join(symptom_lines),
            metadata={**base_meta, "section": "symptoms", "full_schema": full_schema_str}
        ))

    # ---- Causes ----
    cause_lines = []
    for c in safe_list(disease.get("causes")):
        c = safe_dict(c)
        parts = []
        pt = safe_str(c.get("pathogen_type"))
        pn = safe_str(c.get("pathogen_name"))
        desc = safe_str(c.get("description"))

        if pt: parts.append(f"type: {pt}")
        if pn: parts.append(f"name: {pn}")
        if desc: parts.append(f"desc: {desc}")

        for st in safe_list(c.get("strains_or_types")):
            parts.append(f"strain: {st}")

        if parts:
            cause_lines.append("; ".join(parts))

    if cause_lines:
        docs.append(Document(
            page_content=f"Disease name: {name}\nCauses:\n" + "\n".join(cause_lines),
            metadata={**base_meta, "section": "causes", "full_schema": full_schema_str}
        ))

    # ---- Transmission ----
    transmission = safe_dict(disease.get("transmission"))
    trans_lines = []

    for m in safe_list(transmission.get("modes")):
        trans_lines.append(f"Mode: {m}")

    v = safe_str(transmission.get("vector"))
    if v:
        trans_lines.append(f"Vector: {v}")

    if transmission.get("human_to_human"):
        trans_lines.append("Human to human transmission")

    if transmission.get("contagious"):
        trans_lines.append("Contagious")

    inc = safe_str(transmission.get("incubation_period"))
    if inc:
        trans_lines.append(f"Incubation period: {inc}")

    season = safe_str(transmission.get("seasonality"))
    if season:
        trans_lines.append(f"Seasonality: {season}")

    if trans_lines:
        docs.append(Document(
            page_content=f"Disease name: {name}\nTransmission:\n" + "\n".join(trans_lines),
            metadata={**base_meta, "section": "transmission", "full_schema": full_schema_str}
        ))

    # ---- Affected population ----
    ap = safe_dict(disease.get("affected_population"))
    ap_lines = []

    for x in safe_list(ap.get("age_groups")):
        ap_lines.append(f"Age group: {x}")
    for x in safe_list(ap.get("genders")):
        ap_lines.append(f"Gender: {x}")
    for x in safe_list(ap.get("high_risk_groups")):
        ap_lines.append(f"High risk: {x}")

    if ap_lines:
        docs.append(Document(
            page_content=f"Disease name: {name}\nAffected population:\n" + "\n".join(ap_lines),
            metadata={**base_meta, "section": "affected_population", "full_schema": full_schema_str}
        ))

    # ---- Risk factors ----
    rf_lines = []
    for rf in safe_list(disease.get("risk_factors")):
        rf = safe_dict(rf)
        parts = []
        f = safe_str(rf.get("factor"))
        if not f:
            continue
        parts.append(f)

        desc = safe_str(rf.get("description"))
        if desc:
            parts.append(f"desc: {desc}")

        if rf.get("increases_severity"):
            parts.append("increases severity")

        rf_lines.append(" | ".join(parts))

    if rf_lines:
        docs.append(Document(
            page_content=f"Disease name: {name}\nRisk factors:\n" + "\n".join(rf_lines),
            metadata={**base_meta, "section": "risk_factors", "full_schema": full_schema_str}
        ))

    # ---- Diagnosis clues ----
    diag = safe_dict(disease.get("diagnosis_clues"))
    diag_lines = []

    for x in safe_list(diag.get("key_differentiators")):
        diag_lines.append(f"Differentiator: {x}")
    for x in safe_list(diag.get("commonly_confused_with")):
        diag_lines.append(f"Often confused with: {x}")
    for x in safe_list(diag.get("red_flags")):
        diag_lines.append(f"Red flag: {x}")
    for x in safe_list(diag.get("typical_tests")):
        diag_lines.append(f"Test: {x}")

    cc = safe_str(diag.get("clinical_criteria"))
    if cc:
        diag_lines.append(f"Clinical criteria: {cc}")

    if diag_lines:
        docs.append(Document(
            page_content=f"Disease name: {name}\nDiagnosis clues:\n" + "\n".join(diag_lines),
            metadata={**base_meta, "section": "diagnosis", "full_schema": full_schema_str}
        ))

    # ---- Complications ----
    comp_lines = []
    for c in safe_list(disease.get("complications")):
        c = safe_dict(c)
        parts = []
        n = safe_str(c.get("name"))
        if not n:
            continue
        parts.append(n)

        desc = safe_str(c.get("description"))
        if desc:
            parts.append(f"desc: {desc}")

        if c.get("life_threatening"):
            parts.append("life-threatening")

        comp_lines.append(" | ".join(parts))

    if comp_lines:
        docs.append(Document(
            page_content=f"Disease name: {name}\nComplications:\n" + "\n".join(comp_lines),
            metadata={**base_meta, "section": "complications", "full_schema": full_schema_str}
        ))

    # ---- Prognosis ----
    prog = safe_dict(disease.get("prognosis"))
    prog_lines = []

    rt = safe_str(prog.get("typical_recovery_time"))
    if rt:
        prog_lines.append(f"Recovery time: {rt}")

    mr = safe_str(prog.get("mortality_risk"))
    if mr:
        prog_lines.append(f"Mortality risk: {mr}")

    for x in safe_list(prog.get("long_term_effects")):
        prog_lines.append(f"Long term effect: {x}")

    if prog.get("recurrence_possible"):
        prog_lines.append("Recurrence possible")

    notes = safe_str(prog.get("notes"))
    if notes:
        prog_lines.append(f"Notes: {notes}")

    if prog_lines:
        docs.append(Document(
            page_content=f"Disease name: {name}\nPrognosis:\n" + "\n".join(prog_lines),
            metadata={**base_meta, "section": "prognosis", "full_schema": full_schema_str}
        ))

    # ---- Prevention ----
    prev_lines = []
    for p in safe_list(disease.get("prevention")):
        p = safe_dict(p)
        parts = []
        m = safe_str(p.get("method"))
        if not m:
            continue
        parts.append(m)

        desc = safe_str(p.get("description"))
        if desc:
            parts.append(f"desc: {desc}")

        eff = safe_str(p.get("effectiveness"))
        if eff:
            parts.append(f"effectiveness: {eff}")

        prev_lines.append(" | ".join(parts))

    if prev_lines:
        docs.append(Document(
            page_content=f"Disease name: {name}\nPrevention:\n" + "\n".join(prev_lines),
            metadata={**base_meta, "section": "prevention", "full_schema": full_schema_str}
        ))

    # ---- Question flow ----
    qf = safe_dict(disease.get("question_flow"))
    q_lines = []

    for section in ["initial_questions", "refinement_questions", "red_flag_screening_questions"]:
        for q in safe_list(qf.get(section)):
            q = safe_dict(q)
            question = safe_str(q.get("question"))
            purpose = safe_str(q.get("purpose"))
            if question:
                line = f"{section}: {question}"
                if purpose:
                    line += f" (purpose: {purpose})"
                q_lines.append(line)

    if q_lines:
        docs.append(Document(
            page_content=f"Disease name: {name}\nFollow-up questions:\n" + "\n".join(q_lines),
            metadata={**base_meta, "section": "question_flow", "full_schema": full_schema_str}
        ))

    # ---- Differential diagnosis ----
    dd = safe_list(disease.get("differential_diagnosis"))
    if dd:
        docs.append(Document(
            page_content=f"Disease name: {name}\nDifferential diagnosis: " + ", ".join(dd),
            metadata={**base_meta, "section": "differential_diagnosis", "full_schema": full_schema_str}
        ))

    return docs


# ─────────────────────────────────────────────
# Load diseases
# ─────────────────────────────────────────────

def load_diseases_from_folder(folder_path: str) -> List[Dict]:
    folder = Path(folder_path)
    diseases: List[Dict] = []
    files = sorted(folder.glob("*.json"))
    print(f"📂 Found {len(files)} disease JSON files in '{folder_path}'")

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

            # If file contains a list of diseases, extend
            if isinstance(data, list):
                diseases.extend(data)
            # If file contains a single disease object, append
            elif isinstance(data, dict):
                diseases.append(data)
            else:
                print(f"⚠️ Skipping {file}, unsupported JSON format")

    print(f"✅ Loaded {len(diseases)} diseases\n")
    return diseases

# ─────────────────────────────────────────────
# Build FAISS store
# ─────────────────────────────────────────────

def build_faiss_store(diseases: List[Dict], save_path: str = "disease_vector_db") -> FAISS:
    all_docs: List[Document] = []

    for d in diseases:
        chunks = disease_to_chunks(d)
        all_docs.extend(chunks)

    print(f"🔄 Built {len(all_docs)} chunks from {len(diseases)} diseases")

    embeddings = get_embeddings()
    print("⚡ Building FAISS index...")
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)

    print(f"💾 FAISS index saved to '{save_path}/'")
    return vectorstore



def get_full_schema(doc: Document) -> Optional[Dict]:
    """
    Extract and parse full disease schema from Document metadata.
    """
    try:
        raw = doc.metadata.get("full_schema")
        if not raw:
            return None
        if isinstance(raw, dict):
            return raw  # already parsed (just in case)
        return json.loads(raw)
    except Exception:
        return None
# ─────────────────────────────────────────────
# Load FAISS store
# ─────────────────────────────────────────────

def load_faiss_store(save_path: str = "disease_vector_db") -> FAISS:
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    print(f"✅ FAISS index loaded from '{save_path}'")
    return vectorstore


# ─────────────────────────────────────────────
# Search + aggregate
# ─────────────────────────────────────────────

def search_and_aggregate(vectorstore: FAISS, query: str, k: int = 20) -> List[Tuple[str, float]]:
    results = vectorstore.similarity_search_with_score(query, k=k)

    scores_by_disease = defaultdict(list)

    for doc, score in results:
        name = doc.metadata.get("name") or doc.metadata.get("id")
        scores_by_disease[name].append(score)

    aggregated = []
    for disease, scores in scores_by_disease.items():
        avg_score = sum(scores) / len(scores)
        aggregated.append((disease, avg_score))

    aggregated.sort(key=lambda x: x[1])  # lower distance = better
    return aggregated


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR = "data/raw"
    STORE_DIR = "disease_vector_db"

    diseases = load_diseases_from_folder(DATA_DIR)
    vs = build_faiss_store(diseases, STORE_DIR)

    