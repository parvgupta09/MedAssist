# scripts/seed_db.py
# Run once to load metadata into PostgreSQL
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database import engine, SessionLocal, Base
from backend.models.db_models import DiseaseMetadata
from backend.config import METADATA_PATH

def seed():
    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Load metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        diseases = json.load(f)

    db = SessionLocal()
    try:
        inserted = 0
        skipped  = 0

        for disease in diseases:
            existing = db.query(DiseaseMetadata).filter_by(name=disease["name"]).first()
            if existing:
                skipped += 1
                continue

            record = DiseaseMetadata(
                id          = disease.get("id", disease["name"].lower().replace(" ", "_")),
                name        = disease["name"],
                body_system = disease.get("body_system", ""),
                category    = disease.get("category", ""),
                data        = disease
            )
            db.add(record)
            inserted += 1

        db.commit()

    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed()