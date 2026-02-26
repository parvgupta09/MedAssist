"""
scripts/index_diseases.py
Run this ONCE to build the FAISS vector store from your disease JSON files.

Usage:
    python scripts/index_diseases.py
    python scripts/index_diseases.py --input data/raw --output disease_vector_db
"""

import argparse
import sys
import os

# Make sure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.vector_store_setup import load_diseases_from_folder, build_faiss_store


def main():
    parser = argparse.ArgumentParser(description="Index disease JSONs into FAISS vector store")
    parser.add_argument("--input",  default="data/raw",          help="Folder with disease JSON files")
    parser.add_argument("--output", default="disease_vector_db", help="Where to save FAISS index")
    args = parser.parse_args()

    print("=" * 55)
    print("  MedAssist — Disease Vector DB Indexer")
    print("=" * 55 + "\n")

    diseases = load_diseases_from_folder(args.input)

    if not diseases:
        print("❌ No diseases loaded. Check your data folder.")
        sys.exit(1)

    build_faiss_store(diseases, save_path=args.output)

    print("\n✅ Done! You can now run the chatbot.")
    print("   API  : uvicorn api.app:app --reload --port 8000")
    print("   CLI  : python scripts/run_cli.py")


if __name__ == "__main__":
    main()