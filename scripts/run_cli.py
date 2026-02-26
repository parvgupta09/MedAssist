"""
scripts/run_cli.py
Test the chatbot interactively in terminal or Colab.

Usage:
    python scripts/run_cli.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.vector_store_setup import load_faiss_store
from chatbot.chatbot import MedicalChatbot


def main():
    print("\n" + "=" * 60)
    print("  MedAssist — AI Medical Triage Chatbot")
    print("  ⚠️  NOT a medical diagnosis tool.")
    print("      Always consult a licensed doctor.")
    print("=" * 60 + "\n")

    # ── Load FAISS store ─────────────────────────────────────────
    try:
        vs = load_faiss_store("disease_vector_db")
    except Exception as e:
        print(f"❌ Could not load FAISS store: {e}")
        print("   Run: python scripts/index_diseases.py first")
        sys.exit(1)

    # ── Get Groq API key ─────────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("❌ GROQ_API_KEY not set.")
        print("   In Colab run: import os; os.environ['GROQ_API_KEY'] = 'your-key'")
        sys.exit(1)

    # ── Create chatbot ───────────────────────────────────────────
    bot = MedicalChatbot(
        vectorstore  = vs,
        groq_api_key = groq_key,
        top_k        = 25,
        final_top_n  = 4,
    )

    print("MedAssist: Hello! I'm MedAssist, your medical triage assistant.")
    print("MedAssist: What symptoms are you experiencing?\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "bye", "q"):
                print("\nMedAssist: Take care! Please consult a real doctor. Goodbye!")
                break

            response = bot.chat(user_input)
            print(f"\nMedAssist: {response}\n")

            # Show live rankings during followup and result phases
            if bot.state.phase.value in ("followup", "result"):
                rankings = bot.get_rankings()
                if rankings:
                    print("  ┌─ Current Top Candidates ────────────────────┐")
                    for r in rankings[:5]:
                        bar = "█" * int(float(r["probability"].replace("%", "")) / 10)
                        print(f"  │ {r['rank']}. {r['disease']:<28} {r['probability']:>6} {bar}")
                    print("  └─────────────────────────────────────────────┘\n")

            if bot.state.phase.value == "result":
                print("\n" + "=" * 60)
                print("  Session complete. Please see a healthcare provider.")
                print("=" * 60)
                break

        except KeyboardInterrupt:
            print("\n\nGoodbye! Please consult a real doctor.")
            break


if __name__ == "__main__":
    main()
    