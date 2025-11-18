"""
Test the debate system with Google Gemini API.
Much better quality than Llama 3.1 8B, and it's FREE!
"""
import os
os.environ["EXPERT_MODEL"] = "gemini"
os.environ["JUDGE_MODEL"] = "gemini"

from config.settings import settings
from app.graph.graph import build_graph
from langchain_core.messages import HumanMessage


def test_gemini_debate():
    """Test debate system with Gemini."""

    print("=" * 80)
    print("TESTING WITH GOOGLE GEMINI API")
    print("=" * 80)

    # Check API key
    if not settings.GOOGLE_API_KEY or settings.GOOGLE_API_KEY == "your_api_key_here":
        print("\n❌ ERROR: Google API key not configured!")
        print("\nTo use Gemini:")
        print("1. Get API key from: https://aistudio.google.com/app/apikey")
        print("2. Add to .env file:")
        print("   GOOGLE_API_KEY=your_actual_key_here")
        print("3. Run this script again")
        return

    print(f"\n✅ Using Gemini model: {settings.GEMINI_MODEL}")
    print(f"✅ API key configured")

    # Example from the paper (should be INCORRECT)
    medical_note = """54-year-old woman with a painful, rapidly growing leg lesion for 1 month.
History includes Crohn's disease, diabetes, hypertension, and previous anterior uveitis.
Examination revealed a 4-cm tender ulcerative lesion with necrotic base and purplish borders,
along with pitting edema and dilated veins. Diagnosed as a venous ulcer."""

    print("\n" + "=" * 80)
    print("Medical Note:")
    print("-" * 80)
    print(medical_note)
    print("\nExpected: INCORRECT (should be pyoderma gangrenosum, not venous ulcer)")
    print("=" * 80)

    # Build graph with Gemini
    print("\nBuilding debate graph with Gemini...")
    graph = build_graph(settings)

    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content="Analyze this medical note for errors")],
        "medical_note": medical_note,
        "current_round": 1,
        "max_rounds": 2,
        "expertA_arguments": [],
        "expertA_retrieved_docs": [],
        "expertB_arguments": [],
        "expertB_retrieved_docs": [],
        "judge_decision": None,
        "final_answer": None
    }

    # Run debate
    print("\nRunning debate with Gemini (this is much faster than Llama!)...")
    print("=" * 80)

    result = graph.invoke(initial_state)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\nExpert A (Mayo Clinic):")
    print("-" * 80)
    for arg in result.get("expertA_arguments", []):
        print(f"\nRound {arg['round']}:")
        print(arg['content'][:500] + "..." if len(arg['content']) > 500 else arg['content'])

    print("\n\nExpert B (WebMD):")
    print("-" * 80)
    for arg in result.get("expertB_arguments", []):
        print(f"\nRound {arg['round']}:")
        print(arg['content'][:500] + "..." if len(arg['content']) > 500 else arg['content'])

    print("\n\n" + "=" * 80)
    print("JUDGE'S DECISION:")
    print("=" * 80)
    print(result.get("judge_decision", "No decision"))

    print("\n" + "=" * 80)
    final_answer = result.get('final_answer', 'UNKNOWN')
    expected = "INCORRECT"
    status = "✅ CORRECT!" if final_answer == expected else "❌ WRONG"
    print(f"FINAL ANSWER: {final_answer}")
    print(f"EXPECTED: {expected}")
    print(f"RESULT: {status}")
    print("=" * 80)

    return final_answer == expected


if __name__ == "__main__":
    success = test_gemini_debate()

    print("\n" + "=" * 80)
    if success:
        print("🎉 SUCCESS! Gemini correctly identified the error!")
    else:
        print("📝 Gemini gave a different answer. Check the reasoning above.")
    print("=" * 80)

    print("\n💡 TIP: Gemini 1.5 Pro is FREE and much better than Llama 3.1 8B!")
    print("   Get your API key: https://aistudio.google.com/app/apikey")
