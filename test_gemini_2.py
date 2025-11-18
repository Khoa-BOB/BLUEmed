#A 24-year-old woman comes to the emergency department because of a 4-hour history of headaches, nausea, and vomiting. During this time, she has also had recurrent dizziness and palpitations. The symptoms started while she was at a friend's birthday party, where she had one beer. One week ago, the patient was diagnosed with a genitourinary infection and started on antimicrobial therapy. She has no history of major medical illness. Culture tests indicate Neisseria gonorrhoeae. Her pulse is 106/min and blood pressure is 102/73 mm Hg. Physical examination shows facial flushing and profuse sweating.
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
    medical_note = """- A 24-year-old woman comes to the emergency department because of a 4- hour history of headaches, nausea, and vomiting.
- During this time, she has also had recurrent dizziness and palpitations.
- The symptoms started while she was at a friend's birthday party, where she had one beer.
- One week ago, the patient was diagnosed with a genitourinary infection and started on antimicrobial therapy.
- She has no history of major medical illness.
- Culture tests indicate Neisseria gonorrhoeae.
- Her pulse is 106/min and blood pressure is 102/73 mmHg.
- Physical examination shows facial flushing and profuse sweating."""

    print("\n" + "=" * 80)
    print("Medical Note:")
    print("-" * 80)
    print(medical_note)
    print("\nExpected: INCORRECT (should be Trichomonas vaginalis)")
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