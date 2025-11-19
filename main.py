from config.settings import settings
from app.graph.graph import build_graph
from langchain_core.messages import HumanMessage
from datetime import datetime
from pathlib import Path
import json


def save_debate_log(medical_note: str, result: dict, output_dir: str = "logs/debates"):
    """
    Save debate results to a JSON file for analysis and auditing.

    Args:
        medical_note: The input medical note
        result: The graph execution result containing all arguments and decisions
        output_dir: Directory to save logs
    """
    # Create logs directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debate_{timestamp}.json"
    filepath = Path(output_dir) / filename

    # Helper function to convert document objects to dictionaries
    def doc_to_dict(doc):
        """Convert a document object to a JSON-serializable dictionary."""
        return {
            "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
            "content": doc.page_content if hasattr(doc, 'page_content') else str(doc)
        }

    # Build JSON structure
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "final_answer": result.get("final_answer", "UNKNOWN"),
        "max_rounds": result.get("max_rounds", 2),
        "medical_note": medical_note,
        "expert_a": {
            "source": "Mayo Clinic",
            "arguments": result.get("expertA_arguments", []),
            "retrieved_docs": [doc_to_dict(doc) for doc in result.get("expertA_retrieved_docs", [])]
        },
        "expert_b": {
            "source": "WebMD",
            "arguments": result.get("expertB_arguments", []),
            "retrieved_docs": [doc_to_dict(doc) for doc in result.get("expertB_retrieved_docs", [])]
        },
        "judge_decision": result.get("judge_decision", "No decision"),
        "system_metadata": {
            "use_retriever": settings.USE_RETRIEVER,
            "embedding_model": settings.EMBEDDING_MODEL,
            "expert_model": settings.EXPERT_MODEL,
            "judge_model": settings.JUDGE_MODEL
        }
    }

    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"\n📝 Debate log saved to: {filepath}")
    return filepath


def main():
    """
    Main script to run the multi-agent debate system for medical error detection.
    """
    # Build the graph with settings
    print("=" * 80)
    print("MULTI-AGENT DEBATE SYSTEM FOR MEDICAL ERROR DETECTION")
    print("=" * 80)
    print("\nBuilding the debate graph...")
    graph = build_graph(settings)
    print("Graph built successfully!\n")

    # Get medical note input
    print("Enter the medical note to analyze (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    medical_note = "\n".join(lines)

    if not medical_note.strip():
        print("No medical note provided. Using example note...")
        medical_note = """54-year-old woman with a painful, rapidly growing leg lesion for 1 month.
        History includes Crohn's disease, diabetes, hypertension, and previous anterior uveitis.
        Examination revealed a 4-cm tender ulcerative lesion with necrotic base and purplish borders,
        along with pitting edema and dilated veins. Diagnosed as a venous ulcer."""

    # Create initial state for debate
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

    # Run the debate
    print("\n" + "=" * 80)
    print("STARTING MULTI-AGENT DEBATE")
    print("=" * 80)
    print("\nRound 1: Initial arguments from both experts...")

    # Run the debate
    result = graph.invoke(initial_state)

    # Display the results
    print("\n" + "=" * 80)
    print("DEBATE RESULTS")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("EXPERT A (Mayo Clinic) - ARGUMENTS")
    print("-" * 80)
    for arg in result.get("expertA_arguments", []):
        print(f"\nRound {arg['round']}:")
        print(arg['content'])

    print("\n" + "-" * 80)
    print("EXPERT B (WebMD) - ARGUMENTS")
    print("-" * 80)
    for arg in result.get("expertB_arguments", []):
        print(f"\nRound {arg['round']}:")
        print(arg['content'])

    print("\n" + "=" * 80)
    print("JUDGE'S FINAL DECISION")
    print("=" * 80)
    print(result.get("judge_decision", "No decision from Judge"))

    print("\n" + "=" * 80)
    print(f"FINAL ANSWER: {result.get('final_answer', 'UNKNOWN')}")
    print("=" * 80)

    # Save debate log to JSON file
    save_debate_log(medical_note, result)


if __name__ == "__main__":
    main()
