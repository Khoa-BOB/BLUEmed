"""
Single-Agent Medical Error Detection System

Based on the paper "Error Detection in Medical Note through Multi Agent Debate",
this implements the specialized single-agent (S.Agent) approach that:
- Uses few-shot examples and chain-of-thought reasoning
- Retrieves information from a single medical knowledge source (Mayo or WebMD)
- Achieves 70.2% (WebMD) and 72.6% (Mayo) accuracy

Usage:
    python single_agent_main.py --source Mayo
    python single_agent_main.py --source WebMD
"""

from config.settings import settings
from app.graph.single_agent_graph import build_single_agent_graph
from langchain_core.messages import HumanMessage
from datetime import datetime
from pathlib import Path
import json
import argparse


def save_single_agent_log(medical_note: str, result: dict, knowledge_source: str, output_dir: str = "logs/single_agent"):
    """
    Save single agent results to a JSON file.

    Args:
        medical_note: The input medical note
        result: The execution result
        knowledge_source: Which knowledge source was used
        output_dir: Directory to save logs
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"single_agent_{knowledge_source.lower()}_{timestamp}.json"
    filepath = Path(output_dir) / filename

    # Helper function to convert document objects
    def doc_to_dict(doc):
        return {
            "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
            "content": doc.page_content if hasattr(doc, 'page_content') else str(doc)
        }

    # Build JSON structure
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "knowledge_source": knowledge_source,
        "medical_note": medical_note,
        "single_agent_decision": result.get("single_agent_decision", ""),
        "final_answer": result.get("final_answer", "UNKNOWN"),
        "retrieved_docs": [doc_to_dict(doc) for doc in result.get("single_agent_retrieved_docs", [])],
        "system_metadata": {
            "use_retriever": settings.USE_RETRIEVER,
            "embedding_model": settings.EMBEDDING_MODEL,
            "expert_model": settings.EXPERT_MODEL,
        }
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"\n📝 Single agent log saved to: {filepath}")
    return filepath


def main():
    """Main script to run single-agent medical error detection."""
    parser = argparse.ArgumentParser(
        description="Single-Agent Medical Error Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python single_agent_main.py --source Mayo
  python single_agent_main.py --source WebMD --note "Patient diagnosed with..."
        """
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["Mayo", "WebMD"],
        default="Mayo",
        help="Knowledge source to use (Mayo or WebMD). Default: Mayo"
    )
    parser.add_argument(
        "--note",
        type=str,
        help="Medical note to analyze (if not provided, will prompt for input)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print(f"SINGLE-AGENT MEDICAL ERROR DETECTION ({args.source} Clinic)")
    print("=" * 80)
    print(f"\nBuilding single-agent graph with {args.source} knowledge source...")

    # Build graph
    graph = build_single_agent_graph(settings, knowledge_source=args.source)
    print("Graph built successfully!\n")

    # Get medical note input
    if args.note:
        medical_note = args.note
    else:
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

    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content="Analyze this medical note for errors")],
        "medical_note": medical_note,
        "current_round": 1,
        "max_rounds": 1,
        "final_answer": None
    }

    # Run single agent
    print("\n" + "=" * 80)
    print(f"RUNNING SINGLE-AGENT ANALYSIS ({args.source})")
    print("=" * 80)

    result = graph.invoke(initial_state)

    # Display results
    print("\n" + "=" * 80)
    print("SINGLE-AGENT ANALYSIS")
    print("=" * 80)
    print(result.get("single_agent_decision", "No decision"))

    print("\n" + "=" * 80)
    print(f"FINAL ANSWER: {result.get('final_answer', 'UNKNOWN')}")
    print("=" * 80)

    # Save log
    save_single_agent_log(medical_note, result, args.source)


if __name__ == "__main__":
    main()
