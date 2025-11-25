from config.settings import settings
from app.graph.graph import build_graph
from langchain_core.messages import HumanMessage
from datetime import datetime
from pathlib import Path
import json
import re


def extract_judge_decision(judge_decision_raw: str) -> dict:
    """
    Extract structured information from judge's raw decision.

    Args:
        judge_decision_raw: Raw judge decision string

    Returns:
        Dictionary with extracted fields
    """
    extracted = {
        "final_answer": "UNKNOWN",
        "confidence_score": None,
        "winner": None,
        "reasoning": None
    }

    if not judge_decision_raw:
        return extracted

    # Try to parse JSON from judge decision
    decision_json = {}

    def fix_json_string(s: str) -> str:
        """Fix common JSON issues like unescaped newlines in string values."""
        # Replace literal newlines inside strings with escaped newlines
        # This is a simple fix that works for most cases
        import re
        # Find content between quotes and escape newlines
        def escape_newlines_in_strings(match):
            content = match.group(0)
            # Replace actual newlines with \n (but not already escaped ones)
            content = content.replace('\r\n', '\\n').replace('\r', '\\n').replace('\n', '\\n')
            return content
        # Match JSON string values (between quotes, handling escaped quotes)
        fixed = re.sub(r'"(?:[^"\\]|\\.)*"', escape_newlines_in_strings, s, flags=re.DOTALL)
        return fixed

    # Method 1: Try direct JSON parse first
    try:
        decision_json = json.loads(judge_decision_raw)
    except json.JSONDecodeError:
        # Try with fixed JSON
        try:
            fixed_json = fix_json_string(judge_decision_raw)
            decision_json = json.loads(fixed_json)
        except json.JSONDecodeError:
            pass

    # Method 2: Extract JSON from markdown code blocks (```json ... ```)
    if not decision_json:
        markdown_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', judge_decision_raw, re.DOTALL)
        if markdown_match:
            json_str = markdown_match.group(1)
            try:
                decision_json = json.loads(json_str)
            except json.JSONDecodeError:
                # Try with fixed JSON
                try:
                    fixed_json = fix_json_string(json_str)
                    decision_json = json.loads(fixed_json)
                except json.JSONDecodeError:
                    pass

    # Method 3: Find any JSON object containing "Final Answer"
    if not decision_json:
        json_match = re.search(r'(\{[^{}]*"Final Answer"[^{}]*\})', judge_decision_raw, re.DOTALL)
        if json_match:
            try:
                decision_json = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

    # Method 4: Greedy match for any JSON object
    if not decision_json:
        json_match = re.search(r'\{.*\}', judge_decision_raw, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                decision_json = json.loads(json_str)
            except json.JSONDecodeError:
                # Try with fixed JSON
                try:
                    fixed_json = fix_json_string(json_str)
                    decision_json = json.loads(fixed_json)
                except json.JSONDecodeError:
                    pass

    # Extract fields (handle different key formats)
    extracted["final_answer"] = (
        decision_json.get("Final Answer") or
        decision_json.get("final_answer") or
        decision_json.get("final answer") or
        "UNKNOWN"
    )

    # Extract confidence score
    conf_score = (
        decision_json.get("Confidence Score") or
        decision_json.get("confidence_score") or
        decision_json.get("confidence score") or
        decision_json.get("Confidence") or
        decision_json.get("confidence")
    )

    if conf_score is not None:
        try:
            extracted["confidence_score"] = int(conf_score)
        except (ValueError, TypeError):
            pass

    # Extract winner
    extracted["winner"] = (
        decision_json.get("Winner") or
        decision_json.get("winner")
    )

    # Extract reasoning
    extracted["reasoning"] = (
        decision_json.get("Reasoning") or
        decision_json.get("reasoning")
    )

    return extracted


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

    # Extract judge decision into separate fields
    judge_decision_raw = result.get("judge_decision", "")
    judge_info = extract_judge_decision(judge_decision_raw)

    # Build JSON structure
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        # Judge decision - separate key-value pairs
        "final_answer": judge_info["final_answer"],
        "confidence_score": judge_info["confidence_score"],
        "winner": judge_info["winner"],
        "reasoning": judge_info["reasoning"],
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
