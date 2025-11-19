from config.settings import settings
from app.graph.graph import build_graph
from langchain_core.messages import HumanMessage
from datetime import datetime
from pathlib import Path


def save_debate_log(medical_note: str, result: dict, output_dir: str = "logs/debates"):
    """
    Save debate results to a markdown file for analysis and auditing.

    Args:
        medical_note: The input medical note
        result: The graph execution result containing all arguments and decisions
        output_dir: Directory to save logs
    """
    # Create logs directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debate_{timestamp}.md"
    filepath = Path(output_dir) / filename

    # Build markdown content
    markdown_content = f"""# Medical Error Detection Debate Log

**Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Final Answer:** `{result.get("final_answer", "UNKNOWN")}`
**Debate Rounds:** {result.get("max_rounds", 2)}

---

## Medical Note

```
{medical_note}
```

---

## Expert A (Mayo Clinic)

**Sources Retrieved:** {len(result.get("expertA_retrieved_docs", []))}

"""

    # Add Expert A arguments
    for arg in result.get("expertA_arguments", []):
        markdown_content += f"### Round {arg.get('round', '?')}\n\n{arg.get('content', 'No content')}\n\n"

    # Add Expert A retrieved documents
    expertA_docs = result.get("expertA_retrieved_docs", [])
    if expertA_docs:
        markdown_content += "### Retrieved Documents\n\n"
        for i, doc in enumerate(expertA_docs, 1):
            # Extract document metadata
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            file_path = metadata.get('file_path', 'Unknown source')

            markdown_content += f"**{i}. {file_path}**\n\n"

            # Add all metadata fields
            if metadata:
                markdown_content += "**Metadata:**\n```\n"
                for key, value in metadata.items():
                    markdown_content += f"{key}: {value}\n"
                markdown_content += "```\n\n"

            # Add content snippet
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            markdown_content += f"**Content:**\n> {content[:300]}{'...' if len(content) > 300 else ''}\n\n"

    markdown_content += "---\n\n## Expert B (WebMD)\n\n"
    markdown_content += f"**Sources Retrieved:** {len(result.get('expertB_retrieved_docs', []))}\n\n"

    # Add Expert B arguments
    for arg in result.get("expertB_arguments", []):
        markdown_content += f"### Round {arg.get('round', '?')}\n\n{arg.get('content', 'No content')}\n\n"

    # Add Expert B retrieved documents
    expertB_docs = result.get("expertB_retrieved_docs", [])
    if expertB_docs:
        markdown_content += "### Retrieved Documents\n\n"
        for i, doc in enumerate(expertB_docs, 1):
            # Extract document metadata
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            file_path = metadata.get('file_path', 'Unknown source')

            markdown_content += f"**{i}. {file_path}**\n\n"

            # Add all metadata fields
            if metadata:
                markdown_content += "**Metadata:**\n```\n"
                for key, value in metadata.items():
                    markdown_content += f"{key}: {value}\n"
                markdown_content += "```\n\n"

            # Add content snippet
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            markdown_content += f"**Content:**\n> {content[:300]}{'...' if len(content) > 300 else ''}\n\n"

    markdown_content += "---\n\n## Judge's Decision\n\n"
    markdown_content += f"{result.get('judge_decision', 'No decision')}\n\n"

    markdown_content += "---\n\n## Metadata\n\n"
    markdown_content += f"- **Use Retriever:** {settings.USE_RETRIEVER}\n"
    markdown_content += f"- **Embedding Model:** {settings.EMBEDDING_MODEL}\n"
    markdown_content += f"- **Expert Model:** {settings.EXPERT_MODEL}\n"
    markdown_content += f"- **Judge Model:** {settings.JUDGE_MODEL}\n"

    # Save to markdown file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

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
