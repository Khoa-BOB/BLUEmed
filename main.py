from config.settings import settings
from app.graph.graph import build_graph
from langchain_core.messages import HumanMessage

def main():
    """
    Main script to run the medical expert graph.
    Takes user input and applies the graph to get the output.
    """
    # Build the graph with settings
    print("Building the medical expert graph...")
    graph = build_graph(settings)
    print("Graph built successfully!\n")

    # Get user input
    user_question = input("Enter your medical question: ")

    # Create initial state with the user's question
    initial_state = {
        "messages": [HumanMessage(content=user_question)],
        "retrieved_context": ""  # Empty since no retrieval system is active
    }

    # Run the graph
    print("\nProcessing your question...\n")
    result = graph.invoke(initial_state)

    # Display the results
    print("=" * 80)
    print("EXPERT A ANALYSIS:")
    print("=" * 80)
    print(result.get("expert1_analysis", "No analysis from Expert A"))
    print()

    print("=" * 80)
    print("EXPERT B ANALYSIS:")
    print("=" * 80)
    print(result.get("expert2_analysis", "No analysis from Expert B"))
    print()

    print("=" * 80)
    print("JUDGE DECISION:")
    print("=" * 80)
    print(result.get("judge_decision", "No decision from Judge"))
    print()

if __name__ == "__main__":
    main()
