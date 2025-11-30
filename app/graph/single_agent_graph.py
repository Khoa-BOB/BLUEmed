from langgraph.graph import StateGraph, START, END
from app.core.state import MedState
from app.agents.single_agent import single_agent_node
from app.llm.factory import build_llm
from functools import partial


def build_single_agent_graph(config, knowledge_source: str = "Mayo") -> StateGraph:
    """
    Build a simple single-agent graph for medical error detection.

    Args:
        config: Settings configuration
        knowledge_source: Which knowledge source to use ("Mayo" or "WebMD")

    Returns:
        Compiled StateGraph for single agent execution
    """
    # Pre-initialize retriever if enabled
    if config.USE_RETRIEVER:
        from app.rag.smart_hybrid_retriever import get_smart_hybrid_retriever
        print(f"Initializing hybrid retriever for {knowledge_source}...")
        get_smart_hybrid_retriever()
        print()

    # Build LLM
    llm = build_llm(model_name=config.EXPERT_MODEL)

    # Create graph
    builder = StateGraph(MedState)

    # Add single agent node with knowledge source
    builder.add_node(
        "single_agent",
        partial(single_agent_node, llm=llm, knowledge_source=knowledge_source)
    )

    # Simple flow: START -> single_agent -> END
    builder.add_edge(START, "single_agent")
    builder.add_edge("single_agent", END)

    return builder.compile()
