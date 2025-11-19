from langgraph.graph import StateGraph, START, END
from app.core.state import MedState
from app.agents.expertA import expertA_node
from app.agents.expertB import expertB_node
from app.agents.judge import Judge_node
from app.llm.factory import build_llm
from functools import partial


def increment_round(state: MedState) -> dict:
    """Increment the debate round counter."""
    return {"current_round": state["current_round"] + 1}


def should_continue_debate(state: MedState) -> str:
    """Decide whether to continue debate or move to judge."""
    if state["current_round"] >= state["max_rounds"]:
        return "judge"
    else:
        return "continue"


def build_graph(config) -> StateGraph:
    """
    Build the multi-agent debate graph.

    Flow:
    1. Round 1: Both experts provide initial arguments (parallel)
    2. Increment round
    3. Round 2: Both experts provide counter-arguments (parallel)
    4. Judge evaluates all arguments and makes final decision
    """
    # Pre-initialize retriever to avoid race conditions when experts run in parallel
    if config.USE_RETRIEVER:
        from app.rag.smart_retriever import get_smart_retriever
        print("Initializing retriever...")
        get_smart_retriever()  # Initialize singleton before parallel execution
        print()

    llm_expert = build_llm(model_name=config.EXPERT_MODEL)
    llm_judge = build_llm(model_name=config.JUDGE_MODEL)

    builder = StateGraph(MedState)

    # Add nodes
    builder.add_node("expertA_round1", partial(expertA_node, llm=llm_expert))
    builder.add_node("expertB_round1", partial(expertB_node, llm=llm_expert))
    builder.add_node("increment", increment_round)
    builder.add_node("expertA_round2", partial(expertA_node, llm=llm_expert))
    builder.add_node("expertB_round2", partial(expertB_node, llm=llm_expert))
    builder.add_node("judge", partial(Judge_node, llm=llm_judge))

    # Build the flow
    # Round 1: Both experts argue in parallel
    builder.add_edge(START, "expertA_round1")
    builder.add_edge(START, "expertB_round1")

    # After both round 1 arguments, increment round
    builder.add_edge("expertA_round1", "increment")
    builder.add_edge("expertB_round1", "increment")

    # Round 2: Both experts counter-argue in parallel
    builder.add_edge("increment", "expertA_round2")
    builder.add_edge("increment", "expertB_round2")

    # After both round 2 arguments, judge decides
    builder.add_edge("expertA_round2", "judge")
    builder.add_edge("expertB_round2", "judge")

    builder.add_edge("judge", END)

    return builder.compile()