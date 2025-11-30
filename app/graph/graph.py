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

def check_round1_consensus(state: MedState) -> str:
    """
    Check if experts agree in Round 1.
    If they agree, skip Round 2 and go straight to judge.
    If they disagree, continue to Round 2 for debate.
    
    NOTE: Currently disabled - always runs Round 2 for full debate.
    """
    # DISABLED: Always continue to Round 2 for complete debate
    # The original consensus check was too simplistic and caused false positives
    
    # Always continue to Round 2
    if state["current_round"] >= state["max_rounds"]:
        return "judge"
    else:
        return "continue"
    
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
        from app.rag.smart_hybrid_retriever import get_smart_hybrid_retriever
        print("Initializing hybrid retriever...")
        get_smart_hybrid_retriever()  # Initialize singleton before parallel execution
        print()

    llm_expert = build_llm(model_name=config.EXPERT_MODEL)
    llm_judge = build_llm(model_name=config.JUDGE_MODEL, temperature=0.1)

    builder = StateGraph(MedState)

    # Add nodes
    builder.add_node("expertA_round1", partial(expertA_node, llm=llm_expert))
    builder.add_node("expertB_round1", partial(expertB_node, llm=llm_expert))
    builder.add_node("increment", increment_round)
    builder.add_node("expertA_round2", partial(expertA_node, llm=llm_expert))
    builder.add_node("expertB_round2", partial(expertB_node, llm=llm_expert))
    builder.add_node("judge", partial(Judge_node, llm=llm_judge))
    builder.add_node("consensus_check", lambda state: state) 
    # Build the flow
    # Round 1: Both experts argue in parallel
    builder.add_edge(START, "expertA_round1")
    builder.add_edge(START, "expertB_round1")

    # After both round 1 arguments, check for consensus
    builder.add_edge("expertA_round1", "consensus_check")
    builder.add_edge("expertB_round1", "consensus_check")
    # Gate: if both experts has consensus, go to Judge, otherwise continue to next round
    builder.add_conditional_edges(
        "consensus_check",
        check_round1_consensus,
        {
            "judge": "judge",
            "continue": "increment"
        }
    )
    # Round 2: Both experts counter-argue in parallel if no consensus
    builder.add_edge("increment", "expertA_round2")
    builder.add_edge("increment", "expertB_round2")

    # After both round 2 arguments, judge decides
    builder.add_edge("expertA_round2", "judge")
    builder.add_edge("expertB_round2", "judge")

    builder.add_edge("judge", END)

    # Compile without checkpointing (results saved as markdown logs instead)
    return builder.compile()