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

    Consensus criteria (all must be true to skip Round 2):
    1. Both experts agree on classification (CORRECT or INCORRECT)
    2. If INCORRECT, both must agree on the same wrong term
    3. If INCORRECT, both must agree on the same correct term

    If any disagreement exists, continue to Round 2 for debate.
    """
    import re

    if state["current_round"] == 1:
        # Check if both experts have made their arguments
        expertA_args = state.get("expertA_arguments", [])
        expertB_args = state.get("expertB_arguments", [])

        if expertA_args and expertB_args:
            argA = expertA_args[0]["content"]
            argB = expertB_args[0]["content"]

            # Extract classifications
            def extract_classification(arg_text):
                """Extract FINAL CLASSIFICATION from argument."""
                match = re.search(r'FINAL CLASSIFICATION:\s*(CORRECT|INCORRECT)', arg_text, re.IGNORECASE)
                return match.group(1).upper() if match else None

            def extract_terms(arg_text):
                """Extract wrong and correct terms from argument."""
                wrong_match = re.search(r'Wrong term:\s*["\']?([^"\'\n]+)["\']?', arg_text, re.IGNORECASE)
                correct_match = re.search(r'Correct term:\s*["\']?([^"\'\n]+)["\']?', arg_text, re.IGNORECASE)

                wrong_term = wrong_match.group(1).strip() if wrong_match else None
                correct_term = correct_match.group(1).strip() if correct_match else None

                return wrong_term, correct_term

            class_a = extract_classification(argA)
            class_b = extract_classification(argB)

            wrong_a, correct_a = extract_terms(argA)
            wrong_b, correct_b = extract_terms(argB)

            # Debug output
            print(f"\n[CONSENSUS CHECK - Round 1]")
            print(f"  Expert A: class={class_a}, wrong='{wrong_a}', correct='{correct_a}'")
            print(f"  Expert B: class={class_b}, wrong='{wrong_b}', correct='{correct_b}'")

            # Check for consensus
            both_agree_on_classification = (class_a == class_b)

            if both_agree_on_classification:
                # If both say CORRECT, that's full consensus
                if class_a == "CORRECT":
                    print("\n✓ Round 1 Consensus: Both experts agree CORRECT - Skipping Round 2")
                    return "judge"

                # If both say INCORRECT, check if they agree on terms
                elif class_a == "INCORRECT":
                    # Normalize terms for comparison (case-insensitive, strip whitespace)
                    wrong_a_norm = wrong_a.lower().strip() if wrong_a else ""
                    wrong_b_norm = wrong_b.lower().strip() if wrong_b else ""
                    correct_a_norm = correct_a.lower().strip() if correct_a else ""
                    correct_b_norm = correct_b.lower().strip() if correct_b else ""

                    same_wrong_term = (wrong_a_norm == wrong_b_norm) and wrong_a_norm != ""
                    same_correct_term = (correct_a_norm == correct_b_norm) and correct_a_norm != ""

                    if same_wrong_term and same_correct_term:
                        print(f"\n✓ Round 1 Consensus: Both experts agree INCORRECT with same terms")
                        print(f"  Wrong: '{wrong_a}' → Correct: '{correct_a}'")
                        print("  Skipping Round 2")
                        return "judge"
                    else:
                        print(f"\n⚡ Round 1 Disagreement: Experts propose different corrections")
                        print(f"  Expert A: '{wrong_a}' → '{correct_a}'")
                        print(f"  Expert B: '{wrong_b}' → '{correct_b}'")
                        print("  Proceeding to Round 2 for debate")

    # Otherwise continue to Round 2
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