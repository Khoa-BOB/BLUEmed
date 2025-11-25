from app.core.state import MedState, DebateArgument
from app.core.prompts import EXPERT_A_SYSTEM
from langchain_core.messages import SystemMessage, HumanMessage
from app.rag.smart_retriever import get_smart_retriever
from app.rag.metadata_filter import (
    determine_retrieval_strategy,
    get_category_split,
    format_retrieval_summary
)
from config.settings import settings


def expertA_node(state: MedState, llm) -> dict:
    """
    Expert A node with Mayo Clinic access for debate system.
    Generates arguments in rounds, can see Expert B's previous arguments.
    Uses smart retriever with hybrid metadata filtering and query decomposition.
    """
    current_round = state["current_round"]
    medical_note = state["medical_note"]

    # Get smart retriever and fetch Mayo Clinic knowledge with hybrid metadata filtering
    # Toggle via .env: USE_RETRIEVER=True or False
    # Optimization: Only retrieve in round 1, reuse documents in round 2
    if settings.USE_RETRIEVER:
        if current_round == 1:
            # Round 1: Perform retrieval with hybrid metadata filtering
            smart_retriever = get_smart_retriever()

            # Determine optimal retrieval strategy based on note content
            primary_cat, secondary_cat = determine_retrieval_strategy(medical_note)
            k_per_query, primary_max, secondary_max = get_category_split(secondary_cat is not None)

            # Log retrieval strategy
            print(f"\n[Expert A] {format_retrieval_summary(primary_cat, secondary_cat, primary_max, secondary_max)}")

            if secondary_cat:
                # Multi-category retrieval: Get from both primary and secondary categories
                primary_docs = smart_retriever.retrieve_with_decomposition(
                    note=medical_note,
                    expert="A",
                    k_per_query=k_per_query,
                    max_total=primary_max,
                    filter_category=primary_cat
                )

                secondary_docs = smart_retriever.retrieve_with_decomposition(
                    note=medical_note,
                    expert="A",
                    k_per_query=k_per_query,
                    max_total=secondary_max,
                    filter_category=secondary_cat
                )

                retrieved_docs = primary_docs + secondary_docs
            else:
                # Single-category retrieval: Full depth in one category
                retrieved_docs = smart_retriever.retrieve_with_decomposition(
                    note=medical_note,
                    expert="A",
                    k_per_query=k_per_query,
                    max_total=primary_max,
                    filter_category=primary_cat
                )
        else:
            # Round 2+: Reuse documents from state
            retrieved_docs = state.get("expertA_retrieved_docs", [])
    else:
        # No retrieval - experts rely on their training only
        retrieved_docs = []

    # Build context for this round
    prompt_parts = [f"Medical Note to Analyze:\n{medical_note}\n"]

    # Add retrieved Mayo Clinic documents
    if retrieved_docs:
        prompt_parts.append("\n=== Retrieved Mayo Clinic Knowledge ===")
        for i, doc in enumerate(retrieved_docs, 1):
            # Extract content from Document object
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            prompt_parts.append(f"\nSource {i}:\n{content}\n")

    # In round 2, include Expert B's round 1 argument
    if current_round == 2 and state["expertB_arguments"]:
        expert_b_arg = state["expertB_arguments"][0]["content"]
        prompt_parts.append(f"\n=== Expert B's Argument (Round 1) ===\n{expert_b_arg}\n")
        prompt_parts.append("\nNow provide your counter-argument, addressing Expert B's points.")
    else:
        prompt_parts.append("\nProvide your initial argument (Round 1).")

    prompt_parts.append("\n=== Instructions ===")
    prompt_parts.append("Maximum 300 words. Provide evidence-based reasoning using the retrieved Mayo Clinic knowledge.")

    user_prompt = "\n".join(prompt_parts)

    messages = [
        SystemMessage(content=EXPERT_A_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    resp = llm.invoke(messages)

    # Create debate argument
    new_argument: DebateArgument = {
        "round": current_round,
        "content": resp.content
    }

    # Update arguments list
    updated_arguments = state.get("expertA_arguments", []) + [new_argument]

    return {
        "expertA_arguments": updated_arguments,
        "expertA_retrieved_docs": retrieved_docs
    }
