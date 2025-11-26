from app.core.state import MedState, DebateArgument
from app.core.prompts import EXPERT_B_SYSTEM
from langchain_core.messages import SystemMessage, HumanMessage
from app.rag.smart_hybrid_retriever import get_smart_hybrid_retriever
from app.rag.metadata_filter import (
    determine_retrieval_strategy,
    get_category_split,
    format_retrieval_summary
)
from config.settings import settings


def expertB_node(state: MedState, llm) -> dict:
    """
    Expert B node with WebMD access for debate system.
    Generates arguments in rounds, can see Expert A's previous arguments.
    Uses smart retriever with query decomposition (if settings.USE_RETRIEVER=True).
    """
    current_round = state["current_round"]
    medical_note = state["medical_note"]

    # Get smart hybrid retriever and fetch WebMD knowledge
    # Uses: query decomposition + hybrid search (dense + sparse + online)
    # Toggle via .env: USE_RETRIEVER=True or False
    # Optimization: Only retrieve in round 1, reuse documents in round 2
    if settings.USE_RETRIEVER:
        if current_round == 1:
            # Round 1: Perform hybrid retrieval with query decomposition
            smart_hybrid_retriever = get_smart_hybrid_retriever()

            # Determine optimal retrieval strategy based on note content
            primary_cat, secondary_cat = determine_retrieval_strategy(medical_note)
            k_per_query, primary_max, secondary_max = get_category_split(secondary_cat is not None)

            # Log retrieval strategy
            print(f"\n[Expert B] {format_retrieval_summary(primary_cat, secondary_cat, primary_max, secondary_max)}")

            if secondary_cat:
                # Multi-category retrieval: Get from both primary and secondary categories
                primary_docs = smart_hybrid_retriever.retrieve_with_decomposition(
                    note=medical_note,
                    expert="B",
                    k_per_query=k_per_query,
                    max_total=primary_max,
                    filter_category=primary_cat
                )

                secondary_docs = smart_hybrid_retriever.retrieve_with_decomposition(
                    note=medical_note,
                    expert="B",
                    k_per_query=k_per_query,
                    max_total=secondary_max,
                    filter_category=secondary_cat
                )

                retrieved_docs = primary_docs + secondary_docs
            else:
                retrieved_docs = smart_hybrid_retriever.retrieve_with_decomposition(
                    note=medical_note,
                    expert="B",
                    k_per_query=2,
                    max_total=5,
                    use_dense=True,  # Vector search
                    use_sparse=True,  # BM25 keyword search
                    use_online=True   # Online web search
                )
        else:
            # Round 2+: Reuse documents from state
            retrieved_docs = state.get("expertB_retrieved_docs", [])
    else:
        # No retrieval - experts rely on their training only
        retrieved_docs = []

    # Build context for this round
    prompt_parts = [f"Medical Note to Analyze:\n{medical_note}\n"]

    # Add retrieved WebMD documents
    if retrieved_docs:
        prompt_parts.append("\n=== Retrieved WebMD Knowledge ===")
        for i, doc in enumerate(retrieved_docs, 1):
            # Extract content from Document object
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            prompt_parts.append(f"\nSource {i}:\n{content}\n")

    # In round 2, include Expert A's round 1 argument
    if current_round == 2 and state["expertA_arguments"]:
        expert_a_arg = state["expertA_arguments"][0]["content"]
        prompt_parts.append(f"\n=== Expert A's Argument (Round 1) ===\n{expert_a_arg}\n")
        prompt_parts.append("\nNow provide your counter-argument, addressing Expert A's points.")
    else:
        prompt_parts.append("\nProvide your initial argument (Round 1).")

    prompt_parts.append("\n=== Instructions ===")
    prompt_parts.append("Maximum 300 words. Provide evidence-based reasoning using the retrieved WebMD knowledge.")

    user_prompt = "\n".join(prompt_parts)

    messages = [
        SystemMessage(content=EXPERT_B_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    resp = llm.invoke(messages)

    # Create debate argument
    new_argument: DebateArgument = {
        "round": current_round,
        "content": resp.content
    }

    # Update arguments list
    updated_arguments = state.get("expertB_arguments", []) + [new_argument]

    return {
        "expertB_arguments": updated_arguments,
        "expertB_retrieved_docs": retrieved_docs
    }
