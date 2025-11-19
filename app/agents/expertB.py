from app.core.state import MedState, DebateArgument
from app.core.prompts import EXPERT_B_SYSTEM
from langchain_core.messages import SystemMessage, HumanMessage
from app.rag.smart_retriever import get_smart_retriever
from config.settings import settings


def expertB_node(state: MedState, llm) -> dict:
    """
    Expert B node with WebMD access for debate system.
    Generates arguments in rounds, can see Expert A's previous arguments.
    Uses smart retriever with query decomposition (if settings.USE_RETRIEVER=True).
    """
    current_round = state["current_round"]
    medical_note = state["medical_note"]

    # Get smart retriever and fetch WebMD knowledge with query decomposition
    # Toggle via .env: USE_RETRIEVER=True or False
    # Optimization: Only retrieve in round 1, reuse documents in round 2
    if settings.USE_RETRIEVER:
        if current_round == 1:
            # Round 1: Perform retrieval
            smart_retriever = get_smart_retriever()
            retrieved_docs = smart_retriever.retrieve_with_decomposition(
                note=medical_note,
                expert="B",
                k_per_query=2,
                max_total=5
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
