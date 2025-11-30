from app.core.state import MedState
from app.core.prompts import SINGLE_AGENT_SYSTEM
from langchain_core.messages import SystemMessage, HumanMessage
from app.rag.smart_hybrid_retriever import get_smart_hybrid_retriever
from app.rag.metadata_filter import (
    determine_retrieval_strategy,
    get_category_split,
    format_retrieval_summary
)
from config.settings import settings
import json
import re


def single_agent_node(state: MedState, llm, knowledge_source: str = "Mayo") -> dict:
    """
    Single agent node for medical error detection.

    Args:
        state: Current state containing medical note
        llm: Language model to use
        knowledge_source: Which knowledge source to use ("Mayo" or "WebMD")

    Returns:
        Updated state with final decision
    """
    medical_note = state["medical_note"]

    # Determine which expert type to use based on knowledge source
    expert_type = "A" if knowledge_source == "Mayo" else "B"

    # Get smart hybrid retriever and fetch knowledge
    if settings.USE_RETRIEVER:
        smart_hybrid_retriever = get_smart_hybrid_retriever()

        # Determine optimal retrieval strategy based on note content
        primary_cat, secondary_cat = determine_retrieval_strategy(medical_note)
        k_per_query, primary_max, secondary_max = get_category_split(secondary_cat is not None)

        # Log retrieval strategy
        print(f"\n[Single Agent - {knowledge_source}] {format_retrieval_summary(primary_cat, secondary_cat, primary_max, secondary_max)}")

        if secondary_cat:
            # Multi-category retrieval
            primary_docs = smart_hybrid_retriever.retrieve_with_decomposition(
                note=medical_note,
                expert=expert_type,
                k_per_query=k_per_query,
                max_total=primary_max,
                filter_category=primary_cat
            )

            secondary_docs = smart_hybrid_retriever.retrieve_with_decomposition(
                note=medical_note,
                expert=expert_type,
                k_per_query=k_per_query,
                max_total=secondary_max,
                filter_category=secondary_cat
            )

            retrieved_docs = primary_docs + secondary_docs
        else:
            retrieved_docs = smart_hybrid_retriever.retrieve_with_decomposition(
                note=medical_note,
                expert=expert_type,
                k_per_query=2,
                max_total=5,
                use_dense=True,
                use_sparse=True,
                use_online=True
            )
    else:
        retrieved_docs = []

    # Build prompt
    prompt_parts = [f"Medical Note to Analyze:\n{medical_note}\n"]

    # Add retrieved knowledge
    if retrieved_docs:
        prompt_parts.append(f"\n=== Retrieved {knowledge_source} Clinic Knowledge ===")
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            prompt_parts.append(f"\nSource {i}:\n{content}\n")

    prompt_parts.append("\n=== Instructions ===")
    prompt_parts.append(f"Analyze the medical note using the retrieved {knowledge_source} knowledge.")
    prompt_parts.append("Use chain-of-thought reasoning to determine if there is a medical error.")
    prompt_parts.append("Provide your final answer in JSON format:")
    prompt_parts.append('{"Final Answer": "CORRECT" or "INCORRECT", "Confidence Score": <1-10>, "Reasoning": "<explanation>"}')

    user_prompt = "\n".join(prompt_parts)

    messages = [
        SystemMessage(content=SINGLE_AGENT_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    resp = llm.invoke(messages)

    # Parse JSON response
    final_answer = "UNKNOWN"
    decision_json = {}

    # Method 1: Try direct JSON parsing
    try:
        decision_json = json.loads(resp.content)
    except json.JSONDecodeError:
        pass

    # Method 2: Extract JSON from markdown code blocks
    if not decision_json:
        markdown_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', resp.content, re.DOTALL)
        if markdown_match:
            try:
                decision_json = json.loads(markdown_match.group(1))
            except json.JSONDecodeError:
                pass

    # Method 3: Greedy match for any JSON object
    if not decision_json:
        json_match = re.search(r'\{.*\}', resp.content, re.DOTALL)
        if json_match:
            try:
                decision_json = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    # Method 4: Extract Final Answer directly using regex
    if not decision_json:
        answer_match = re.search(r'"Final Answer":\s*"(CORRECT|INCORRECT)"', resp.content)
        if answer_match:
            final_answer = answer_match.group(1)

    # Extract final_answer from parsed JSON
    if decision_json:
        final_answer = decision_json.get("Final Answer") or decision_json.get("final_answer") or "UNKNOWN"

    return {
        "final_answer": final_answer,
        "single_agent_decision": resp.content,
        "single_agent_retrieved_docs": retrieved_docs,
        "knowledge_source": knowledge_source
    }
