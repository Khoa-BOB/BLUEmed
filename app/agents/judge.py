from app.core.state import MedState
from app.core.prompts import JUDGE_SYSTEM
from langchain_core.messages import SystemMessage, HumanMessage
import json


def Judge_node(state: MedState, llm) -> dict:
    """
    Judge node that evaluates arguments ONLY (no access to medical note).
    Makes final decision based on argument quality.
    """
    # Collect all arguments from both experts
    expertA_args = state.get("expertA_arguments", [])
    expertB_args = state.get("expertB_arguments", [])

    # Build prompt with ONLY the arguments (no medical note)
    prompt_parts = ["Here is the complete debate between Expert A and Expert B:\n"]

    prompt_parts.append("\n=== EXPERT A (Mayo Clinic) ARGUMENTS ===")
    for arg in expertA_args:
        prompt_parts.append(f"\nRound {arg['round']}:")
        prompt_parts.append(arg['content'])

    prompt_parts.append("\n\n=== EXPERT B (WebMD) ARGUMENTS ===")
    for arg in expertB_args:
        prompt_parts.append(f"\nRound {arg['round']}:")
        prompt_parts.append(arg['content'])

    prompt_parts.append("\n\nNow evaluate both experts' arguments and make your final decision.")
    prompt_parts.append("Respond in JSON format as specified in your instructions.")

    user_prompt = "\n".join(prompt_parts)

    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    resp = llm.invoke(messages)

    # Try to parse JSON response
    final_answer = "UNKNOWN"
    try:
        # Try direct JSON parsing first
        decision_json = json.loads(resp.content)
        final_answer = decision_json.get("Final Answer", "UNKNOWN")
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{[^}]*"Final Answer"[^}]*\}', resp.content, re.DOTALL)
        if json_match:
            try:
                decision_json = json.loads(json_match.group(0))
                final_answer = decision_json.get("Final Answer", "UNKNOWN")
            except json.JSONDecodeError:
                # Extract directly using regex as fallback
                answer_match = re.search(r'"Final Answer":\s*"(CORRECT|INCORRECT)"', resp.content)
                if answer_match:
                    final_answer = answer_match.group(1)

    return {
        "judge_decision": resp.content,
        "final_answer": final_answer
    }
