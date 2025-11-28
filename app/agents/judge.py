from app.core.state import MedState
from app.core.prompts import JUDGE_SYSTEM
from app.utils.safety_rules import hybrid_safety_predict
from langchain_core.messages import SystemMessage, HumanMessage
import json


def Judge_node(state: MedState, llm) -> dict:
    """
    Judge node that evaluates arguments ONLY (no access to medical note).
    Makes final decision based on argument quality.
    Applies rule-based safety checks to prevent obvious false positives.
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

    # Try to parse JSON response with robust multi-method extraction
    import re
    final_answer = "UNKNOWN"
    decision_json = {}

    # Method 1: Try direct JSON parsing first
    try:
        decision_json = json.loads(resp.content)
    except json.JSONDecodeError:
        pass

    # Method 2: Extract JSON from markdown code blocks (```json ... ```)
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

    # Method 4: Extract Final Answer directly using regex as last fallback
    if not decision_json:
        answer_match = re.search(r'"Final Answer":\s*"(CORRECT|INCORRECT)"', resp.content)
        if answer_match:
            final_answer = answer_match.group(1)

    # Extract final_answer from parsed JSON
    if decision_json:
        final_answer = decision_json.get("Final Answer") or decision_json.get("final_answer") or "UNKNOWN"

    # ========================================
    # APPLY HYBRID SAFETY LAYER
    # ========================================

    # Get the medical note from state
    medical_note = state.get("medical_note", "")

    # Combine expert arguments for analysis
    expertA_content = "\n".join([arg['content'] for arg in expertA_args])
    expertB_content = "\n".join([arg['content'] for arg in expertB_args])

    # Apply comprehensive hybrid safety predictor
    safety_result = hybrid_safety_predict(
        note=medical_note,
        llm_final_classification=final_answer,
        expert_a_content=expertA_content,
        expert_b_content=expertB_content,
        judge_content=resp.content
    )

    final_classification = safety_result["final_classification"]

    # ========================================
    # LOGGING & DIAGNOSTICS
    # ========================================

    print("\n" + "="*60)
    print("HYBRID SAFETY LAYER RESULTS")
    print("="*60)

    # Show initial vs final
    if final_classification != safety_result["initial_llm_classification"]:
        print(f"OVERRIDE: {safety_result['initial_llm_classification']} → {final_classification}")
    else:
        print(f"✓ CONFIRMED: {final_classification}")

    # Show rule information
    if safety_result["rules"]["rule_applied"]:
        print(f"\nRULE TRIGGERED: {safety_result['rules']['rule_applied']}")
        print(f"   Reason: {safety_result['rules']['reason']}")

    if safety_result["rules"]["disagreement_override"]:
        print(f"\nDISAGREEMENT OVERRIDE:")
        print(f"   {safety_result['rules']['disagreement_override']}")

    # Show confidence adjustment
    if safety_result["confidence_adjustment"]["adjustment_applied"]:
        print(f"\nCONFIDENCE FILTER:")
        print(f"   {safety_result['confidence_adjustment']['reason']}")

    # Show term evidence
    print(f"\nTERM EVIDENCE:")
    expert_a_terms = safety_result["term_evidence"]["expert_a"]
    expert_b_terms = safety_result["term_evidence"]["expert_b"]

    print(f"   Expert A: Wrong='{expert_a_terms['wrong_term']}', Correct='{expert_a_terms['correct_term']}'")
    print(f"   Expert B: Wrong='{expert_b_terms['wrong_term']}', Correct='{expert_b_terms['correct_term']}'")

    # Show expert signals
    print(f"\nEXPERT SIGNALS:")
    expert_a_sig = safety_result["expert_signals"]["expert_a"]
    expert_b_sig = safety_result["expert_signals"]["expert_b"]

    print(f"   Expert A: Class={expert_a_sig['classification']}, Conf={expert_a_sig['confidence']}")
    print(f"   Expert B: Class={expert_b_sig['classification']}, Conf={expert_b_sig['confidence']}")

    if safety_result["expert_signals"]["disagreement"]:
        print(f"   ⚡ Experts disagree!")

    print(f"\nFINAL: {final_classification}")
    print("="*60 + "\n")

    return {
        "judge_decision": resp.content,
        "final_answer": final_classification,
        "safety_diagnostics": safety_result
    }
