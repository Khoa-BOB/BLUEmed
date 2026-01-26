"""
Weighted Voting System for Multi-Agent Decisions.

Provides an optional layer to combine expert and judge decisions with configurable weights.
Default weights: Judge 40%, Expert A 30%, Expert B 30%
"""
import re
from typing import Dict, Optional
from config.settings import settings


def extract_expert_classification(expert_argument: str) -> Optional[str]:
    """
    Extract classification from expert argument.

    Args:
        expert_argument: Expert's argument text

    Returns:
        "CORRECT" or "INCORRECT" or None if not found
    """
    patterns = [
        r'FINAL CLASSIFICATION:\s*(CORRECT|INCORRECT)',
        r'5\.\s*FINAL CLASSIFICATION:\s*(CORRECT|INCORRECT)',
        r'5\.\s*(CORRECT|INCORRECT)',
        r'Based on my analysis,\s*this note is\s*(CORRECT|INCORRECT)',
        r'Classification:\s*(CORRECT|INCORRECT)',
        r'My classification:\s*(CORRECT|INCORRECT)',
    ]

    for pattern in patterns:
        match = re.search(pattern, expert_argument, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


def calculate_weighted_decision(
    judge_classification: str,
    expert_a_arguments: list,
    expert_b_arguments: list,
    weight_judge: float = None,
    weight_expert_a: float = None,
    weight_expert_b: float = None
) -> Dict:
    """
    Calculate weighted decision based on all agents' classifications.

    Args:
        judge_classification: Judge's final decision (CORRECT or INCORRECT)
        expert_a_arguments: List of Expert A arguments
        expert_b_arguments: List of Expert B arguments
        weight_judge: Weight for judge (default from settings)
        weight_expert_a: Weight for expert A (default from settings)
        weight_expert_b: Weight for expert B (default from settings)

    Returns:
        Dictionary with weighted decision results
    """
    # Use settings weights if not provided
    if weight_judge is None:
        weight_judge = settings.WEIGHT_JUDGE
    if weight_expert_a is None:
        weight_expert_a = settings.WEIGHT_EXPERT_A
    if weight_expert_b is None:
        weight_expert_b = settings.WEIGHT_EXPERT_B

    # Normalize weights to sum to 1.0
    total_weight = weight_judge + weight_expert_a + weight_expert_b
    weight_judge = weight_judge / total_weight
    weight_expert_a = weight_expert_a / total_weight
    weight_expert_b = weight_expert_b / total_weight

    # Extract expert classifications from their latest arguments
    expert_a_class = None
    if expert_a_arguments:
        # Use the latest argument (could be round 1 or 2)
        latest_arg = expert_a_arguments[-1]["content"]
        expert_a_class = extract_expert_classification(latest_arg)

    expert_b_class = None
    if expert_b_arguments:
        # Use the latest argument
        latest_arg = expert_b_arguments[-1]["content"]
        expert_b_class = extract_expert_classification(latest_arg)

    # Calculate weighted scores for INCORRECT (error detected)
    # Score of 1.0 = INCORRECT, Score of 0.0 = CORRECT
    judge_score = 1.0 if judge_classification == "INCORRECT" else 0.0
    expert_a_score = 1.0 if expert_a_class == "INCORRECT" else 0.0
    expert_b_score = 1.0 if expert_b_class == "INCORRECT" else 0.0

    # Calculate weighted total
    weighted_total = (
        judge_score * weight_judge +
        expert_a_score * weight_expert_a +
        expert_b_score * weight_expert_b
    )

    # Final decision: if weighted total >= 0.5, classify as INCORRECT
    final_decision = "INCORRECT" if weighted_total >= 0.5 else "CORRECT"

    # Calculate confidence based on how decisive the vote was
    # If all agree: high confidence (close to 1.0 or 0.0)
    # If split: lower confidence (close to 0.5)
    confidence = abs(weighted_total - 0.5) * 2.0  # Scale to 0-1 range

    result = {
        "final_decision": final_decision,
        "weighted_score": weighted_total,  # 0.0 = CORRECT, 1.0 = INCORRECT
        "confidence": confidence,
        "weights": {
            "judge": weight_judge,
            "expert_a": weight_expert_a,
            "expert_b": weight_expert_b
        },
        "individual_classifications": {
            "judge": judge_classification,
            "expert_a": expert_a_class or "UNKNOWN",
            "expert_b": expert_b_class or "UNKNOWN"
        },
        "individual_scores": {
            "judge": judge_score,
            "expert_a": expert_a_score,
            "expert_b": expert_b_score
        },
        "agreement": {
            "all_agree": (judge_classification == expert_a_class == expert_b_class),
            "majority": sum([
                judge_classification == "INCORRECT",
                expert_a_class == "INCORRECT",
                expert_b_class == "INCORRECT"
            ]) >= 2
        }
    }

    return result


def print_weighted_decision_summary(weighted_result: Dict):
    """
    Print a formatted summary of the weighted decision.

    Args:
        weighted_result: Result from calculate_weighted_decision
    """
    print("\n" + "="*60)
    print("WEIGHTED VOTING RESULTS")
    print("="*60)

    # Individual classifications
    ind = weighted_result["individual_classifications"]
    print(f"\nIndividual Classifications:")
    print(f"  Judge:    {ind['judge']}")
    print(f"  Expert A: {ind['expert_a']} (Mayo Clinic)")
    print(f"  Expert B: {ind['expert_b']} (WebMD)")

    # Weights
    weights = weighted_result["weights"]
    print(f"\nWeights Applied:")
    print(f"  Judge:    {weights['judge']:.1%}")
    print(f"  Expert A: {weights['expert_a']:.1%}")
    print(f"  Expert B: {weights['expert_b']:.1%}")

    # Weighted score
    score = weighted_result["weighted_score"]
    print(f"\nWeighted Score: {score:.3f}")
    print(f"  (0.0 = CORRECT, 1.0 = INCORRECT, threshold = 0.5)")

    # Agreement info
    agreement = weighted_result["agreement"]
    if agreement["all_agree"]:
        print(f"\n All agents agree!")
    elif agreement["majority"]:
        print(f"\n Majority decision")
    else:
        print(f"\n Split decision")

    # Final decision
    print(f"\n WEIGHTED FINAL DECISION: {weighted_result['final_decision']}")
    print(f"   Confidence: {weighted_result['confidence']:.1%}")
    print("="*60 + "\n")
