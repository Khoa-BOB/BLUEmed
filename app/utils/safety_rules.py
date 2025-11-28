"""
Hybrid safety layer for substitution-error classification.

Goal:
- Preserve high recall for true INCORRECT notes
- Dramatically reduce false positives caused by:
  - process issues (missing labs, premature diagnosis)
  - vague/uncertain reasoning
  - hallucinated substitutions
"""

import re
from typing import Dict, Optional, Tuple, Any


# =========================
# Helper: Parse expert classification + confidence
# =========================

def parse_expert_classification(text: str) -> Optional[str]:
    """
    Parse expert's own final classification from their content.

    Looks for:
      FINAL CLASSIFICATION: CORRECT
      FINAL CLASSIFICATION: INCORRECT
    """
    m = re.search(r'FINAL CLASSIFICATION:\s*(CORRECT|INCORRECT)', text, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).upper()


def extract_confidence_from_text(text: str) -> Optional[str]:
    """
    Extract confidence level from expert arguments.

    Returns:
        "HIGH", "MEDIUM", "LOW", or None if not found
    """
    text_upper = text.upper()

    if re.search(r'CONFIDENCE:\s*HIGH', text_upper):
        return "HIGH"
    elif re.search(r'CONFIDENCE:\s*MEDIUM', text_upper):
        return "MEDIUM"
    elif re.search(r'CONFIDENCE:\s*LOW', text_upper):
        return "LOW"

    return None


# =========================
# Helper: Extract wrong/correct term pairs
# =========================

def extract_substitution_terms(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to extract (wrong_term, correct_term) from an expert's explanation.

    We rely on patterns like:
      "The note states: "X""
      "The note says: "X""
      "This should be: "Y""
      "The correct term is: "Y""
      "This should instead be: "Y""

    Returns:
        (wrong_term, correct_term) or (None, None) if not found
    """
    # Normalize quotes a bit
    cleaned = text.replace("“", "\"").replace("”", "\"")

    # WRONG term patterns
    wrong_patterns = [
        r'The note (states|says)\s*[:\-]?\s*"([^"]+)"',
        r'Stated diagnosis\s*[:\-]?\s*"([^"]+)"',
        r'Wrong term\s*[:\-]?\s*"([^"]+)"',
    ]

    # CORRECT term patterns
    correct_patterns = [
        r'(should be|should instead be|correct term (is|should be))\s*[:\-]?\s*"([^"]+)"',
        r'should have been documented as\s*[:\-]?\s*"([^"]+)"',
        r'should instead say\s*[:\-]?\s*"([^"]+)"',
    ]

    wrong_term = None
    correct_term = None

    for p in wrong_patterns:
        m = re.search(p, cleaned, re.IGNORECASE)
        if m:
            # last group usually the quoted term
            wrong_term = m.groups()[-1].strip()
            break

    for p in correct_patterns:
        m = re.search(p, cleaned, re.IGNORECASE)
        if m:
            # last group usually the quoted term
            correct_term = m.groups()[-1].strip()
            break

    # sanity check: identical terms don't count as substitution
    if wrong_term and correct_term and wrong_term.lower() == correct_term.lower():
        return None, None

    return wrong_term, correct_term


# =========================
# Rule-based safety checks (improved)
# =========================

def rule_based_safety_check(
    note: str,
    llm_classification: str,
    expert_a_content: str = "",
    expert_b_content: str = "",
) -> Dict[str, Any]:
    """
    Apply rule-based safety checks to catch obvious false positives.

    Returns:
        Dict with:
            - classification: Possibly overridden classification
            - rule_applied: Name of rule (or None)
            - reason: Explanation
    """

    all_content = f"{expert_a_content} {expert_b_content}".lower()
    note_lower = note.lower()

    # RULE 1: Culture / lab results explicitly confirm the diagnosis → not substitution
    if re.search(r'culture (test[s]?|result[s]?)?\s*(show[s]?|indicate[s]?|confirm[s]?|positive for|grew|grows)', note_lower):
        if llm_classification == "INCORRECT":
            return {
                "classification": "CORRECT",
                "rule_applied": "RULE_1_CULTURE_DEFINITIVE",
                "reason": "Culture/lab results indicate definitive confirmation, not a substitution error"
            }

    # RULE 2: Process-issue language: missing tests, premature diagnosis → not substitution
    process_patterns = [
        r'should have (done|ordered|performed|confirmed|tested)',
        r'did not (order|perform|confirm|test)',
        r'failed to (order|perform|confirm|test)',
        r'without (testing|confirmation|lab work|imaging)',
        r'premature (diagnosis|conclusion)',
        r'lack[s]? (of )?confirmation',
        r'unconfirmed (diagnosis|infection)'
    ]

    for pattern in process_patterns:
        if re.search(pattern, all_content):
            if llm_classification == "INCORRECT":
                return {
                    "classification": "CORRECT",
                    "rule_applied": "RULE_2_PROCESS_ISSUE",
                    "reason": "Detected process-related critique (missing tests, premature diagnosis), not term substitution"
                }

    # RULE 3: Side-effect / reaction language → usually medication reaction, not wrong diagnosis
    side_effect_patterns = [
        r'(side effect[s]?|adverse effect[s]?|reaction[s]?) (of|from|to)',
        r'disulfiram.{0,30}reaction',
        r'expected (side effect|reaction|symptom)',
        r'(medication|drug).{0,30}(interaction|reaction)',
    ]

    for pattern in side_effect_patterns:
        if re.search(pattern, all_content):
            if llm_classification == "INCORRECT":
                return {
                    "classification": "CORRECT",
                    "rule_applied": "RULE_3_SIDE_EFFECT",
                    "reason": "Discussion centers on medication side effects/reactions, not substituted diagnosis/term"
                }

    # RULE 4: Hierarchical broader/specific terminology → not substitution
    hierarchical_patterns = [
        r'(more|less) specific',
        r'broader (term|category)',
        r'(genus|species|class) vs',
        r'hierarchically related',
        r'same (organism|pathogen|disease|condition)',
        r'(is|are) just (more|less) specific'
    ]

    for pattern in hierarchical_patterns:
        if re.search(pattern, all_content):
            if llm_classification == "INCORRECT":
                return {
                    "classification": "CORRECT",
                    "rule_applied": "RULE_4_HIERARCHICAL_TERMS",
                    "reason": "Expert reasoning indicates hierarchical relation (broad vs specific), not a true substitution"
                }

    # RULE 5: Strong uncertainty language across expert arguments → not definitive error
    uncertainty_patterns = [
        r'could (be|have been|also be)',
        r'might (be|have been)',
        r'possibly',
        r'perhaps',
        r'may be',
        r'alternative (but valid|approach|option)',
        r'both (are )?valid'
    ]

    uncertainty_count = 0
    for pattern in uncertainty_patterns:
        if re.search(pattern, all_content):
            uncertainty_count += 1

    if llm_classification == "INCORRECT" and uncertainty_count >= 2:
        return {
            "classification": "CORRECT",
            "rule_applied": "RULE_5_UNCERTAINTY",
            "reason": f"Multiple uncertainty phrases ({uncertainty_count}) suggest lack of definitive substitution error"
        }

    # No rules triggered
    return {
        "classification": llm_classification,
        "rule_applied": None,
        "reason": None
    }


# =========================
# Confidence-based adjustment (weak supervision)
# =========================

def confidence_adjusted_predict(
    classification: str,
    expert_a_content: str = "",
    expert_b_content: str = "",
) -> Dict[str, Any]:
    """
    Flip low/medium confidence INCORRECT predictions to CORRECT.

    This reduces false positives by requiring high confidence for INCORRECT.
    """

    if classification != "INCORRECT":
        return {
            "classification": classification,
            "adjustment_applied": False,
            "reason": None
        }

    conf_a = extract_confidence_from_text(expert_a_content)
    conf_b = extract_confidence_from_text(expert_b_content)

    # If either expert is LOW/MEDIUM → be conservative and flip to CORRECT
    if conf_a in ["LOW", "MEDIUM"] or conf_b in ["LOW", "MEDIUM"]:
        return {
            "classification": "CORRECT",
            "adjustment_applied": True,
            "reason": f"Flipped INCORRECT to CORRECT due to low/medium confidence (A: {conf_a}, B: {conf_b})"
        }

    # If we can't read confidence reliably → conservative default
    if conf_a is None or conf_b is None:
        return {
            "classification": "CORRECT",
            "adjustment_applied": True,
            "reason": "Flipped INCORRECT to CORRECT due to missing confidence scores (conservative default)"
        }

    return {
        "classification": classification,
        "adjustment_applied": False,
        "reason": None
    }


# =========================
# Hybrid predictor (Option C)
# =========================

def hybrid_safety_predict(
    note: str,
    llm_final_classification: str,
    expert_a_content: str,
    expert_b_content: str,
    judge_content: str = "",
) -> Dict[str, Any]:
    """
    Hybrid safety layer that combines:
    - rule-based checks
    - weak supervision (confidence, expert disagreement)
    - two-term extraction
    - judge-independent override

    Returns:
        {
          "final_classification": "CORRECT"/"INCORRECT",
          "initial_llm_classification": ...,
          "rules": {...},
          "confidence_adjustment": {...},
          "term_evidence": {...},
          "expert_signals": {...}
        }
    """

    # Step 1: Start with LLM/Judge classification
    current = llm_final_classification.upper()

    # Step 2: Rule-based overrides (process issues, uncertainty, etc.)
    rule_result = rule_based_safety_check(
        note=note,
        llm_classification=current,
        expert_a_content=expert_a_content,
        expert_b_content=expert_b_content,
    )
    current = rule_result["classification"]

    # Step 3: Extract wrong/correct terms from each expert
    wrong_a, correct_a = extract_substitution_terms(expert_a_content)
    wrong_b, correct_b = extract_substitution_terms(expert_b_content)

    # Step 4: Expert classifications + confidence (weak supervision signals)
    class_a = parse_expert_classification(expert_a_content)
    class_b = parse_expert_classification(expert_b_content)
    conf_a = extract_confidence_from_text(expert_a_content)
    conf_b = extract_confidence_from_text(expert_b_content)

    # Step 5: Enforce hard two-term rule for INCORRECT
    any_strong_pair = (wrong_a and correct_a) or (wrong_b and correct_b)
    if current == "INCORRECT" and not any_strong_pair:
        # No explicit substitution pair found → override to CORRECT
        return {
            "final_classification": "CORRECT",
            "initial_llm_classification": llm_final_classification.upper(),
            "rules": {
                "rule_applied": rule_result["rule_applied"] or "RULE_HARD_TWO_TERM",
                "reason": rule_result["reason"] or "INCORRECT requires explicit wrong+correct term pair from at least one expert"
            },
            "confidence_adjustment": None,
            "term_evidence": {
                "expert_a": {"wrong_term": wrong_a, "correct_term": correct_a},
                "expert_b": {"wrong_term": wrong_b, "correct_term": correct_b},
            },
            "expert_signals": {
                "expert_a": {"classification": class_a, "confidence": conf_a},
                "expert_b": {"classification": class_b, "confidence": conf_b},
                "disagreement": class_a is not None and class_b is not None and class_a != class_b
            }
        }

    # Step 6: Handle expert disagreement as weak supervision
    experts_disagree = (class_a is not None and class_b is not None and class_a != class_b)

    if current == "INCORRECT" and experts_disagree and not any_strong_pair:
        # If experts disagree and we don't have strong term evidence, be conservative
        current = "CORRECT"
        disagreement_reason = "Experts disagree on classification and no strong term pair present → conservative CORRECT"
    else:
        disagreement_reason = None

    # Step 7: Confidence-based adjustment
    conf_result = confidence_adjusted_predict(
        classification=current,
        expert_a_content=expert_a_content,
        expert_b_content=expert_b_content,
    )
    current = conf_result["classification"]

    return {
        "final_classification": current,
        "initial_llm_classification": llm_final_classification.upper(),
        "rules": {
            "rule_applied": rule_result["rule_applied"],
            "reason": rule_result["reason"],
            "disagreement_override": disagreement_reason,
        },
        "confidence_adjustment": conf_result,
        "term_evidence": {
            "expert_a": {"wrong_term": wrong_a, "correct_term": correct_a},
            "expert_b": {"wrong_term": wrong_b, "correct_term": correct_b},
        },
        "expert_signals": {
            "expert_a": {"classification": class_a, "confidence": conf_a},
            "expert_b": {"classification": class_b, "confidence": conf_b},
            "disagreement": experts_disagree
        }
    }
