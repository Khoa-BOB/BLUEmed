"""
Hybrid Safety Layer (Option C v2)
--------------------------------
This module implements:

1. Improved substitution term extraction
2. Improved rule-based safety checks
3. Full hybrid predictor for LLM expert debate systems

Usage:
------
result = hybrid_safety_predict(
    note=note_text,
    llm_final_classification=judge_label,
    expert_a_content=expert_a_output,
    expert_b_content=expert_b_output,
    judge_content=judge_output,
)

final_label = result["final_classification"]
"""

import re
from typing import Dict, Optional, Tuple, Any


# =====================================================================
# SECTION 1 — HELPERS: PARSE CLASSIFICATION + CONFIDENCE
# =====================================================================

def parse_expert_classification(text: str) -> Optional[str]:
    """Extract the expert's own stated final classification."""
    # Try multiple patterns to handle different expert response formats
    patterns = [
        r'FINAL CLASSIFICATION:\s*(CORRECT|INCORRECT)',  # Standard format
        r'5\.\s*FINAL CLASSIFICATION:\s*(CORRECT|INCORRECT)',  # Numbered with label
        r'5\.\s*(CORRECT|INCORRECT)',  # Just numbered line 5
        r'(?:^|\n)5\.\s+(CORRECT|INCORRECT)',  # Line starting with 5.
    ]

    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()

    return None


def extract_confidence_from_text(text: str) -> Optional[str]:
    """
    Extract confidence level (HIGH, MEDIUM, LOW) or percentage.
    Handles multiple formats:
    - CONFIDENCE: HIGH
    - CONFIDENCE: 90%
    - 6. CONFIDENCE: 10
    - 6. 90%
    """
    text_u = text.upper()

    # literal labels
    if "CONFIDENCE:" in text_u:
        if "HIGH" in text_u:
            return "HIGH"
        if "MEDIUM" in text_u:
            return "MEDIUM"
        if "LOW" in text_u:
            return "LOW"

    # percentage-based with CONFIDENCE: label
    m = re.search(r'CONFIDENCE:\s*(\d+)\s*%?', text_u)
    if m:
        val = int(m.group(1))
        if val >= 90:
            return "HIGH"
        elif val >= 70:
            return "MEDIUM"
        else:
            return "LOW"

    # percentage at line 6 (Expert B format: "6. 90%" or "6. 10")
    m = re.search(r'6\.\s*(\d+)\s*%?', text_u)
    if m:
        val = int(m.group(1))
        # If it's a score out of 10, convert to percentage
        if val <= 10:
            val = val * 10
        if val >= 90:
            return "HIGH"
        elif val >= 70:
            return "MEDIUM"
        else:
            return "LOW"

    return None


# =====================================================================
# SECTION 2 — IMPROVED SUBSTITUTION TERM EXTRACTION
# =====================================================================

CANONICAL_SUBSTITUTIONS = {
    "pseudoseizures": "seizures",
    "pseudo-seizures": "seizures",
    "pseudo seizure": "seizure",
    "pseudo seizures": "seizures",
    # add more canonical pairs here...
}


def extract_substitution_terms(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract explicit wrong_term and correct_term pairs from expert content.
    Handles both straight quotes and smart/curly quotes.
    """

    # Normalize all quote types to straight quotes for consistent matching
    cleaned = (
        text.replace(""", "\"")
            .replace(""", "\"")
            .replace("'", "'")
            .replace("'", "'")
    )

    # WRONG TERM PATTERNS
    # Match patterns with flexible quote handling (both straight " and any remaining curly quotes)
    wrong_patterns = [
        r'(?:the note (?:states|says)|diagnosed as|labeled as|called|documented as|referred to as|misdiagnosed as)\s*[:\-]?\s*["\u201c]([^"\u201d]+)["\u201d]',
        r'(?:wrong term|1\.\s*wrong term)\s*[:\-]?\s*["\u201c]([^"\u201d]+)["\u201d]',
        r'1\.\s*[Ww]rong\s+[Tt]erm\s*[:\-]?\s*["\u201c]([^"\u201d]+)["\u201d]',
        r'1\.\s*[:\-]?\s*["\u201c]([^"\u201d]+)["\u201d]',
    ]

    # CORRECT TERM PATTERNS
    correct_patterns = [
        r'(?:should be|should instead be|correct term (?:is|should be))\s*[:\-]?\s*["\u201c]([^"\u201d]+)["\u201d]',
        r'(?:correct term|2\.\s*correct term)\s*[:\-]?\s*["\u201c]([^"\u201d]+)["\u201d]',
        r'2\.\s*[Cc]orrect\s+[Tt]erm\s*[:\-]?\s*["\u201c]([^"\u201d]+)["\u201d]',
        r'should have been (?:diagnosed|labeled|documented) as\s*[:\-]?\s*["\u201c]([^"\u201d]+)["\u201d]',
        r'true (?:diagnosis|condition) (?:is|would be)\s*[:\-]?\s*["\u201c]([^"\u201d]+)["\u201d]',
        r'2\.\s*[:\-]?\s*["\u201c]([^"\u201d]+)["\u201d]',
    ]

    wrong = None
    correct = None

    # extract wrong
    for p in wrong_patterns:
        m = re.search(p, cleaned, re.IGNORECASE)
        if m:
            wrong = m.group(1).strip()
            break

    # extract correct
    for p in correct_patterns:
        m = re.search(p, cleaned, re.IGNORECASE)
        if m:
            correct = m.group(1).strip()
            break

    # canonical fallback
    lowered = cleaned.lower()
    if wrong is None:
        for k, v in CANONICAL_SUBSTITUTIONS.items():
            if k in lowered:
                wrong = k
                correct = v
                break

    # sanity
    if wrong and correct and wrong.lower() == correct.lower():
        return None, None

    return wrong, correct


# =====================================================================
# SECTION 3 — IMPROVED RULE-BASED SAFETY CHECKS (v2)
# =====================================================================

def rule_based_safety_check(
    note: str,
    llm_classification: str,
    expert_a_content: str,
    expert_b_content: str,
    wrong_a: Optional[str],
    correct_a: Optional[str],
    wrong_b: Optional[str],
    correct_b: Optional[str],
) -> Dict[str, Any]:
    """
    Rule-based overrides ONLY for CLEAR false positives.
    Conservative approach - only override when absolutely certain.
    """

    all_c = f"{expert_a_content} {expert_b_content}".lower()
    note_l = note.lower()
    has_pair = (wrong_a and correct_a) or (wrong_b and correct_b)

    classification = llm_classification.upper()

    # IMPORTANT: Only apply these rules if classification is INCORRECT
    if classification != "INCORRECT":
        return {
            "classification": classification,
            "rule_applied": None,
            "reason": None,
        }

    # RULE 1 — Culture/lab confirmation (ONLY if NO term pair AND explicit culture in note)
    # Be strict: only trigger if culture is in the note AND no terms extracted
    if not has_pair:
        if re.search(r'culture (test[s]?|result[s]?)?\s*(show[s]?|indicate[s]?|confirm[s]?|positive for)\s+[\w\s]+', note_l):
            return {
                "classification": "CORRECT",
                "rule_applied": "RULE_1_CULTURE_DEFINITIVE",
                "reason": "Lab/culture explicitly confirms diagnosis in note; likely not a substitution.",
            }

    # RULE 2 — Process issues (ONLY if NO term pair AND strong process language)
    # Require BOTH: no term pair AND multiple process indicators
    if not has_pair:
        process_patterns = [
            r'should have (done|ordered|performed|confirmed|tested)',
            r'did not (order|perform|confirm|test)',
            r'failed to (order|perform|confirm|test)',
            r'without (testing|confirmation|lab work|imaging)',
            r'premature (diagnosis|conclusion)',
            r'lack of confirmation',
            r'unconfirmed (diagnosis|infection)',
        ]

        process_count = sum(1 for p in process_patterns if re.search(p, all_c))

        # Only override if 2+ process phrases (stronger signal)
        if process_count >= 2:
            return {
                "classification": "CORRECT",
                "rule_applied": "RULE_2_PROCESS_ISSUE",
                "reason": f"Multiple process-issue indicators ({process_count}); not a substitution error.",
            }

    # RULE 3 — Side-effect / reaction (ONLY if NO term pair AND clear side effect language)
    if not has_pair:
        side_patterns = [
            r'(side effect|adverse effect|adverse reaction)',
            r'expected (reaction|symptom)',
            r'(drug|medication).{0,40}(interaction|reaction)',
        ]
        side_count = sum(1 for p in side_patterns if re.search(p, all_c))

        if side_count >= 1:
            return {
                "classification": "CORRECT",
                "rule_applied": "RULE_3_SIDE_EFFECT",
                "reason": "Side-effect discussion rather than wrong-term substitution.",
            }

    # RULE 4 — Hierarchical terms (ONLY if NO term pair AND hierarchical language)
    if not has_pair:
        hierarchical = [
            r'more specific',
            r'less specific',
            r'broader term',
            r'hierarchically related',
            r'same (organism|pathogen)',
        ]

        if any(re.search(p, all_c) for p in hierarchical):
            return {
                "classification": "CORRECT",
                "rule_applied": "RULE_4_HIERARCHICAL",
                "reason": "Experts describe hierarchical differences; not substitution.",
            }

    # RULE 5 — Strong uncertainty (ONLY if NO term pair AND high uncertainty)
    if not has_pair:
        unc_patterns = [
            r'could be',
            r'might be',
            r'possibly',
            r'perhaps',
            r'may be',
            r'alternative but valid',
            r'both valid',
        ]
        unc_count = sum(1 for p in unc_patterns if re.search(p, all_c))

        # Require 3+ uncertainty phrases (very conservative)
        if unc_count >= 3:
            return {
                "classification": "CORRECT",
                "rule_applied": "RULE_5_UNCERTAINTY",
                "reason": f"Very high uncertainty ({unc_count} phrases); not definitive error.",
            }

    # No rules triggered - keep original classification
    return {
        "classification": classification,
        "rule_applied": None,
        "reason": None,
    }


# =====================================================================
# SECTION 4 — CONFIDENCE-BASED ADJUSTMENT
# =====================================================================

def confidence_adjusted_predict(
    classification: str,
    conf_a: Optional[str],
    conf_b: Optional[str],
) -> Dict[str, Any]:
    """
    Flip INCORRECT → CORRECT ONLY IF BOTH experts have very low confidence.
    More conservative - trusts experts more.
    """

    if classification != "INCORRECT":
        return {"classification": classification, "adjustment_applied": False, "reason": None}

    # Only flip if BOTH are missing confidence scores (very conservative)
    # We trust experts even if they say LOW - they still identified a substitution
    both_missing = (conf_a is None) and (conf_b is None)

    if both_missing:
        return {
            "classification": "CORRECT",
            "adjustment_applied": True,
            "reason": f"Both experts missing confidence scores - conservative override (A={conf_a}, B={conf_b})."
        }

    return {"classification": classification, "adjustment_applied": False, "reason": None}


# =====================================================================
# SECTION 5 — FULL HYBRID SAFETY PREDICTOR (Option C v2)
# =====================================================================

def hybrid_safety_predict(
    note: str,
    llm_final_classification: str,
    expert_a_content: str,
    expert_b_content: str,
    judge_content: str = "",
) -> Dict[str, Any]:
    """
    Full hybrid substitution-error safety predictor.
    """

    current = llm_final_classification.upper()

    # extract terms
    wrong_a, correct_a = extract_substitution_terms(expert_a_content)
    wrong_b, correct_b = extract_substitution_terms(expert_b_content)
    any_pair = (wrong_a and correct_a) or (wrong_b and correct_b)

    # expert signals
    class_a = parse_expert_classification(expert_a_content)
    class_b = parse_expert_classification(expert_b_content)
    conf_a = extract_confidence_from_text(expert_a_content)
    conf_b = extract_confidence_from_text(expert_b_content)

    # rules
    rule_result = rule_based_safety_check(
        note,
        current,
        expert_a_content,
        expert_b_content,
        wrong_a,
        correct_a,
        wrong_b,
        correct_b,
    )
    current = rule_result["classification"]

    # CRITICAL: If both experts agree on INCORRECT and BOTH have term pairs, TRUST THEM
    # This overrides BOTH the Judge AND the rule-based checks
    both_have_pairs = bool(wrong_a and correct_a) and bool(wrong_b and correct_b)
    both_agree_incorrect = (class_a == "INCORRECT") and (class_b == "INCORRECT")

    # Debug logging
    print(f"\n[DEBUG CONSENSUS CHECK]")
    print(f"  Expert A: class={class_a}, wrong={wrong_a}, correct={correct_a}, conf={conf_a}")
    print(f"  Expert B: class={class_b}, wrong={wrong_b}, correct={correct_b}, conf={conf_b}")
    print(f"  both_agree_incorrect={both_agree_incorrect}, both_have_pairs={both_have_pairs}")
    print(f"  current={current}")
    print(f"  llm_classification={llm_final_classification}")

    # OVERRIDE: If both experts agree INCORRECT with evidence, trust them regardless of Judge
    if both_agree_incorrect and both_have_pairs:
        # Strong consensus with evidence - don't override
        print(f"  → TRIGGERING CONSENSUS_WITH_EVIDENCE - keeping INCORRECT")
        return {
            "final_classification": "INCORRECT",
            "initial_llm_classification": llm_final_classification.upper(),
            "rules": {
                "rule_applied": "CONSENSUS_WITH_EVIDENCE",
                "reason": "Both experts agree INCORRECT and both provided term pairs - strong consensus",
            },
            "confidence_adjustment": {"classification": "INCORRECT", "adjustment_applied": False, "reason": None},
            "term_evidence": {
                "expert_a": {"wrong": wrong_a, "correct": correct_a},
                "expert_b": {"wrong": wrong_b, "correct": correct_b},
            },
            "expert_signals": {
                "expert_a": {"classification": class_a, "confidence": conf_a},
                "expert_b": {"classification": class_b, "confidence": conf_b},
                "disagreement": False,
            }
        }

    # hard two-term rule (only apply if no consensus or no pairs)
    if current == "INCORRECT" and not any_pair:
        return {
            "final_classification": "CORRECT",
            "initial_llm_classification": llm_final_classification.upper(),
            "rules": {
                "rule_applied": rule_result["rule_applied"] or "RULE_HARD_TWO_TERM",
                "reason": rule_result["reason"] or "No wrong+correct term pair detected.",
            },
            "confidence_adjustment": None,
            "term_evidence": {
                "expert_a": {"wrong": wrong_a, "correct": correct_a},
                "expert_b": {"wrong": wrong_b, "correct": correct_b},
            },
            "expert_signals": {
                "expert_a": {"classification": class_a, "confidence": conf_a},
                "expert_b": {"classification": class_b, "confidence": conf_b},
                "disagreement": class_a != class_b if class_a and class_b else False,
            }
        }

    # expert disagreement handling (only override without pair)
    experts_disagree = (class_a and class_b and class_a != class_b)
    disagreement_override = None
    if current == "INCORRECT" and experts_disagree and not any_pair:
        current = "CORRECT"
        disagreement_override = "Experts disagree with no strong substitution evidence."

    # confidence adjustment
    conf_result = confidence_adjusted_predict(current, conf_a, conf_b)
    current = conf_result["classification"]

    return {
        "final_classification": current,
        "initial_llm_classification": llm_final_classification.upper(),
        "rules": {
            "rule_applied": rule_result["rule_applied"],
            "reason": rule_result["reason"],
            "disagreement_override": disagreement_override,
        },
        "confidence_adjustment": conf_result,
        "term_evidence": {
            "expert_a": {"wrong": wrong_a, "correct": correct_a},
            "expert_b": {"wrong": wrong_b, "correct": correct_b},
        },
        "expert_signals": {
            "expert_a": {"classification": class_a, "confidence": conf_a},
            "expert_b": {"classification": class_b, "confidence": conf_b},
            "disagreement": experts_disagree,
        }
    }
