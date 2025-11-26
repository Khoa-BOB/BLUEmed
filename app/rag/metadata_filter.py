"""
Hybrid metadata filtering strategy for RAG retrieval.

Dynamically determines whether to use single-category or multi-category
retrieval based on medical note analysis.
"""
from typing import Tuple, Optional


def determine_retrieval_strategy(medical_note: str) -> Tuple[str, Optional[str]]:
    """
    Analyze medical note to determine optimal retrieval strategy.

    Strategy Logic:
    1. If both diagnosis AND medication keywords present → Multi-category (diseases + drugs)
    2. If medication keywords dominate → Single category (drugs_supplements)
    3. Otherwise (default) → Single category (diseases_conditions)

    Args:
        medical_note: Medical note text to analyze

    Returns:
        Tuple of (primary_category, secondary_category)
        - If secondary_category is None: Use single-category retrieval
        - If secondary_category is set: Use multi-category retrieval

    Examples:
        >>> determine_retrieval_strategy("Patient diagnosed with pneumonia")
        ("diseases_conditions", None)

        >>> determine_retrieval_strategy("Prescribed metformin 500mg for diabetes")
        ("drugs_supplements", "diseases_conditions")

        >>> determine_retrieval_strategy("Patient with kidney disease prescribed metformin")
        ("diseases_conditions", "drugs_supplements")
    """
    note_lower = medical_note.lower()

    # Diagnosis/disease-related keywords
    diagnosis_keywords = [
        "diagnosed", "diagnosis", "suspected", "infection", "disease",
        "condition", "syndrome", "presents with", "symptoms are",
        "clinical picture", "pathogen", "organism", "virus", "bacteria",
        "parasite", "differential diagnosis", "ruled out", "confirmed",
        "manifestation", "complication", "etiology"
    ]

    # Medication/treatment-related keywords
    medication_keywords = [
        "prescribed", "medication", "drug", "treatment with", "started on",
        "administered", "therapy", "dose", "dosage", "mg", "ml", "units",
        "contraindicated", "interaction", "adverse", "side effect",
        "antibiotic", "antiviral", "antimicrobial", "pharmaceutical",
        "prescription", "regimen", "titrated"
    ]

    # Count keyword matches
    diagnosis_score = sum(1 for keyword in diagnosis_keywords if keyword in note_lower)
    medication_score = sum(1 for keyword in medication_keywords if keyword in note_lower)

    # Decision logic with thresholds
    if diagnosis_score > 0 and medication_score >= 2:
        # Mixed case: Both diagnosis and significant medication mentions
        # Use multi-category: 60% disease + 40% drug
        return ("diseases_conditions", "drugs_supplements")

    elif medication_score > diagnosis_score and medication_score >= 2:
        # Medication-focused case
        # But also get some disease context
        return ("drugs_supplements", "diseases_conditions")

    else:
        # Default: Diagnosis-focused case (most common for error detection)
        # Single category retrieval for maximum depth
        return ("diseases_conditions", None)


def get_category_split(has_secondary: bool) -> Tuple[int, int, int]:
    """
    Get optimal document count split for single vs multi-category retrieval.

    Args:
        has_secondary: Whether secondary category is present

    Returns:
        Tuple of (k_per_query, primary_max, secondary_max)

    Single-category: (2, 5, 0) → 5 docs from 1 category
    Multi-category: (1, 3, 2) → 3 docs from primary + 2 from secondary
    """
    if has_secondary:
        # Multi-category: Split budget 60/40
        return (1, 3, 2)  # k_per_query=1, primary=3, secondary=2
    else:
        # Single-category: Full budget on one category
        return (2, 5, 0)  # k_per_query=2, primary=5, secondary=0


def format_retrieval_summary(
    primary_category: str,
    secondary_category: Optional[str],
    primary_count: int,
    secondary_count: int = 0
) -> str:
    """
    Format a human-readable summary of the retrieval strategy.

    Args:
        primary_category: Primary category used
        secondary_category: Secondary category (if any)
        primary_count: Number of docs from primary
        secondary_count: Number of docs from secondary

    Returns:
        Formatted summary string
    """
    if secondary_category:
        return (
            f"Hybrid retrieval: {primary_count} from '{primary_category}' + "
            f"{secondary_count} from '{secondary_category}'"
        )
    else:
        return f"Single-category retrieval: {primary_count} from '{primary_category}'"
