"""
Smart RAG retriever with query decomposition for medical notes.

This retriever improves upon basic similarity search by:
1. Extracting key medical entities from the note
2. Decomposing into multiple focused queries
3. Retrieving relevant documents for each query
4. Combining and deduplicating results
"""
import re
from typing import List, Dict, Set
from app.rag.chroma_retriever import MedicalKnowledgeRetriever


class SmartMedicalRetriever:
    """
    Smart retriever that decomposes medical notes into focused queries.
    """

    def __init__(self, base_retriever: MedicalKnowledgeRetriever):
        """
        Initialize smart retriever.

        Args:
            base_retriever: Base Chroma retriever
        """
        self.base_retriever = base_retriever

        # Medical entity patterns
        self.diagnosis_patterns = [
            r"diagnosed with ([^.,]+)",
            r"diagnosis[:\s]+([^.,]+)",
            r"culture.*indicate[s]?\s+([^.,]+)",
            r"confirmed ([^.,]+)",
        ]

        self.symptom_keywords = [
            "headache", "nausea", "vomiting", "dizziness", "palpitations",
            "flushing", "sweating", "pain", "fever", "rash", "ulcer",
            "bleeding", "cough", "dyspnea", "chest pain", "abdominal pain",
            "confusion", "seizure", "weakness", "fatigue", "edema"
        ]

        self.medication_patterns = [
            r"prescribed ([^.,]+)",
            r"started on ([^.,]+)",
            r"treated with ([^.,]+)",
            r"antimicrobial therapy",
            r"antibiotic",
            r"medication",
        ]

        self.context_keywords = {
            "alcohol": ["beer", "wine", "alcohol", "drinking", "ethanol"],
            "pregnancy": ["pregnant", "pregnancy", "gestation", "trimester"],
            "kidney": ["kidney", "renal", "creatinine", "GFR", "dialysis"],
            "liver": ["liver", "hepatic", "cirrhosis", "ALT", "AST"],
            "elderly": ["elderly", "geriatric", "older adult"],
            "immunocompromised": ["immunocompromised", "HIV", "transplant"],
        }

    def extract_diagnosis(self, note: str) -> List[str]:
        """
        Extract diagnosis/condition from medical note.

        Args:
            note: Medical note text

        Returns:
            List of extracted diagnoses
        """
        diagnoses = []
        note_lower = note.lower()

        for pattern in self.diagnosis_patterns:
            matches = re.findall(pattern, note_lower, re.IGNORECASE)
            diagnoses.extend(matches)

        # Clean up
        diagnoses = [d.strip() for d in diagnoses]
        return list(set(diagnoses))  # Remove duplicates

    def extract_symptoms(self, note: str) -> List[str]:
        """
        Extract symptoms from medical note.

        Args:
            note: Medical note text

        Returns:
            List of found symptoms
        """
        note_lower = note.lower()
        found_symptoms = []

        for symptom in self.symptom_keywords:
            if symptom in note_lower:
                found_symptoms.append(symptom)

        return found_symptoms

    def extract_medications(self, note: str) -> List[str]:
        """
        Extract medication mentions from note.

        Args:
            note: Medical note text

        Returns:
            List of medication mentions
        """
        medications = []
        note_lower = note.lower()

        for pattern in self.medication_patterns:
            matches = re.findall(pattern, note_lower, re.IGNORECASE)
            medications.extend(matches)

        # Clean up
        medications = [m.strip() for m in medications]
        return list(set(medications))

    def extract_context_clues(self, note: str) -> List[str]:
        """
        Extract important contextual clues (alcohol, pregnancy, etc.).

        Args:
            note: Medical note text

        Returns:
            List of context categories found
        """
        note_lower = note.lower()
        contexts = []

        for context_type, keywords in self.context_keywords.items():
            for keyword in keywords:
                if keyword in note_lower:
                    contexts.append(context_type)
                    break  # Only add each context type once

        return contexts

    def decompose_query(self, note: str) -> List[str]:
        """
        Decompose medical note into focused queries.

        Enhanced to include differential diagnosis queries for better error detection.

        Args:
            note: Medical note text

        Returns:
            List of focused query strings
        """
        queries = []
        note_lower = note.lower()

        # Extract entities
        diagnoses = self.extract_diagnosis(note)
        symptoms = self.extract_symptoms(note)
        medications = self.extract_medications(note)
        contexts = self.extract_context_clues(note)

        # Detect travel history
        travel_keywords = ["travel", "returned from", "trip to", "visited", "missionary"]
        has_travel = any(keyword in note_lower for keyword in travel_keywords)

        # Detect suspected/uncertain diagnosis
        is_suspected = "suspected" in note_lower or "suspect" in note_lower

        # Priority 1: Differential diagnosis queries (MOST IMPORTANT for error detection)
        if diagnoses:
            for diagnosis in diagnoses[:1]:  # Primary diagnosis
                # Query for differential diagnosis
                queries.append(f"{diagnosis} differential diagnosis similar conditions")

                # Query for when NOT to diagnose this
                if is_suspected:
                    queries.append(f"when to suspect {diagnosis} versus alternatives")

                # Query for typical presentation
                queries.append(f"{diagnosis} typical symptoms presentation characteristics")

        # Priority 2: Symptom-based differential (especially with travel)
        if symptoms:
            symptom_cluster = " ".join(symptoms[:4])  # Top 4 symptoms
            queries.append(f"{symptom_cluster} causes differential diagnosis")

            # Travel-related infections
            if has_travel and diagnoses:
                queries.append(f"travel infections {symptom_cluster} differential {diagnoses[0]}")
            elif has_travel:
                queries.append(f"travel related infections {symptom_cluster}")

        # Priority 3: Diagnosis-specific features (to verify match)
        if diagnoses and symptoms:
            diagnosis = diagnoses[0]
            # Check if key symptoms match the diagnosis
            queries.append(f"{diagnosis} does it cause {' '.join(symptoms[:3])}")

        # Priority 4: Context-specific contraindications
        if diagnoses and contexts:
            for diagnosis in diagnoses[:1]:
                for context in contexts[:1]:  # Most important context only
                    queries.append(f"{diagnosis} {context} contraindications complications")

        # Priority 5: Medication interactions (if medications mentioned)
        if medications and contexts:
            for med in medications[:1]:
                for context in contexts[:1]:
                    queries.append(f"{med} {context} interaction contraindications")

        # Priority 6: Treatment guidelines (lower priority)
        if diagnoses:
            queries.append(f"{diagnoses[0]} treatment guidelines")

        # Always include full note as fallback (lowest priority)
        queries.append(note[:500])

        return queries

    def retrieve_with_decomposition(
        self,
        note: str,
        expert: str,
        k_per_query: int = 2,
        max_total: int = 5,
        filter_category: str = None
    ) -> List:
        """
        Retrieve documents using query decomposition.

        Args:
            note: Medical note
            expert: "A" for Mayo Clinic, "B" for WebMD
            k_per_query: Documents to retrieve per query
            max_total: Maximum total documents to return
            filter_category: Optional category filter
                           ("drugs_supplements", "diseases_conditions", "symptoms")

        Returns:
            List of relevant documents with metadata (deduplicated)
        """
        # Decompose into focused queries
        queries = self.decompose_query(note)

        expert_name = "Mayo Clinic" if expert == "A" else "WebMD"
        filter_msg = f" (filtered: {filter_category})" if filter_category else ""
        print(f"\n[Smart Retriever - {expert_name}] Decomposed into {len(queries)} queries{filter_msg}:")
        for i, q in enumerate(queries[:5], 1):  # Show first 5
            print(f"  {i}. {q[:80]}...")

        # Retrieve for each query
        all_docs = []
        seen_contents: Set[str] = set()

        for query in queries:
            docs = self.base_retriever.retrieve_for_expert(
                query=query,
                expert=expert,
                k=k_per_query,
                filter_category=filter_category
            )

            # Deduplicate based on content
            for doc in docs:
                # Use first 200 chars as fingerprint
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                fingerprint = content[:200].strip()
                if fingerprint not in seen_contents:
                    seen_contents.add(fingerprint)
                    all_docs.append(doc)

                    # Stop if we have enough
                    if len(all_docs) >= max_total:
                        break

            if len(all_docs) >= max_total:
                break

        print(f"[Smart Retriever - {expert_name}] Retrieved {len(all_docs)} unique documents\n")
        return all_docs[:max_total]


# Global smart retriever instance
_smart_retriever = None


def get_smart_retriever(
    persist_dir: str = None,
    embedding_model: str = None,
    auto_build: bool = True,
    data_dir: str = "BlueMed_data"
) -> SmartMedicalRetriever:
    """
    Get or create global smart retriever instance.

    Args:
        persist_dir: Directory containing vector databases
        embedding_model: Embedding model to use
        auto_build: If True, automatically build databases if they don't exist
        data_dir: Directory containing source documents

    Returns:
        SmartMedicalRetriever instance
    """
    global _smart_retriever
    if _smart_retriever is None:
        from app.rag.chroma_retriever import get_retriever
        base_retriever = get_retriever(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            auto_build=auto_build,
            data_dir=data_dir
        )
        _smart_retriever = SmartMedicalRetriever(base_retriever)
    return _smart_retriever
