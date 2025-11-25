"""
JSON result logger for debate system.
Saves all debate results with confidence scores for evaluation and ROC-AUC calculation.
"""
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import re


class ResultLogger:
    """Logger for saving debate results in JSON format."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize result logger.

        Args:
            output_dir: Directory to save result JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_judge_decision(self, judge_decision_raw: str) -> Dict[str, Any]:
        """
        Extract structured information from judge's raw decision.

        Args:
            judge_decision_raw: Raw judge decision string

        Returns:
            Dictionary with extracted fields
        """
        extracted = {
            "final_answer": "UNKNOWN",
            "confidence_score": None,
            "confidence_normalized": None,  # For ROC-AUC (0-1 scale)
            "winner": None,
            "reasoning": None
        }

        if not judge_decision_raw:
            return extracted

        # Try to parse JSON from judge decision
        decision_json = {}

        # Method 1: Try direct JSON parse first
        try:
            decision_json = json.loads(judge_decision_raw)
        except json.JSONDecodeError:
            pass

        # Method 2: Extract JSON from markdown code blocks (```json ... ```)
        if not decision_json:
            markdown_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', judge_decision_raw, re.DOTALL)
            if markdown_match:
                try:
                    decision_json = json.loads(markdown_match.group(1))
                except json.JSONDecodeError:
                    pass

        # Method 3: Find any JSON object containing "Final Answer"
        if not decision_json:
            json_match = re.search(r'(\{[^{}]*"Final Answer"[^{}]*\})', judge_decision_raw, re.DOTALL)
            if json_match:
                try:
                    decision_json = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

        # Method 4: Greedy match for any JSON object
        if not decision_json:
            json_match = re.search(r'\{.*\}', judge_decision_raw, re.DOTALL)
            if json_match:
                try:
                    decision_json = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

        # Extract fields (handle different key formats)
        extracted["final_answer"] = (
            decision_json.get("Final Answer") or
            decision_json.get("final_answer") or
            decision_json.get("final answer") or
            "UNKNOWN"
        )

        # Extract confidence score
        conf_score = (
            decision_json.get("Confidence Score") or
            decision_json.get("confidence_score") or
            decision_json.get("confidence score") or
            decision_json.get("Confidence") or
            decision_json.get("confidence")
        )

        if conf_score is not None:
            try:
                extracted["confidence_score"] = int(conf_score)
                # Normalize to 0-1 scale for ROC-AUC
                extracted["confidence_normalized"] = float(conf_score) / 10.0
            except (ValueError, TypeError):
                pass

        # Extract winner
        extracted["winner"] = (
            decision_json.get("Winner") or
            decision_json.get("winner")
        )

        # Extract reasoning
        extracted["reasoning"] = (
            decision_json.get("Reasoning") or
            decision_json.get("reasoning")
        )

        return extracted

    def _serialize_documents(self, docs: list) -> list:
        """
        Serialize LangChain Document objects to JSON-compatible format.

        Args:
            docs: List of Document objects or strings

        Returns:
            List of serialized document dictionaries
        """
        serialized = []
        for doc in docs:
            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                # LangChain Document object
                serialized.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            elif isinstance(doc, str):
                # Plain string
                serialized.append({
                    "content": doc,
                    "metadata": {}
                })
            elif isinstance(doc, dict):
                # Already serialized
                serialized.append(doc)
            else:
                # Unknown format, convert to string
                serialized.append({
                    "content": str(doc),
                    "metadata": {}
                })
        return serialized

    def save_result(
        self,
        medical_note: str,
        state: Dict[str, Any],
        execution_time: float,
        ground_truth_label: Optional[int] = None,
        case_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save debate result to JSON file.

        Args:
            medical_note: Original medical note
            state: Final state from graph execution
            execution_time: Time taken to execute (seconds)
            ground_truth_label: Optional ground truth (0=CORRECT, 1=INCORRECT)
            case_id: Optional case identifier
            metadata: Optional additional metadata

        Returns:
            Path to saved JSON file
        """
        # Generate IDs
        request_id = str(uuid.uuid4())
        if case_id is None:
            case_id = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Extract judge decision
        judge_decision_raw = state.get("judge_decision", "")
        judge_info = self.extract_judge_decision(judge_decision_raw)

        # Serialize retrieved documents (convert Document objects to dicts)
        expertA_docs = self._serialize_documents(state.get("expertA_retrieved_docs", []))
        expertB_docs = self._serialize_documents(state.get("expertB_retrieved_docs", []))

        # Build result structure with separate key-value pairs for judge decision
        result = {
            "id": case_id,
            "timestamp": datetime.now().isoformat(),
            "note": medical_note,

            # Labels
            "ground_truth": ground_truth_label,
            "predicted": 1 if judge_info["final_answer"] == "INCORRECT" else 0,

            # Judge decision - separate key-value pairs
            "final_answer": judge_info["final_answer"],
            "confidence_score": judge_info["confidence_score"],
            "confidence_normalized": judge_info["confidence_normalized"],  # 0.0-1.0 scale for ROC-AUC
            "winner": judge_info["winner"],
            "reasoning": judge_info["reasoning"],

            # Execution time
            "execution_time": execution_time
        }

        # Save to file
        filename = f"result_{case_id}_{request_id[:8]}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Result saved to: {filepath}")
        return filepath

    def save_batch_results(self, results: list, filename: str = None) -> Path:
        """
        Save multiple results to a single JSON file.

        Args:
            results: List of result dictionaries
            filename: Optional filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_results_{timestamp}.json"

        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Batch results ({len(results)} cases) saved to: {filepath}")
        return filepath

    def load_result(self, filepath: str) -> Dict[str, Any]:
        """Load result from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_batch_results(self, filepath: str) -> list:
        """Load batch results from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
