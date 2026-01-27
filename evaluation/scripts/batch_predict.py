#!/usr/bin/env python3
"""
Batch prediction script for medical error detection.

This script processes test cases in chunks and saves results for evaluation.
Features:
- Processes cases in configurable chunks (default: 20)
- Tracks progress and can resume from where it left off
- Saves detailed results to logs/debates with case_id and ground_truth
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)  # Ensure relative paths work correctly

from config.settings import settings
from app.graph.graph import build_graph
from langchain_core.messages import HumanMessage
from main import extract_judge_decision


class BatchPredictor:
    """Handles batch predictions with progress tracking and resume capability."""

    def __init__(
        self,
        test_file: str,
        output_dir: str = "logs/debates",
        checkpoint_file: str = "evaluation/checkpoint.json",
        chunk_size: int = 20,
        delay_between_cases: float = 15.0
    ):
        """
        Initialize batch predictor.

        Args:
            test_file: Path to JSON file containing test cases
            output_dir: Directory to save debate results
            checkpoint_file: File to track progress for resume capability
            chunk_size: Number of cases to process per chunk
            delay_between_cases: Seconds to wait between cases (for rate limiting)
        """
        self.test_file = Path(test_file)
        self.output_dir = Path(output_dir)
        self.checkpoint_file = Path(checkpoint_file)
        self.chunk_size = chunk_size
        self.delay_between_cases = delay_between_cases

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        # Load test cases
        self.test_cases = self._load_test_cases()

        # Load or initialize checkpoint
        self.checkpoint = self._load_checkpoint()

        # Build graph once
        print("Building debate graph...")
        self.graph = build_graph(settings)
        print("Graph built successfully!\n")

    def _load_test_cases(self) -> List[Dict]:
        """Load test cases from JSON file."""
        with open(self.test_file, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} test cases from {self.test_file}")
        return cases

    def _load_checkpoint(self) -> Dict:
        """Load checkpoint for resume capability."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            print(f"Resuming from checkpoint: {checkpoint['processed_count']}/{len(self.test_cases)} cases completed")
            return checkpoint
        return {
            "processed_ids": [],
            "processed_count": 0,
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat()
        }

    def _save_checkpoint(self):
        """Save current progress to checkpoint file."""
        self.checkpoint["last_update"] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.checkpoint, f, indent=2)

    def _doc_to_dict(self, doc) -> Dict:
        """Convert document object to JSON-serializable dictionary."""
        return {
            "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
            "content": doc.page_content if hasattr(doc, 'page_content') else str(doc)
        }

    def process_single_case(self, case: Dict) -> Dict:
        """
        Process a single test case through the debate system.

        Args:
            case: Test case with 'id', 'text', 'label' fields

        Returns:
            Result dictionary with prediction and metadata
        """
        case_id = case.get("id", f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        medical_note = case.get("text", "")
        ground_truth = case.get("label", -1)  # 0=CORRECT, 1=INCORRECT
        error_type = case.get("error_type", "NA")

        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content="Analyze this medical note for errors")],
            "medical_note": medical_note,
            "current_round": 1,
            "max_rounds": 2,
            "expertA_arguments": [],
            "expertA_retrieved_docs": [],
            "expertB_arguments": [],
            "expertB_retrieved_docs": [],
            "judge_decision": None,
            "final_answer": None
        }

        # Run debate
        start_time = time.time()
        result = self.graph.invoke(initial_state)
        execution_time = time.time() - start_time

        # Extract judge decision
        judge_decision_raw = result.get("judge_decision", "")
        judge_info = extract_judge_decision(judge_decision_raw)

        # Build result structure
        result_data = {
            "case_id": case_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "medical_note": medical_note,
            "ground_truth": ground_truth,  # 0=CORRECT, 1=INCORRECT
            "error_type": error_type,

            # Prediction
            "predicted_label": 1 if judge_info["final_answer"] == "INCORRECT" else 0,
            "final_answer": judge_info["final_answer"],
            "confidence_score": judge_info["confidence_score"],
            "confidence_normalized": judge_info["confidence_score"] / 10.0 if judge_info["confidence_score"] else None,
            "winner": judge_info["winner"],
            "reasoning": judge_info["reasoning"],

            # Expert arguments
            "expert_a": {
                "source": "Mayo Clinic",
                "arguments": result.get("expertA_arguments", []),
                "retrieved_docs": [self._doc_to_dict(doc) for doc in result.get("expertA_retrieved_docs", [])]
            },
            "expert_b": {
                "source": "WebMD",
                "arguments": result.get("expertB_arguments", []),
                "retrieved_docs": [self._doc_to_dict(doc) for doc in result.get("expertB_retrieved_docs", [])]
            },

            # Metadata
            "execution_time": execution_time,
            "system_metadata": {
                "use_retriever": settings.USE_RETRIEVER,
                "use_judge_retriever": settings.USE_JUDGE_RETRIEVER,
                "use_weighted_voting": settings.USE_WEIGHTED_VOTING,
                "embedding_model": settings.EMBEDDING_MODEL,
                "expert_model": settings.EXPERT_MODEL,
                "judge_model": settings.JUDGE_MODEL
            }
        }

        # Add weighted voting results if enabled
        if settings.USE_WEIGHTED_VOTING and result.get("weighted_decision"):
            weighted = result["weighted_decision"]
            result_data["weighted_voting"] = {
                "final_decision": weighted["final_decision"],
                "weighted_score": weighted["weighted_score"],
                "confidence": weighted["confidence"],
                "weights": weighted["weights"],
                "individual_classifications": weighted["individual_classifications"],
                "agreement": weighted["agreement"]
            }
            # Override prediction with weighted result
            result_data["predicted_label"] = 1 if weighted["final_decision"] == "INCORRECT" else 0
            result_data["final_answer"] = weighted["final_decision"]

        # Add judge retrieval info if enabled
        if settings.USE_JUDGE_RETRIEVER:
            mayo_docs = result.get("judge_retrieved_mayo_docs", [])
            webmd_docs = result.get("judge_retrieved_webmd_docs", [])
            result_data["judge_retrieval"] = {
                "mayo_docs_count": len(mayo_docs),
                "webmd_docs_count": len(webmd_docs),
                "mayo_docs": [self._doc_to_dict(doc) for doc in mayo_docs],
                "webmd_docs": [self._doc_to_dict(doc) for doc in webmd_docs]
            }

        return result_data

    def save_result(self, result: Dict) -> Path:
        """Save single result to JSON file."""
        case_id = result["case_id"]
        filename = f"result_{case_id}.json"
        filepath = self.output_dir / filename

        # Ensure output directory exists (critical for file writing)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Write file with explicit error handling
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise IOError(f"Failed to write result file {filepath}: {e}")

        return filepath

    def run_batch(self, start_idx: int = None, end_idx: int = None) -> List[Dict]:
        """
        Run batch prediction on test cases.

        Args:
            start_idx: Starting index (default: resume from checkpoint)
            end_idx: Ending index (default: process all remaining)

        Returns:
            List of results
        """
        # Determine start index
        if start_idx is None:
            start_idx = self.checkpoint["processed_count"]

        # Determine end index
        if end_idx is None:
            end_idx = len(self.test_cases)

        # Filter out already processed cases
        cases_to_process = []
        for i, case in enumerate(self.test_cases[start_idx:end_idx], start=start_idx):
            case_id = case.get("id", f"case_{i}")
            if case_id not in self.checkpoint["processed_ids"]:
                cases_to_process.append((i, case))

        total = len(cases_to_process)
        print(f"\n{'='*60}")
        print(f"Starting batch prediction: {total} cases to process")
        print(f"Chunk size: {self.chunk_size}")
        print(f"{'='*60}\n")

        results = []

        for batch_num, batch_start in enumerate(range(0, total, self.chunk_size)):
            batch_end = min(batch_start + self.chunk_size, total)
            batch_cases = cases_to_process[batch_start:batch_end]

            print(f"\n{'='*60}")
            print(f"Processing chunk {batch_num + 1}: cases {batch_start + 1} to {batch_end} of {total}")
            print(f"{'='*60}")

            for j, (idx, case) in enumerate(batch_cases):
                case_id = case.get("id", f"case_{idx}")
                print(f"\n[{batch_start + j + 1}/{total}] Processing case: {case_id}")

                try:
                    # Process case
                    result = self.process_single_case(case)

                    # CRITICAL: Save result FIRST before updating checkpoint
                    filepath = self.save_result(result)

                    # Verify file was actually written
                    if not filepath.exists():
                        raise IOError(f"File was not saved: {filepath}")

                    print(f"  ✓ Saved to: {filepath}")
                    print(f"  Prediction: {result['final_answer']} (ground truth label: {result['ground_truth']})")
                    print(f"  Confidence: {result['confidence_score']}")
                    print(f"  Time: {result['execution_time']:.2f}s")

                    # Update checkpoint ONLY after successful file save
                    self.checkpoint["processed_ids"].append(case_id)
                    self.checkpoint["processed_count"] += 1
                    self._save_checkpoint()

                    results.append(result)

                    # Rate limiting delay
                    if self.delay_between_cases > 0 and (batch_start + j + 1) < total:
                        print(f"  ⏳ Waiting {self.delay_between_cases}s (rate limit)...")
                        time.sleep(self.delay_between_cases)

                except Exception as e:
                    print(f"  ✗ Error processing case {case_id}: {e}")
                    import traceback
                    print(f"  Full traceback:")
                    traceback.print_exc()
                    # Wait before retrying next case to avoid rate limits
                    time.sleep(self.delay_between_cases)
                    continue

            # Summary for chunk
            print(f"\nChunk {batch_num + 1} complete. Total processed: {self.checkpoint['processed_count']}/{len(self.test_cases)}")

        return results

    def get_progress(self) -> Dict:
        """Get current progress information."""
        return {
            "total_cases": len(self.test_cases),
            "processed": self.checkpoint["processed_count"],
            "remaining": len(self.test_cases) - self.checkpoint["processed_count"],
            "percentage": (self.checkpoint["processed_count"] / len(self.test_cases)) * 100 if self.test_cases else 0
        }


def main():
    """Main entry point for batch prediction."""
    parser = argparse.ArgumentParser(description="Batch prediction for medical error detection")
    parser.add_argument(
        "--test-file", "-t",
        type=str,
        default="test_data/test_cases_10.json",
        help="Path to JSON file containing test cases"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="logs/debates",
        help="Directory to save results"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=20,
        help="Number of cases to process per chunk"
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=None,
        help="Starting index (default: resume from checkpoint)"
    )
    parser.add_argument(
        "--end", "-e",
        type=int,
        default=None,
        help="Ending index (default: process all)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset checkpoint and start from beginning"
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=15.0,
        help="Seconds to wait between cases (for API rate limiting). Default: 15s for free tier."
    )

    args = parser.parse_args()

    # Reset checkpoint if requested
    if args.reset:
        checkpoint_path = Path("evaluation/checkpoint.json")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("Checkpoint reset.")

    # Initialize predictor
    predictor = BatchPredictor(
        test_file=args.test_file,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        delay_between_cases=args.delay
    )

    # Show progress
    progress = predictor.get_progress()
    print(f"\nProgress: {progress['processed']}/{progress['total_cases']} ({progress['percentage']:.1f}%)")

    if progress['remaining'] == 0:
        print("All cases already processed!")
        return

    # Run batch prediction
    results = predictor.run_batch(start_idx=args.start, end_idx=args.end)

    # Final summary
    print(f"\n{'='*60}")
    print("BATCH PREDICTION COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {len(results)} cases")
    print(f"Results saved to: {args.output_dir}")
    print(f"Run evaluation with: python evaluation/evaluate.py")


if __name__ == "__main__":
    main()
