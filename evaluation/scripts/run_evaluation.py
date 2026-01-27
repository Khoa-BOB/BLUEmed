#!/usr/bin/env python3
"""
Complete evaluation pipeline script.

This script runs the full evaluation pipeline:
1. Batch prediction on test cases
2. Evaluation of predictions
3. Generation of metrics report

Usage:
    python evaluation/run_evaluation.py --test-file test_data/all_cases.json
"""

import argparse
import sys
import os
from pathlib import Path

# Resolve paths relative to current directory BEFORE changing to project root
# This is needed because batch_predict.py changes cwd to project root
_original_cwd = Path.cwd()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from batch_predict import BatchPredictor
from evaluate import Evaluator


def resolve_path(path_str: str) -> str:
    """Resolve a path that may be relative to the original working directory."""
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    # Try relative to original cwd first
    resolved = _original_cwd / path
    if resolved.exists() or not (PROJECT_ROOT / path).exists():
        return str(resolved)
    # Fall back to project root relative
    return str(PROJECT_ROOT / path)


def main():
    """Run complete evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Run complete evaluation pipeline")
    parser.add_argument(
        "--test-file", "-t",
        type=str,
        required=True,
        help="Path to JSON file containing test cases"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="logs/debates",
        help="Directory to save prediction results"
    )
    parser.add_argument(
        "--eval-output", "-e",
        type=str,
        default="evaluation/results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=20,
        help="Number of cases to process per chunk"
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Skip prediction step and only run evaluation"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset checkpoint and start predictions from beginning"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MEDICAL ERROR DETECTION - EVALUATION PIPELINE")
    print("=" * 60)

    # Step 1: Batch Prediction
    if not args.skip_prediction:
        print("\n--- STEP 1: BATCH PREDICTION ---\n")

        # Reset checkpoint if requested
        if args.reset:
            checkpoint_path = Path("evaluation/checkpoint.json")
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print("Checkpoint reset.")

        predictor = BatchPredictor(
            test_file=resolve_path(args.test_file),
            output_dir=resolve_path(args.output_dir),
            chunk_size=args.chunk_size
        )

        progress = predictor.get_progress()
        print(f"Progress: {progress['processed']}/{progress['total_cases']} ({progress['percentage']:.1f}%)")

        if progress['remaining'] > 0:
            predictor.run_batch()
        else:
            print("All cases already processed!")
    else:
        print("\n--- SKIPPING PREDICTION STEP ---\n")

    # Step 2: Evaluation
    print("\n--- STEP 2: EVALUATION ---\n")

    evaluator = Evaluator(results_dir=resolve_path(args.output_dir))
    count = evaluator.load_results()

    if count == 0:
        print("No valid results found for evaluation.")
        return

    metrics = evaluator.calculate_metrics()
    evaluator.print_summary(metrics)

    json_path, csv_path = evaluator.save_results(
        metrics,
        output_dir=resolve_path(args.eval_output)
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nPrediction results: {args.output_dir}")
    print(f"Evaluation results: {args.eval_output}")


if __name__ == "__main__":
    main()
