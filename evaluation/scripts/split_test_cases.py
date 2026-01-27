#!/usr/bin/env python3
"""
Utility script to split large test case files into smaller chunks.

This is useful for:
- Managing large test sets (e.g., 597 cases)
- Running parallel evaluation
- Testing with subsets
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def split_test_cases(
    input_file: str,
    output_dir: str,
    chunk_size: int = 20,
    output_prefix: str = "test_cases_chunk"
) -> List[Path]:
    """
    Split a JSON test cases file into smaller chunks.

    Args:
        input_file: Path to input JSON file with test cases
        output_dir: Directory to save chunk files
        chunk_size: Number of cases per chunk
        output_prefix: Prefix for output filenames

    Returns:
        List of paths to created chunk files
    """
    # Load test cases
    with open(input_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases from {input_file}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Split into chunks
    chunk_files = []
    num_chunks = (len(test_cases) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(test_cases))
        chunk = test_cases[start_idx:end_idx]

        # Save chunk
        filename = f"{output_prefix}_{i:03d}.json"
        filepath = output_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)

        chunk_files.append(filepath)
        print(f"Created {filepath}: {len(chunk)} cases (indices {start_idx}-{end_idx-1})")

    print(f"\nCreated {len(chunk_files)} chunk files in {output_dir}")
    return chunk_files


def merge_test_cases(
    input_dir: str,
    output_file: str,
    pattern: str = "test_cases_chunk_*.json"
) -> int:
    """
    Merge multiple chunk files back into a single file.

    Args:
        input_dir: Directory containing chunk files
        output_file: Path to output merged file
        pattern: Glob pattern for chunk files

    Returns:
        Total number of cases merged
    """
    input_path = Path(input_dir)
    chunk_files = sorted(input_path.glob(pattern))

    all_cases = []
    for filepath in chunk_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        all_cases.extend(cases)
        print(f"Loaded {len(cases)} cases from {filepath.name}")

    # Save merged file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)

    print(f"\nMerged {len(all_cases)} cases into {output_file}")
    return len(all_cases)


def get_test_case_stats(input_file: str) -> Dict:
    """
    Get statistics about test cases.

    Args:
        input_file: Path to JSON test cases file

    Returns:
        Dictionary with statistics
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    stats = {
        "total_cases": len(test_cases),
        "label_distribution": {},
        "error_type_distribution": {},
        "split_distribution": {}
    }

    for case in test_cases:
        # Label distribution
        label = case.get("label", "unknown")
        label_str = "INCORRECT" if label == 1 else "CORRECT" if label == 0 else str(label)
        stats["label_distribution"][label_str] = stats["label_distribution"].get(label_str, 0) + 1

        # Error type distribution
        error_type = case.get("error_type", "NA")
        stats["error_type_distribution"][error_type] = stats["error_type_distribution"].get(error_type, 0) + 1

        # Split distribution
        split = case.get("split", "unknown")
        stats["split_distribution"][split] = stats["split_distribution"].get(split, 0) + 1

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Split or merge test case files")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Split command
    split_parser = subparsers.add_parser("split", help="Split test cases into chunks")
    split_parser.add_argument("input_file", help="Input JSON file")
    split_parser.add_argument("--output-dir", "-o", default="test_data/chunks", help="Output directory")
    split_parser.add_argument("--chunk-size", "-c", type=int, default=20, help="Cases per chunk")
    split_parser.add_argument("--prefix", "-p", default="test_cases_chunk", help="Output filename prefix")

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge chunk files")
    merge_parser.add_argument("input_dir", help="Directory containing chunk files")
    merge_parser.add_argument("output_file", help="Output merged file")
    merge_parser.add_argument("--pattern", "-p", default="test_cases_chunk_*.json", help="Glob pattern")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show test case statistics")
    stats_parser.add_argument("input_file", help="Input JSON file")

    args = parser.parse_args()

    if args.command == "split":
        split_test_cases(
            args.input_file,
            args.output_dir,
            args.chunk_size,
            args.prefix
        )

    elif args.command == "merge":
        merge_test_cases(
            args.input_dir,
            args.output_file,
            args.pattern
        )

    elif args.command == "stats":
        stats = get_test_case_stats(args.input_file)
        print(f"\n{'='*40}")
        print("TEST CASE STATISTICS")
        print(f"{'='*40}")
        print(f"\nTotal cases: {stats['total_cases']}")

        print("\nLabel distribution:")
        for label, count in stats["label_distribution"].items():
            pct = count / stats["total_cases"] * 100
            print(f"  {label}: {count} ({pct:.1f}%)")

        print("\nError type distribution:")
        for error_type, count in stats["error_type_distribution"].items():
            pct = count / stats["total_cases"] * 100
            print(f"  {error_type}: {count} ({pct:.1f}%)")

        print("\nSplit distribution:")
        for split, count in stats["split_distribution"].items():
            pct = count / stats["total_cases"] * 100
            print(f"  {split}: {count} ({pct:.1f}%)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
