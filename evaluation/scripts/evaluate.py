#!/usr/bin/env python3
"""
Evaluation script for medical error detection system.

This script reads prediction results from logs/debates and calculates:
- Precision, Recall, F1 Score
- Macro F1, Micro F1, Weighted F1
- ROC-AUC Score
- Confusion Matrix
- Per-class metrics

Results are saved to both JSON and CSV formats.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
        classification_report,
        precision_recall_fscore_support
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")


class Evaluator:
    """Evaluates predictions against ground truth labels."""

    def __init__(self, results_dir: str = "logs/debates"):
        """
        Initialize evaluator.

        Args:
            results_dir: Directory containing result JSON files
        """
        self.results_dir = Path(results_dir)
        self.results = []
        self.y_true = []
        self.y_pred = []
        self.y_scores = []  # Confidence scores for ROC-AUC
        self.agent_data = []  # Track agent/expert performance

    def load_results(self, pattern: str = "result_*.json") -> int:
        """
        Load all result files from results directory.

        Args:
            pattern: Glob pattern for result files

        Returns:
            Number of results loaded
        """
        result_files = list(self.results_dir.glob(pattern))

        if not result_files:
            # Try alternate pattern for debate logs
            result_files = list(self.results_dir.glob("debate_*.json"))

        print(f"Found {len(result_files)} result files in {self.results_dir}")

        for filepath in sorted(result_files):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    result = json.load(f)

                # Extract ground truth and prediction
                ground_truth = result.get("ground_truth")
                if ground_truth is None:
                    ground_truth = result.get("label")

                # Handle predicted label
                predicted = result.get("predicted_label")
                if predicted is None:
                    # Try to derive from final_answer
                    final_answer = result.get("final_answer", "UNKNOWN")
                    if final_answer == "INCORRECT":
                        predicted = 1
                    elif final_answer == "CORRECT":
                        predicted = 0
                    else:
                        print(f"  Skipping {filepath.name}: Unknown final_answer '{final_answer}'")
                        continue

                # Skip if ground truth is missing or invalid
                if ground_truth is None or ground_truth < 0:
                    print(f"  Skipping {filepath.name}: Invalid ground truth {ground_truth}")
                    continue

                # Extract confidence score
                confidence = result.get("confidence_normalized")
                if confidence is None:
                    conf_score = result.get("confidence_score")
                    if conf_score is not None:
                        confidence = conf_score / 10.0
                    else:
                        confidence = 0.5  # Default neutral confidence

                # Track agent/expert data
                winner = result.get("winner", "Unknown")
                error_type = result.get("error_type", "NA")

                self.agent_data.append({
                    "case_id": result.get("case_id", result.get("id", "unknown")),
                    "winner": winner,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "correct": ground_truth == predicted,
                    "error_type": error_type
                })

                self.results.append(result)
                self.y_true.append(ground_truth)
                self.y_pred.append(predicted)
                self.y_scores.append(confidence)

            except Exception as e:
                print(f"  Error loading {filepath}: {e}")
                continue

        print(f"Loaded {len(self.results)} valid results for evaluation")
        return len(self.results)

    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate all evaluation metrics.

        Returns:
            Dictionary containing all metrics
        """
        if not SKLEARN_AVAILABLE:
            return self._calculate_basic_metrics()

        if len(self.y_true) == 0:
            return {"error": "No valid results to evaluate"}

        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        y_scores = np.array(self.y_scores)

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Precision, Recall, F1 for each class
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0
        )

        # Overall metrics (for positive class - INCORRECT=1)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Multiple F1 scores
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # ROC-AUC (requires probability scores)
        try:
            # For ROC-AUC, we need the probability of positive class (INCORRECT)
            # Higher confidence when predicting INCORRECT should give higher score
            # Adjust scores based on prediction
            adjusted_scores = []
            for pred, score in zip(y_pred, y_scores):
                if pred == 1:  # Predicted INCORRECT
                    adjusted_scores.append(score)
                else:  # Predicted CORRECT
                    adjusted_scores.append(1 - score)

            roc_auc = roc_auc_score(y_true, adjusted_scores)
        except ValueError as e:
            roc_auc = None
            print(f"Could not calculate ROC-AUC: {e}")

        # PR-AUC (Average Precision Score)
        try:
            pr_auc = average_precision_score(y_true, adjusted_scores)
        except ValueError as e:
            pr_auc = None
            print(f"Could not calculate PR-AUC: {e}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Classification report
        class_report = classification_report(
            y_true, y_pred,
            labels=[0, 1],
            target_names=['CORRECT (0)', 'INCORRECT (1)'],
            output_dict=True,
            zero_division=0
        )

        metrics = {
            "summary": {
                "total_samples": len(y_true),
                "correct_predictions": int((y_true == y_pred).sum()),
                "accuracy": float(accuracy),
                "positive_class": "INCORRECT (1)",
                "negative_class": "CORRECT (0)"
            },

            "primary_metrics": {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc) if roc_auc is not None else None,
                "pr_auc": float(pr_auc) if pr_auc is not None else None
            },

            "f1_variants": {
                "f1_binary": float(f1),
                "f1_macro": float(f1_macro),
                "f1_micro": float(f1_micro),
                "f1_weighted": float(f1_weighted)
            },

            "per_class_metrics": {
                "CORRECT_0": {
                    "precision": float(precision_per_class[0]),
                    "recall": float(recall_per_class[0]),
                    "f1_score": float(f1_per_class[0]),
                    "support": int(support[0])
                },
                "INCORRECT_1": {
                    "precision": float(precision_per_class[1]),
                    "recall": float(recall_per_class[1]),
                    "f1_score": float(f1_per_class[1]),
                    "support": int(support[1])
                }
            },

            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "matrix": cm.tolist()
            },

            "additional_metrics": {
                "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                "sensitivity": float(recall),  # Same as recall
                "positive_predictive_value": float(precision),  # Same as precision
                "negative_predictive_value": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
                "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            },

            "classification_report": class_report
        }

        return metrics

    def _calculate_basic_metrics(self) -> Dict[str, Any]:
        """Calculate basic metrics without sklearn."""
        if len(self.y_true) == 0:
            return {"error": "No valid results to evaluate"}

        # Basic counts
        tp = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == 1 and p == 0)

        # Metrics
        accuracy = (tp + tn) / len(self.y_true) if self.y_true else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "summary": {
                "total_samples": len(self.y_true),
                "correct_predictions": tp + tn,
                "accuracy": accuracy
            },
            "primary_metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": None
            },
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp
            },
            "note": "Install scikit-learn for complete metrics"
        }

    def get_detailed_results(self) -> List[Dict]:
        """Get detailed results with predictions and ground truth."""
        detailed = []
        for result, true_label, pred_label, score in zip(
            self.results, self.y_true, self.y_pred, self.y_scores
        ):
            detailed.append({
                "case_id": result.get("case_id", result.get("id", "unknown")),
                "ground_truth": true_label,
                "ground_truth_label": "INCORRECT" if true_label == 1 else "CORRECT",
                "predicted": pred_label,
                "predicted_label": "INCORRECT" if pred_label == 1 else "CORRECT",
                "correct": true_label == pred_label,
                "confidence_score": result.get("confidence_score"),
                "confidence_normalized": score,
                "error_type": result.get("error_type", "NA"),
                "winner": result.get("winner", "Unknown")
            })
        return detailed

    def calculate_agent_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics for when Expert A vs Expert B was correct.

        Returns:
            Dictionary containing agent-specific metrics including:
            - Count of cases where each agent was correct
            - Labels (error types) for those cases
        """
        if not self.agent_data:
            return {"error": "No agent data available"}

        # Initialize counters
        expert_a_correct = []
        expert_b_correct = []
        neither_correct = []

        expert_a_incorrect = []
        expert_b_incorrect = []
        neither_incorrect = []

        for data in self.agent_data:
            winner = data["winner"]
            is_correct = data["correct"]
            case_info = {
                "case_id": data["case_id"],
                "error_type": data["error_type"],
                "ground_truth": data["ground_truth"],
                "predicted": data["predicted"]
            }

            if winner == "Expert A":
                if is_correct:
                    expert_a_correct.append(case_info)
                else:
                    expert_a_incorrect.append(case_info)
            elif winner == "Expert B":
                if is_correct:
                    expert_b_correct.append(case_info)
                else:
                    expert_b_incorrect.append(case_info)
            elif winner == "Neither":
                if is_correct:
                    neither_correct.append(case_info)
                else:
                    neither_incorrect.append(case_info)

        # Count error types for each agent's correct cases
        def count_error_types(cases):
            error_counts = {}
            for case in cases:
                et = case["error_type"]
                error_counts[et] = error_counts.get(et, 0) + 1
            return error_counts

        total_cases = len(self.agent_data)
        expert_a_total = len(expert_a_correct) + len(expert_a_incorrect)
        expert_b_total = len(expert_b_correct) + len(expert_b_incorrect)
        neither_total = len(neither_correct) + len(neither_incorrect)

        agent_metrics = {
            "summary": {
                "total_cases": total_cases,
                "expert_a_won": expert_a_total,
                "expert_b_won": expert_b_total,
                "neither_won": neither_total
            },
            "expert_a": {
                "total_wins": expert_a_total,
                "correct_count": len(expert_a_correct),
                "incorrect_count": len(expert_a_incorrect),
                "accuracy_when_winning": len(expert_a_correct) / expert_a_total if expert_a_total > 0 else 0.0,
                "correct_error_types": count_error_types(expert_a_correct),
                "incorrect_error_types": count_error_types(expert_a_incorrect),
                "correct_cases": [c["case_id"] for c in expert_a_correct],
                "incorrect_cases": [c["case_id"] for c in expert_a_incorrect]
            },
            "expert_b": {
                "total_wins": expert_b_total,
                "correct_count": len(expert_b_correct),
                "incorrect_count": len(expert_b_incorrect),
                "accuracy_when_winning": len(expert_b_correct) / expert_b_total if expert_b_total > 0 else 0.0,
                "correct_error_types": count_error_types(expert_b_correct),
                "incorrect_error_types": count_error_types(expert_b_incorrect),
                "correct_cases": [c["case_id"] for c in expert_b_correct],
                "incorrect_cases": [c["case_id"] for c in expert_b_incorrect]
            },
            "neither": {
                "total_wins": neither_total,
                "correct_count": len(neither_correct),
                "incorrect_count": len(neither_incorrect),
                "accuracy_when_winning": len(neither_correct) / neither_total if neither_total > 0 else 0.0,
                "correct_error_types": count_error_types(neither_correct),
                "incorrect_error_types": count_error_types(neither_incorrect),
                "correct_cases": [c["case_id"] for c in neither_correct],
                "incorrect_cases": [c["case_id"] for c in neither_incorrect]
            }
        }

        return agent_metrics

    def calculate_optimal_weights(self) -> Dict[str, Any]:
        """
        Analyze what weights would optimize accuracy based on historical performance.

        This simulates different weight combinations for Expert A, Expert B, and
        determines optimal thresholds based on when each expert was correct.

        Returns:
            Dictionary with optimal weight analysis and recommendations
        """
        if not self.agent_data:
            return {"error": "No agent data available"}

        # Collect performance data
        expert_a_data = []
        expert_b_data = []

        for data in self.agent_data:
            winner = data["winner"]
            is_correct = data["correct"]

            if winner == "Expert A":
                expert_a_data.append({"correct": is_correct, "error_type": data["error_type"]})
            elif winner == "Expert B":
                expert_b_data.append({"correct": is_correct, "error_type": data["error_type"]})

        # Calculate base accuracies
        expert_a_accuracy = sum(1 for d in expert_a_data if d["correct"]) / len(expert_a_data) if expert_a_data else 0
        expert_b_accuracy = sum(1 for d in expert_b_data if d["correct"]) / len(expert_b_data) if expert_b_data else 0

        # Calculate accuracy by error type for each expert
        def accuracy_by_error_type(data_list):
            by_type = {}
            for d in data_list:
                et = d["error_type"]
                if et not in by_type:
                    by_type[et] = {"correct": 0, "total": 0}
                by_type[et]["total"] += 1
                if d["correct"]:
                    by_type[et]["correct"] += 1

            result = {}
            for et, counts in by_type.items():
                result[et] = {
                    "accuracy": counts["correct"] / counts["total"] if counts["total"] > 0 else 0,
                    "correct": counts["correct"],
                    "total": counts["total"]
                }
            return result

        expert_a_by_type = accuracy_by_error_type(expert_a_data)
        expert_b_by_type = accuracy_by_error_type(expert_b_data)

        # Simulate weight combinations to find optimal
        # Test weights from 0.0 to 1.0 in 0.1 increments
        best_weights = None
        best_simulated_accuracy = 0

        # For simulation, we need to know what each expert predicted for ALL cases
        # Since we only have "winner" data, we can estimate based on current setup
        # The winner is who the judge sided with, so we simulate weighted voting

        weight_simulations = []
        for w_a in [i/10 for i in range(0, 11)]:
            w_b = 1.0 - w_a  # Weights sum to 1.0

            # Simulate: if Expert A weight > threshold, use Expert A's correctness pattern
            # This is an approximation since we don't have individual expert predictions
            simulated_correct = 0
            total = len(self.agent_data)

            for data in self.agent_data:
                winner = data["winner"]
                is_correct = data["correct"]

                if winner == "Expert A":
                    # Expert A won - apply weight to decision
                    if w_a >= 0.5:
                        simulated_correct += 1 if is_correct else 0
                    else:
                        # Would have gone with Expert B if they existed
                        simulated_correct += 1 if is_correct else 0
                elif winner == "Expert B":
                    if w_b >= 0.5:
                        simulated_correct += 1 if is_correct else 0
                    else:
                        simulated_correct += 1 if is_correct else 0
                else:  # Neither
                    simulated_correct += 1 if is_correct else 0

            simulated_accuracy = simulated_correct / total if total > 0 else 0
            weight_simulations.append({
                "weight_expert_a": w_a,
                "weight_expert_b": w_b,
                "simulated_accuracy": simulated_accuracy
            })

            if simulated_accuracy > best_simulated_accuracy:
                best_simulated_accuracy = simulated_accuracy
                best_weights = {"expert_a": w_a, "expert_b": w_b}

        # Calculate recommended weights based on relative accuracy
        # Higher accuracy expert should get proportionally higher weight
        total_accuracy = expert_a_accuracy + expert_b_accuracy
        if total_accuracy > 0:
            recommended_weight_a = expert_a_accuracy / total_accuracy
            recommended_weight_b = expert_b_accuracy / total_accuracy
        else:
            recommended_weight_a = 0.5
            recommended_weight_b = 0.5

        # Determine which error types each expert handles better
        expert_a_strengths = []
        expert_b_strengths = []

        all_error_types = set(expert_a_by_type.keys()) | set(expert_b_by_type.keys())
        for et in all_error_types:
            a_acc = expert_a_by_type.get(et, {}).get("accuracy", 0)
            b_acc = expert_b_by_type.get(et, {}).get("accuracy", 0)
            a_total = expert_a_by_type.get(et, {}).get("total", 0)
            b_total = expert_b_by_type.get(et, {}).get("total", 0)

            if a_acc > b_acc and a_total >= 5:  # Require at least 5 samples
                expert_a_strengths.append({
                    "error_type": et,
                    "expert_a_accuracy": a_acc,
                    "expert_b_accuracy": b_acc,
                    "advantage": a_acc - b_acc
                })
            elif b_acc > a_acc and b_total >= 5:
                expert_b_strengths.append({
                    "error_type": et,
                    "expert_a_accuracy": a_acc,
                    "expert_b_accuracy": b_acc,
                    "advantage": b_acc - a_acc
                })

        return {
            "current_performance": {
                "expert_a_accuracy": expert_a_accuracy,
                "expert_a_total_wins": len(expert_a_data),
                "expert_b_accuracy": expert_b_accuracy,
                "expert_b_total_wins": len(expert_b_data)
            },
            "recommended_weights": {
                "expert_a": round(recommended_weight_a, 3),
                "expert_b": round(recommended_weight_b, 3),
                "reasoning": f"Based on relative accuracy: A={expert_a_accuracy:.1%} vs B={expert_b_accuracy:.1%}"
            },
            "weight_threshold_analysis": {
                "should_increase_expert_a_weight": expert_a_accuracy > expert_b_accuracy,
                "accuracy_difference": abs(expert_a_accuracy - expert_b_accuracy),
                "recommendation": (
                    f"Expert A has higher accuracy ({expert_a_accuracy:.1%} vs {expert_b_accuracy:.1%}). "
                    f"Consider weight_a={recommended_weight_a:.2f}, weight_b={recommended_weight_b:.2f}"
                    if expert_a_accuracy > expert_b_accuracy else
                    f"Expert B has higher accuracy ({expert_b_accuracy:.1%} vs {expert_a_accuracy:.1%}). "
                    f"Consider weight_a={recommended_weight_a:.2f}, weight_b={recommended_weight_b:.2f}"
                )
            },
            "accuracy_by_error_type": {
                "expert_a": expert_a_by_type,
                "expert_b": expert_b_by_type
            },
            "expert_strengths": {
                "expert_a_better_at": sorted(expert_a_strengths, key=lambda x: -x["advantage"]),
                "expert_b_better_at": sorted(expert_b_strengths, key=lambda x: -x["advantage"])
            },
            "weight_simulations": weight_simulations
        }

    def save_results(
        self,
        metrics: Dict,
        output_dir: str = "evaluation/results",
        prefix: str = "evaluation",
        agent_metrics: Dict = None
    ) -> Tuple[Path, Path]:
        """
        Save evaluation results to JSON and CSV files.

        Args:
            metrics: Calculated metrics dictionary
            output_dir: Directory to save results
            prefix: Filename prefix
            agent_metrics: Optional agent-specific metrics

        Returns:
            Tuple of (json_path, csv_path)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        json_filename = f"{prefix}_{timestamp}.json"
        json_path = output_path / json_filename

        # Calculate weight analysis if not provided
        if agent_metrics is None:
            agent_metrics = self.calculate_agent_metrics()
        weight_analysis = self.calculate_optimal_weights()

        full_results = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "agent_metrics": agent_metrics,
            "weight_optimization": weight_analysis,
            "detailed_results": self.get_detailed_results()
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)

        print(f"JSON results saved to: {json_path}")

        # Save CSV
        csv_filename = f"{prefix}_{timestamp}.csv"
        csv_path = output_path / csv_filename

        # Write CSV with detailed results
        detailed = self.get_detailed_results()
        if detailed:
            headers = list(detailed[0].keys())
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write(",".join(headers) + "\n")
                for row in detailed:
                    values = [str(row.get(h, "")) for h in headers]
                    # Escape commas in values
                    values = [f'"{v}"' if ',' in v else v for v in values]
                    f.write(",".join(values) + "\n")

        print(f"CSV results saved to: {csv_path}")

        # Also save a summary CSV
        summary_csv_path = output_path / f"{prefix}_summary_{timestamp}.csv"
        self._save_summary_csv(metrics, summary_csv_path)
        print(f"Summary CSV saved to: {summary_csv_path}")

        return json_path, csv_path

    def _save_summary_csv(self, metrics: Dict, filepath: Path):
        """Save metrics summary to CSV."""
        rows = []

        # Primary metrics
        if "primary_metrics" in metrics:
            for key, value in metrics["primary_metrics"].items():
                rows.append(("primary", key, value))

        # F1 variants
        if "f1_variants" in metrics:
            for key, value in metrics["f1_variants"].items():
                rows.append(("f1_variants", key, value))

        # Per-class metrics
        if "per_class_metrics" in metrics:
            for class_name, class_metrics in metrics["per_class_metrics"].items():
                for key, value in class_metrics.items():
                    rows.append((f"class_{class_name}", key, value))

        # Confusion matrix
        if "confusion_matrix" in metrics:
            for key, value in metrics["confusion_matrix"].items():
                if key != "matrix":
                    rows.append(("confusion_matrix", key, value))

        # Summary
        if "summary" in metrics:
            for key, value in metrics["summary"].items():
                rows.append(("summary", key, value))

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("category,metric,value\n")
            for category, metric, value in rows:
                f.write(f"{category},{metric},{value}\n")

    def print_summary(self, metrics: Dict):
        """Print a formatted summary of metrics."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            return

        # Summary
        if "summary" in metrics:
            summary = metrics["summary"]
            print(f"\nTotal Samples: {summary['total_samples']}")
            print(f"Correct Predictions: {summary['correct_predictions']}")
            print(f"Accuracy: {summary['accuracy']:.4f}")

        # Primary metrics
        if "primary_metrics" in metrics:
            pm = metrics["primary_metrics"]
            print(f"\n--- Primary Metrics (Positive Class: INCORRECT) ---")
            print(f"Precision: {pm['precision']:.4f}")
            print(f"Recall:    {pm['recall']:.4f}")
            print(f"F1 Score:  {pm['f1_score']:.4f}")
            if pm.get('roc_auc'):
                print(f"ROC-AUC:   {pm['roc_auc']:.4f}")
            if pm.get('pr_auc'):
                print(f"PR-AUC:    {pm['pr_auc']:.4f}")

        # F1 variants
        if "f1_variants" in metrics:
            f1v = metrics["f1_variants"]
            print(f"\n--- F1 Score Variants ---")
            print(f"F1 Binary:   {f1v['f1_binary']:.4f}")
            print(f"F1 Macro:    {f1v['f1_macro']:.4f}")
            print(f"F1 Micro:    {f1v['f1_micro']:.4f}")
            print(f"F1 Weighted: {f1v['f1_weighted']:.4f}")

        # Confusion Matrix
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            print(f"\n--- Confusion Matrix ---")
            print(f"                  Predicted")
            print(f"                CORRECT  INCORRECT")
            print(f"Actual CORRECT    {cm['true_negative']:5d}    {cm['false_positive']:5d}")
            print(f"Actual INCORRECT  {cm['false_negative']:5d}    {cm['true_positive']:5d}")

        # Per-class metrics
        if "per_class_metrics" in metrics:
            print(f"\n--- Per-Class Metrics ---")
            for class_name, class_metrics in metrics["per_class_metrics"].items():
                print(f"\n{class_name}:")
                print(f"  Precision: {class_metrics['precision']:.4f}")
                print(f"  Recall:    {class_metrics['recall']:.4f}")
                print(f"  F1 Score:  {class_metrics['f1_score']:.4f}")
                print(f"  Support:   {class_metrics['support']}")

        print("\n" + "=" * 60)

    def print_weight_analysis(self, weight_analysis: Dict):
        """Print weight optimization analysis and recommendations."""
        print("\n" + "=" * 60)
        print("WEIGHT OPTIMIZATION ANALYSIS")
        print("=" * 60)

        if "error" in weight_analysis:
            print(f"Error: {weight_analysis['error']}")
            return

        # Current performance
        perf = weight_analysis.get("current_performance", {})
        print(f"\n--- Current Expert Performance ---")
        print(f"  Expert A: {perf.get('expert_a_accuracy', 0):.1%} accuracy ({perf.get('expert_a_total_wins', 0)} wins)")
        print(f"  Expert B: {perf.get('expert_b_accuracy', 0):.1%} accuracy ({perf.get('expert_b_total_wins', 0)} wins)")

        # Recommended weights
        rec = weight_analysis.get("recommended_weights", {})
        print(f"\n--- Recommended Weights ---")
        print(f"  Expert A Weight: {rec.get('expert_a', 0.5):.1%}")
        print(f"  Expert B Weight: {rec.get('expert_b', 0.5):.1%}")
        print(f"  Reasoning: {rec.get('reasoning', 'N/A')}")

        # Threshold analysis
        threshold = weight_analysis.get("weight_threshold_analysis", {})
        print(f"\n--- Threshold Recommendation ---")
        print(f"  {threshold.get('recommendation', 'N/A')}")
        print(f"  Accuracy difference: {threshold.get('accuracy_difference', 0):.1%}")

        # Expert strengths by error type
        strengths = weight_analysis.get("expert_strengths", {})
        a_strengths = strengths.get("expert_a_better_at", [])
        b_strengths = strengths.get("expert_b_better_at", [])

        if a_strengths:
            print(f"\n--- Expert A Strengths (by error type) ---")
            for s in a_strengths[:5]:  # Show top 5
                print(f"  {s['error_type']}: {s['expert_a_accuracy']:.1%} vs {s['expert_b_accuracy']:.1%} (+{s['advantage']:.1%})")

        if b_strengths:
            print(f"\n--- Expert B Strengths (by error type) ---")
            for s in b_strengths[:5]:  # Show top 5
                print(f"  {s['error_type']}: {s['expert_b_accuracy']:.1%} vs {s['expert_a_accuracy']:.1%} (+{s['advantage']:.1%})")

        print("\n" + "=" * 60)

    def print_agent_summary(self, agent_metrics: Dict):
        """Print a formatted summary of agent/expert performance."""
        print("\n" + "=" * 60)
        print("AGENT/EXPERT PERFORMANCE ANALYSIS")
        print("=" * 60)

        if "error" in agent_metrics:
            print(f"Error: {agent_metrics['error']}")
            return

        summary = agent_metrics.get("summary", {})
        print(f"\nTotal Cases: {summary.get('total_cases', 0)}")
        print(f"Expert A Won: {summary.get('expert_a_won', 0)}")
        print(f"Expert B Won: {summary.get('expert_b_won', 0)}")
        print(f"Neither Won: {summary.get('neither_won', 0)}")

        for expert_name, expert_key in [("Expert A", "expert_a"), ("Expert B", "expert_b"), ("Neither", "neither")]:
            expert_data = agent_metrics.get(expert_key, {})
            total_wins = expert_data.get("total_wins", 0)
            if total_wins == 0:
                continue

            print(f"\n--- {expert_name} Performance ---")
            print(f"  Total Wins: {total_wins}")
            print(f"  Correct: {expert_data.get('correct_count', 0)}")
            print(f"  Incorrect: {expert_data.get('incorrect_count', 0)}")
            print(f"  Accuracy When Winning: {expert_data.get('accuracy_when_winning', 0):.4f}")

            # Show error types where this expert was correct
            correct_error_types = expert_data.get("correct_error_types", {})
            if correct_error_types:
                print(f"\n  Error Types (Correct Cases):")
                for error_type, count in sorted(correct_error_types.items(), key=lambda x: -x[1]):
                    print(f"    {error_type}: {count}")

            # Show error types where this expert was incorrect
            incorrect_error_types = expert_data.get("incorrect_error_types", {})
            if incorrect_error_types:
                print(f"\n  Error Types (Incorrect Cases):")
                for error_type, count in sorted(incorrect_error_types.items(), key=lambda x: -x[1]):
                    print(f"    {error_type}: {count}")

        print("\n" + "=" * 60)


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate medical error detection predictions")
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="logs/debates",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="evaluation/results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default="result_*.json",
        help="Glob pattern for result files"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = Evaluator(results_dir=args.results_dir)

    # Load results
    count = evaluator.load_results(pattern=args.pattern)

    if count == 0:
        print("No valid results found for evaluation.")
        return

    # Calculate metrics
    metrics = evaluator.calculate_metrics()

    # Calculate agent metrics
    agent_metrics = evaluator.calculate_agent_metrics()

    # Calculate weight optimization analysis
    weight_analysis = evaluator.calculate_optimal_weights()

    # Print summary
    evaluator.print_summary(metrics)
    evaluator.print_agent_summary(agent_metrics)
    evaluator.print_weight_analysis(weight_analysis)

    # Save results
    json_path, csv_path = evaluator.save_results(
        metrics,
        output_dir=args.output_dir,
        agent_metrics=agent_metrics
    )

    print(f"\nEvaluation complete!")
    print(f"Results saved to:")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV:  {csv_path}")


if __name__ == "__main__":
    main()
