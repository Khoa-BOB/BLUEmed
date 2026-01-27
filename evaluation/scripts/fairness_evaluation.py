#!/usr/bin/env python3
"""
Fairness Evaluation using Equality Difference (ED) Metrics.

This script evaluates demographic fairness by measuring equality differences
of true positive/negative and false positive/negative rates across demographic groups.

Reference:
    Dixon et al. (2018), Park et al. (2018), Garg et al. (2019)

Equality Difference (ED) Formula:
    ED = Σ_{d∈D} |Rate_d - Rate_overall|

where D is a demographic factor (e.g., gender, age group) and d is a specific
demographic group (e.g., Male, Female).

Usage:
    python evaluation/fairness_evaluation.py
    python evaluation/fairness_evaluation.py --results-dir logs/debates --output-dir evaluation/fairness
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Some statistics will be limited.")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plots will not be generated.")

try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Some metrics will be limited.")

# Import the demographic analyzer
import sys
sys.path.append(str(Path(__file__).parent))
from demographic_analysis import DemographicAnalyzer


class FairnessEvaluator:
    """Evaluates fairness using Equality Difference (ED) metrics."""

    def __init__(self, analyzer: DemographicAnalyzer):
        """
        Initialize fairness evaluator.

        Args:
            analyzer: DemographicAnalyzer instance with loaded data
        """
        self.analyzer = analyzer
        self.merged_data = analyzer.merged_data
        self.fairness_results = {}

    def calculate_confusion_metrics(self, records: List[Dict]) -> Dict[str, float]:
        """
        Calculate TP, TN, FP, FN and their rates for a group of records.

        Args:
            records: List of prediction records

        Returns:
            Dictionary with counts and rates (TPR, TNR, FPR, FNR)
        """
        if not records:
            return {
                'count': 0,
                'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
                'TPR': None, 'TNR': None, 'FPR': None, 'FNR': None
            }

        y_true = []
        y_pred = []
        for r in records:
            if r['ground_truth'] is not None and r['predicted_label'] is not None:
                y_true.append(r['ground_truth'])
                y_pred.append(r['predicted_label'])

        if not y_true:
            return {
                'count': 0,
                'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
                'TPR': None, 'TNR': None, 'FPR': None, 'FNR': None
            }

        # Calculate confusion matrix
        # Labels: 0 = CORRECT, 1 = INCORRECT
        if SKLEARN_AVAILABLE:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        else:
            # Manual calculation
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
            tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        # Calculate rates
        # TPR (True Positive Rate / Recall / Sensitivity) = TP / (TP + FN)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else None

        # TNR (True Negative Rate / Specificity) = TN / (TN + FP)
        tnr = tn / (tn + fp) if (tn + fp) > 0 else None

        # FPR (False Positive Rate) = FP / (FP + TN)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None

        # FNR (False Negative Rate) = FN / (FN + TP)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else None

        # Additional metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else None
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        f1 = 2 * (precision * tpr) / (precision + tpr) if (precision and tpr and (precision + tpr) > 0) else None

        return {
            'count': len(y_true),
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TPR': float(tpr) if tpr is not None else None,
            'TNR': float(tnr) if tnr is not None else None,
            'FPR': float(fpr) if fpr is not None else None,
            'FNR': float(fnr) if fnr is not None else None,
            'Accuracy': float(accuracy) if accuracy is not None else None,
            'Precision': float(precision) if precision is not None else None,
            'F1': float(f1) if f1 is not None else None,
        }

    def calculate_equality_difference(
        self,
        dimension: str,
        metric_name: str = 'FPR'
    ) -> Dict[str, Any]:
        """
        Calculate Equality Difference (ED) for a specific metric across demographic groups.

        ED = Σ_{d∈D} |Rate_d - Rate_overall|

        Args:
            dimension: Demographic dimension (e.g., 'gender', 'age_group')
            metric_name: Metric to evaluate ('TPR', 'TNR', 'FPR', 'FNR')

        Returns:
            Dictionary with ED score and detailed breakdown
        """
        # Calculate overall rate
        overall_metrics = self.calculate_confusion_metrics(self.merged_data)
        overall_rate = overall_metrics.get(metric_name)

        if overall_rate is None:
            return {
                'dimension': dimension,
                'metric': metric_name,
                'ED': None,
                'overall_rate': None,
                'groups': {},
                'error': 'Could not calculate overall rate'
            }

        # Group records by dimension
        groups = defaultdict(list)
        for record in self.merged_data:
            key = record.get(dimension, 'Unknown')
            if key is None:
                key = 'Unknown'
            groups[key].append(record)

        # Calculate rate for each group and ED
        group_metrics = {}
        ed_sum = 0.0

        for group_name, group_records in groups.items():
            metrics = self.calculate_confusion_metrics(group_records)
            group_rate = metrics.get(metric_name)

            group_metrics[group_name] = {
                'count': metrics['count'],
                'rate': group_rate,
                'difference': abs(group_rate - overall_rate) if group_rate is not None else None,
                'TP': metrics['TP'],
                'TN': metrics['TN'],
                'FP': metrics['FP'],
                'FN': metrics['FN'],
                'all_metrics': metrics
            }

            if group_rate is not None:
                ed_sum += abs(group_rate - overall_rate)

        return {
            'dimension': dimension,
            'metric': metric_name,
            'ED': float(ed_sum),
            'overall_rate': float(overall_rate),
            'groups': group_metrics,
            'num_groups': len(group_metrics)
        }

    def calculate_cross_demographic_ed(
        self,
        dimension1: str,
        dimension2: str,
        metric_name: str = 'FPR',
        min_sample_size: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate Equality Difference (ED) for cross-demographic groups.

        Args:
            dimension1: First demographic dimension (e.g., 'gender')
            dimension2: Second demographic dimension (e.g., 'age_group')
            metric_name: Metric to evaluate ('TPR', 'TNR', 'FPR', 'FNR')
            min_sample_size: Minimum samples required for a group to be included

        Returns:
            Dictionary with ED score and detailed breakdown
        """
        # Calculate overall rate
        overall_metrics = self.calculate_confusion_metrics(self.merged_data)
        overall_rate = overall_metrics.get(metric_name)

        if overall_rate is None:
            return {
                'dimension1': dimension1,
                'dimension2': dimension2,
                'metric': metric_name,
                'ED': None,
                'overall_rate': None,
                'groups': {},
                'error': 'Could not calculate overall rate'
            }

        # Group records by both dimensions
        cross_groups = defaultdict(list)
        for record in self.merged_data:
            val1 = record.get(dimension1, 'Unknown')
            val2 = record.get(dimension2, 'Unknown')
            if val1 is None:
                val1 = 'Unknown'
            if val2 is None:
                val2 = 'Unknown'

            # Create cross-group key: "Male × Young Adult (18-35)"
            cross_key = f"{val1} × {val2}"
            cross_groups[cross_key].append(record)

        # Calculate rate for each cross-group and ED
        group_metrics = {}
        ed_sum = 0.0
        excluded_groups = []

        for group_name, group_records in cross_groups.items():
            # Skip groups with insufficient samples
            if len(group_records) < min_sample_size:
                excluded_groups.append((group_name, len(group_records)))
                continue

            metrics = self.calculate_confusion_metrics(group_records)
            group_rate = metrics.get(metric_name)

            group_metrics[group_name] = {
                'count': metrics['count'],
                'rate': group_rate,
                'difference': abs(group_rate - overall_rate) if group_rate is not None else None,
                'TP': metrics['TP'],
                'TN': metrics['TN'],
                'FP': metrics['FP'],
                'FN': metrics['FN'],
                'all_metrics': metrics
            }

            if group_rate is not None:
                ed_sum += abs(group_rate - overall_rate)

        return {
            'dimension1': dimension1,
            'dimension2': dimension2,
            'metric': metric_name,
            'ED': float(ed_sum),
            'overall_rate': float(overall_rate),
            'groups': group_metrics,
            'num_groups': len(group_metrics),
            'excluded_groups': excluded_groups,
            'min_sample_size': min_sample_size
        }

    def evaluate_all_fairness(self, include_cross_demographic: bool = True) -> Dict[str, Any]:
        """
        Evaluate fairness across all demographic dimensions and metrics.

        Args:
            include_cross_demographic: Whether to include cross-demographic analysis

        Returns:
            Complete fairness evaluation results
        """
        print("\n" + "=" * 70)
        print("FAIRNESS EVALUATION (Equality Difference)")
        print("=" * 70)

        results = {
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(self.merged_data),
            'metrics_evaluated': ['TPR', 'TNR', 'FPR', 'FNR'],
            'dimensions': {},
            'cross_demographic': {}
        }

        # Demographic dimensions to evaluate
        dimensions = ['age_group', 'gender']

        # Metrics to evaluate
        metrics = ['TPR', 'TNR', 'FPR', 'FNR']

        for dimension in dimensions:
            print(f"\nEvaluating fairness for dimension: {dimension.upper()}")
            results['dimensions'][dimension] = {}

            for metric in metrics:
                print(f"  Calculating ED for {metric}...")
                ed_result = self.calculate_equality_difference(dimension, metric)
                results['dimensions'][dimension][metric] = ed_result

                # Print summary
                ed_score = ed_result['ED']
                if ed_score is not None:
                    print(f"    {metric} ED: {ed_score:.4f}")
                    for group, gmetrics in ed_result['groups'].items():
                        if gmetrics['rate'] is not None:
                            print(f"      {group}: {gmetrics['rate']:.4f} "
                                  f"(diff: {gmetrics['difference']:.4f}, n={gmetrics['count']})")

        # Cross-demographic analysis (Gender × Age)
        if include_cross_demographic:
            print("\n" + "=" * 70)
            print("CROSS-DEMOGRAPHIC ANALYSIS (Gender × Age)")
            print("=" * 70)

            results['cross_demographic']['gender_x_age'] = {}

            for metric in metrics:
                print(f"\n  Calculating cross-demographic ED for {metric}...")
                cross_ed = self.calculate_cross_demographic_ed(
                    'gender', 'age_group', metric, min_sample_size=10
                )
                results['cross_demographic']['gender_x_age'][metric] = cross_ed

                # Print summary
                ed_score = cross_ed['ED']
                if ed_score is not None:
                    print(f"    {metric} Cross-Demographic ED: {ed_score:.4f}")
                    print(f"    Included groups: {cross_ed['num_groups']}")

                    # Show top 5 groups with largest disparities
                    sorted_groups = sorted(
                        cross_ed['groups'].items(),
                        key=lambda x: x[1]['difference'] if x[1]['difference'] is not None else 0,
                        reverse=True
                    )

                    print(f"    Top 5 groups with largest disparities:")
                    for group, gmetrics in sorted_groups[:5]:
                        if gmetrics['rate'] is not None:
                            print(f"      {group:35s}: {gmetrics['rate']:.4f} "
                                  f"(diff: {gmetrics['difference']:.4f}, n={gmetrics['count']})")

                    # Show excluded groups
                    if cross_ed['excluded_groups']:
                        print(f"    Excluded groups (n < {cross_ed['min_sample_size']}): "
                              f"{len(cross_ed['excluded_groups'])}")

        # Calculate summary statistics
        results['summary'] = self._calculate_summary_statistics(results)

        self.fairness_results = results
        return results

    def _calculate_summary_statistics(self, results: Dict) -> Dict[str, Any]:
        """Calculate summary statistics across all ED scores."""
        all_eds = []

        for dimension, metrics in results['dimensions'].items():
            for metric, ed_result in metrics.items():
                if ed_result['ED'] is not None:
                    all_eds.append({
                        'dimension': dimension,
                        'metric': metric,
                        'ED': ed_result['ED']
                    })

        if not all_eds:
            return {'count': 0}

        ed_values = [item['ED'] for item in all_eds]

        summary = {
            'count': len(ed_values),
            'mean_ED': float(np.mean(ed_values)) if NUMPY_AVAILABLE else sum(ed_values) / len(ed_values),
            'max_ED': float(max(ed_values)),
            'min_ED': float(min(ed_values)),
            'all_scores': all_eds
        }

        if NUMPY_AVAILABLE:
            summary['std_ED'] = float(np.std(ed_values))
            summary['median_ED'] = float(np.median(ed_values))

        return summary

    def generate_visualizations(self, output_dir: str = "evaluation/fairness"):
        """
        Generate visualization plots for fairness evaluation.

        Args:
            output_dir: Directory to save plots
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Skipping visualizations.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("GENERATING FAIRNESS VISUALIZATIONS")
        print("=" * 70)

        # 1. ED scores heatmap
        self._plot_ed_heatmap(output_path)

        # 2. Rate comparison plots
        self._plot_rate_comparisons(output_path)

        # 3. Confusion matrix breakdown
        self._plot_confusion_breakdown(output_path)

        # 4. ED bar charts
        self._plot_ed_barcharts(output_path)

        # 5. Cross-demographic visualizations
        if 'cross_demographic' in self.fairness_results and self.fairness_results['cross_demographic']:
            self._plot_cross_demographic_heatmaps(output_path)

        print(f"\nVisualizations saved to: {output_path}")

    def _plot_ed_heatmap(self, output_path: Path):
        """Plot heatmap of ED scores across dimensions and metrics."""
        print("  Generating ED heatmap...")

        dimensions = list(self.fairness_results['dimensions'].keys())
        metrics = self.fairness_results['metrics_evaluated']

        # Create matrix of ED scores
        ed_matrix = []
        for dimension in dimensions:
            row = []
            for metric in metrics:
                ed_score = self.fairness_results['dimensions'][dimension][metric]['ED']
                row.append(ed_score if ed_score is not None else 0)
            ed_matrix.append(row)

        ed_matrix = np.array(ed_matrix) if NUMPY_AVAILABLE else ed_matrix

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(ed_matrix, cmap='YlOrRd', aspect='auto')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(dimensions)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels([d.replace('_', ' ').title() for d in dimensions])

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Equality Difference (ED)", rotation=-90, va="bottom")

        # Add text annotations
        for i in range(len(dimensions)):
            for j in range(len(metrics)):
                value = ed_matrix[i][j] if NUMPY_AVAILABLE else ed_matrix[i][j]
                text = ax.text(j, i, f'{value:.3f}',
                             ha="center", va="center", color="black", fontsize=11, weight='bold')

        ax.set_title("Fairness Evaluation: Equality Difference (ED) Scores\n"
                     "Lower scores indicate better fairness", fontsize=14, weight='bold')
        fig.tight_layout()

        filepath = output_path / "fairness_ed_heatmap.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {filepath}")

    def _plot_rate_comparisons(self, output_path: Path):
        """Plot rate comparisons across demographic groups."""
        print("  Generating rate comparison plots...")

        metrics = ['FPR', 'FNR', 'TPR', 'TNR']

        for dimension in self.fairness_results['dimensions'].keys():
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Rate Comparison by {dimension.replace("_", " ").title()}',
                        fontsize=16, weight='bold')

            for idx, metric in enumerate(metrics):
                ax = axes[idx // 2, idx % 2]

                ed_result = self.fairness_results['dimensions'][dimension][metric]
                overall_rate = ed_result['overall_rate']

                groups = []
                rates = []
                counts = []

                for group, gmetrics in ed_result['groups'].items():
                    if gmetrics['rate'] is not None:
                        groups.append(group)
                        rates.append(gmetrics['rate'])
                        counts.append(gmetrics['count'])

                # Plot bars
                x_pos = np.arange(len(groups)) if NUMPY_AVAILABLE else list(range(len(groups)))
                bars = ax.bar(x_pos, rates, alpha=0.7, color='steelblue', edgecolor='black')

                # Add overall rate line
                ax.axhline(y=overall_rate, color='red', linestyle='--', linewidth=2,
                          label=f'Overall {metric}: {overall_rate:.3f}')

                # Customize
                ax.set_xlabel('Demographic Group', fontsize=11)
                ax.set_ylabel(f'{metric} Rate', fontsize=11)
                ax.set_title(f'{metric} (ED: {ed_result["ED"]:.4f})', fontsize=12, weight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(groups, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)

                # Add count labels on bars
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'n={count}',
                           ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            filepath = output_path / f"fairness_rates_{dimension}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {filepath}")

    def _plot_confusion_breakdown(self, output_path: Path):
        """Plot confusion matrix breakdown by demographic groups."""
        print("  Generating confusion matrix breakdown...")

        for dimension in self.fairness_results['dimensions'].keys():
            # Get FPR data (contains all confusion metrics)
            ed_result = self.fairness_results['dimensions'][dimension]['FPR']

            groups = list(ed_result['groups'].keys())
            tp_vals = [ed_result['groups'][g]['TP'] for g in groups]
            tn_vals = [ed_result['groups'][g]['TN'] for g in groups]
            fp_vals = [ed_result['groups'][g]['FP'] for g in groups]
            fn_vals = [ed_result['groups'][g]['FN'] for g in groups]

            # Stacked bar chart
            fig, ax = plt.subplots(figsize=(12, 6))

            x_pos = np.arange(len(groups)) if NUMPY_AVAILABLE else list(range(len(groups)))
            width = 0.6

            # Stack bars
            p1 = ax.bar(x_pos, tp_vals, width, label='True Positives (TP)', color='green', alpha=0.8)
            p2 = ax.bar(x_pos, tn_vals, width, bottom=tp_vals, label='True Negatives (TN)',
                       color='lightgreen', alpha=0.8)

            bottom = [tp + tn for tp, tn in zip(tp_vals, tn_vals)]
            p3 = ax.bar(x_pos, fp_vals, width, bottom=bottom, label='False Positives (FP)',
                       color='orange', alpha=0.8)

            bottom = [tp + tn + fp for tp, tn, fp in zip(tp_vals, tn_vals, fp_vals)]
            p4 = ax.bar(x_pos, fn_vals, width, bottom=bottom, label='False Negatives (FN)',
                       color='red', alpha=0.8)

            ax.set_xlabel('Demographic Group', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'Confusion Matrix Breakdown by {dimension.replace("_", " ").title()}',
                        fontsize=14, weight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(groups, rotation=45, ha='right')
            ax.legend(loc='upper right')
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            filepath = output_path / f"confusion_breakdown_{dimension}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {filepath}")

    def _plot_ed_barcharts(self, output_path: Path):
        """Plot ED scores as bar charts for each dimension."""
        print("  Generating ED bar charts...")

        metrics = self.fairness_results['metrics_evaluated']

        for dimension in self.fairness_results['dimensions'].keys():
            fig, ax = plt.subplots(figsize=(10, 6))

            ed_scores = []
            for metric in metrics:
                ed_score = self.fairness_results['dimensions'][dimension][metric]['ED']
                ed_scores.append(ed_score if ed_score is not None else 0)

            x_pos = np.arange(len(metrics)) if NUMPY_AVAILABLE else list(range(len(metrics)))
            bars = ax.bar(x_pos, ed_scores, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'],
                         alpha=0.8, edgecolor='black')

            # Add value labels
            for bar, score in zip(bars, ed_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.4f}',
                       ha='center', va='bottom', fontsize=11, weight='bold')

            ax.set_xlabel('Metric', fontsize=12)
            ax.set_ylabel('Equality Difference (ED)', fontsize=12)
            ax.set_title(f'Fairness Scores by {dimension.replace("_", " ").title()}\n'
                        f'(Lower is more fair)',
                        fontsize=14, weight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics)
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            filepath = output_path / f"ed_barchart_{dimension}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {filepath}")

    def _plot_cross_demographic_heatmaps(self, output_path: Path):
        """Plot heatmaps for cross-demographic analysis."""
        print("  Generating cross-demographic heatmaps...")

        if 'gender_x_age' not in self.fairness_results['cross_demographic']:
            return

        metrics = self.fairness_results['metrics_evaluated']

        for metric in metrics:
            cross_ed = self.fairness_results['cross_demographic']['gender_x_age'][metric]

            if not cross_ed['groups']:
                continue

            # Extract gender and age from cross-group names
            # Format: "Male × Young Adult (18-35)"
            groups_data = {}
            genders = set()
            ages = set()

            for group_name, gmetrics in cross_ed['groups'].items():
                if ' × ' in group_name:
                    gender, age = group_name.split(' × ')
                    genders.add(gender)
                    ages.add(age)
                    groups_data[(gender, age)] = gmetrics['rate']

            genders = sorted(list(genders))
            ages = sorted(list(ages))

            # Create matrix
            matrix = []
            for gender in genders:
                row = []
                for age in ages:
                    rate = groups_data.get((gender, age), None)
                    row.append(rate if rate is not None else 0)
                matrix.append(row)

            if not matrix:
                continue

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

            # Set ticks and labels
            ax.set_xticks(np.arange(len(ages)) if NUMPY_AVAILABLE else list(range(len(ages))))
            ax.set_yticks(np.arange(len(genders)) if NUMPY_AVAILABLE else list(range(len(genders))))
            ax.set_xticklabels(ages, fontsize=10)
            ax.set_yticklabels(genders, fontsize=11)

            # Rotate x labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel(f"{metric} Rate", rotation=-90, va="bottom", fontsize=11)

            # Add text annotations with counts
            for i, gender in enumerate(genders):
                for j, age in enumerate(ages):
                    rate = matrix[i][j]
                    # Get count
                    count = 0
                    group_name = f"{gender} × {age}"
                    if group_name in cross_ed['groups']:
                        count = cross_ed['groups'][group_name]['count']

                    if rate > 0:
                        color = "white" if rate < 0.5 else "black"
                        text = ax.text(j, i, f'{rate:.3f}\n(n={count})',
                                     ha="center", va="center", color=color,
                                     fontsize=9, weight='bold')

            ax.set_xlabel('Age Group', fontsize=12)
            ax.set_ylabel('Gender', fontsize=12)
            ax.set_title(f'Cross-Demographic {metric} Rates (Gender × Age)\n'
                        f'Overall {metric}: {cross_ed["overall_rate"]:.3f} | '
                        f'ED: {cross_ed["ED"]:.4f}',
                        fontsize=13, weight='bold')

            plt.tight_layout()
            filepath = output_path / f"cross_demographic_{metric}_heatmap.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {filepath}")

        # Also create a summary comparison plot
        self._plot_cross_demographic_ed_comparison(output_path)

    def _plot_cross_demographic_ed_comparison(self, output_path: Path):
        """Plot comparison of ED scores: simple vs cross-demographic."""
        print("  Generating ED comparison (simple vs cross-demographic)...")

        if 'gender_x_age' not in self.fairness_results['cross_demographic']:
            return

        metrics = self.fairness_results['metrics_evaluated']

        # Collect ED scores
        gender_eds = []
        age_eds = []
        cross_eds = []

        for metric in metrics:
            gender_ed = self.fairness_results['dimensions']['gender'][metric]['ED']
            age_ed = self.fairness_results['dimensions']['age_group'][metric]['ED']
            cross_ed = self.fairness_results['cross_demographic']['gender_x_age'][metric]['ED']

            gender_eds.append(gender_ed)
            age_eds.append(age_ed)
            cross_eds.append(cross_ed)

        # Plot grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 7))

        x = np.arange(len(metrics)) if NUMPY_AVAILABLE else list(range(len(metrics)))
        width = 0.25

        bars1 = ax.bar(x - width, gender_eds, width, label='Gender', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x, age_eds, width, label='Age Group', color='orange', alpha=0.8)
        bars3 = ax.bar(x + width, cross_eds, width, label='Gender × Age (Cross)', color='crimson', alpha=0.8)

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, weight='bold')

        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Equality Difference (ED)', fontsize=12)
        ax.set_title('Fairness Comparison: Single-Dimension vs Cross-Demographic\n'
                    '(Lower ED = More Fair)',
                    fontsize=14, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filepath = output_path / "ed_comparison_cross_demographic.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {filepath}")

    def save_results(self, output_dir: str = "evaluation/fairness"):
        """
        Save fairness evaluation results to JSON and CSV.

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        json_path = output_path / f"fairness_evaluation_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.fairness_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nJSON results saved to: {json_path}")

        # Save CSV summary
        csv_path = output_path / f"fairness_summary_{timestamp}.csv"
        self._save_csv_summary(csv_path)
        print(f"CSV summary saved to: {csv_path}")

        return json_path, csv_path

    def _save_csv_summary(self, filepath: Path):
        """Save ED scores summary to CSV."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('dimension,metric,ED_score,overall_rate,num_groups\n')

            for dimension, metrics in self.fairness_results['dimensions'].items():
                for metric, ed_result in metrics.items():
                    ed = ed_result['ED']
                    overall = ed_result['overall_rate']
                    num_groups = ed_result['num_groups']
                    f.write(f'{dimension},{metric},{ed},{overall},{num_groups}\n')

    def print_summary(self):
        """Print formatted summary of fairness evaluation."""
        print("\n" + "=" * 70)
        print("FAIRNESS EVALUATION SUMMARY")
        print("=" * 70)

        summary = self.fairness_results.get('summary', {})

        print(f"\nTotal Cases Evaluated: {self.fairness_results['total_cases']}")
        print(f"Total ED Scores Calculated: {summary.get('count', 0)}")

        if summary.get('count', 0) > 0:
            print(f"\nEquality Difference (ED) Statistics:")
            print(f"  Mean ED:   {summary.get('mean_ED', 0):.4f}")
            print(f"  Median ED: {summary.get('median_ED', 'N/A')}")
            print(f"  Min ED:    {summary.get('min_ED', 0):.4f}")
            print(f"  Max ED:    {summary.get('max_ED', 0):.4f}")
            if 'std_ED' in summary:
                print(f"  Std ED:    {summary.get('std_ED', 0):.4f}")

        print("\n" + "-" * 70)
        print("DETAILED RESULTS BY DIMENSION AND METRIC")
        print("-" * 70)

        for dimension, metrics in self.fairness_results['dimensions'].items():
            print(f"\n{dimension.upper().replace('_', ' ')}:")

            for metric in ['FPR', 'FNR', 'TPR', 'TNR']:
                ed_result = metrics[metric]
                print(f"\n  {metric}:")
                print(f"    ED Score: {ed_result['ED']:.4f}")
                print(f"    Overall Rate: {ed_result['overall_rate']:.4f}")
                print(f"    Group Breakdown:")

                for group, gmetrics in ed_result['groups'].items():
                    if gmetrics['rate'] is not None:
                        print(f"      {group:20s}: {gmetrics['rate']:.4f} "
                              f"(diff: {gmetrics['difference']:+.4f}, n={gmetrics['count']})")

        print("\n" + "=" * 70)
        print("INTERPRETATION:")
        print("  - Lower ED scores indicate better fairness (less disparity)")
        print("  - FPR ED: Measures false positive rate equality")
        print("  - FNR ED: Measures false negative rate equality")
        print("  - TPR ED: Measures true positive rate (recall) equality")
        print("  - TNR ED: Measures true negative rate (specificity) equality")
        print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fairness evaluation using Equality Difference (ED) metrics"
    )
    parser.add_argument(
        "--validation-file", "-v",
        type=str,
        default="test_data/validation.json",
        help="Path to validation.json"
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="logs/debates",
        help="Directory containing result_*.json files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="evaluation/fairness",
        help="Directory to save fairness evaluation results"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DEMOGRAPHIC FAIRNESS EVALUATION")
    print("Using Equality Difference (ED) Metrics")
    print("=" * 70)

    # Step 1: Load data using DemographicAnalyzer
    print("\nStep 1: Loading and processing data...")
    analyzer = DemographicAnalyzer(
        validation_file=args.validation_file,
        results_dir=args.results_dir
    )

    count = analyzer.load_data()
    if count == 0:
        print("No prediction results found. Run batch_predict.py first.")
        return

    analyzer.merge_data()

    # Step 2: Run fairness evaluation
    print("\nStep 2: Running fairness evaluation...")
    evaluator = FairnessEvaluator(analyzer)
    results = evaluator.evaluate_all_fairness()

    # Step 3: Print summary
    print("\nStep 3: Printing summary...")
    evaluator.print_summary()

    # Step 4: Generate visualizations
    print("\nStep 4: Generating visualizations...")
    evaluator.generate_visualizations(output_dir=args.output_dir)

    # Step 5: Save results
    print("\nStep 5: Saving results...")
    evaluator.save_results(output_dir=args.output_dir)

    print("\n" + "=" * 70)
    print("FAIRNESS EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
