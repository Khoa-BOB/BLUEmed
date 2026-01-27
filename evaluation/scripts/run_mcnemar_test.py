import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def mcnemar_test(y_true, y_pred_a, y_pred_b, exact_threshold=25):
    """
    Compute McNemar's test p-value between two classifiers.

    Parameters:
        y_true (array-like): Ground truth labels (0/1)
        y_pred_a (array-like): Predictions from model A (e.g., BLUEmed)
        y_pred_b (array-like): Predictions from model B (baseline)
        exact_threshold (int): Use exact test if b + c < threshold

    Returns:
        p_value (float)
        contingency_table (list)
    """

    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)

    # Contingency table counts
    a = np.sum((y_pred_a == y_true) & (y_pred_b == y_true))
    b = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
    c = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
    d = np.sum((y_pred_a != y_true) & (y_pred_b != y_true))

    table = [[a, b],
             [c, d]]

    # Choose exact or approximate McNemar
    exact = (b + c) < exact_threshold

    result = mcnemar(table, exact=exact, correction=not exact)

    return result.pvalue, table


def create_baseline_predictions(y_true, baseline_type='majority'):
    """
    Create baseline predictions for comparison.

    Parameters:
        y_true: Ground truth labels
        baseline_type: Type of baseline ('majority', 'random', 'stratified')

    Returns:
        Baseline predictions
    """
    if baseline_type == 'majority':
        # Predict the majority class for all samples
        majority_class = np.bincount(y_true).argmax()
        return np.full_like(y_true, majority_class)

    elif baseline_type == 'random':
        # Random predictions with same class distribution
        np.random.seed(42)
        return np.random.randint(0, 2, size=len(y_true))

    elif baseline_type == 'stratified':
        # Random predictions matching the true distribution
        np.random.seed(42)
        class_ratio = np.mean(y_true)
        return np.random.binomial(1, class_ratio, size=len(y_true))

    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


def main():
    # Load the evaluation results
    eval_file = 'results/evaluation_20251124_090504.csv'

    print("=" * 80)
    print("McNemar's Test for BLUEmed Model Evaluation")
    print("=" * 80)
    print(f"\nLoading evaluation results from: {eval_file}")

    # Read the CSV file
    df = pd.read_csv(eval_file)

    print(f"Total samples: {len(df)}")
    print(f"\nClass distribution:")
    print(df['ground_truth_label'].value_counts())

    # Extract predictions and ground truth
    y_true = df['ground_truth'].values
    y_pred_bluemed = df['predicted'].values

    # Calculate BLUEmed accuracy
    bluemed_correct = np.sum(y_pred_bluemed == y_true)
    bluemed_accuracy = bluemed_correct / len(y_true)

    print(f"\nBLUEmed Model Performance:")
    print(f"  Correct predictions: {bluemed_correct}/{len(y_true)}")
    print(f"  Accuracy: {bluemed_accuracy:.4f}")

    # Compare against different baselines
    baselines = ['majority', 'random', 'stratified']

    for baseline_type in baselines:
        print("\n" + "-" * 80)
        print(f"\nBaseline: {baseline_type.upper()}")
        print("-" * 80)

        # Create baseline predictions
        y_pred_baseline = create_baseline_predictions(y_true, baseline_type)

        # Calculate baseline accuracy
        baseline_correct = np.sum(y_pred_baseline == y_true)
        baseline_accuracy = baseline_correct / len(y_true)

        print(f"  Correct predictions: {baseline_correct}/{len(y_true)}")
        print(f"  Accuracy: {baseline_accuracy:.4f}")

        # Perform McNemar's test
        p_value, table = mcnemar_test(y_true, y_pred_bluemed, y_pred_baseline)

        print(f"\nMcNemar's Test Results:")
        print(f"  Contingency Table:")
        print(f"    Both correct (a):     {table[0][0]:4d}")
        print(f"    BLUEmed only (b):     {table[0][1]:4d}")
        print(f"    Baseline only (c):    {table[1][0]:4d}")
        print(f"    Both wrong (d):       {table[1][1]:4d}")
        print(f"\n  McNemar statistic (b + c): {table[0][1] + table[1][0]}")
        print(f"  p-value: {p_value:.6f}")

        # Interpretation
        alpha = 0.05
        if p_value < alpha:
            if table[0][1] > table[1][0]:
                print(f"  ✓ BLUEmed is SIGNIFICANTLY BETTER than {baseline_type} baseline (p < {alpha})")
            else:
                print(f"  ✗ BLUEmed is SIGNIFICANTLY WORSE than {baseline_type} baseline (p < {alpha})")
        else:
            print(f"  → No significant difference between BLUEmed and {baseline_type} baseline (p >= {alpha})")

    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
