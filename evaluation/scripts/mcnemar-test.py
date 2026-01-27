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
