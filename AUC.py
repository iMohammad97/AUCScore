import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def compute_auc_roc(y_true, anomaly_scores):
    """
    Compute AUC-ROC for time-series anomaly detection.

    Parameters:
        y_true (array-like): Ground truth binary labels for each time point (0 = normal, 1 = anomaly).
        anomaly_scores (array-like): Anomaly scores (higher indicates more anomalous).

    Returns:
        float: AUC-ROC score.
    """
    try:
        auc_roc = roc_auc_score(y_true, anomaly_scores)
    except ValueError:
        print("AUC-ROC computation failed: Ensure both classes (0 and 1) are present in y_true.")
        auc_roc = np.nan
    return auc_roc

def compute_auc_pr(y_true, anomaly_scores):
    """
    Compute AUC-PR for time-series anomaly detection.

    Parameters:
        y_true (array-like): Ground truth binary labels for each time point (0 = normal, 1 = anomaly).
        anomaly_scores (array-like): Anomaly scores (higher indicates more anomalous).

    Returns:
        float: AUC-PR score.
    """
    try:
        precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
        auc_pr = auc(recall, precision)
    except ValueError:
        print("AUC-PR computation failed: Ensure both classes (0 and 1) are present in y_true.")
        auc_pr = np.nan
    return auc_pr

# Example usage
if __name__ == "__main__":
    # Ground truth binary labels
    y_true = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]

    # Anomaly scores (higher means more anomalous)
    anomaly_scores = [0.1, 0.2, 0.9, 0.8, 0.3, 0.2, 0.7, 0.9, 0.4, 0.1]

    # Compute AUC-ROC
    auc_roc = compute_auc_roc(y_true, anomaly_scores)
    print(f"AUC-ROC: {auc_roc:.2f}")

    # Compute AUC-PR
    auc_pr = compute_auc_pr(y_true, anomaly_scores)
    print(f"AUC-PR: {auc_pr:.2f}")
