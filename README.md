# AUC-ROC and AUC-PR Calculation Module

This repository provides a Python module for calculating **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve) and **AUC-PR** (Area Under the Precision-Recall Curve). These metrics are designed to evaluate time-series anomaly detection models effectively, especially for imbalanced datasets and varying anomaly scoring thresholds.

## Features

- **AUC-ROC**:
  - Measures the trade-off between the true positive rate (sensitivity) and false positive rate (1-specificity) at various thresholds.
  - Suitable for datasets with balanced classes.
  
- **AUC-PR**:
  - Measures the trade-off between precision and recall at different thresholds.
  - Especially effective for imbalanced datasets where anomalies are rare.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/iMohammad97/AUCScore.git
   ```
2. Navigate to the directory:
   ```bash
   cd AUCScore
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Import the Module
```python
from AUC import compute_auc_roc, compute_auc_pr
```

### 2. Input Formats

#### Ground Truth Labels (`y_true`)
A binary array where `1` indicates an anomaly and `0` indicates normalcy:
```python
y_true = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
```

#### Anomaly Scores (`anomaly_scores`)
A continuous array of anomaly scores where higher values indicate more anomalous behavior:
```python
anomaly_scores = [0.1, 0.2, 0.9, 0.8, 0.3, 0.2, 0.7, 0.9, 0.4, 0.1]
```

### 3. Example Code

```python
from AUCScore import compute_auc_roc, compute_auc_pr

# Ground truth labels
y_true = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]

# Anomaly scores
anomaly_scores = [0.1, 0.2, 0.9, 0.8, 0.3, 0.2, 0.7, 0.9, 0.4, 0.1]

# Calculate AUC-ROC
auc_roc = compute_auc_roc(y_true, anomaly_scores)
print(f"AUC-ROC: {auc_roc:.2f}")

# Calculate AUC-PR
auc_pr = compute_auc_pr(y_true, anomaly_scores)
print(f"AUC-PR: {auc_pr:.2f}")
```

### 4. Output
- **AUC-ROC**: Outputs the Area Under the Receiver Operating Characteristic Curve.
- **AUC-PR**: Outputs the Area Under the Precision-Recall Curve.

### 5. Error Handling
- Ensure `y_true` contains both classes (0 and 1); otherwise, a warning is printed, and `np.nan` is returned.

## Contribution

We welcome contributions to improve this module! Fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

This implementation was inspired by the need for robust evaluation metrics in time-series anomaly detection, focusing on imbalanced datasets and continuous anomaly scoring.
