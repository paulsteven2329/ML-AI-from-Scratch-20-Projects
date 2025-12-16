# Logistic Regression — Customer Churn (From Scratch)

“Accuracy is not enough—metrics matter.”

## Goal
Train logistic regression from scratch to predict churn and understand classification metrics: Precision, Recall, F1.

## What You’ll Learn
- Sigmoid function and probabilities
- Decision boundary at threshold 0.5 (or tuned)
- Binary cross-entropy loss and gradients
- Evaluating with Precision/Recall/F1 beyond accuracy

## Run the Project
```bash
python "PHASE_01/logistic_regression_churn/train_logistic_regression.py" --out "PHASE_01/logistic_regression_churn/outputs"
```
Use a CSV with columns `tenure, monthly, support_calls, churn`:
```bash
python "PHASE_01/logistic_regression_churn/train_logistic_regression.py" --csv "/path/to/churn.csv" --out "PHASE_01/logistic_regression_churn/outputs"
```

## Files Generated
- `outputs/training_curve.csv`: epoch vs. cross-entropy loss
- Console metrics: Precision, Recall, F1, Accuracy

## Technical Doc (Code Understanding)
- `generate_synthetic(...)`: creates plausible churn-like features.
- `standardize(...)`: feature scaling for stable optimization.
- `train_logreg(...)`: gradient descent on cross-entropy loss.
- `metrics(...)`: computes confusion-matrix-based metrics.

## Practice
- Tune threshold (e.g., 0.4 vs 0.6) and observe precision/recall trade-offs.
- Plot ROC curve and compute AUC (extension).
- Add L2 regularization to reduce overfitting.

## Common Pitfalls
- Class imbalance: accuracy can be misleading—use recall/precision.
- Unscaled features: can slow convergence or cause instability.
- Overfitting: add regularization or early stopping.

## References
- Logistic Regression: https://en.wikipedia.org/wiki/Logistic_regression
- Precision/Recall/F1: https://en.wikipedia.org/wiki/Precision_and_recall
- Cross-Entropy: https://en.wikipedia.org/wiki/Cross-entropy
