# KNN & Decision Trees — Visual Intuition

“Simple models can be surprisingly powerful.”

## Goal
Build intuition for K-Nearest Neighbors (KNN) and Decision Trees via a simple student performance dataset and visual summaries.

## What You’ll Learn
- KNN: distance metrics, neighbor voting
- Decision Trees: splits, thresholds, impurity (Gini)
- Bias-variance tradeoff: K too small vs too large; deep vs shallow trees

## Run the Project
```bash
python "PHASE_01/knn_decision_trees/run_models.py" --out "PHASE_01/knn_decision_trees/outputs" --k 5
```

## Files Generated
- `outputs/summary.txt`: KNN accuracy, stump accuracy, best split feature/threshold, post angle

## Technical Doc (Code Understanding)
- `generate_student(...)`: synthetic dataset with `study_hours`, `attendance`, `prior_grade`, and `passed`.
- `knn_predict(...)`: Euclidean distance + majority vote among k nearest neighbors.
- `decision_stump_train(...)`: finds the best single split using Gini impurity.
- `decision_stump_predict(...)`: applies the learned threshold to predict.

## Practice
- Change `--k` and watch accuracy; plot decision regions (extension).
- Replace stump with a deeper tree; compare bias-variance.
- Try different distance metrics (Manhattan, Minkowski) for KNN.

## Common Pitfalls
- Unscaled features distort distances in KNN; standardize when needed.
- Overfitting with very small K or overly deep trees.
- Ignoring class imbalance; consider weighted votes or balanced splits.

## References
- KNN: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
- Decision Trees: https://en.wikipedia.org/wiki/Decision_tree_learning
- Gini Impurity: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
