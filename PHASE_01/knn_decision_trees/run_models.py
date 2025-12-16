import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Technical notes:
# - Visual, simple implementations of KNN and a basic Decision Tree (stump) for interpretability.
# - Uses a small dataset (e.g., seaborn 'student' like features via synthetic gen) to predict pass/fail.


def generate_student(n=300, seed=123):
    rng = np.random.default_rng(seed)
    study_hours = rng.uniform(0, 10, size=n)
    attendance = rng.integers(50, 100, size=n)
    prior_grade = rng.integers(40, 100, size=n)
    logits = -5 + 0.5 * study_hours + 0.05 * attendance + 0.06 * prior_grade + rng.normal(0, 0.5, size=n)
    prob = 1 / (1 + np.exp(-logits))
    passed = (prob > 0.5).astype(int)
    return pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "prior_grade": prior_grade,
        "passed": passed
    })


def knn_predict(X_train, y_train, x, k=5):
    # Euclidean distance; simple vote
    dists = np.linalg.norm(X_train - x, axis=1)
    idx = np.argsort(dists)[:k]
    votes = y_train[idx]
    most_common = Counter(votes).most_common(1)[0][0]
    return most_common


def decision_stump_train(X, y):
    # Train a single-split tree: choose feature and threshold that best separates classes by Gini impurity
    n, d = X.shape
    best = {"gini": 1e9, "feature": None, "threshold": None}
    for j in range(d):
        thresholds = np.unique(X[:, j])
        for thr in thresholds:
            left = y[X[:, j] <= thr]
            right = y[X[:, j] > thr]
            def gini(arr):
                if len(arr) == 0:
                    return 0.0
                p = arr.mean()
                return 2 * p * (1 - p)
            g = (len(left) * gini(left) + len(right) * gini(right)) / (len(left) + len(right) + 1e-8)
            if g < best["gini"]:
                best = {"gini": g, "feature": j, "threshold": thr}
    return best


def decision_stump_predict(stump, x):
    j = stump["feature"]
    thr = stump["threshold"]
    return 1 if x[j] > thr else 0


def evaluate_knn_tree(df, k=5):
    features = [c for c in df.columns if c != "passed"]
    X = df[features].values.astype(float)
    y = df["passed"].values.astype(int)

    # Simple train/test split
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    tr_idx, te_idx = idx[:split], idx[split:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    # KNN
    preds_knn = np.array([knn_predict(X_tr, y_tr, X_te[i], k=k) for i in range(len(X_te))])
    acc_knn = (preds_knn == y_te).mean()

    # Decision stump (visual split)
    stump = decision_stump_train(X_tr, y_tr)
    preds_tree = np.array([decision_stump_predict(stump, X_te[i]) for i in range(len(X_te))])
    acc_tree = (preds_tree == y_te).mean()

    return acc_knn, acc_tree, stump, features


def main():
    parser = argparse.ArgumentParser(description="KNN & Decision Tree (stump) explained visually")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    parser.add_argument("--k", type=int, default=5, help="K for KNN")
    args = parser.parse_args()

    df = generate_student()
    acc_knn, acc_tree, stump, features = evaluate_knn_tree(df, k=args.k)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(f"KNN accuracy: {acc_knn:.3f}\n")
        f.write(f"Decision stump accuracy: {acc_tree:.3f}\n")
        f.write(f"Best split feature: {features[stump['feature']]} at threshold {stump['threshold']:.3f}\n")
        f.write("Post Angle: Simple models can be surprisingly powerful.\n")

    print("Saved:", os.path.join(args.out, "summary.txt"))


if __name__ == "__main__":
    main()
