import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Technical notes:
# - Implements logistic regression training via gradient descent on a simple churn dataset.
# - If no CSV provided, generates a synthetic dataset with a few features.
# - Reports Precision, Recall, and F1 alongside accuracy.


def generate_synthetic(n=500, seed=7):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, size=n)
    monthly = rng.uniform(20, 120, size=n)
    support_calls = rng.integers(0, 10, size=n)
    # Churn propensity modeled by a linear combination with noise
    logits = -3.0 + (-0.03 * tenure) + (0.02 * monthly) + (0.2 * support_calls) + rng.normal(0, 0.5, size=n)
    prob = 1 / (1 + np.exp(-logits))
    churn = (prob > 0.5).astype(int)
    return pd.DataFrame({
        "tenure": tenure,
        "monthly": monthly,
        "support_calls": support_calls,
        "churn": churn
    })


def standardize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0) + 1e-8
    return (X - mu) / sigma, mu, sigma


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_logreg(X, y, lr=0.1, epochs=2000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    history = []
    for t in range(epochs):
        z = X @ w + b
        p = sigmoid(z)
        error = p - y
        # Binary cross-entropy gradient
        grad_w = (X.T @ error) / n
        grad_b = error.mean()
        w -= lr * grad_w
        b -= lr * grad_b
        if t % 100 == 0:
            loss = -(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8)).mean()
            history.append((t, loss))
    return w, b, history


def metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return precision, recall, f1, accuracy


def main():
    parser = argparse.ArgumentParser(description="Logistic Regression for Customer Churn (from scratch)")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV with columns: tenure, monthly, support_calls, churn")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = generate_synthetic()

    features = [c for c in df.columns if c != "churn"]
    X_raw = df[features].values.astype(float)
    y = df["churn"].values.astype(int)

    X, mu, sigma = standardize(X_raw)
    w, b, history = train_logreg(X, y, lr=args.lr, epochs=args.epochs)

    # Predictions at threshold 0.5
    p = sigmoid(X @ w + b)
    y_hat = (p >= 0.5).astype(int)

    precision, recall, f1, accuracy = metrics(y, y_hat)
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(history, columns=["epoch", "loss"]).to_csv(f"{args.out}/training_curve.csv", index=False)

    print("Metrics:")
    print(f"Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}\nAccuracy: {accuracy:.3f}")
    print("Post Angle: Accuracy is not enough—metrics matter.")


if __name__ == "__main__":
    main()
