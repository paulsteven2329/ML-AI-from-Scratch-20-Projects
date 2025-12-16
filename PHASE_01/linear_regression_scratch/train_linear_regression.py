import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Technical notes:
# - Implements univariate linear regression from scratch using gradient descent.
# - No sklearn used for model training; only numpy/pandas.
# - House price toy example with synthetic data or CSV input.


def generate_synthetic(n=200, seed=42):
    rng = np.random.default_rng(seed)
    sqft = rng.uniform(500, 3500, size=n)
    noise = rng.normal(0, 20000, size=n)
    price = 50 * sqft + 50000 + noise  # true slope ~50, intercept 50k
    return pd.DataFrame({"sqft": sqft, "price": price})


def normalize(x):
    mu = x.mean()
    sigma = x.std(ddof=0)
    return (x - mu) / (sigma + 1e-8), mu, sigma


def gradient_descent(x, y, lr=0.01, epochs=2000):
    # x, y are numpy arrays (x normalized), learn w (slope) and b (intercept)
    w = 0.0
    b = 0.0
    history = []
    n = len(x)
    for t in range(epochs):
        y_pred = w * x + b
        error = y_pred - y
        # MSE gradient
        dw = (2 / n) * np.dot(error, x)
        db = (2 / n) * error.sum()
        w -= lr * dw
        b -= lr * db
        loss = (error ** 2).mean()
        if t % 100 == 0:
            history.append((t, loss, w, b))
    return w, b, history


def train(df, feature="sqft", target="price", lr=0.01, epochs=2000, out_dir="outputs"):
    x_raw = df[feature].values.astype(float)
    y = df[target].values.astype(float)
    x, mu, sigma = normalize(x_raw)
    w, b, history = gradient_descent(x, y, lr=lr, epochs=epochs)

    # Map back to original scale: y = w*(x- mu)/sigma + b
    slope = w / (sigma + 1e-8)
    intercept = b - w * mu / (sigma + 1e-8)

    # Save training curve
    import os
    os.makedirs(out_dir, exist_ok=True)
    hist_df = pd.DataFrame(history, columns=["epoch", "loss", "w_norm", "b_norm"])
    hist_df.to_csv(f"{out_dir}/training_curve.csv", index=False)

    # Plot fit
    plt.figure(figsize=(6, 4))
    plt.scatter(x_raw, y, s=12, alpha=0.6, label="data")
    xx = np.linspace(x_raw.min(), x_raw.max(), 100)
    yy = slope * xx + intercept
    plt.plot(xx, yy, color="red", label="fit")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title("Linear Regression From Scratch (GD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fit.png")
    plt.close()

    print("Learned parameters:")
    print(f"slope = {slope:.2f}, intercept = {intercept:.2f}")
    return slope, intercept


def main():
    parser = argparse.ArgumentParser(description="Univariate Linear Regression from scratch (gradient descent)")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV with columns: sqft, price")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = generate_synthetic()

    train(df, lr=args.lr, epochs=args.epochs, out_dir=args.out)


if __name__ == "__main__":
    main()
