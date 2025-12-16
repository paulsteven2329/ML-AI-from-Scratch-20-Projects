import argparse
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def load_data(source: str) -> pd.DataFrame:
    """
    Load a dataset.
    - If source is a file path, read CSV via pandas.
    - If source is a known seaborn dataset name, load via seaborn.
    Defaults to seaborn 'penguins' if source is empty.

    Technical note:
    - Using seaborn datasets ensures a consistent schema for quick demos.
    - CSVs can include mixed types; pandas infers dtypes, which affects downstream
        numeric vs. categorical handling.
    """
    if not source:
        return sns.load_dataset("penguins").drop(columns=["species"], errors="ignore")

    if os.path.isfile(source):
        df = pd.read_csv(source)
        return df

    # try seaborn dataset by name for convenience and reproducibility
    try:
        df = sns.load_dataset(source)
        return df
    except Exception as e:
        print(f"Failed to load dataset '{source}': {e}")
        sys.exit(1)


def summarize(df: pd.DataFrame) -> dict:
    summary = {}
    summary["shape"] = df.shape
    summary["dtypes"] = df.dtypes.astype(str).to_dict()
    summary["missing_counts"] = df.isna().sum().to_dict()
    summary["missing_pct"] = (df.isna().mean() * 100).round(2).to_dict()
    # pandas describe() provides numeric stats and categorical counts/unique/top/freq
    summary["basic_stats"] = df.describe(include="all").to_dict()
    return summary


def detect_outliers_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    outlier_counts = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            outlier_counts[col] = 0
            continue
        # IQR rule: flag values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_counts[col] = int(((series < lower) | (series > upper)).sum())
    return pd.DataFrame({"feature": list(outlier_counts.keys()), "outliers": list(outlier_counts.values())})


def plot_distributions(df: pd.DataFrame, out_dir: str, numeric_cols: list[str], categorical_cols: list[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Numeric distributions
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"dist_{col}.png"))
        plt.close()

    # Categorical counts
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(y=df[col])
        plt.title(f"Counts: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"counts_{col}.png"))
        plt.close()


def plot_correlations(df: pd.DataFrame, out_dir: str, numeric_cols: list[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Pearson correlation on numeric features; captures linear relationships only
    corr = df[numeric_cols].corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlations")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlations_heatmap.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis (EDA) on a dataset")
    parser.add_argument("--source", type=str, default="penguins", help="CSV path or seaborn dataset name (default: penguins)")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory for figures and reports")
    args = parser.parse_args()

    df = load_data(args.source)

    # Basic type splits: informs appropriate handling (numeric vs categorical)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Summary artifacts saved for auditability and iteration
    summary = summarize(df)
    os.makedirs(args.out, exist_ok=True)
    pd.Series(summary["missing_counts"]).to_csv(os.path.join(args.out, "missing_counts.csv"))
    pd.Series(summary["missing_pct"]).to_csv(os.path.join(args.out, "missing_pct.csv"))
    pd.DataFrame(summary["basic_stats"]).to_csv(os.path.join(args.out, "basic_stats.csv"))

    # Outliers via IQR: robust to non-normal distributions; simple heuristic
    outliers_df = detect_outliers_iqr(df, numeric_cols)
    outliers_df.to_csv(os.path.join(args.out, "outliers_iqr.csv"), index=False)

    # Visualizations: quick shapes of distributions and category frequencies
    figs_dir = os.path.join(args.out, "figs")
    plot_distributions(df, figs_dir, numeric_cols, categorical_cols)

    if len(numeric_cols) >= 2:
        plot_correlations(df, figs_dir, numeric_cols)

    # Print concise console report
    print("=== EDA Summary ===")
    print(f"Rows, Columns: {summary['shape']}")
    print("Missing values (top 5):")
    print(pd.Series(summary["missing_counts"]).sort_values(ascending=False).head())
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)
    print(f"Outputs saved to: {args.out}")
    print("Post Angle: Before any ML model, understanding data is everything.")


if __name__ == "__main__":
    main()
