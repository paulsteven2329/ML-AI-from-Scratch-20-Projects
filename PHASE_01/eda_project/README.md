# Data Exploration & Visualization Project

Goal: Show strong fundamentals, data intuition, and model understanding.

Project: Exploratory Data Analysis on a Real Dataset

Concepts:
- Pandas, NumPy
- Missing values, outliers
- Data distributions
- Feature correlations

Post Angle: "Before any ML model, understanding data is everything."

## Tutorial: Step-by-Step EDA

Follow this mini tutorial to explore a dataset end-to-end. You can use the seaborn `penguins` dataset or your own CSV.

### 1) Setup your environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r "PHASE_01/eda_project/requirements.txt"
```

### 2) Load and summarize the data

Run the EDA script to produce core summaries and plots:

```bash
python "PHASE_01/eda_project/eda.py" --source penguins --out "PHASE_01/eda_project/outputs"
```

Outputs created:
- `outputs/missing_counts.csv`: per-column count of missing values
- `outputs/missing_pct.csv`: per-column percent of missing values
- `outputs/basic_stats.csv`: `describe()` summary including numeric stats and categorical counts

Why this matters: Understanding missingness guides imputation vs. dropping strategies and informs downstream model robustness.

### 3) Examine distributions and outliers

- `outputs/figs/dist_<feature>.png`: numeric feature distributions with KDE overlay
- `outputs/outliers_iqr.csv`: IQR-based outlier counts per numeric feature

Tips:
- Heavy tails or multimodal distributions may suggest transformations or feature splitting.
- Many outliers can indicate measurement issues or rare but important cases.

### 4) Check feature correlations

- `outputs/figs/correlations_heatmap.png`: Pearson correlations among numeric features

Interpreting correlations:
- High absolute correlation (|r| > 0.8) can imply redundancy; consider feature selection or dimensionality reduction.
- Low correlation does not mean independence; explore non-linear relationships and interactions.

### 5) Use your own CSV

```bash
python "PHASE_01/eda_project/eda.py" --source "/absolute/path/to/your.csv" --out "PHASE_01/eda_project/outputs"
```

Guidance:
- Ensure the CSV has headers. Non-numeric columns will be treated as categorical.
- If the file is large, start with a sample to speed up (e.g., pre-filter rows).

## How to Run

- With seaborn dataset:
```bash
python "PHASE_01/eda_project/eda.py" --source penguins --out "PHASE_01/eda_project/outputs"
```

- With local CSV:
```bash
python "PHASE_01/eda_project/eda.py" --source "/path/to/data.csv" --out "PHASE_01/eda_project/outputs"
```

## What It Does

- Loads a dataset from CSV or seaborn by name
- Summarizes shape, dtypes, missing values, and `describe()` stats
- Detects outliers per numeric feature via IQR rule
- Visualizes numeric distributions, categorical counts
- Plots a correlation heatmap for numeric features

## Notes & Extensions

- Correlation uses pairwise Pearson on numeric features.
- For wide datasets, consider sampling to speed up plots.
- Extend with domain checks (e.g., valid ranges), statistical tests, and target-variable analysis.

## References (Deeper Dive)

- Pandas User Guide: https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html
- NumPy Fundamentals: https://numpy.org/doc/stable/user/basics.html
- Seaborn Tutorials: https://seaborn.pydata.org/tutorial.html
- Matplotlib Gallery: https://matplotlib.org/stable/gallery/index.html
- Outlier detection (IQR): https://en.wikipedia.org/wiki/Interquartile_range
- Pearson correlation: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

Post Angle Reminder: "Before any ML model, understanding data is everything."
