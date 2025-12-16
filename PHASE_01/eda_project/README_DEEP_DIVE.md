# Exploratory Data Analysis (EDA) — Beginner Friendly Deep Dive

Before any ML model, understanding data is everything.

## What is EDA?
EDA is the process of inspecting, cleaning, and visualizing data to understand its structure, quality, and relationships. The goal is to build intuition and spot issues early.

## Why it matters
- Better features: You discover transformations and new signals.
- Fewer surprises: You find missing values, outliers, and odd distributions early.
- Smarter modeling: You choose models and metrics that match the data.

## Concepts Covered
- Pandas & NumPy basics
- Missing values: counts, percentages, handling strategies
- Outliers: IQR rule and when to care
- Distributions: histograms, KDE, category counts
- Correlations: Pearson heatmap for numeric features

## Hands-on Practice
1) Setup and run using the seaborn `penguins` dataset:
```bash
python "PHASE_01/eda_project/eda.py" --source penguins --out "PHASE_01/eda_project/outputs"
```
2) Review outputs:
- `outputs/missing_counts.csv`, `outputs/missing_pct.csv`
- `outputs/basic_stats.csv`
- `outputs/outliers_iqr.csv`
- `outputs/figs/` plots

3) Try your own CSV:
```bash
python "PHASE_01/eda_project/eda.py" --source "/absolute/path/to/your.csv" --out "PHASE_01/eda_project/outputs"
```

## Interpreting Results
- High missing percentages: consider imputation (median/mean/mode), domain defaults, or dropping.
- Skewed distributions: consider log transform, binning, or robust models.
- Strong correlations: potential redundancy; consider feature selection or PCA.

## Technical Notes (Code Understanding)
- `load_data(...)`: loads from CSV or seaborn; pandas infers dtypes which drive numeric vs categorical processing.
- `summarize(...)`: uses `describe(include="all")` for numeric and categorical summaries.
- `detect_outliers_iqr(...)`: flags values outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` per feature.
- `plot_distributions(...)`: histograms + KDE for numeric, count plots for categorical.
- `plot_correlations(...)`: Pearson correlation on numeric columns only.

## Practice Ideas
- Add Spearman correlation for monotonic relationships.
- Create pairplots to visually inspect feature relationships.
- Build a simple data quality checklist (ranges, duplicates, invalid categories).

## References
- Pandas User Guide: https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html
- NumPy Basics: https://numpy.org/doc/stable/user/basics.html
- Seaborn Tutorials: https://seaborn.pydata.org/tutorial.html
- IQR rule: https://en.wikipedia.org/wiki/Interquartile_range
- Pearson correlation: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
