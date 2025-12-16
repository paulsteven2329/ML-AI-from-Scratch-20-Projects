# PHASE 1: ML Foundations (Core Concepts)

Goal: Show strong fundamentals, data intuition, and model understanding.

## 1️⃣ Data Exploration & Visualization Project
- Project: Exploratory Data Analysis on a Real Dataset
- Concepts: Pandas, NumPy; Missing values, Outliers; Data distributions; Feature correlations
- Post Angle: "Before any ML model, understanding data is everything."
- Folder: `eda_project`
- Run:
  ```bash
  python "PHASE_01/eda_project/eda.py" --source penguins --out "PHASE_01/eda_project/outputs"
  ```

## 2️⃣ Linear Regression From Scratch
- Project: Predict House Prices Without Using Sklearn
- Concepts: Gradient descent; Loss functions; Overfitting vs underfitting
- Post Angle: "If you can build Linear Regression from scratch, you understand ML."
- Folder: `linear_regression_scratch`
- Run:
  ```bash
  python "PHASE_01/linear_regression_scratch/train_linear_regression.py" --out "PHASE_01/linear_regression_scratch/outputs"
  ```

## 3️⃣ Classification with Logistic Regression
- Project: Customer Churn Prediction
- Concepts: Sigmoid function; Decision boundary; Precision, Recall, F1
- Post Angle: "Accuracy is not enough—metrics matter."
- Folder: `logistic_regression_churn`
- Run:
  ```bash
  python "PHASE_01/logistic_regression_churn/train_logistic_regression.py" --out "PHASE_01/logistic_regression_churn/outputs"
  ```

## 4️⃣ KNN & Decision Trees Explained Visually
- Project: Student Performance Prediction
- Concepts: Distance metrics; Tree splits; Bias-variance tradeoff
- Post Angle: "Simple models can be surprisingly powerful."
- Folder: `knn_decision_trees`
- Run:
  ```bash
  python "PHASE_01/knn_decision_trees/run_models.py" --out "PHASE_01/knn_decision_trees/outputs"
  ```

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r "PHASE_01/requirements.txt"
```
