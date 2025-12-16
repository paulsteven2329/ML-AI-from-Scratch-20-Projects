# Linear Regression From Scratch — Beginner Friendly

“If you can build Linear Regression from scratch, you understand ML.”

## Goal
Implement univariate linear regression using gradient descent without sklearn. Learn how loss, gradients, and parameters interact.

## What You’ll Learn
- Gradient descent mechanics
- Mean Squared Error (MSE) loss
- Normalization and parameter scaling
- Underfitting vs. overfitting intuition

## Run the Project
```bash
python "PHASE_01/linear_regression_scratch/train_linear_regression.py" --out "PHASE_01/linear_regression_scratch/outputs"
```
Optionally use your CSV with columns `sqft, price`:
```bash
python "PHASE_01/linear_regression_scratch/train_linear_regression.py" --csv "/path/to/house.csv" --out "PHASE_01/linear_regression_scratch/outputs"
```

## Files Generated
- `outputs/training_curve.csv`: epoch, loss, parameters
- `outputs/fit.png`: data scatter with fitted line

## Technical Doc (Code Understanding)
- `generate_synthetic(...)`: creates a toy linear relationship plus noise.
- `normalize(...)`: standardizes features to 0 mean, unit variance.
- `gradient_descent(...)`: updates slope `w` and intercept `b` using MSE gradients.
- Parameter mapping: learned `w, b` in normalized space are converted back to original scale for interpretability.

## Practice
- Change learning rate `--lr` and epochs `--epochs` to see training behavior.
- Add a validation split; measure MAE/MSE on validation.
- Extend to multivariate regression (multiple features).

## Common Pitfalls
- Learning rate too high: divergence.
- Not normalizing features: slow or unstable convergence.
- Too few epochs: underfitting; too many with noise: overfitting.

## References
- Gradient Descent: https://en.wikipedia.org/wiki/Gradient_descent
- MSE Loss: https://en.wikipedia.org/wiki/Mean_squared_error
- Feature Scaling: https://en.wikipedia.org/wiki/Feature_scaling
