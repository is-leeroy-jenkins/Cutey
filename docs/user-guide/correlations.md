# Correlations

The Correlations tab calculates and visualizes pairwise relationships among selected numeric columns.

## 🧭 Purpose

Correlation analysis helps identify related variables, redundant predictors, potential multicollinearity, and relationships worth investigating before regression or forecasting.

## 🧮 Methods

| Method | Description | Use |
|---|---|---|
| Pearson | Linear correlation. | Best for approximately linear relationships. |
| Spearman | Rank-based monotonic correlation. | Useful for nonlinear monotonic relationships or skewed data. |
| Kendall | Rank association. | Conservative rank-based association measure. |

## 📊 Outputs

Cutey displays:

- Correlation matrix.
- Correlation heatmap.
- Numeric values inside heatmap cells.

## ⚠️ Interpretation Notes

- Correlation does not prove causation.
- High correlation among predictors can affect regression stability.
- Outliers can distort Pearson correlation.
- Spearman and Kendall can be more robust for skewed financial data.

## ✅ Recommended Next Step

Use correlation results to refine predictor choices before moving to [Regression](regression.md).
