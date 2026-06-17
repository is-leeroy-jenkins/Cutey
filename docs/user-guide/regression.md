# Regression

The Regression tab compares multiple supervised regression models using selected numeric predictors and a selected numeric target.

## 🧭 Purpose

Regression analysis estimates how well selected numeric fields predict a target field and provides diagnostics to review model fit and residual behavior.

## 🧱 Requirements

| Requirement | Reason |
|---|---|
| At least two numeric columns | One or more predictors and one target are required. |
| Complete rows | Rows with missing predictor or target values are removed. |
| At least 20 complete rows | The app requires enough data for train/test evaluation. |

## ⚙️ Controls

| Control | Description |
|---|---|
| Predictors `X` | Numeric columns used as model inputs. |
| Target `y` | Numeric field the models attempt to predict. |

## 🧪 Models Compared

| Model | Family |
|---|---|
| Linear Regression | Ordinary linear model |
| Ridge | Regularized linear model |
| Lasso | Sparse regularized linear model |
| Bayesian Ridge | Bayesian linear regression |
| ElasticNet | Mixed L1/L2 regularized model |
| HuberRegressor | Robust regression |
| SGDRegressor | Iterative linear model |
| RandomForestRegressor | Tree ensemble |
| GradientBoostingRegressor | Boosted tree ensemble |

## 📏 Metrics

| Metric | Meaning |
|---|---|
| MSE | Mean squared error. Lower is better. |
| R² | Share of target variance explained. Higher is better. |

## 📈 Diagnostics

Cutey plots diagnostics for the best model by R².

| Diagnostic | Purpose |
|---|---|
| Actual vs Predicted | Shows how close predictions are to actual target values. |
| Residuals vs Predicted | Shows error patterns, bias, heteroskedasticity, or outlier behavior. |

## ⚠️ Interpretation Notes

- Do not use R² alone to select a model.
- Review residual patterns before trusting model output.
- Tree models can fit nonlinear patterns but may overfit small datasets.
- Regularized models can help when predictors are correlated.
- Results are exploratory unless validated against a formal analytical protocol.

## ✅ Recommended Practice

1. Start with a small, defensible predictor set.
2. Avoid using columns that directly encode the target.
3. Compare linear and ensemble models.
4. Review diagnostics.
5. Validate results with domain knowledge.
