# Distributions

The Distributions tab visualizes numeric fields with histograms and Q-Q plots.

## 🧭 Purpose

Distribution review helps determine whether values are skewed, heavy-tailed, sparse, zero-heavy, or approximately normal. These properties influence transformations, tests, and modeling choices.

## 📊 Histograms

Users select one or more numeric columns and choose a bin count. Cutey plots each selected field as a histogram.

Histograms help identify:

- Skewed balances.
- Extreme values.
- Multi-modal patterns.
- Zero-heavy fields.
- Scale differences across accounts or measures.

## 📈 Q-Q Plot

The Q-Q plot compares a selected numeric field against a normal distribution.

| Pattern | Interpretation |
|---|---|
| Points close to line | More normal-like distribution. |
| Curved pattern | Skew or heavy tails. |
| Extreme endpoint deviations | Outliers or tail risk. |
| Sparse points | Insufficient observations or discrete data. |

## ⚠️ Minimum Data

Cutey requires enough non-null values for the Q-Q plot. If the selected column has too few values, the app displays a warning.

## ✅ Recommended Next Step

Use this tab before [Transformations](transformations.md). Skewed or scale-heavy fields may benefit from `log1p`, `StandardScaler`, or `MinMaxScaler`.
