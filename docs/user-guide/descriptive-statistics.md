# Descriptive Statistics

The Descriptive Stats tab summarizes selected numeric columns.

## 🧭 Purpose

Use descriptive statistics to understand central tendency, spread, missingness, skew, and the overall shape of financial or account-balance fields before deeper analysis.

## 🧮 Column Selection

Cutey identifies numeric columns automatically. If no typed numeric columns exist, it attempts to infer numeric columns by coercing values and checking whether enough values are numeric.

## 📊 Core Summary

The core summary uses pandas descriptive statistics and includes selected percentiles.

| Statistic | Meaning |
|---|---|
| Count | Non-null observations. |
| Mean | Average value. |
| Standard deviation | Spread around the mean. |
| Minimum | Smallest observed value. |
| Percentiles | Distribution cut points. |
| Maximum | Largest observed value. |

## 📈 Extended Summary

The extended summary includes:

| Metric | Meaning |
|---|---|
| `missing_count` | Number of missing values. |
| `missing_pct` | Percentage of missing values. |
| `zero_count` | Number of zero values. |
| `unique_count` | Number of unique values. |
| `skewness` | Distribution asymmetry. |
| `kurtosis` | Tail weight and peak behavior. |

## 📊 Grouped Bar Chart

The grouped bar chart visualizes the first five rows of selected numeric columns. This is a quick visual check for scale differences and obvious anomalies.

## ✅ Recommended Interpretation

- High missing percentages may require filtering, imputation, or exclusion.
- High skewness may justify a log transformation.
- Extreme maximum values may dominate regression or PCA.
- Zero-heavy fields may require special interpretation in budget execution analysis.
