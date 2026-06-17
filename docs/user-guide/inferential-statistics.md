# Inferential Statistics

The Inferential Statistics tab provides normality checks, confidence intervals, and group-comparison tests for selected variables.

## 🧭 Purpose

Inferential tests help analysts evaluate assumptions, compare groups, and determine whether observed differences may be statistically meaningful.

## 🧮 Numeric Variable

Select a numeric variable for testing. Cutey coerces the selected field to numeric values and removes missing values before running tests.

## 📏 Normality and Confidence Metrics

| Metric | Purpose |
|---|---|
| Shapiro-Wilk W | Tests whether the sample appears normally distributed. |
| Kolmogorov-Smirnov statistic | Compares standardized values against a standard normal distribution. |
| 95% confidence interval | Estimates a plausible range for the population mean. |

## 👥 Optional Grouping Column

When a nonnumeric grouping column is selected, Cutey compares values across groups.

## 🧪 Two-Group Tests

If the grouping column contains two valid groups, Cutey runs:

| Test | Purpose |
|---|---|
| Welch t-test | Compares group means without assuming equal variance. |
| Mann-Whitney U | Compares group distributions nonparametrically. |
| Levene test | Tests equality of variances. |

## 🧪 Three-or-More Group Tests

If the grouping column contains three or more valid groups, Cutey runs:

| Test | Purpose |
|---|---|
| One-way ANOVA | Tests whether group means differ across groups. |

## ⚠️ Interpretation Notes

- Statistical significance does not guarantee operational significance.
- Very small samples can make tests unreliable.
- Very large samples can make small differences statistically significant.
- Group definitions should be meaningful and documented.
- Confirm assumptions before using results in formal reporting.

## ✅ Recommended Practice

1. Inspect distributions first.
2. Confirm group labels are clean and meaningful.
3. Check sample sizes by group.
4. Interpret p-values alongside effect size, context, and data quality.
