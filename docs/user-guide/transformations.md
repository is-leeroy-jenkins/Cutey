# Transformations

The Transforms tab applies selected numeric transformations to a cleaned numeric view.

## 🧭 Purpose

Transformations help prepare numeric data for exploratory analysis, PCA, clustering, regression, and visual comparison.

## 🧮 Available Transforms

| Transform | Description | Typical Use |
|---|---|---|
| `None` | Keeps original values. | Baseline inspection. |
| `log1p` | Applies `log(1 + x)`. | Reducing right skew in non-negative fields. |
| `StandardScaler` | Centers values and scales to unit variance. | PCA, clustering, and models sensitive to scale. |
| `MinMaxScaler` | Scales values to a bounded range. | Comparing fields on a common range. |

## 🧹 Cleaning Behavior

The tab coerces selected fields to numeric values and drops rows with missing values in the selected columns before applying the transform.

## ⚠️ Interpretation Notes

- `log1p` is best for non-negative values.
- Standard scaling changes units to standardized values.
- Min-max scaling is sensitive to extreme values.
- Transformed values are displayed for review; they do not overwrite the original loaded dataframe in the export tab.

## ✅ Recommended Sequence

1. Review descriptive statistics.
2. Review distributions.
3. Identify skew or scale issues.
4. Apply a transformation.
5. Compare transformed values before modeling.
