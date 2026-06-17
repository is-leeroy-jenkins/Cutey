# Export

The Export tab downloads the currently loaded dataframe as a CSV file.

## 🧭 Purpose

Export provides a simple way to save the working dataset after loading and review.

## 📤 Download

The export button creates:

```text
balance_projection_export.csv
```

The file is encoded as UTF-8 CSV.

## 📦 Export Scope

The export contains the loaded dataframe. It does not currently include:

- Descriptive statistics tables.
- Transformed data views.
- PCA components.
- Cluster labels.
- Correlation matrices.
- Regression results.
- Diagnostic plots.
- Inferential test outputs.
- Time-series aggregation tables.

## ✅ Recommended Practice

Use export to preserve the input dataframe used during the session. Save analytical outputs separately if they are needed for reports, audit trails, or reproducibility packages.

## 🔗 Related Pages

- [Data Loading](data-loading.md)
- [Data Overview](data-overview.md)
- [Regression](regression.md)
- [Time Series](time-series.md)
