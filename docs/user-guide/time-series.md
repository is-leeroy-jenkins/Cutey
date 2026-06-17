# Time Series

The Time Series tab aggregates selected numeric values by a period-like field and visualizes trends.

## 🧭 Purpose

This tab helps analysts inspect period-based movement in account balances, obligations, outlays, or other numeric measures.

## 📦 Optional Dependency

The tab requires `statsmodels` to be installed. If it is not available, Cutey displays a warning.

```powershell
pip install statsmodels
```

## 🧱 Period Column Detection

Cutey searches for columns whose names contain:

- `year`
- `period`
- `availability`

If no matching column exists, the tab displays a warning.

## ⚙️ Controls

| Control | Description |
|---|---|
| Period Column | Field used to group observations by fiscal year, period, or availability-like value. |
| Value Columns | Numeric columns summed by the selected period. |

## 📊 Output

Cutey displays:

- Aggregated table by period.
- Multi-series trend chart.
- Distinct line styles and markers for selected value fields.

## ⚠️ Forecasting Boundary

The current visible workflow performs aggregation and trend visualization. Although the source imports forecasting classes from `statsmodels`, user-facing ARIMA or Exponential Smoothing forecast controls should not be documented as implemented until those controls exist in the application.

## ✅ Recommended Practice

1. Confirm the period field is numeric or can be coerced to numeric.
2. Use consistent fiscal periods.
3. Select value fields that can be meaningfully summed.
4. Review trends for abnormal spikes, drop-offs, or missing periods.
5. Validate period totals against source systems before reporting.
