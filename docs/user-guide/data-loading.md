# Data Loading

The sidebar controls data loading. Cutey can load uploaded CSV or Excel files, or use the fallback workbook when no file is uploaded.

## 🧭 Purpose

The data-loading workflow establishes the working pandas dataframe used by every analytical tab.

## 📥 Supported Uploads

| Upload Type | Extension | Loader |
|---|---|---|
| CSV | `.csv` | `pandas.read_csv` |
| Excel | `.xlsx` | `pandas.read_excel` |
| Excel | `.xls` | `pandas.read_excel` |

## 🧾 Excel Sheet Selection

When an Excel workbook is loaded, Cutey reads available sheet names and shows a sheet selector. The selected sheet becomes the working dataframe.

## 📂 Fallback File

When no upload is provided, Cutey attempts to load:

```text
data/Account Balances.xlsx
```

If this file is missing, the app displays a data-load failure message and stops.

## ⚠️ Duplicate Headers

If duplicate column headers are detected, Cutey displays a warning in the sidebar. Duplicate headers can create ambiguity because selecting a column by name may return more than one column. Cutey handles this in selected helper functions by using the first matching column when a one-dimensional series is required.

## ❌ Unsupported Files

Unsupported file types are rejected. Use `.csv`, `.xlsx`, or `.xls`.

## ✅ Recommended Practice

1. Start with a clean worksheet or CSV.
2. Remove title rows and explanatory notes above the header.
3. Ensure the first row contains column names.
4. Avoid duplicate column names.
5. Convert financial fields to numeric values where possible.
6. Keep fiscal year, period, or availability fields in a consistent format.
