# Data Overview

The Data tab displays the loaded dataframe and its column data types.

## 🧭 Purpose

Use this tab first. It verifies that the file loaded correctly and that the dataframe structure is suitable for downstream analytics.

## 🧱 Data Preview

The data preview shows the loaded dataframe in an editable Streamlit grid. This view helps confirm:

- Row count.
- Column count.
- Column names.
- Obvious data-quality issues.
- Missing values.
- Numeric fields imported as text.
- Unexpected repeated headers.

## 🧾 Column Types

The column-type table lists each column and its pandas data type.

Common types include:

| Type | Meaning |
|---|---|
| `int64` | Integer numeric field. |
| `float64` | Decimal numeric field. |
| `object` | Text or mixed-type field. |
| `datetime64` | Date/time field. |
| `bool` | Boolean field. |

## ⚠️ What to Check

| Check | Why It Matters |
|---|---|
| Numeric fields imported as text | May require coercion before modeling. |
| Duplicate headers | Can produce ambiguous column selection. |
| Empty columns | Add noise to statistical summaries. |
| Period fields | Needed for time-series aggregation. |
| Target variable | Needed for regression workflows. |

## ✅ Recommended Next Step

After reviewing this tab, continue to [Descriptive Statistics](descriptive-statistics.md) to understand the numeric fields.
