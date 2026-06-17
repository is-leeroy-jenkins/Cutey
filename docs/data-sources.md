# Data Sources

Cutey is designed to work with tabular financial, budget-execution, and account-balance datasets. The application is flexible enough to accept generic CSV or Excel files, but it is most useful when the dataset contains numeric balance fields and period-like attributes.

## 🧭 Purpose

Below describes the data that Cutey can load, the fields that improve analytical value, and the recommended preparation checks before using statistical or machine-learning tabs.

## 📥 Supported File Types

| File Type             |                    Extension | Behavior                                                 |
|-----------------------|-----------------------------:|----------------------------------------------------------|
| CSV                   |                       `.csv` | Loaded directly into a pandas dataframe.                 |
| Excel workbook        |                      `.xlsx` | Sheet names are listed and the selected sheet is loaded. |
| Legacy Excel workbook |                       `.xls` | Sheet names are listed and the selected sheet is loaded. |
| Fallback Excel file   | `data/Account Balances.xlsx` | Loaded when no upload is provided and the file exists.   |

## 🧾 Recommended Dataset Structure

Cutey works best with datasets that include:

| Field Type            | Examples                                                          | Why It Helps                                                                         |
|-----------------------|-------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Numeric balances      | obligations, outlays, recoveries, unobligated balances, resources | Required for statistics, transformations, PCA, correlations, regression, and trends. |
| Period fields         | fiscal year, period, availability period                          | Required for the Time Series tab to group and visualize trends.                      |
| Account identifiers   | TAS, main account, agency, bureau, account name                   | Useful for filtering, grouping, and interpretation outside the app.                  |
| Classification fields | account type, availability category, program activity             | Useful for inferential grouping and analyst review.                                  |
| Target variables      | future balance, outlay, obligation rate, execution rate           | Required for supervised regression workflows.                                        |

## 🏛️ Federal Financial Context

Cutey's domain is federal budget execution and accounting. Suitable source concepts include:

| Data Concept                    | Typical Use                                          |
|---------------------------------|------------------------------------------------------|
| SF-133                          | Status of budget execution and budgetary resources.  |
| GTAS                            | Trial balance and Treasury-account-level actuals.    |
| Account A / account balances    | Balance projection and execution analysis.           |
| Agency apportionments           | Budget execution controls and availability context.  |
| USAspending / Data.gov extracts | Public financial and programmatic context.           |
| Object class / program activity | Execution analysis by spending category or activity. |

These sources are not hard-coded into the current Streamlit app. They should be prepared as CSV or Excel files before loading into Cutey.

## 🔎 Data Quality Checks

Before modeling, confirm:

| Check                     | Reason                                                                              |
|---------------------------|-------------------------------------------------------------------------------------|
| Duplicate column headers  | Duplicate headers can cause pandas to return multiple columns for one name.         |
| Missing values            | Missing values affect summary statistics, tests, PCA, clustering, and regression.   |
| Numeric coercion          | Some numeric-looking fields may be stored as text.                                  |
| Period field quality      | Time-series aggregation requires a clean year, period, or availability-like column. |
| Extreme values            | Large balances or outliers can dominate regressions and transformations.            |
| Target leakage            | Predictors should not include fields that directly encode the target.               |

## 🧮 Numeric Columns

The application first checks pandas numeric data types. If no typed numeric columns exist, it attempts to infer numeric columns by coercing values and checking whether a sufficient share of values are numeric.

This behavior allows Cutey to work with spreadsheets where financial fields are imported as text, but it does not eliminate the need for data validation.

## 🧱 Duplicate Headers

When duplicate headers are detected, Cutey warns the user. Several helper functions use the first matching column when a requested column name maps to multiple columns. This protects calculations that require a one-dimensional series.

## 🕰️ Period Fields

The Time Series tab looks for columns whose names contain:

- `year`
- `period`
- `availability`

The selected period field is converted to numeric where possible, and selected numeric values are summed by period.

## ✅ Recommended Preparation Sequence

1. Remove unnecessary title rows, notes, and merged-header artifacts from the spreadsheet.
2. Use one row per observation.
3. Use one column per variable.
4. Ensure numeric fields do not contain currency symbols or footnote markers.
5. Keep period or fiscal-year fields consistent.
6. Avoid duplicate column names when possible.
7. Save the dataset as `.csv` or `.xlsx`.
8. Load into Cutey and review the Data tab before running models.

## 🔗 Related Pages

- [Data Loading](user-guide/data-loading.md)
- [Data Overview](user-guide/data-overview.md)
- [Transformations](user-guide/transformations.md)
- [Regression](user-guide/regression.md)
- [Time Series](user-guide/time-series.md)
