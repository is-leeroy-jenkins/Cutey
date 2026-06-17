![](./img/cutey-project.png)

___

Cutey is a python application for exploring, validating, modeling, and exporting federal account-balance data. The application is organized around a practical balance-projection workflow: load data, inspect structure, evaluate distributions, transform numeric fields, explore relationships, compare regression models, test statistical assumptions, review period trends, and export the working dataset.

## 🧭 Purpose

Cutey supports analysts who need a local, interactive environment for reviewing account-balance, budget-execution, and financial-management datasets. It is especially useful when the data contains numeric balance fields, fiscal periods, Treasury-account-style identifiers, or agency financial attributes that need to be inspected before formal modeling or reporting.

## 🧱 Core Workflow

| Stage     | Application Area                         | Purpose                                                                      |
|-----------|------------------------------------------|------------------------------------------------------------------------------|
| Load      | Sidebar upload or fallback Excel file    | Load CSV, XLSX, or XLS data into the working dataframe.                      |
| Inspect   | Data tab                                 | Review rows, columns, data types, and duplicate headers.                     |
| Summarize | Descriptive Stats tab                    | Compute core and extended descriptive statistics.                            |
| Diagnose  | Distributions and Inferential Stats tabs | Review distribution shape, normality, confidence intervals, and group tests. |
| Transform | Transforms tab                           | Apply log, standard-scaling, or min-max transformations.                     |
| Explore   | PCA + Clustering and Correlations tabs   | Identify structure, clusters, and variable relationships.                    |
| Model     | Regression tab                           | Compare multiple regression estimators and diagnostics.                      |
| Trend     | Time Series tab                          | Aggregate numeric values by fiscal or period-like fields.                    |
| Export    | Export tab                               | Download the working dataframe as CSV.                                       |

## 🧪 Implemented Capabilities

Cutey currently includes:

- CSV, XLSX, and XLS data loading.
- Fallback loading from `data/Account Balances.xlsx`.
- Duplicate column-header detection.
- Numeric-column inference and coercion.
- Descriptive statistics with missing-value and shape diagnostics.
- Histogram and Q-Q plot generation.
- Numeric transformations using `log1p`, `StandardScaler`, and `MinMaxScaler`.
- Two-component PCA with KMeans clustering.
- Pearson, Spearman, and Kendall correlation analysis.
- Multi-model regression comparison.
- Regression diagnostics for actual-vs-predicted and residual behavior.
- Inferential statistics, including normality checks, confidence intervals, two-group tests, and ANOVA.
- Period-based aggregation and trend visualization when period-like columns exist.
- CSV export.

## 🏛️ Intended Users

Cutey is designed for:

- Budget analysts reviewing execution data.
- Data scientists preparing financial datasets for modeling.
- Auditors inspecting unusual balances, trends, or execution patterns.
- Program analysts comparing account-level relationships.
- Developers extending Streamlit analytics into a documented MkDocs site.

## 📦 Inputs and Outputs

| Type                     | Supported                                    |
|--------------------------|----------------------------------------------|
| Input files              | `.csv`, `.xlsx`, `.xls`                      |
| Fallback data            | `data/Account Balances.xlsx`                 |
| Primary in-memory object | pandas `DataFrame`                           |
| Visual outputs           | Streamlit tables, metrics, matplotlib charts |
| Download output          | `balance_projection_export.csv`              |

## ✅ Recommended Use

1. Start with a clean CSV or Excel file.
2. Confirm the file has meaningful numeric columns.
3. Use the Data tab to check data types and duplicate headers.
4. Use Descriptive Stats and Distributions before modeling.
5. Apply transformations only after reviewing skew, missingness, and scale.
6. Use regression and inferential results as analytical diagnostics, not as final official determinations.
7. Validate final conclusions against authoritative budget, accounting, or financial-management sources.

## 🔗 Documentation Map

| Page                              | Description                                                      |
|-----------------------------------|------------------------------------------------------------------|
| [Architecture](architecture.md)   | Application layers, data flow, and workflow design.              |
| [Data Sources](data-sources.md)   | Supported data inputs and recommended dataset structure.         |
| [User Guide](user-guide/index.md) | Step-by-step application usage.                                  |
| [API Reference](api/index.md)     | Source-generated documentation for helper functions in `app.py`. |
| [Development](development.md)     | Local setup, validation workflow, and MkDocs build process.      |

!!! warning "Analytical use"
    Cutey is intended for analytical exploration and documentation-supported review. Validate official financial, budgetary, or accounting conclusions against authoritative systems and governing guidance.
