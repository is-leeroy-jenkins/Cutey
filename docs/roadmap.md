# Roadmap

Cutey already provides a practical Streamlit workflow for data loading, statistical review, regression comparison, and period-based trend analysis. The roadmap identifies enhancements that would deepen forecasting, interpretability, deployment, and documentation maturity.

## 🧭 Purpose

This page separates implemented application behavior from planned project direction. This prevents the documentation from overstating the current source while preserving the broader vision described in the project README.

## ✅ Implemented

| Capability | Status |
|---|---|
| CSV/XLSX/XLS loading | Implemented |
| Fallback Excel loading | Implemented |
| Data preview and data-type review | Implemented |
| Descriptive statistics | Implemented |
| Missing-value and zero-count summary | Implemented |
| Histograms and Q-Q plots | Implemented |
| Numeric transformations | Implemented |
| PCA with KMeans clustering | Implemented |
| Correlation matrix and heatmap | Implemented |
| Multi-model regression comparison | Implemented |
| Regression diagnostics | Implemented |
| Inferential statistics | Implemented |
| Period-based aggregation and trends | Implemented |
| CSV export | Implemented |

## 🔮 Planned Enhancements

| Enhancement | Description |
|---|---|
| Full forecasting controls | Add user-facing ARIMA, Exponential Smoothing, or related forecasting controls. |
| Prophet integration | Add optional forecasting support if project dependencies and deployment environment allow it. |
| GridSearchCV tuning | Add controlled hyperparameter search for supported models. |
| SHAP interpretability | Explain model predictions and feature effects. |
| Model confidence intervals | Add uncertainty bands or prediction intervals where statistically appropriate. |
| Budget-rule diagnostics | Flag potential execution, availability, or balance-risk conditions. |
| API deployment | Add Flask or FastAPI endpoints for forecast services. |
| LLM summarization | Summarize analytical outputs in plain language after validation safeguards are defined. |
| Batch reporting | Export model results, charts, and diagnostic summaries as documentation-ready artifacts. |
| Module refactor | Split `app.py` into import-safe modules for cleaner API documentation. |

## 🧱 Documentation Roadmap

| Documentation Item | Purpose |
|---|---|
| Architecture diagram | Visualize the Streamlit/data/model/export workflow. |
| Workflow diagram | Show analyst steps from loading data through export. |
| Source-generated API pages | Render helper functions through mkdocstrings. |
| Data dictionary guidance | Define recommended columns for federal balance datasets. |
| Validation guide | Document analytical QA checks before model interpretation. |
| GitHub Pages guide | Publish built MkDocs site from the repository. |

## 🧪 Recommended Development Order

1. Stabilize and document current helper functions.
2. Generate MkDocs pages for the implemented Streamlit workflow.
3. Add CSS and JavaScript documentation enhancements.
4. Validate `mkdocs build`.
5. Add architecture and workflow diagrams.
6. Refactor helper logic into import-safe modules.
7. Expand API documentation after the refactor.
8. Add forecasting controls only after the current trend workflow is validated.

## ⚠️ Implementation Boundary

Planned features should not be documented as current features until the corresponding source code exists, runs, and is validated. This is especially important for forecasting, SHAP, LLM summarization, API deployment, and rule-based budget diagnostics.
