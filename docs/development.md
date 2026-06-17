# Development

This page documents the local development workflow for Cutey and the MkDocs documentation site.

## 🧭 Purpose

The development process should preserve the Streamlit application behavior while improving documentation quality, source readability, validation discipline, and MkDocs build reliability.

## 🧱 Project Layout

Recommended layout:

```text
Cutey/
├── app.py
├── config.py
├── requirements.txt
├── mkdocs.yml
├── data/
│   └── Account Balances.xlsx
├── resources/
│   ├── cutey_logo.ico
│   └── favicon.ico
└── docs/
    ├── index.md
    ├── architecture.md
    ├── data-sources.md
    ├── development.md
    ├── roadmap.md
    ├── api/
    │   ├── index.md
    │   └── app.md
    ├── user-guide/
    │   └── *.md
    └── assets/
        ├── css/
        │   └── cutey.css
        └── js/
            └── cutey.js
```

## 🧪 Local Environment

Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install runtime dependencies.

```powershell
pip install streamlit pandas numpy matplotlib scikit-learn scipy statsmodels openpyxl
```

Install documentation dependencies.

```powershell
pip install mkdocs mkdocs-material "mkdocstrings[python]" pymdown-extensions
```

## ▶️ Run the Application

From the project root:

```powershell
streamlit run app.py
```

The default local Streamlit URL is usually:

```text
http://localhost:8501
```

## 📚 Run the Documentation Site

Preview locally:

```powershell
mkdocs serve
```

Build static documentation:

```powershell
mkdocs build
```

## ✅ Validation Commands

Run these checks before committing source or documentation changes.

```powershell
python -m py_compile .pp.py
python -m compileall .
mkdocs build
```

If the API page fails, test direct import behavior.

```powershell
python -c "import app; print('app import ok')"
```

Because Streamlit applications often execute UI code at import time, API documentation should be tested carefully. If import-time execution causes problems, move helper functions into a separate module such as `cutey_core.py` and document that module instead.

## 🧾 Docstring Standard

Use griffe-compatible Google-style docstrings.

Required sections where applicable:

```text
Purpose:
Args:
Returns:
Raises:
Notes:
Examples:
```

Example:

```python
def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce selected dataframe columns to numeric values.

    Purpose:
        Creates a copy of the dataframe and converts selected columns to numeric
        values so downstream statistical, transformation, and modeling operations
        receive numeric inputs.

    Args:
        df: Dataframe containing source columns.
        cols: Column names selected for numeric coercion.

    Returns:
        pd.DataFrame: Dataframe copy with selected columns converted to numeric values.
    """
```

## 🚫 Source Preservation Rules

When updating documentation comments, do not change:

- imports
- constants
- function names
- function signatures
- Streamlit layout
- tab order
- estimator configuration
- model parameters
- plotting behavior
- fallback file behavior
- output file names

Documentation improvements should not alter analytical results.

## 🧩 Suggested Refactor Path

The current application is compact and works as a single Streamlit file. If the project grows, split the code gradually.

| Future Module | Purpose |
|---|---|
| `data.py` | File loading, sheet detection, dataframe validation. |
| `statistics.py` | Descriptive and inferential statistics helpers. |
| `visualization.py` | Matplotlib chart builders. |
| `models.py` | Regression and clustering model helpers. |
| `forecasting.py` | Time-series forecasting and trend methods. |
| `ui.py` | Streamlit layout, tabs, and controls. |

This refactor would improve mkdocstrings output because API pages could document import-safe modules without executing Streamlit UI code.

## 🔍 Common MkDocs Issues

| Symptom | Likely Cause | Fix |
|---|---|---|
| Page missing from nav | File exists but is not listed | Add it to `mkdocs.yml` nav. |
| Nav points to missing page | Nav entry has no matching file | Create the file or remove the nav entry. |
| Griffe warning on args | Malformed `Args:` section | Use `name: description` format. |
| Return warning | Missing type or explicit return type | Add return annotation or explicit `Returns:` type. |
| Import failure | mkdocstrings cannot import `app.py` | Install dependencies or document helper module instead. |

## 🔗 Related Pages

- [Architecture](architecture.md)
- [Application API](api/app.md)
- [User Guide](user-guide/index.md)
