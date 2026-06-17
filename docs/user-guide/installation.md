# Installation

This page explains how to install and run Cutey locally.

## 🧭 Purpose

Cutey is a Streamlit application. It runs from the project root with Python dependencies installed in a virtual environment.

## 🧱 Create a Virtual Environment

From the project root, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 📦 Install Runtime Dependencies

```powershell
pip install streamlit pandas numpy matplotlib scikit-learn scipy statsmodels openpyxl
```

Minimum packages:

| Package | Purpose |
|---|---|
| `streamlit` | Web application interface. |
| `pandas` | Dataframe loading and manipulation. |
| `numpy` | Numeric arrays and calculations. |
| `matplotlib` | Charts and diagnostic plots. |
| `scipy` | Statistical tests. |
| `scikit-learn` | PCA, clustering, scaling, and regression models. |
| `statsmodels` | Optional time-series dependency. |
| `openpyxl` | Excel workbook support. |

## ▶️ Run the App

```powershell
streamlit run app.py
```

The app usually opens at:

```text
http://localhost:8501
```

## ⚙️ Required Configuration Values

The source references `config.py`. Confirm that it defines the values used by `app.py`.

```python
LOGO = "resources/cutey_logo.ico"

TABS = [
    "Data",
    "Descriptive Stats",
    "Distributions",
    "Transforms",
    "PCA + Clustering",
    "Correlations",
    "Regression",
    "Inferential Stats",
    "Time Series",
    "Export",
]

BLUE_DIVIDER = "<div style='height:2px;align:left;background:#0078FC;margin:6px 0 10px 0;'></div>"
```

## 📚 Optional Documentation Dependencies

To build the MkDocs site:

```powershell
pip install mkdocs mkdocs-material "mkdocstrings[python]" pymdown-extensions
```

Preview documentation:

```powershell
mkdocs serve
```

Build documentation:

```powershell
mkdocs build
```

## ✅ Installation Check

After installation, verify:

1. `streamlit run app.py` starts successfully.
2. The sidebar appears.
3. A CSV or Excel file can be uploaded.
4. The Data tab displays rows and columns.
5. The Export tab can download a CSV.
