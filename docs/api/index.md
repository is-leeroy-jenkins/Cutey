# API Reference

The Cutey API reference is generated from application source code using mkdocstrings.

## 🧭 Purpose

The API reference documents source-level helper functions used by the Streamlit workflow. The current project is centered on `app.py`, so the API section starts with a single application page.

## 🧱 Modules

| Module | Description |
|---|---|
| [`app`](app.md) | Streamlit application, data-loading helpers, numeric-column helpers, plotting helper, and tab workflow. |

## ⚠️ Import-Time Note

Streamlit applications often execute UI code at import time. If mkdocstrings has trouble importing `app.py`, move reusable helper functions into an import-safe module such as `cutey_core.py`, then point the API page to that module.

## ✅ Recommended Future API Split

| Future API Page | Source Module |
|---|---|
| Data | `data.py` |
| Statistics | `statistics.py` |
| Visualization | `visualization.py` |
| Models | `models.py` |
| Forecasting | `forecasting.py` |
| UI | `ui.py` |
