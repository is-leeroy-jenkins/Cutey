'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                name.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="guro.py" company="Terry D. Eppler">

	     name.py
	     Copyright ©  2022  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    name.py
  </summary>
  ******************************************************************************************
'''
"""
******************************************************************************************
  Assembly:                Cutey (Streamlit)
  Filename:                app.py
  Author:                  Terry D. Eppler (adapted by ChatGPT)
  Created:                 12-17-2025

  Last Modified By:        ChatGPT
  Last Modified On:        12-17-2025
******************************************************************************************
<copyright>
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the “Software”),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
</copyright>
******************************************************************************************
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

from scipy import stats

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

try:
    # Optional, but recommended for the time-series tab
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


# -------------------------------------------------------------------------------------------------
# Guard / helpers
# -------------------------------------------------------------------------------------------------

def throw_if(name: str, value: object) -> None:
    """
    Purpose:
    --------
    Simple guard that raises ValueError if `value` is falsy.

    Parameters:
    -----------
    name (str): Variable name used in error message.
    value (object): Value to validate.

    Returns:
    --------
    None
    """
    if value is None:
        raise ValueError(f'Argument "{name}" cannot be null.')
    if isinstance(value, str) and not value.strip():
        raise ValueError(f'Argument "{name}" cannot be empty.')


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Purpose:
    --------
    Convert selected columns to numeric safely.

    Parameters:
    -----------
    df (pd.DataFrame): Input dataframe.
    cols (List[str]): Column names to coerce.

    Returns:
    --------
    pd.DataFrame: Copy with coerced numeric columns.
    """
    df_out = df.copy()
    for c in cols:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce")
    return df_out


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Purpose:
    --------
    Identify numeric columns (including coercible columns with some numeric values).

    Parameters:
    -----------
    df (pd.DataFrame): Input dataframe.

    Returns:
    --------
    List[str]: Candidate numeric columns.
    """
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if numeric_cols:
        return numeric_cols

    # Fallback: attempt coercion-based detection
    candidates: List[str] = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.6:
            candidates.append(c)
    return candidates


def _safe_fig() -> plt.Figure:
    """
    Purpose:
    --------
    Create a single matplotlib figure with standard sizing.

    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(10, 5))
    return fig


@dataclass
class RegressionResult:
    model: str
    mse: float
    r2: float


# -------------------------------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_table_from_upload(
    uploaded: io.BytesIO,
    filename: str,
    sheet_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Purpose:
    --------
    Load an uploaded CSV or Excel file into a DataFrame.

    Parameters:
    -----------
    uploaded (io.BytesIO): Uploaded file bytes.
    filename (str): Original filename.
    sheet_name (Optional[str]): Excel sheet to load.

    Returns:
    --------
    pd.DataFrame
    """
    throw_if("uploaded", uploaded)
    throw_if("filename", filename)

    name = filename.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        if sheet_name:
            return pd.read_excel(uploaded, sheet_name=sheet_name)
        return pd.read_excel(uploaded)

    raise ValueError("Unsupported file type. Please upload .csv, .xlsx, or .xls.")


def _list_excel_sheets(uploaded: io.BytesIO) -> List[str]:
    """
    Purpose:
    --------
    Get Excel sheet names.

    Parameters:
    -----------
    uploaded (io.BytesIO): Uploaded file bytes.

    Returns:
    --------
    List[str]
    """
    try:
        xls = pd.ExcelFile(uploaded)
        return list(xls.sheet_names)
    except Exception:
        return []


# -------------------------------------------------------------------------------------------------
# Analytics / plots
# -------------------------------------------------------------------------------------------------

def plot_histograms(df: pd.DataFrame, cols: List[str], bins: int = 30) -> None:
    """
    Purpose:
    --------
    Plot histograms for selected numeric columns.

    Parameters:
    -----------
    df (pd.DataFrame): Input data.
    cols (List[str]): Columns to plot.
    bins (int): Histogram bins.

    Returns:
    --------
    None
    """
    throw_if("df", df)
    throw_if("cols", cols)

    for c in cols:
        fig = _safe_fig()
        ax = fig.add_subplot(111)
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        ax.hist(s.values, bins=bins)
        ax.set_title(f"Histogram: {c}")
        ax.set_xlabel(c)
        ax.set_ylabel("Count")
        st.pyplot(fig)


def plot_qq(df: pd.DataFrame, col: str) -> None:
    """
    Purpose:
    --------
    Plot a Q-Q plot for normality assessment.

    Parameters:
    -----------
    df (pd.DataFrame): Input data.
    col (str): Column to evaluate.

    Returns:
    --------
    None
    """
    throw_if("df", df)
    throw_if("col", col)

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        st.warning(f"No numeric data available for '{col}'.")
        return

    fig = _safe_fig()
    ax = fig.add_subplot(111)
    stats.probplot(s.values, dist="norm", plot=ax)
    ax.set_title(f"Q-Q Plot (Normality): {col}")
    st.pyplot(fig)


def compute_correlations(df: pd.DataFrame, cols: List[str], method: str) -> pd.DataFrame:
    """
    Purpose:
    --------
    Compute correlations for selected columns.

    Parameters:
    -----------
    df (pd.DataFrame): Input data.
    cols (List[str]): Numeric columns.
    method (str): 'pearson', 'spearman', or 'kendall'.

    Returns:
    --------
    pd.DataFrame
    """
    throw_if("df", df)
    throw_if("cols", cols)
    throw_if("method", method)

    df_num = _coerce_numeric(df[cols], cols)
    return df_num.corr(method=method)


def run_regressions(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float,
    random_state: int,
    use_poly: bool,
    poly_degree: int
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Purpose:
    --------
    Train and evaluate a set of regressors similar to the notebook section.

    Parameters:
    -----------
    df (pd.DataFrame): Data.
    feature_cols (List[str]): Predictors.
    target_col (str): Target variable.
    test_size (float): Test fraction.
    random_state (int): RNG seed.
    use_poly (bool): Include polynomial regression model.
    poly_degree (int): Polynomial degree for polynomial regression.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
        (results_df, predictions_df, fitted_models)
    """
    throw_if("df", df)
    throw_if("feature_cols", feature_cols)
    throw_if("target_col", target_col)

    df_work = df.copy()
    df_work = _coerce_numeric(df_work, feature_cols + [target_col])
    df_work = df_work.dropna(subset=feature_cols + [target_col])

    X = df_work[feature_cols]
    y = df_work[target_col]

    if len(df_work) < 20:
        raise ValueError("Not enough complete rows (>= 20) after numeric coercion and NA filtering.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models: Dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=random_state),
        "Lasso Regression": Lasso(alpha=10.0, max_iter=10000, random_state=random_state),
        "Bayesian Ridge Regression": BayesianRidge(),
    }

    if use_poly:
        models[f"Polynomial Regression (Degree={poly_degree})"] = make_pipeline(
            PolynomialFeatures(poly_degree),
            LinearRegression()
        )

    results: List[RegressionResult] = []
    preds_table: Dict[str, np.ndarray] = {}

    fitted_models: Dict[str, object] = {}

    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)

        mse = float(mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        results.append(RegressionResult(model=name, mse=mse, r2=r2))
        preds_table[name] = y_pred
        fitted_models[name] = mdl

    results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("r2", ascending=False)
    results_df = results_df.rename(columns={"model": "Model", "mse": "MSE", "r2": "R2 Score"}).round(4)

    pred_df = pd.DataFrame({"Actual": y_test.values})
    for k, v in preds_table.items():
        pred_df[k] = v
    pred_df = pred_df.reset_index(drop=True)

    return results_df, pred_df, fitted_models


def plot_actual_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    """
    Purpose:
    --------
    Scatter plot of actual vs predicted.

    Parameters:
    -----------
    y_true (np.ndarray): Actual values.
    y_pred (np.ndarray): Predicted values.
    title (str): Chart title.

    Returns:
    --------
    None
    """
    fig = _safe_fig()
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred)
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    # 45-degree reference line
    lo = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
    hi = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
    ax.plot([lo, hi], [lo, hi])
    st.pyplot(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    """
    Purpose:
    --------
    Residual plot: predicted vs residual.

    Parameters:
    -----------
    y_true (np.ndarray): Actual values.
    y_pred (np.ndarray): Predicted values.
    title (str): Chart title.

    Returns:
    --------
    None
    """
    residuals = y_true - y_pred

    fig = _safe_fig()
    ax = fig.add_subplot(111)
    ax.scatter(y_pred, residuals)
    ax.axhline(0.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    st.pyplot(fig)


def run_pca(df: pd.DataFrame, cols: List[str], n_components: int, scale: str) -> Tuple[pd.DataFrame, PCA]:
    """
    Purpose:
    --------
    Run PCA on selected numeric columns.

    Parameters:
    -----------
    df (pd.DataFrame): Input data.
    cols (List[str]): Numeric columns to include.
    n_components (int): PCA components.
    scale (str): 'none', 'standard', 'minmax'.

    Returns:
    --------
    Tuple[pd.DataFrame, PCA]: (components_df, pca_model)
    """
    throw_if("df", df)
    throw_if("cols", cols)

    df_num = _coerce_numeric(df[cols], cols).dropna()
    if df_num.empty or len(df_num) < 10:
        raise ValueError("Not enough numeric rows for PCA (need >= 10).")

    X = df_num.values

    if scale == "standard":
        X = StandardScaler().fit_transform(X)
    elif scale == "minmax":
        X = MinMaxScaler().fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    comps = pca.fit_transform(X)

    comp_cols = [f"PC{i+1}" for i in range(comps.shape[1])]
    df_comps = pd.DataFrame(comps, columns=comp_cols)

    return df_comps, pca


def run_kmeans(df_features: pd.DataFrame, k: int) -> np.ndarray:
    """
    Purpose:
    --------
    Run KMeans clustering.

    Parameters:
    -----------
    df_features (pd.DataFrame): Feature matrix.
    k (int): Number of clusters.

    Returns:
    --------
    np.ndarray: Cluster labels.
    """
    throw_if("df_features", df_features)
    if k < 2:
        raise ValueError("k must be >= 2")

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    return km.fit_predict(df_features.values)


def forecast_time_series(
    ts: pd.Series,
    steps: int,
    arima_order: Tuple[int, int, int],
    seasonal_period: Optional[int]
) -> Dict[str, pd.Series]:
    """
    Purpose:
    --------
    Forecast a univariate time series with ARIMA and Holt-Winters.

    Parameters:
    -----------
    ts (pd.Series): Time series indexed by year/period.
    steps (int): Forecast horizon.
    arima_order (Tuple[int,int,int]): (p,d,q).
    seasonal_period (Optional[int]): If provided, Holt-Winters seasonal period.

    Returns:
    --------
    Dict[str, pd.Series]: forecasts by model name.
    """
    if not _HAS_STATSMODELS:
        raise RuntimeError("statsmodels is not installed; cannot run time-series forecasting.")

    throw_if("ts", ts)

    ts_clean = ts.dropna()
    if len(ts_clean) < 8:
        raise ValueError("Need at least 8 time points for forecasting.")

    forecasts: Dict[str, pd.Series] = {}

    # ARIMA
    arima = ARIMA(ts_clean, order=arima_order)
    arima_fit = arima.fit()
    arima_fc = arima_fit.forecast(steps=steps)
    forecasts["ARIMA"] = arima_fc

    # Holt-Winters (Exponential Smoothing)
    if seasonal_period and seasonal_period >= 2:
        hw = ExponentialSmoothing(
            ts_clean,
            trend="add",
            seasonal="add",
            seasonal_periods=int(seasonal_period)
        ).fit()
    else:
        hw = ExponentialSmoothing(
            ts_clean,
            trend="add",
            seasonal=None
        ).fit()

    hw_fc = hw.forecast(steps=steps)
    forecasts["Holt-Winters"] = hw_fc

    return forecasts


# -------------------------------------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Cutey — Balance Projector (Streamlit)", layout="wide")

st.title("Cutey — Balance Projector (Streamlit)")
st.caption(
    "A Streamlit adaptation of the balances notebook: load SF-133/GTAS-like balance data, explore "
    "distributions, correlations, PCA/clustering, regression comparisons, and time-series forecasts."
)

with st.expander("What this app expects", expanded=False):
    st.markdown(
        """
- Upload a **CSV** or **Excel** file containing account/budget balance fields.
- The original notebook uses an Excel input and common fields such as obligations, outlays, unobligated balance,
  total resources, and a fiscal year-like period field (for time-series aggregation). :contentReference[oaicite:1]{index=1}
        """
    )

# Sidebar: upload
st.sidebar.header("1) Load Data")

uploaded = st.sidebar.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"])

df: Optional[pd.DataFrame] = None

if uploaded is not None:
    sheet_name: Optional[str] = None
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        sheets = _list_excel_sheets(uploaded)
        if sheets:
            sheet_name = st.sidebar.selectbox("Excel sheet", sheets, index=0)
        else:
            sheet_name = None

    try:
        df = load_table_from_upload(uploaded, uploaded.name, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

if df is None:
    st.info("Upload a dataset to begin.")
    st.stop()

st.sidebar.success(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

tabs = st.tabs(
    [
        "Data",
        "Descriptive Stats",
        "Distributions",
        "Transforms",
        "PCA + Clustering",
        "Correlations",
        "Regression",
        "Time Series",
        "Export",
    ]
)

# -----------------------
# Data tab
# -----------------------
with tabs[0]:
    st.subheader("Data Preview")
    st.dataframe(df, use_container_width=True, height=420)

    st.subheader("Column Types")
    df_types = pd.DataFrame(
        {"Column": df.columns, "DType": [str(t) for t in df.dtypes]}
    )
    st.dataframe(df_types, use_container_width=True, height=320)

# -----------------------
# Descriptive stats
# -----------------------
with tabs[1]:
    st.subheader("Descriptive Statistics")

    numeric_cols = _numeric_columns(df)
    if not numeric_cols:
        st.warning("No numeric columns detected.")
    else:
        selected = st.multiselect(
            "Numeric columns to summarize",
            options=numeric_cols,
            default=numeric_cols[: min(10, len(numeric_cols))],
        )

        if selected:
            df_num = _coerce_numeric(df[selected], selected)
            st.dataframe(df_num.describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).round(3),
                         use_container_width=True)

# -----------------------
# Distributions
# -----------------------
with tabs[2]:
    st.subheader("Distributions / Normality Checks")

    numeric_cols = _numeric_columns(df)
    if not numeric_cols:
        st.warning("No numeric columns detected.")
    else:
        cols = st.multiselect(
            "Columns",
            options=numeric_cols,
            default=numeric_cols[: min(6, len(numeric_cols))],
        )
        bins = st.slider("Histogram bins", min_value=10, max_value=100, value=30, step=5)

        if cols:
            plot_histograms(df, cols, bins=bins)

        st.markdown("---")
        qq_col = st.selectbox("Q-Q plot column", options=numeric_cols)
        if qq_col:
            plot_qq(df, qq_col)

# -----------------------
# Transforms
# -----------------------
with tabs[3]:
    st.subheader("Transformations (log1p / scaling)")

    numeric_cols = _numeric_columns(df)
    if not numeric_cols:
        st.warning("No numeric columns detected.")
    else:
        cols = st.multiselect(
            "Columns to transform",
            options=numeric_cols,
            default=numeric_cols[: min(6, len(numeric_cols))],
        )

        transform = st.selectbox("Transform", options=["None", "log1p", "StandardScaler", "MinMaxScaler"])

        if cols:
            df_work = _coerce_numeric(df[cols], cols)

            if transform == "log1p":
                df_out = np.log1p(df_work)
            elif transform == "StandardScaler":
                df_out = pd.DataFrame(StandardScaler().fit_transform(df_work.dropna()), columns=cols)
            elif transform == "MinMaxScaler":
                df_out = pd.DataFrame(MinMaxScaler().fit_transform(df_work.dropna()), columns=cols)
            else:
                df_out = df_work

            st.dataframe(df_out.head(50), use_container_width=True)

# -----------------------
# PCA + Clustering
# -----------------------
with tabs[4]:
    st.subheader("PCA + KMeans Clustering")

    numeric_cols = _numeric_columns(df)
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for PCA/clustering.")
    else:
        cols = st.multiselect(
            "Columns for PCA",
            options=numeric_cols,
            default=numeric_cols[: min(8, len(numeric_cols))],
        )
        scale = st.selectbox("Scaling", options=["none", "standard", "minmax"], index=1)
        n_components = st.slider("PCA components", min_value=2, max_value=min(10, len(cols) if cols else 2), value=2)
        k = st.slider("KMeans clusters (k)", min_value=2, max_value=10, value=3)

        if cols and len(cols) >= 2:
            try:
                df_comps, pca = run_pca(df, cols, n_components=n_components, scale=scale)

                labels = run_kmeans(df_comps, k=k)
                df_plot = df_comps.copy()
                df_plot["Cluster"] = labels

                st.write("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))

                if "PC1" in df_plot.columns and "PC2" in df_plot.columns:
                    fig = _safe_fig()
                    ax = fig.add_subplot(111)
                    ax.scatter(df_plot["PC1"], df_plot["PC2"])
                    ax.set_title("PCA Scatter (PC1 vs PC2)")
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    st.pyplot(fig)

                st.dataframe(df_plot.head(200), use_container_width=True)
            except Exception as e:
                st.error(f"PCA/Clustering failed: {e}")

# -----------------------
# Correlations
# -----------------------
with tabs[5]:
    st.subheader("Correlation Matrices")

    numeric_cols = _numeric_columns(df)
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlations.")
    else:
        cols = st.multiselect(
            "Columns",
            options=numeric_cols,
            default=numeric_cols[: min(12, len(numeric_cols))],
        )
        method = st.selectbox("Method", options=["pearson", "spearman", "kendall"], index=0)

        if cols and len(cols) >= 2:
            corr = compute_correlations(df, cols, method=method)
            st.dataframe(corr.round(4), use_container_width=True)

# -----------------------
# Regression
# -----------------------
with tabs[6]:
    st.subheader("Regression Model Comparison")

    numeric_cols = _numeric_columns(df)
    if len(numeric_cols) < 2:
        st.warning("Need numeric columns for regression.")
    else:
        feature_cols = st.multiselect(
            "Predictors (X)",
            options=numeric_cols,
            default=[c for c in ["Obligations", "UnobligatedBalance", "Outlays"] if c in df.columns]
                    or numeric_cols[: min(3, len(numeric_cols))],
        )
        target_col = st.selectbox(
            "Target (y)",
            options=numeric_cols,
            index=(numeric_cols.index("TotalResources") if "TotalResources" in numeric_cols else 0),
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        with c2:
            random_state = st.number_input("Random state", min_value=0, max_value=10_000, value=42, step=1)
        with c3:
            use_poly = st.checkbox("Include polynomial regression", value=True)
        with c4:
            poly_degree = st.slider("Polynomial degree", 2, 5, 2)

        if feature_cols and target_col:
            try:
                results_df, pred_df, fitted = run_regressions(
                    df=df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    use_poly=bool(use_poly),
                    poly_degree=int(poly_degree),
                )

                st.markdown("#### Model Metrics (sorted by R²)")
                st.dataframe(results_df, use_container_width=True)

                st.markdown("#### Predictions Preview")
                st.dataframe(pred_df.head(200), use_container_width=True)

                best_model = results_df.iloc[0]["Model"]
                st.markdown(f"#### Diagnostics for Best Model: `{best_model}`")

                y_true = pred_df["Actual"].values
                y_pred = pred_df[best_model].values

                plot_actual_vs_pred(y_true, y_pred, title=f"Actual vs Predicted — {best_model}")
                plot_residuals(y_true, y_pred, title=f"Residuals — {best_model}")

            except Exception as e:
                st.error(f"Regression failed: {e}")

# -----------------------
# Time series
# -----------------------
with tabs[7]:
    st.subheader("Time Series (Aggregate by Period → Forecast)")

    if not _HAS_STATSMODELS:
        st.warning(
            "statsmodels is not available in this environment. Install it to enable ARIMA/Holt-Winters."
        )
    else:
        # Period column selection (the notebook references BeginningPeriodOfAvailability)
        period_candidates = [c for c in df.columns if "year" in c.lower() or "period" in c.lower() or "availability" in c.lower()]
        period_col = st.selectbox(
            "Period column (year/period)",
            options=period_candidates if period_candidates else list(df.columns),
            index=(period_candidates.index("BeginningPeriodOfAvailability")
                   if "BeginningPeriodOfAvailability" in period_candidates else 0),
        )

        numeric_cols = _numeric_columns(df)
        value_candidates = [c for c in numeric_cols if c not in [period_col]]
        value_cols = st.multiselect(
            "Values to aggregate (sum) by period",
            options=value_candidates,
            default=[c for c in ["AnnualAppropriations", "CarryoverAuthority", "UnobligatedBalance", "TotalResources"]
                     if c in value_candidates] or value_candidates[: min(3, len(value_candidates))],
        )

        target_ts_col = st.selectbox(
            "Forecast which aggregated series?",
            options=value_cols if value_cols else value_candidates,
            index=0 if value_cols else 0,
        )

        steps = st.slider("Forecast steps", min_value=1, max_value=10, value=2)
        p = st.slider("ARIMA p", 0, 5, 1)
        d = st.slider("ARIMA d", 0, 2, 1)
        q = st.slider("ARIMA q", 0, 5, 1)
        seasonal_period = st.number_input("Holt-Winters seasonal period (optional)", min_value=0, max_value=24, value=0, step=1)

        if period_col and value_cols and target_ts_col:
            try:
                df_ts = df[[period_col] + value_cols].copy()
                df_ts[period_col] = pd.to_numeric(df_ts[period_col], errors="coerce")
                df_ts = _coerce_numeric(df_ts, value_cols)
                df_ts = df_ts.dropna(subset=[period_col])

                agg = df_ts.groupby(period_col)[value_cols].sum().sort_index()
                st.markdown("#### Aggregated Time Series")
                st.dataframe(agg, use_container_width=True)

                # Plot target series
                fig = _safe_fig()
                ax = fig.add_subplot(111)
                ax.plot(agg.index.values, agg[target_ts_col].values, marker="o")
                ax.set_title(f"Trend: {target_ts_col} by {period_col}")
                ax.set_xlabel(period_col)
                ax.set_ylabel(target_ts_col)
                st.pyplot(fig)

                # Forecast
                ts = agg[target_ts_col]
                forecasts = forecast_time_series(
                    ts=ts,
                    steps=int(steps),
                    arima_order=(int(p), int(d), int(q)),
                    seasonal_period=(int(seasonal_period) if int(seasonal_period) > 0 else None),
                )

                st.markdown("#### Forecasts")
                df_fc = pd.DataFrame({k: v.values for k, v in forecasts.items()})
                st.dataframe(df_fc.round(4), use_container_width=True)

                fig2 = _safe_fig()
                ax2 = fig2.add_subplot(111)
                ax2.plot(ts.index.values, ts.values, marker="o", label="Actual")
                # Extend x-axis labels for forecast
                last_x = float(ts.index.values[-1])
                x_fc = np.arange(last_x + 1, last_x + 1 + int(steps))
                for name, series_fc in forecasts.items():
                    ax2.plot(x_fc, series_fc.values, marker="o", label=name)

                ax2.set_title(f"Forecast: {target_ts_col}")
                ax2.set_xlabel(period_col)
                ax2.set_ylabel(target_ts_col)
                ax2.legend()
                st.pyplot(fig2)

            except Exception as e:
                st.error(f"Time series analysis failed: {e}")

# -----------------------
# Export
# -----------------------
with tabs[8]:
    st.subheader("Export")

    st.write("Download the currently loaded dataset (as CSV) after any upstream cleaning you performed externally.")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="cutey_export.csv",
        mime="text/csv",
    )

