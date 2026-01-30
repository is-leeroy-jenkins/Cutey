from __future__ import annotations

# -------------------------------------------------------------------------------------------------
# Standard library
# -------------------------------------------------------------------------------------------------
import io
from pathlib import Path
from typing import Optional, List, Dict

# -------------------------------------------------------------------------------------------------
# Third-party
# -------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    SGDRegressor,
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# -------------------------------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------------------------------
LOGO = r'resources\cutey_logo.png'

BLUE_DIVIDER = "<div style='height:2px;align:left;background:#0078FC;margin:6px 0 10px 0;'></div>"

# -------------------------------------------------------------------------------------------------
# Streamlit config
# -------------------------------------------------------------------------------------------------
st.logo( LOGO, size='large' )
st.set_page_config(page_title="Cutey-Py", layout="wide", page_icon=r'resources\favicon.ico')
st.subheader( "Balance Projection" )


# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------
FALLBACK_DATA_PATH = Path("data") / "Account Balances.xlsx"

# A consistent palette/marker/linestyle cycle (Pogi-like clarity)
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]


# -------------------------------------------------------------------------------------------------
# Helpers (robust to duplicate headers)
# -------------------------------------------------------------------------------------------------
def numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric columns; if none are typed numeric, infer coercible columns."""
    cols = list(df.select_dtypes(include=[np.number]).columns)
    if cols:
        return cols
    inferred: List[str] = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.6:
            inferred.append(c)
    return inferred


def new_fig(width: float = 10.0, height: float = 5.0) -> plt.Figure:
    """Create a standard matplotlib figure."""
    return plt.figure(figsize=(width, height))


def series_from_column(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Return a 1-D Series from a column name, even if duplicate headers exist.
    If df[col_name] is a DataFrame (duplicate headers), take the first.
    """
    col = df[col_name]
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0]
    return col


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Safely coerce selected columns to numeric; supports duplicate headers (DataFrame -> Series).
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = series_from_column(out, c)
        out[c] = pd.to_numeric(s, errors="coerce")
    return out


# -------------------------------------------------------------------------------------------------
# Excel helpers
# -------------------------------------------------------------------------------------------------
def list_excel_sheets_from_upload(uploaded: io.BytesIO) -> List[str]:
    try:
        return list(pd.ExcelFile(uploaded).sheet_names)
    except Exception:
        return []


def list_excel_sheets_from_path(path: Path) -> List[str]:
    try:
        return list(pd.ExcelFile(path).sheet_names)
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def load_table(
    uploaded_file: io.BytesIO | None,
    uploaded_name: str | None,
    sheet_name: str | None
) -> pd.DataFrame:
    """Load data from either an uploaded file or the fallback Excel file."""
    if uploaded_file is not None and uploaded_name:
        name = uploaded_name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file, sheet_name=sheet_name)
        raise ValueError("Unsupported uploaded file type. Use .csv, .xlsx, or .xls.")
    if not FALLBACK_DATA_PATH.exists():
        raise FileNotFoundError(f"Fallback file not found: {FALLBACK_DATA_PATH.resolve()}")
    return pd.read_excel(FALLBACK_DATA_PATH, sheet_name=sheet_name)


# -------------------------------------------------------------------------------------------------
# Sidebar – data loading
# -------------------------------------------------------------------------------------------------
st.sidebar.header("Account-A Data")

uploaded = st.sidebar.file_uploader(
    "Upload CSV/XLSX (optional — fallback will be used if omitted)",
    type=["csv", "xlsx", "xls"]
)

sheet_name: Optional[str] = None

if uploaded is not None and uploaded.name.lower().endswith((".xlsx", ".xls")):
    sheets = list_excel_sheets_from_upload(uploaded)
elif uploaded is None and FALLBACK_DATA_PATH.exists():
    sheets = list_excel_sheets_from_path(FALLBACK_DATA_PATH)
else:
    sheets = []

if sheets:
    sheet_name = st.sidebar.selectbox("Excel Sheet", sheets, index=0)

try:
    df = load_table(
        uploaded_file=uploaded,
        uploaded_name=uploaded.name if uploaded else None,
        sheet_name=sheet_name
    )
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

st.sidebar.success(
    f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns"
    + (" (fallback)" if uploaded is None else "")
)

dup_count = int(pd.Index(df.columns).duplicated().sum())
if dup_count > 0:
    st.sidebar.warning(f"Detected {dup_count} duplicate column header(s) in this dataset.")


# -------------------------------------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------------------------------------
tabs = st.tabs([
    "Data",
    "Descriptive Stats",
    "Inferential Stats",
    "Distributions",
    "Transforms",
    "PCA + Clustering",
    "Correlations",
    "Regression",
    "Time Series",
    "Export",
])


# -------------------------------------------------------------------------------------------------
# TAB 0 — Data
# -------------------------------------------------------------------------------------------------
with tabs[0]:
    st.markdown( "##### Data Preview" )
    st.data_editor( df, width='content', height='auto', num_rows='dynamic' )
    st.markdown( BLUE_DIVIDER, unsafe_allow_html=True )
    
    st.markdown( "##### Column Types" )
    st.data_editor(
        pd.DataFrame( {"Column": df.columns, "DType": df.dtypes.astype(str)} ),
	    use_container_width=True, height='auto', num_rows='dynamic' )

if dup_count > 0:
		st.info(
			"Duplicate headers detected. When a single column is requested by name, "
			"the first matching column is used to ensure 1-D inputs for computations." )


# -------------------------------------------------------------------------------------------------
# TAB 1 — Descriptive Stats
# -------------------------------------------------------------------------------------------------
with tabs[1]:
    cols = numeric_columns(df)
    if not cols:
        st.warning("No numeric columns detected.")
    else:
        selected = st.multiselect("Columns", cols, default=cols[: min(8, len(cols))])
        if selected:
            num = coerce_numeric(df[selected], selected)

            c1, c2 = st.columns( 2, border=True)
            with c1:
                st.markdown("**Core Summary**")
                st.data_editor(
                    num.describe( percentiles=[ 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95 ] ).round( 3 ),
                    use_container_width=True, height='auto', num_rows='dynyamic' )
            with c2:
                st.markdown("**Extended Summary**")
                extended = pd.DataFrame({
                    "missing_count": num.isna().sum(),
                    "missing_pct": (num.isna().mean() * 100.0).round(2),
                    "zero_count": (num == 0).sum(),
                    "unique_count": num.nunique(),
                    "skewness": num.skew(numeric_only=True),
                    "kurtosis": num.kurtosis(numeric_only=True),
                }).round(3)
                st.data_editor(extended, use_container_width=True, height='auto', num_rows='dynamic' )
	            
	        
            st.markdown( BLUE_DIVIDER, unsafe_allow_html=True )
	            
            st.markdown("**Grouped Bar (first 5 rows)**")
            head = num.head(5).reset_index(drop=True)
            fig = new_fig(12, 5)
            ax = fig.add_subplot(111)
            x = np.arange(head.shape[0])
            width = max(0.8 / max(1, head.shape[1]), 0.12)
            for i, col in enumerate(head.columns):
                offset = (i - (head.shape[1] - 1) / 2) * width
                ax.bar(
                    x + offset,
                    head[col].values,
                    width=width,
                    label=col,
                    color=PALETTE[i % len(PALETTE)],
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.85,
                    hatch="/\\"[i % 2],
                )
            ax.set_xticks(x)
            ax.set_xticklabels([ f"r{i}" for i in range( 1, head.shape[ 0 ] + 1 ) ] )
            ax.set_title( "Grouped Bars" )
            ax.legend( ncol=3, fontsize=8 )
            st.pyplot( fig )

# -------------------------------------------------------------------------------------------------
# TAB 2 — Inferential Stats
# -------------------------------------------------------------------------------------------------
with tabs[7]:
    st.markdown( "##### Inferential Statistics")
    num_cols = numeric_columns( df )
    if not num_cols:
        st.warning("No numeric columns detected for inferential tests.")
    else:
        y_col = st.selectbox( "Numeric variable", num_cols, key="infer_y" )
        group_col_opts = [c for c in df.columns if c not in num_cols]
        group_col = st.selectbox( "Grouping column (optional)", ["<None>"] + group_col_opts, index=0)

        y = pd.to_numeric( series_from_column( df, y_col ), errors="coerce" )
        y = y.dropna( )
        c1, c2, c3 = st.columns( 3 )
        with c1:
            k2, p_sw = stats.shapiro(y.sample(min(len(y), 500), random_state=42)) if len(y) > 3 else (np.nan, np.nan)
            st.metric( "Shapiro–Wilk W (Normality)", f"{k2:.4f}" if not np.isnan(k2) else "n/a" )
            st.caption( "H0: data are normally distributed" )
        with c2:
            ys = (y - y.mean()) / (y.std(ddof=1) or 1.0)
            ks, p_ks = stats.kstest(ys, "norm")
            st.metric("KS statistic vs N(0,1)", f"{ks:.4f}")
            st.caption("H0: distribution equals standard normal")
        with c3:
            mu = y.mean()
            se = y.std(ddof=1) / np.sqrt(len(y))
            tcrit = stats.t.ppf(0.975, df=max(len(y) - 1, 1))
            lo = mu - tcrit * se
            hi = mu + tcrit * se
            st.metric("95% CI for mean", f"[{lo:.3f}, {hi:.3f}]")
        if group_col != "<None>":
            gser = series_from_column(df, group_col)
            tmp = pd.DataFrame({"y": y, "g": gser}).dropna()
            groups = [grp["y"].values for _, grp in tmp.groupby("g")]

            if len(groups) == 2:
                tstat, p_t = stats.ttest_ind(groups[0], groups[1], equal_var=False, nan_policy="omit")
                ustat, p_u = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                lstat, p_l = stats.levene(groups[0], groups[1], center="median")

                st.write("**Two-group tests**")
                st.table(pd.DataFrame({
                    "Test": ["Welch t-test", "Mann-Whitney U", "Levene (equal variances)"],
                    "Statistic": [tstat, ustat, lstat],
                    "p-value": [p_t, p_u, p_l],
                }).round(5))
            elif len(groups) >= 3:
                fstat, p_a = stats.f_oneway(*groups)
                st.write("**One-way ANOVA**")
                st.table(pd.DataFrame({"F": [fstat], "p-value": [p_a]}).round(5))
            else:
                st.info("Grouping column does not contain multiple levels after cleaning.")
			    
# -------------------------------------------------------------------------------------------------
# TAB 3 — Distributions 
# -------------------------------------------------------------------------------------------------
with tabs[2]:
    cols = numeric_columns(df)
    if not cols:
        st.warning("No numeric columns detected.")
    else:
        chosen = st.multiselect("Histogram Columns", cols, default=cols[: min(4, len(cols))])
        bins = st.slider("Bins", 10, 100, 36)
        if chosen:
            fig = new_fig(12, 5)
            ax = fig.add_subplot(111)
            for i, c in enumerate(chosen):
                s = pd.to_numeric(series_from_column(df, c), errors="coerce").dropna()
                ax.hist(
                    s,
                    bins=bins,
                    alpha=0.40,
                    label=c,
                    color=PALETTE[i % len(PALETTE)],
                    edgecolor="black",
                    linewidth=0.5,
                )
            ax.set_title("Histograms (alpha + edges)")
            ax.legend(ncol=3, fontsize=8)
            st.pyplot(fig)

        st.markdown( BLUE_DIVIDER, unsafe_allow_html=True )
        qq_col = st.selectbox("Q-Q Plot Column", cols)
        sqq = pd.to_numeric(series_from_column(df, qq_col), errors="coerce").dropna()
        if len(sqq) < 5:
            st.warning("Not enough values for a Q-Q plot.")
        else:
            fig = new_fig()
            ax = fig.add_subplot(111)
            (osm, osr), (slope, intercept, r) = stats.probplot(sqq, dist="norm")
            ax.scatter(osm, osr, marker="o", s=18, edgecolor="black", linewidths=0.4, alpha=0.8)
            ax.plot(osm, slope * np.asarray(osm) + intercept, linestyle="--", color="#444444")
            ax.set_title(f"Q-Q Plot – {qq_col} (r={r:.3f})")
            st.pyplot(fig)

# -------------------------------------------------------------------------------------------------
# TAB 3 — Transforms
# -------------------------------------------------------------------------------------------------
with tabs[3]:
    cols = numeric_columns(df)
    if not cols:
        st.warning("No numeric columns detected.")
    else:
        selected = st.multiselect("Columns", cols, default=cols[: min(4, len(cols))])
        transform = st.selectbox("Transform", ["None", "log1p", "StandardScaler", "MinMaxScaler"])

        if selected:
            data = coerce_numeric(df[selected], selected).dropna()
            if data.empty:
                st.warning("No complete numeric rows after coercion and NA filtering.")
            else:
                if transform == "log1p":
                    data_out = np.log1p(data)
                elif transform == "StandardScaler":
                    data_out = pd.DataFrame(StandardScaler().fit_transform(data), columns=selected)
                elif transform == "MinMaxScaler":
                    data_out = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=selected)
                else:
                    data_out = data
                st.data_editor(data_out.head(100), use_container_width=True, height='auto', num_rows='dynamic' )

# -------------------------------------------------------------------------------------------------
# TAB 4 — PCA + Clustering
# -------------------------------------------------------------------------------------------------
with tabs[4]:
    cols = numeric_columns(df)
    if len(cols) < 2:
        st.warning("Need at least two numeric columns.")
    else:
        selected = st.multiselect("PCA Columns", cols, default=cols[: min(6, len(cols))])
        if len(selected) >= 2:
            scale = st.selectbox("Scaling", ["standard", "minmax", "none"], index=0)
            k = st.slider("Clusters (k)", 2, 10, 3)

            Xdf = coerce_numeric(df[selected], selected).dropna()
            if Xdf.shape[0] < 10:
                st.warning("Not enough numeric rows for PCA/clustering.")
            else:
                X = Xdf.values
                if scale == "standard":
                    X = StandardScaler().fit_transform(X)
                elif scale == "minmax":
                    X = MinMaxScaler().fit_transform(X)

                pca = PCA(n_components=2, random_state=42)
                comps = pca.fit_transform(X)
                labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(comps)

                fig = new_fig()
                ax = fig.add_subplot(111)
                for lab in np.unique(labels):
                    idx = labels == lab
                    color = PALETTE[int(lab) % len(PALETTE)]
                    marker = MARKERS[int(lab) % len(MARKERS)]
                    ax.scatter(
                        comps[idx, 0],
                        comps[idx, 1],
                        label=f"Cluster {lab}",
                        marker=marker,
                        edgecolor="black",
                        linewidth=0.4,
                        alpha=0.90,
                        c=color,
                    )
                ax.set_title("PCA (PC1 vs PC2) with KMeans")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.legend(ncol=3, fontsize=8)
                st.pyplot(fig)
                st.write("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
			    
			    
        st.markdown( BLUE_DIVIDER, unsafe_allow_html=True )

# -------------------------------------------------------------------------------------------------
# TAB 5 — Correlations
# -------------------------------------------------------------------------------------------------
with tabs[5]:
    cols = numeric_columns(df)
    if len(cols) < 2:
        st.warning("Need at least two numeric columns.")
    else:
        selected = st.multiselect("Correlation Columns", cols, default=cols[: min(8, len(cols))])
        if len(selected) >= 2:
            method = st.selectbox("Method", ["pearson", "spearman", "kendall"], index=0)
            cdf = coerce_numeric(df[selected], selected)
            corr = cdf.corr(method=method)
            st.data_editor(corr.round(4), use_container_width=True, height='auto', num_rows='dynamic' )

            st.markdown( BLUE_DIVIDER, unsafe_allow_html=True )
        
            # Heatmap for better visual delineation
            fig = new_fig(8, 6)
            ax = fig.add_subplot(111)
            im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(np.arange(len(selected)))
            ax.set_yticks(np.arange(len(selected)))
            ax.set_xticklabels(selected, rotation=45, ha="right")
            ax.set_yticklabels(selected)
            for i in range(len(selected)):
                for j in range(len(selected)):
                    ax.text(j, i, f"{corr.values[i, j]:.2f}",
                            ha="center", va="center", fontsize=8, color="black")
            ax.set_title("Correlation Heatmap")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)

# -------------------------------------------------------------------------------------------------
# TAB 6 — Regression
# -------------------------------------------------------------------------------------------------
with tabs[6]:
    cols = numeric_columns(df)
    if len(cols) < 2:
        st.warning("Need numeric columns for regression.")
    else:
        X_cols = st.multiselect("Predictors (X)", cols, default=cols[: min(4, len(cols))])
        y_col = st.selectbox("Target (y)", cols)

        if X_cols and y_col:
            # Build X safely (duplicate headers supported)
            X_parts: Dict[str, pd.Series] = {
                c: pd.to_numeric(series_from_column(df, c), errors="coerce") for c in X_cols
            }
            X = pd.DataFrame(X_parts)

            # Build y as guaranteed 1-D
            y = pd.to_numeric(series_from_column(df, y_col), errors="coerce")

            # Align and clean
            train = X.copy()
            train["__target__"] = y
            train = train.dropna()

            if train.shape[0] < 20:
                st.warning("Not enough complete numeric rows for regression (need at least 20).")
            else:
                X_clean = train[X_cols]
                y_clean = train["__target__"]

                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y_clean, test_size=0.2, random_state=42
                )

                models = {
                    "Linear": LinearRegression(),
                    "Ridge": Ridge(alpha=10.0),
                    "Lasso": Lasso(alpha=10.0, max_iter=10000),
                    "BayesianRidge": BayesianRidge(),
                    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
                    "HuberRegressor": HuberRegressor(),
                    "SGDRegressor": SGDRegressor(max_iter=2000, tol=1e-3),
                    "RandomForestRegressor": RandomForestRegressor(
                        n_estimators=200, random_state=42
                    ),
                    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
                }

                results = []
                for name, model in models.items():
                    model.fit(X_train, y_train)     # y is 1-D by construction
                    pred = model.predict(X_test)
                    results.append({
                        "Model": name,
                        "MSE": float(mean_squared_error(y_test, pred)),
                        "R²": float(r2_score(y_test, pred)),
                    })

                res_df = pd.DataFrame(results).sort_values("R²", ascending=False).round(4)
                st.data_editor(res_df, use_container_width=True, height='auto', num_rows='dynamic' )

                # Diagnostics plot for top 1 model
                top = res_df.iloc[0]["Model"]
                st.markdown(f"**Diagnostics: {top}**")
                top_model = models[top]
                y_pred = top_model.predict(X_test)

                # Actual vs Predicted
                fig = new_fig()
                ax = fig.add_subplot(111)
                ax.scatter(y_test, y_pred, c="#1f77b4", edgecolor="black", linewidth=0.5, alpha=0.85)
                lo = float(np.nanmin([np.nanmin(y_test), np.nanmin(y_pred)]))
                hi = float(np.nanmax([np.nanmax(y_test), np.nanmax(y_pred)]))
                ax.plot([lo, hi], [lo, hi], linestyle="--", color="#444444")
                ax.set_title("Actual vs Predicted")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)
		        
                st.markdown( BLUE_DIVIDER, unsafe_allow_html=True )

                # Residuals
                residuals = y_test - y_pred
                fig = new_fig()
                ax = fig.add_subplot(111)
                ax.scatter(y_pred, residuals, c="#d62728", edgecolor="black", linewidth=0.5, alpha=0.85)
                ax.axhline(0.0, linestyle="--", color="#444444")
                ax.set_title("Residuals vs Predicted")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Residual (Actual - Predicted)")
                st.pyplot(fig)

# -------------------------------------------------------------------------------------------------
# TAB 8 — Time Series
# -------------------------------------------------------------------------------------------------
with tabs[8]:
    if not HAS_STATSMODELS:
        st.warning("statsmodels is not installed. Install it to enable time-series forecasting.")
    else:
        period_candidates = [
            c for c in df.columns
            if "year" in str(c).lower() or "period" in str(c).lower() or "availability" in str(c).lower()
        ]
        if not period_candidates:
            st.warning("No period/year-like column detected.")
        else:
            period_col = st.selectbox("Period Column", period_candidates)
            value_cols = st.multiselect("Value Columns (sum by period)", numeric_columns(df))
            if value_cols:
                tmp = df[[period_col] + value_cols].copy()
                tmp[period_col] = pd.to_numeric(series_from_column(tmp, period_col), errors="coerce")
                tmp = coerce_numeric(tmp, value_cols).dropna(subset=[period_col])
                agg = tmp.groupby(period_col)[value_cols].sum().sort_index()
                st.data_editor(agg, use_container_width=True, height='auto', num_rows='dynamic' )

                # Trend lines (distinct linestyles/markers)
                fig = new_fig(12, 5)
                ax = fig.add_subplot(111)
                x = agg.index.values
                for i, col in enumerate(value_cols):
                    y = agg[col].values
                    ax.plot(
                        x, y,
                        label=col,
                        color=PALETTE[i % len(PALETTE)],
                        linestyle=LINESTYLES[i % len(LINESTYLES)],
                        marker=MARKERS[i % len(MARKERS)],
                        linewidth=1.6,
                        alpha=0.95,
                    )
                ax.set_title("Period Trends (distinct lines/markers)")
                ax.legend(ncol=3, fontsize=8)
                st.pyplot(fig)

# -------------------------------------------------------------------------------------------------
# TAB 9 — Export
# -------------------------------------------------------------------------------------------------
with tabs[9]:
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="balance_projection_export.csv",
        mime="text/csv",
    )
