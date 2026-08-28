###### Cutey
![](https://github.com/is-leeroy-jenkins/Cutey/blob/main/resources/assets/img/git/Cutey.png)
___

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-0078FC?style=for-the-badge&logo=github)](https://is-leeroy-jenkins.github.io/Cutey/)

A machine-learning toolkit for federal budget execution & accounting implemented in Python. Built using `Scikit`, `TensorFlow`, and `PyTorch`, the notebook integrates structured budget execution data—such as SF-133 reports from OMB and agency-specific datasets from Data.gov—to inform predictive models across multiple federal financial scenarios.

## ☁️ Cloud

<table>
<tr>
<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://cutey-py.streamlit.app/">
<img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit App">
</a>
</td>

<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://colab.research.google.com/github/is-leeroy-jenkins/Cutey/blob/main/balances.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>
</td>

<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://dbc-a0c21f80-7bb3.cloud.databricks.com/editor/notebooks/1460524320197786?o=7474645703081351">
<img src="https://img.shields.io/badge/Databricks%20Repo-Cutey--Py-FF3621?logo=databricks&logoColor=white" alt="Databricks Notebook">
</a>
</td>

<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://leeroy.usw-16.palantirfoundry.com/shares/links/tq6dokgd3ezi2">
<img src="https://img.shields.io/badge/Palantir%20Foundry-Repo-101113?logo=palantir&logoColor=white" alt="Repo">
</a>
</td>
</tr>
</table>

## 🎥 Demo

![](https://github.com/is-leeroy-jenkins/Cutey/blob/main/resources/assets/cutey-demo.gif)

## ☁️ Google

![](https://github.com/is-leeroy-jenkins/Cutey/blob/main/resources/assets/Cutey-nb.gif)



## 🕸️ Streamlit (Web)

![](https://github.com/is-leeroy-jenkins/Cutey/blob/main/resources/assets/Cutey-Py.gif)


## 🔍 Predictive Pipeline

#### Balance Projector provides a complete pipeline from raw data ingestion to model deployment, including:

- **Data Cleaning & Normalization**: Built-in functions allow for preprocessing of SF-133 data, including handling missing values, filtering TAS codes, and aggregating over fiscal quarters or years.
- **Feature Engineering Templates**: Add lag features, growth rates, fiscal flags (e.g., end-of-year), and obligation-to-appropriation ratios with reusable code blocks.
- **Time-Aware Modeling Support**: While inherently tabular, Balance Projector supports datasets structured as rolling fiscal snapshots, making it viable for time series forecasting over fixed federal periods (e.g., P01–P12).

## 🧪 How to Install

```bash
git clone https://github.com/your-username/balance-projector.git
cd balance-projector
pip install -r requirements.txt
jupyter notebook balances.ipynb
```

#### Option A — Google Colab (no local setup)


```
1. Click the **Open In Colab** badge above.
2. Upload your CSV or mount Google Drive.
3. Set `DATA_PATH` near the top of the notebook.
4. **Runtime → Run all**.
```

#### Option B — Local (conda or venv)

```
bash
# 1) Create environment
conda create -n schedx python=3.11 -y
conda activate schedx

# 2) Install dependencies
pip install -U pip wheel setuptools
pip install pandas numpy scipy matplotlib seaborn scikit-learn jupyter

# 3) Launch Jupyter
jupyter notebook
```

> Open `ipynb/balances.ipynb` and run cells top-to-bottom.


### 1️⃣ Clone the Repository

First, clone the GitHub repository to your local machine:

    git clone https://github.com/<your-username>/<your-repo-name>.git
    cd <your-repo-name>

Replace `<your-username>` and `<your-repo-name>` with the actual GitHub path.

---

### 2️⃣ Create and Activate a Virtual Environment (Recommended)

Using a virtual environment is strongly recommended to avoid dependency conflicts.

**Windows (PowerShell):**

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

**macOS / Linux:**

    python3 -m venv .venv
    source .venv/bin/activate

Once activated, your shell prompt should indicate the virtual environment is in use.

---

### 3️⃣ Install Dependencies

Install the required Python packages using `pip`:

    pip install --upgrade pip
    pip install -r requirements.txt

If a `requirements.txt` file is not provided, the minimum required packages are:

- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `statsmodels` (required for time-series forecasting features)

You can install them directly with:

    pip install streamlit pandas numpy matplotlib scikit-learn scipy statsmodels

---

### 4️⃣ Run the Streamlit App

From the root of the repository (where `app.py` is located), run:

    streamlit run app.py

Streamlit will start a local development server and automatically open the app
in your default web browser. If it does not open automatically, look for a URL
similar to:

    http://localhost:8501

and open it manually.

---

### 5️⃣ Using the Application

1. Upload a **CSV** or **Excel (.xlsx / .xls)** file containing balance or budget data.
2. Navigate through the tabs to explore:
   - Data preview and descriptive statistics
   - Distributions and normality checks
   - PCA and clustering
   - Correlation analysis
   - Regression model comparisons
   - Time-series aggregation and forecasting
3. Download processed data from the **Export** tab if needed.

---

### 6️⃣ Stopping the App

To stop the application, return to the terminal where Streamlit is running and press:

    Ctrl + C

This will shut down the local Streamlit server safely.

---

### Notes

- The app runs entirely **locally**; no data is uploaded to external services.
- For best results, use datasets with clearly defined numeric balance fields and
  a period or fiscal-year column for time-series analysis.
- The time-series tab requires `statsmodels`; if it is not installed, those
  features will be disabled automatically.


## 🧠 Machine Learning in a Regulatory Environment

- Forecast balances for **No-year, Multi-year, and Expiring Accounts**
- Incorporate **Period of Availability (PoA)**, fiscal flags, and **transfer actions**
- Embed OMB guidance into modeling features (e.g., apportionment limits, program activities)

## 📦 Fine-tuning Datasets


| File Name                                                                                                                                                                 | Description                                                                                                            |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| [Balanced Budget and Emergency Deficit Control Act of 1985](https://huggingface.co/datasets/leeroy-jankins/The-Balanced-Budget-And-Emergency-Deficit-Control-Act-of-1985) | Establishes statutory limits on federal spending and deficit control mechanisms, including sequestration procedures.   |
| [Budget Control Act of 2011](https://huggingface.co/datasets/leeroy-jankins/The-Budget-Control-Act-2011)                                                                  | Sets discretionary spending caps and establishes enforcement mechanisms to control federal deficits.                   |
| [Digital Accountability And Transparency Act of 2014](https://huggingface.co/datasets/leeroy-jankins/Data-Act-2014)                                                       | Requires standardized federal spending data and improved transparency through government-wide financial reporting.     |
| [Federal Account Symbols And Titles Book](https://huggingface.co/datasets/leeroy-jankins/FastBook)                                                                        | Defines Treasury account symbols and official titles used for federal budgetary and accounting purposes.               |
| [Federal Acquisition Regulation](https://huggingface.co/datasets/leeroy-jankins/Federal-Acquisition-Regulation)                                                           | Establishes uniform policies and procedures governing the acquisition of goods and services by federal agencies.       |
| [Federal Government Standards For Internal Controls](https://huggingface.co/datasets/leeroy-jankins/Federal-Government-Standards-For-Internal-Controls)                   | Defines the internal control framework for federal agencies to ensure accountability, integrity, and compliance.       |
| [Federal Managers Financial Integrity Act of 1982](https://huggingface.co/datasets/leeroy-jankins/FMFIA-1982)                                                             | Requires agencies to establish internal controls and report annually on their effectiveness.                           |
| [Federal Trust Fund Accounting Guide](https://huggingface.co/datasets/leeroy-jankins/Federal-Trust-Fund-Accounting-Guide)                                                 | Provides accounting guidance for the management and reporting of federal trust funds.                                  |
| [Financial Management Regulations DOD 7000-14-R](https://huggingface.co/datasets/leeroy-jankins/DOD-7000-14-Financial-Management-Regulation)                                                                                                                        | Establishes DoD-specific financial management policies, procedures, and accounting requirements.                       |
| [Fiscal Responsibility Act](https://huggingface.co/datasets/leeroy-jankins/The-Fiscal-Responsibility-Act-of-2023)                                                                                                                                                 | Establishes statutory measures intended to improve fiscal discipline and control federal spending.                     |
| [Government Auditing Standards](https://huggingface.co/datasets/leeroy-jankins/Government-Auditing-Standards)                                                                                                                                             | Sets professional standards for audits of government organizations, programs, activities, and functions.               |
| [Government Invoicing User Guide](https://huggingface.co/datasets/leeroy-jankins/Government-Performance-and-Results-Act)                                                                                                                                           | Provides guidance on federal invoicing standards and processes for government transactions.                            |
| [Government Performance and Results Act of 1993](https://huggingface.co/datasets/leeroy-jankins/Government-Performance-and-Results-Act)                                                                                                                            | Requires agencies to engage in strategic planning and performance measurement to improve program effectiveness.        |
| [GPRA Modernization Act of 2010](https://huggingface.co/datasets/leeroy-jankins/The-GPRA-Modernization-Act-Of-2010)                                                                                                                                            | Updates GPRA by strengthening performance management, cross-agency goals, and accountability.                          |
| [OMB Circular A-11 Preparation Submission And Execution Of The Budget](https://huggingface.co/datasets/leeroy-jankins/OMB-Circular-A-11)                                                                                                      | Provides comprehensive guidance for preparing, submitting, and executing the President’s Budget.                       |
| [OMB Circular A-11 Section 120 Apportionment Process](https://huggingface.co/datasets/leeroy-jankins/OMB-Circular-A11-Section-120-Apportionment-Process)                                                                                                                       | Defines the apportionment process used to control the rate of obligation of budgetary resources.                       |
| [OMB Circular A-123 Managements Responsibility for Enterprise Risk Management and Internal Control](https://huggingface.co/datasets/leeroy-jankins/OMB-Circular-A-123)                                                                         | Defines management responsibilities for internal control and enterprise risk management across federal agencies.       |
| [Federal Trust Fund Accounting Guide](https://huggingface.co/datasets/leeroy-jankins/Federal-Trust-Fund-Accounting-Guide)                                                                                                                       | Establishes requirements for federal agency financial statements and reporting.                                        |
| [Principles Of Federal Appropriations Law Volume One](https://huggingface.co/datasets/leeroy-jankins/Principles-Of-Federal-Appropriations-Law)                                                                                                                       | Authoritative GAO guidance on foundational principles governing the use of federal appropriations.                     |
| [Statements of Federal Federal Financial Accounting Concepts and Standards](https://huggingface.co/datasets/leeroy-jankins/Statements-Of-Federal-Financial-Accounting-Concepts-And-Standards)                                                                                                 | Establishes accounting concepts and standards for federal financial reporting.                                         |
| [The Anti-Deficiency Act PL 97-258](https://huggingface.co/datasets/leeroy-jankins/The-Anti-Deficiency-Act)                                                                                                                                         | Prohibits federal agencies from obligating or expending funds in excess of appropriations or before enactment.         |
| [The Anti-Deficiency Reform and Enforcement Act of 2018](https://huggingface.co/datasets/leeroy-jankins/The-Anti-Deficiency-Reform-And-Enforcement-Act-Of-2018)                                                                                                                    | Strengthens Anti-Deficiency Act enforcement and reporting requirements to improve fiscal accountability.               |
| [The Chief Financial Officers Act of 1990](https://huggingface.co/datasets/leeroy-jankins/The-Chief-Financial-Officers-Act-1990)                                                                                                                                  | Establishes agency Chief Financial Officers and modernizes federal financial management practices.                     |
| [The Congressional Budget and Impoundment Control Act of 1974](https://huggingface.co/datasets/leeroy-jankins/The-Congressional-Budget-And-Impoundment-Control-Act-Of-1974)                                                                                                              | Establishes the congressional budget process and restricts executive impoundment of appropriated funds.                |
| [Statutory Pay As You Go Act of 2010](https://huggingface.co/datasets/leeroy-jankins/Statutory-Pay-As-You-Go-Act-of-2010)                                                                                                                                                   | Authorizes interagency agreements for the provision of goods and services on a reimbursable basis.                     |
| [The Stafford Act](https://huggingface.co/datasets/leeroy-jankins/The-Stafford-Act)                                                                                                                                                          | Provides the statutory framework for federal disaster response and emergency assistance.                               |
| [Federal Trust Fund Accounting Guide](https://huggingface.co/datasets/leeroy-jankins/Federal-Trust-Fund-Accounting-Guide)                                                                                                                                  | Provides additional appropriations authority beyond regular annual funding acts.                                       |
| [Title 2 Code of Federal Regulations – Uniform Administrative Requirements, Cost Principles, and Audit](https://huggingface.co/datasets/leeroy-jankins/Title-2-CFR-Uniform-Administrative-Requirements-Cost-Principles-And-Audit)                                                                     | Establishes uniform administrative, cost, and audit requirements for federal financial assistance.                     |
| [Title 31 Code of Federal Regulations – Money and Finance](https://huggingface.co/datasets/leeroy-jankins/Title-31-CFR-Money-and-Finance)                                                                                                                  | Codifies Treasury and federal financial management regulations governing money and finance.                            |
| [US Standard General Ledger Account Definitions](https://huggingface.co/datasets/leeroy-jankins/US-Standard-General-Ledger-Accounts-And-Definitions)                                                                                                                            | Defines standardized account structures used for federal accounting and financial reporting.                           |


## 🧮 Multi-Model Performance Comparison

Models are evaluated using a consistent schema with a visual and tabular dashboard for:

- MAE, MSE, RMSE, and R² across all accounts
- Dynamic bar plots sorted by error metrics
- Actual vs. predicted overlays and error bands

Future enhancements will support model confidence intervals and budget violation detection.


## ✅ Regression Models

| Model                        | Module                                   |
|-----------------------------|------------------------------------------|
| Linear Regression           | `sklearn.linear_model.LinearRegression`  |
| Decision Tree Regressor     | `sklearn.tree.DecisionTreeRegressor`     |
| Random Forest Regressor     | `sklearn.ensemble.RandomForestRegressor`|
| Gradient Boosting Regressor | `sklearn.ensemble.GradientBoostingRegressor` |
| XGBoost Regressor *(optional)* | `xgboost.XGBRegressor`             |

> Each model shares a unified interface and is trained and evaluated under a standardized loop for fairness.



## 🧠 Regulation-Aware Context Modeling

#### The models are aligned with federal fiscal law and budget practice, allowing for:

- Modeling **Expired vs. Unexpired balances**
- Use of **Period of Availability** to segment funding windows
- Tracking **Obligational Authority**, **Reapportioned Amounts**, and **Transfer-In/Transfer-Out**

#### This makes Balance Projector suitable for use by:

- **Agency Budget Formulation Teams**
- **OIG Auditors**
- **Congressional Appropriation Staff**


#### Balance Projector implements and compares the following regression models:

| Model                        | Module                              |
|-----------------------------|--------------------------------------|
| Linear Regression           | `sklearn.linear_model.LinearRegression` |
| Random Forest Regressor     | `sklearn.ensemble.RandomForestRegressor` |
| Gradient Boosting Regressor | `sklearn.ensemble.GradientBoostingRegressor` |

#### Each model is trained using unified logic for fair and comparable evaluation across time series-structured or tabular datasets.


## 🧪 Feature Engineering

- **Lag Variables**: Capture temporal dependencies in obligations/outlays
- **Growth Rates**: Quarter-over-quarter and year-over-year trends
- **Ratios**: Unobligated-to-appropriated balance, obligation rates
- **Time Markers**: Flags for EOY, CR periods, seasonal inflections
- **Log and Power Transforms**: Handle skew and nonlinear exposure


## 📈 Visual & Quantitative Evaluation
- Line plots of forecasted vs. actual balances per TAS
- Residual scatterplots + histograms
- Error bars across time segments
- Comparative bar charts for MAE, MSE, RMSE

## 📏 Metrics
- **R²**: Variance explained
- **MAE**: Absolute error
- **MSE / RMSE**: Penalize large residuals
- **MAPE (optional)**: Scaled percent error
- **Execution Time**: For training and inference



## 🏛️ Use Cases in Government

- **OMB Reporting**: Model PoA burn rates, EOY projections
- **IG/OIG**: Track abnormal obligation behaviors
- **Agencies**: Budget execution validation and outlay pacing
- **Congressional Staff**: Score reprogramming risk, monitor carryover balances



## 📊 Forecasting Federal Account Balances

Balance Projector is purpose-built to support the **federal appropriations community** by forecasting balances in Treasury accounts based on data from:

- **OMB SF-133**: Status of Budget Execution and Budgetary Resources
- **Account A** (from `MAX A-11` or Treasury Appropriation Fund Symbol tables)
- **Agency-submitted apportionments and execution reports**
- **Publicly accessible datasets from [Data.gov](https://www.data.gov/)**, including:
  - GTAS (Governmentwide Treasury Account Symbol Adjusted Trial Balance System)
  - USAspending
  - Budget Object Class and Program Activity by Treasury Account

## 📦 What This Means in Practice

- The notebook is pre-structured to ingest **SF-133 extracts** in tabular CSV format. Key fields include:
  - `Treasury Account Symbol (TAS)`
  - `Period of Availability`
  - `Obligations`, `Outlays`, `Unobligated Balances`, `Appropriations`, etc.

- Forecasts can be made at:
  - **Account level (e.g., 012-1234)** using aggregated totals
  - **Budget Object Class (BOC) level**
  - **Program Activity level**
  - **Quarterly, Monthly, or Annual** time frequencies

- The models can be trained to predict:
  - Future unobligated balances (for expired or current-year accounts)
  - Fiscal year close-out positions
  - Anticipated outlays or expenditure curves




## 📊 Descriptive Statistics

| Statistic         | Description                             | Use in Budget Analysis                                               |
|------------------|-----------------------------------------|----------------------------------------------------------------------|
| **Mean**         | Average value                           | Avg. Outlays, Obligations, etc., across accounts                |
| **Median**       | Middle value                            | Robust central tendency in skewed financial data                    |
| **Mode**         | Most frequent value                     | Identify common MainAccountCodes or Availability categories     |
| **Standard Deviation** | Spread around the mean                | Indicates variability in execution rates or balances                |
| **Variance**     | Square of standard deviation            | Used in statistical tests and model diagnostics                     |
| **Range**        | Difference between max and min          | Measures total spread of financial metrics                          |
| **Interquartile Range (IQR)** | Spread of middle 50% of data           | Identifies budget outliers and extreme accounts                     |
| **Skewness**     | Asymmetry of distribution               | Skewed obligations suggest few accounts dominate totals             |
| **Kurtosis**     | "Peakedness" of distribution            | High values indicate outlier-prone financial data                   |





## 🔍 Inferrential Statistics


| Metric           | Description                                            | Use in Budget Analysis                                               |
|-------------------------|--------------------------------------------------------|----------------------------------------------------------------------|
| **Pearson Correlation** | Linear relationship between variables                  | E.g., TotalResources vs. Obligations                                 |
| **Spearman Correlation**| Monotonic (rank-based) relationship                    | More robust to non-linear trends in financial execution              |
| **t-test**              | Compare means between 2 groups                         | Discretionary vs. Mandatory accounts' execution rates                |
| **ANOVA**               | Compare means across multiple groups                   | Obligations across availability periods or account types             |
| **Chi-square Test**     | Categorical independence                               | Are Main Account Codes related to availability or a specific agency? |
| **Confidence Intervals**| Estimate range of a population mean                    | Upper and lower bound expected obligations or recoveries             |
| **Regression Coefficients (p-values)** | Test variable significance                             | Are Recoveries a significant predictor of UnobligatedBalance?        |
| **F-statistic (overall regression)**   | Test whole model fit                                   | Determines the combined influence of all predictors                  |
| **Z-score / Outlier Tests** | Deviation from standard mean                           | Identify abnormal balances or lapse rates                            |
| **Boxplots**            | Visual outlier detection                               | Discover obligation anomalies within agencies                        |
#### This makes the Balance Projector highly useful for:
- **Accountants and Budget Officers** projecting funding needs or lapsing balances
- **OMB/Agency Analysts** building models for apportionments or reprogrammings
- **Inspectors General or Auditors** analyzing obligation trends


## 📊 Comprehensive Evaluation Metrics
- **R² Score** – Measures variance explained
- **Mean Absolute Error (MAE)** – Average prediction error
- **Mean Squared Error (MSE)** – Penalizes larger errors
- **Residual analysis and plots**
- **Model comparison summaries**



## 📁 Flexible Dataset Input
- Accepts both default examples and custom CSVs
- Auto-preprocessing compatible with `pandas`
- Structured to allow easy injection of Treasury Account Symbols or SF-133 data



## 📈 Visual Analytics
- Side-by-side actual vs. predicted plots
- Residual scatter charts
- Metric bar plots across models



## 🧰 Modular and Extensible
- Add regressors in less than 10 lines of code
- Separate training, prediction, and evaluation logic
- Easy to integrate with other government financial models



## 📦 Dependencies

| Package          | Description                                                      | Link                                                  |
|------------------|------------------------------------------------------------------|-------------------------------------------------------|
| numpy            | Numerical computing library                                      | [numpy.org](https://numpy.org/)                      |
| pandas           | Data manipulation and DataFrames                                 | [pandas.pydata.org](https://pandas.pydata.org/)      |
| matplotlib       | Plotting and visualization                                       | [matplotlib.org](https://matplotlib.org/)            |
| seaborn          | Statistical data visualization                                   | [seaborn.pydata.org](https://seaborn.pydata.org/)    |
| scikit-learn     | ML modeling and metrics                                          | [scikit-learn.org](https://scikit-learn.org/stable/) |
| xgboost          | Gradient boosting framework (optional)                          | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) |
| torch            | PyTorch deep learning library                                    | [pytorch.org](https://pytorch.org/)                  |
| tensorflow       | End-to-end ML platform                                           | [tensorflow.org](https://www.tensorflow.org/)        |
| openai           | OpenAI’s Python API client                                       | [openai-python](https://github.com/openai/openai-python) |
| requests         | HTTP requests for API and web access                             | [requests.readthedocs.io](https://requests.readthedocs.io/) |
| PySimpleGUI      | GUI framework for desktop apps                                   | [pysimplegui.readthedocs.io](https://pysimplegui.readthedocs.io/) |
| typing           | Type hinting standard library                                    | [typing Docs](https://docs.python.org/3/library/typing.html) |
| pyodbc           | ODBC database connector                                          | [pyodbc GitHub](https://github.com/mkleehammer/pyodbc) |
| fitz             | PDF document parser via PyMuPDF                                  | [pymupdf](https://pymupdf.readthedocs.io/)           |
| pillow           | Image processing library                                         | [python-pillow.org](https://python-pillow.org/)       |
| openpyxl         | Excel file processing                                            | [openpyxl Docs](https://openpyxl.readthedocs.io/)     |
| soundfile        | Read/write sound file formats                                    | [pysoundfile](https://pysoundfile.readthedocs.io/)    |
| sounddevice      | Audio I/O interface                                              | [sounddevice Docs](https://python-sounddevice.readthedocs.io/) |
| loguru           | Structured, elegant logging                                      | [loguru GitHub](https://github.com/Delgan/loguru)     |
| statsmodels      | Statistical tests and regression diagnostics                     | [statsmodels.org](https://www.statsmodels.org/)       |
| dotenv           | Load environment variables from `.env`                          | [python-dotenv GitHub](https://github.com/theskumar/python-dotenv) |
| python-dotenv    | Same as above (modern usage)                                     | [python-dotenv](https://saurabh-kumar.com/python-dotenv/) |



## 📁 Customize Dataset

Replace dataset ingestion cell with:

```python
import pandas as pd
df = pd.read_csv("your_dataset.csv")
X = df.drop("target_column", axis=1)
y = df["target_column"]
```



## 📊 Outputs

- R², MAE, MSE for each model
- Bar plots of performance scores
- Visual predicted vs. actual scatter charts
- Residual error analysis



## 🔮 Roadmap

- [ ] Add time series models (Prophet, ARIMA)
- [ ] Integrate GridSearchCV for model tuning
- [ ] SHAP-based interpretability
- [ ] Flask/FastAPI API for deploying forecasts
- [ ] LLM summarization of forecast outcomes




> **Disclaimer**: This is for analytical exploration and research purposes.  
> This is **not** an official government product; validate against authoritative sources before use.

## 📝 License

- Cutey is published under the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Cutey/blob/main/LICENSE.txt).


