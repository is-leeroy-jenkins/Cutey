##### Cutey
![](https://github.com/is-leeroy-jenkins/Cutey/blob/main/resources/assets/img/git/Cutey.png)
# 


The Cutey **Balance Projector** is a machine learning toolkit for federal budget execution & finance implemented in Python. It empowers data scientists, budget analysts, and federal agency personnel to train, compare, and visualize multiple regression models for **forecasting the balances of federal appropriation accounts**. Built using `Scikit`, `TensorFlow`, and `PyTorch`, the notebook integrates structured budget execution data—such as SF-133 reports from OMB and agency-specific datasets from Data.gov—to inform predictive models across multiple federal financial scenarios.

---
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/is-leeroy-jenkins/Cutey/blob/main/balances.ipynb)




## 🚀 Features

### 🔍 End-to-End Forecasting Workflow

Balance Projector provides a complete pipeline from raw data ingestion to model deployment, including:

- **Data Cleaning & Normalization**: Built-in functions allow for preprocessing of SF-133 data, including handling missing values, filtering TAS codes, and aggregating over fiscal quarters or years.
- **Feature Engineering Templates**: Add lag features, growth rates, fiscal flags (e.g., end-of-year), and obligation-to-appropriation ratios with reusable code blocks.
- **Time-Aware Modeling Support**: While inherently tabular, Balance Projector supports datasets structured as rolling fiscal snapshots, making it viable for time series forecasting over fixed federal periods (e.g., P01–P12).

---

### 🧠 Machine Learning in a Regulatory Environment

- Forecast balances for **Expired vs. Unexpired accounts**
- Incorporate **Period of Availability (PoA)**, fiscal flags, and **transfer actions**
- Embed OMB guidance into modeling features (e.g., apportionment limits, program activities)

### 🧾 Integrated Federal Data Templates

- **[SF-133](https://portal.max.gov/portal/document/SF133/Budget/FACTS%20II%20-%20SF%20133%20Report%20on%20Budget%20Execution%20and%20Budgetary%20Resources.html)**: Status of Budget Execution CSV import + reshaping
- **[GTAS](https://fiscal.treasury.gov/gtas/)**: Trial balance integration for actuals by TAS
- **[Agency Apportionments](https://openomb.org/)**: XML or CSV-based loader
- **[Data.gov](https://data.gov/)**: USAspending, object class, and program activity support

### Vectorized Datasets

- [Appropriations](https://huggingface.co/datasets/leeroy-jankins/Appropriations) - Enacted appropriations from 1996-2024 available for fine-tuning learning models
- [Regulations](https://huggingface.co/datasets/leeroy-jankins/Regulations/tree/main) - Collection of federal regulations on the use of appropriatied funds
- [SF-133](https://huggingface.co/datasets/leeroy-jankins/SF133) - The Report on Budget Execution and Budgetary Resources
- [Balances](https://huggingface.co/datasets/leeroy-jankins/Balances) -  U.S. federal agency Account Balances (File A) submitted as part of the DATA Act 2014.
- [Outlays](https://huggingface.co/datasets/leeroy-jankins/Outlays) -  The actual disbursements of funds by the U.S. federal government from 1962 to 2025
- [SF-133](https://huggingface.co/datasets/leeroy-jankins/SF133) The Report on Budget Execution and Budgetary Resources
- [Balances](https://huggingface.co/datasets/leeroy-jankins/Balances) - U.S. federal agency Account Balances (File A) submitted as part of the DATA Act 2014.
- [Circular A11](https://huggingface.co/datasets/leeroy-jankins/OMB-Circular-A-11) - Guidance from OMB on the preparation, submission, and execution of the federal budget
- [Fastbook](https://huggingface.co/datasets/leeroy-jankins/FastBook) - Treasury guidance on federal ledger accouts
- [Redbook](https://huggingface.co/datasets/leeroy-jankins/RedBook) - The Principles of Appropriations Law (Volumes I & II).
---


### 🧮 Multi-Model Performance Comparison

Models are evaluated using a consistent schema with a visual and tabular dashboard for:

- MAE, MSE, RMSE, and R² across all accounts
- Dynamic bar plots sorted by error metrics
- Actual vs. predicted overlays and error bands

Future enhancements will support model confidence intervals and budget violation detection.

---

## 🧮 Supported Models

### ✅ Regression Models

| Model                        | Module                                   |
|-----------------------------|------------------------------------------|
| Linear Regression           | `sklearn.linear_model.LinearRegression`  |
| Decision Tree Regressor     | `sklearn.tree.DecisionTreeRegressor`     |
| Random Forest Regressor     | `sklearn.ensemble.RandomForestRegressor`|
| Gradient Boosting Regressor | `sklearn.ensemble.GradientBoostingRegressor` |
| XGBoost Regressor *(optional)* | `xgboost.XGBRegressor`             |

> Each model shares a unified interface and is trained and evaluated under a standardized loop for fairness.

---

### 🧠 Regulatory Context-Aware Modeling

The models are aligned with federal fiscal law and budget practice, allowing for:

- Modeling **Expired vs. Unexpired balances**
- Use of **Period of Availability** to segment funding windows
- Tracking **Obligational Authority**, **Reapportioned Amounts**, and **Transfer-In/Transfer-Out**

This makes Balance Projector suitable for use by:

- **Agency Budget Formulation Teams**
- **OIG Auditors**
- **Congressional Appropriation Staff**

---


### 🧠 Model Benchmarking Framework
Balance Projector implements and compares the following regression models:

| Model                        | Module                              |
|-----------------------------|--------------------------------------|
| Linear Regression           | `sklearn.linear_model.LinearRegression` |
| Random Forest Regressor     | `sklearn.ensemble.RandomForestRegressor` |
| Gradient Boosting Regressor | `sklearn.ensemble.GradientBoostingRegressor` |

Each model is trained using unified logic for fair and comparable evaluation across time series-structured or tabular datasets.

---
## 🧪 Feature Engineering

- **Lag Variables**: Capture temporal dependencies in obligations/outlays
- **Growth Rates**: Quarter-over-quarter and year-over-year trends
- **Ratios**: Unobligated-to-appropriated balance, obligation rates
- **Time Markers**: Flags for EOY, CR periods, seasonal inflections
- **Log and Power Transforms**: Handle skew and nonlinear exposure

---

## 📊 Visual & Quantitative Evaluation

### 📈 Charts
- Line plots of forecasted vs. actual balances per TAS
- Residual scatterplots + histograms
- Error bars across time segments
- Comparative bar charts for MAE, MSE, RMSE

### 📏 Metrics
- **R²**: Variance explained
- **MAE**: Absolute error
- **MSE / RMSE**: Penalize large residuals
- **MAPE (optional)**: Scaled percent error
- **Execution Time**: For training and inference

---

## 🏛️ Use Cases in Government

- **OMB Analysts**: Model PoA burn rates, EOY projections
- **IG/OIG**: Track abnormal obligation behaviors
- **Agencies**: Budget execution validation and outlay pacing
- **Congressional Staff**: Score reprogramming risk, monitor carryover balances

---

### 📊 Forecasting Federal Account Balances

Balance Projector is purpose-built to support the **federal appropriations community** by forecasting balances in Treasury accounts based on data from:

- **OMB SF-133**: Status of Budget Execution and Budgetary Resources
- **Account A** (from `MAX A-11` or Treasury Appropriation Fund Symbol tables)
- **Agency-submitted apportionments and execution reports**
- **Publicly accessible datasets from [Data.gov](https://www.data.gov/)**, including:
  - GTAS (Governmentwide Treasury Account Symbol Adjusted Trial Balance System)
  - USAspending
  - Budget Object Class and Program Activity by Treasury Account

#### 📦 What This Means in Practice

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

This makes the Balance Projector highly useful for:
- **Accountants and Budget Officers** projecting funding needs or lapsing balances
- **OMB/Agency Analysts** building models for apportionments or reprogrammings
- **Inspectors General or Auditors** analyzing obligation trends

---

### 📊 Comprehensive Evaluation Metrics
- **R² Score** – Measures variance explained
- **Mean Absolute Error (MAE)** – Average prediction error
- **Mean Squared Error (MSE)** – Penalizes larger errors
- **Residual analysis and plots**
- **Model comparison summaries**

---

### 📁 Flexible Dataset Input
- Accepts both default examples and custom CSVs
- Auto-preprocessing compatible with `pandas`
- Structured to allow easy injection of Treasury Account Symbols or SF-133 data

---

### 📈 Visual Analytics
- Side-by-side actual vs. predicted plots
- Residual scatter charts
- Metric bar plots across models

---

### 🧰 Modular and Extensible
- Add regressors in less than 10 lines of code
- Separate training, prediction, and evaluation logic
- Easy to integrate with other government financial models

---

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

---

## 🧪 How to Run

```bash
git clone https://github.com/your-username/balance-projector.git
cd balance-projector
pip install -r requirements.txt
jupyter notebook balances.ipynb
```

---

### 📁 Customize Dataset

Replace dataset ingestion cell with:

```python
import pandas as pd
df = pd.read_csv("your_dataset.csv")
X = df.drop("target_column", axis=1)
y = df["target_column"]
```

---

### 📊 Outputs

- R², MAE, MSE for each model
- Bar plots of performance scores
- Visual predicted vs. actual scatter charts
- Residual error analysis

---

## 🔮 Roadmap

- [ ] Add time series models (Prophet, ARIMA)
- [ ] Integrate GridSearchCV for model tuning
- [ ] SHAP-based interpretability
- [ ] Flask/FastAPI API for deploying forecasts
- [ ] LLM summarization of forecast outcomes

---

## 🤝 Contributing

1. 🍴 Fork the project
2. 🔧 Create a branch: `git checkout -b feat/new-feature`
3. ✅ Commit and push changes
4. 📬 Submit a pull request

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

> “All models are wrong, but some are useful.” — George E. P. Box
