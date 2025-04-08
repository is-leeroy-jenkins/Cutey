##### Cutey
![](https://github.com/is-leeroy-jenkins/Cutey/blob/main/resources/assets/img/git/Cutey.png)
# 


The Cutey **Balance Projector** is a machine learning toolkit for federal budget execution & finance implemented in Python. It empowers data scientists, budget analysts, and federal agency personnel to train, compare, and visualize multiple regression models for **forecasting the balances of federal appropriation accounts**. Built using `Scikit`, `TensorFlow`, and `PyTorch`, the notebook integrates structured budget execution data—such as SF-133 reports from OMB and agency-specific datasets from Data.gov—to inform predictive models across multiple federal financial scenarios.

---
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/is-leeroy-jenkins/Cutey/blob/main/ipynb/balances.ipynb)




## 🚀 Features

### 🔍 End-to-End Forecasting Workflow

Balance Projector provides a complete pipeline from raw data ingestion to model deployment, including:

- **Data Cleaning & Normalization**: Built-in functions allow for preprocessing of SF-133 data, including handling missing values, filtering TAS codes, and aggregating over fiscal quarters or years.
- **Feature Engineering Templates**: Add lag features, growth rates, fiscal flags (e.g., end-of-year), and obligation-to-appropriation ratios with reusable code blocks.
- **Time-Aware Modeling Support**: While inherently tabular, Balance Projector supports datasets structured as rolling fiscal snapshots, making it viable for time series forecasting over fixed federal periods (e.g., P01–P12).

---

### 🧾 Federal Dataset Integration Templates

Custom code templates are included for parsing and aligning:

- **OMB SF-133 Reports**: Join by TAS or Account A and reshape to fiscal month/quarter granularity.
- **MAX Schedule-X**: Capture enacted budget data and program activity descriptions.
- **GTAS Trial Balances**: Align actual trial balances by fiscal year and subfunction codes.
- **Apportionment Snapshots**: For agencies using internal XML-based budget execution reports.

All templates are provided using `pandas` and are notebook-integrated with placeholder paths.

---

### 🧮 Multi-Model Performance Comparison

Models are evaluated using a consistent schema with a visual and tabular dashboard for:

- MAE, MSE, RMSE, and R² across all accounts
- Dynamic bar plots sorted by error metrics
- Actual vs. predicted overlays and error bands

Future enhancements will support model confidence intervals and budget violation detection.

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
