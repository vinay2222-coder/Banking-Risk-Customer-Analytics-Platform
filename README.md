# Banking Risk & Customer Analytics Platform

## Objective
A complete end-to-end data analytics project simulating a real-world scenario at a major financial institution. This project analyzes banking customer behavior, loan risk, credit usage, transactions, customer churn, fraud patterns, and profitability to help the bank improve decision-making and reduce financial risk.

## Project Architecture
- **Data Generation**: Synthesizing a 50,000+ row dataset of realistic banking data including customers, loans, transactions, and demographics.
- **Data Cleaning & EDA**: Handling missing values, outliers, feature engineering, and extracting insights on customer segments, loan defaults, and fraud patterns.
- **SQL Analysis**: Advanced SQL queries utilizing CTEs, Window Functions, and Joins to extract critical business KPIs.
- **Machine Learning**: Predictive models (Logistic Regression, Random Forest, XGBoost) for:
  - Loan Default Risk Prediction
  - Customer Churn Prediction
  - Fraud Detection
- **Power BI Dashboard**: A professional dark-themed dashboard with dynamic KPI cards, drill-through pages, and actionable business insights.

## Tech Stack
- **Languages**: Python, SQL
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Faker
- **Tools**: Power BI, Excel, Jupyter Notebook
- **Database**: SQLite / PostgreSQL (Simulated via SQL scripts)

## Folder Structure
```text
.
├── data/
│   ├── raw/                 # Generated raw datasets
│   └── processed/           # Cleaned datasets ready for BI/ML
├── docs/                    # Output Pngs
├── powerbi/                 # Power BI Dashboard
├── sql/                     # Advanced SQL scripts for analysis
├── src/                     # Python source code
│   ├── data_generation.py   # Script to synthesize 50k+ rows of banking data
│   ├── data_cleaning.py     # Data wrangling and feature engineering
│   ├── eda.py               # Exploratory Data Analysis & visualizations
│   ├── ml_models.py         # Churn, Fraud, and Default prediction models
│   └── segmentation_and_forecasting.py # K-Means segmentation & time series forecasting
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Setup Instructions

1. **Clone the repository / Navigate to project folder**:
   ```bash
   cd "Banking Risk & Customer Analytics Platform"
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute the pipeline**:
   - Run `python src/data_generation.py` to create the dataset.
   - Run `python src/data_cleaning.py` to process the data.
   - Run `python src/eda.py` to generate analysis plots.
   - Run `python src/ml_models.py` to train and evaluate ML models.
   - Run `python src/segmentation_and_forecasting.py` to run clustering and revenue forecasting.

## Business Impact
- **Risk Mitigation**: Proactive identification of potential loan defaults using ML.
- **Revenue Optimization**: Identification of high-value segments and branch performance.
- **Customer Retention**: Strategic interventions based on churn probability scoring.
- **Fraud Prevention**: Detection of anomalous transaction patterns to reduce losses.
