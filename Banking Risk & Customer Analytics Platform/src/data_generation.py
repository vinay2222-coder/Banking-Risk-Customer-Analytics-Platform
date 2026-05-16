import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

# Initialize Faker
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

NUM_ROWS = 50000

def generate_banking_data():
    print(f"Generating {NUM_ROWS} rows of banking data...")
    
    # 1. Basic Demographics
    # Who are these people? Let's build out their basic info.
    customer_ids = [f"CUST_{str(i).zfill(6)}" for i in range(1, NUM_ROWS + 1)]
    ages = np.random.randint(18, 75, size=NUM_ROWS)
    genders = np.random.choice(["Male", "Female", "Other"], p=[0.52, 0.46, 0.02], size=NUM_ROWS)
    
    # Income usually follows a log-normal distribution (lots of middle class, a few ultra-rich)
    incomes = np.random.lognormal(mean=10.8, sigma=0.6, size=NUM_ROWS).round(2)
    incomes = np.clip(incomes, 20000, 250000) # Keep it grounded, no billionaires for this dataset
    
    occupations = ["Salaried", "Self-Employed", "Business", "Student", "Retired"]
    occupation_arr = np.random.choice(occupations, p=[0.5, 0.2, 0.15, 0.05, 0.1], size=NUM_ROWS)
    
    regions = ["North", "South", "East", "West", "Central"]
    region_arr = np.random.choice(regions, size=NUM_ROWS)
    
    branches = [f"Branch_{str(i).zfill(3)}" for i in range(1, 51)] # 50 Branches total
    branch_arr = np.random.choice(branches, size=NUM_ROWS)

    # 2. Financial Metrics
    # Now let's look at their bank accounts.
    credit_scores = np.random.normal(loc=650, scale=80, size=NUM_ROWS)
    credit_scores = np.clip(credit_scores, 300, 850).astype(int)
    
    account_types = ["Savings", "Current", "Salary"]
    account_type_arr = np.random.choice(account_types, p=[0.6, 0.1, 0.3], size=NUM_ROWS)
    
    account_balances = np.random.exponential(scale=15000, size=NUM_ROWS).round(2)
    
    card_types = ["Standard", "Gold", "Platinum", "None"]
    card_type_arr = np.random.choice(card_types, p=[0.4, 0.3, 0.1, 0.2], size=NUM_ROWS)

    # 3. Loan Metrics
    # Let's say about 40% of our customer base is currently holding a loan with us.
    has_loan = np.random.choice([True, False], p=[0.4, 0.6], size=NUM_ROWS)
    loan_amounts = np.where(has_loan, np.random.uniform(5000, 100000, size=NUM_ROWS), 0).round(2)
    
    # Realism check: Better credit score = better interest rate. We'll simulate that here.
    base_rate = 15
    rate_adjustment = (credit_scores - 300) / 550 * 10
    interest_rates = np.where(has_loan, np.clip(base_rate - rate_adjustment + np.random.normal(0, 1, NUM_ROWS), 5, 24), 0).round(2)
    
    # Let's rough out an EMI calculation assuming a standard 5-year payback period.
    tenure_months = 60
    monthly_rate = interest_rates / (12 * 100)
    emi = np.where(has_loan, (loan_amounts * monthly_rate * ((1 + monthly_rate)**tenure_months)) / (((1 + monthly_rate)**tenure_months) - 1), 0).round(2)

    # 4. Transaction & Spend Behavior
    # How active are they? Poisson distribution works well for event counts like monthly transactions.
    transaction_counts = np.random.poisson(lam=15, size=NUM_ROWS) 
    avg_monthly_spends = np.random.uniform(500, 10000, size=NUM_ROWS).round(2)
    
    last_tx_dates = [fake.date_between(start_date='-60d', end_date='today') for _ in range(NUM_ROWS)]

    # 5. Risk, Fraud, and Churn Indicators 
    # This is the secret sauce. We need to bake in logical correlations so our ML models actually have patterns to find.
    
    # Default Risk: People with crazy high EMIs compared to their income, and bad credit, are more likely to default.
    emi_to_income = emi / (incomes / 12)
    default_prob = 1 / (1 + np.exp(-(-2 + 5 * emi_to_income - 0.005 * (credit_scores - 600))))
    default_risk = np.where(has_loan, np.random.binomial(1, default_prob), 0)
    
    loan_statuses = np.where(has_loan, np.where(default_risk == 1, "Defaulted", np.random.choice(["Active", "Closed"], p=[0.8, 0.2], size=NUM_ROWS)), "No Loan")

    # Fraud Flag: Extremely high transaction counts usually warrant an investigation.
    fraud_prob = np.where(transaction_counts > 40, 0.15, 0.01)
    fraud_flag = np.random.binomial(1, fraud_prob)

    # Churn: Why do people leave? Usually because they barely use the account (low balance, no loans, few transactions).
    churn_prob = 1 / (1 + np.exp(-(-3 + 0.5 * (account_balances < 1000) - 0.5 * has_loan + 0.5 * (transaction_counts < 5))))
    customer_churn = np.random.binomial(1, churn_prob)

    # Smash it all together into a DataFrame
    df = pd.DataFrame({
        'Customer_ID': customer_ids,
        'Age': ages,
        'Gender': genders,
        'Income': incomes,
        'Occupation': occupation_arr,
        'Region': region_arr,
        'Branch': branch_arr,
        'Account_Type': account_type_arr,
        'Account_Balance': account_balances,
        'Credit_Score': credit_scores,
        'Card_Type': card_type_arr,
        'Has_Loan': has_loan.astype(int),
        'Loan_Amount': loan_amounts,
        'Interest_Rate': interest_rates,
        'EMI': emi,
        'Loan_Status': loan_statuses,
        'Transaction_Count': transaction_counts,
        'Avg_Monthly_Spend': avg_monthly_spends,
        'Last_Transaction_Date': last_tx_dates,
        'Default_Risk': default_risk,
        'Fraud_Flag': fraud_flag,
        'Customer_Churn': customer_churn
    })
    
    # Let's mess the data up a bit! Real-world data is never perfect, so we'll intentionally punch some holes in it.
    idx_missing_income = np.random.choice(df.index, size=int(NUM_ROWS * 0.02), replace=False)
    df.loc[idx_missing_income, 'Income'] = np.nan
    
    idx_missing_score = np.random.choice(df.index, size=int(NUM_ROWS * 0.01), replace=False)
    df.loc[idx_missing_score, 'Credit_Score'] = np.nan

    # Throwing in some duplicates for the cleaning script to catch.
    duplicates = df.sample(n=int(NUM_ROWS * 0.005))
    df = pd.concat([df, duplicates], ignore_index=True)

    # Give it a good shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Ensure our save destinations exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # Ship it!
    output_path = 'data/raw/banking_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully! Shape: {df.shape}. Saved to {output_path}")

if __name__ == "__main__":
    generate_banking_data()
