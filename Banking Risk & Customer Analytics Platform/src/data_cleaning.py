import pandas as pd
import numpy as np
import os

def clean_data(input_path='data/raw/banking_data.csv', output_path='data/processed/cleaned_banking_data.csv'):
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Initial Shape: {df.shape}")
    
    # 1. Remove Duplicates
    # Sometimes data engineering pipelines double-count things. Let's make sure each row is unique.
    print("Removing duplicates...")
    duplicates_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Removed {duplicates_count} duplicates.")

    # 2. Handle Missing Values
    # Dealing with empty data! 
    print("Handling missing values...")
    # It makes more sense to guess someone's missing income based on what others in their profession make, rather than a global average.
    df['Income'] = df.groupby('Occupation')['Income'].transform(lambda x: x.fillna(x.median()))
    
    # For credit score, we'll just use the overall median as a safe bet.
    df['Credit_Score'] = df['Credit_Score'].fillna(df['Credit_Score'].median())
    
    # 3. Outlier Detection & Treatment
    # We don't want a few billionaires or data glitches heavily skewing our models, so we'll cap the extreme ends (1st and 99th percentiles).
    print("Treating outliers...")
    def cap_outliers(series, lower_quantile=0.01, upper_quantile=0.99):
        lower_limit = series.quantile(lower_quantile)
        upper_limit = series.quantile(upper_quantile)
        return np.clip(series, lower_limit, upper_limit)
    
    df['Account_Balance'] = cap_outliers(df['Account_Balance'])
    df['Avg_Monthly_Spend'] = cap_outliers(df['Avg_Monthly_Spend'])
    
    # 4. Feature Engineering
    # This is where we create new columns that tell a better story than the raw data alone.
    print("Performing feature engineering...")
    # How much of their income goes to paying off loans? High ratio = high stress.
    df['Debt_to_Income_Ratio'] = np.where(df['Income'] > 0, (df['EMI'] * 12) / df['Income'], 0)
    # How much are they spending compared to their balance?
    df['Credit_Utilization_Proxy'] = np.where(df['Account_Balance'] > 0, df['Avg_Monthly_Spend'] / df['Account_Balance'], 0)
    
    # Recency: How long has it been since they actually used their account?
    df['Last_Transaction_Date'] = pd.to_datetime(df['Last_Transaction_Date'])
    reference_date = df['Last_Transaction_Date'].max()
    df['Days_Since_Last_Tx'] = (reference_date - df['Last_Transaction_Date']).dt.days

    # Quick estimate of their total value to the bank over a standard 5-year period.
    df['CLV_Proxy'] = df['Avg_Monthly_Spend'] * 12 * 5 
    
    # Let's bucket people into Risk Segments. This is gold for business folks who want simple labels.
    conditions = [
        (df['Credit_Score'] >= 750) & (df['Debt_to_Income_Ratio'] < 0.3),
        (df['Credit_Score'] >= 650) & (df['Debt_to_Income_Ratio'] < 0.5),
        (df['Credit_Score'] < 650) | (df['Debt_to_Income_Ratio'] >= 0.5)
    ]
    choices = ['Low Risk', 'Medium Risk', 'High Risk']
    df['Risk_Segment'] = np.select(conditions, choices, default='Unknown')

    # 5. Data Validation
    # Let's do a final sanity check before saving. If any of these fail, we stop here.
    assert df['Income'].isnull().sum() == 0, "Wait, there are still missing values in Income!"
    assert df['Credit_Score'].isnull().sum() == 0, "Wait, there are still missing values in Credit_Score!"
    assert df.duplicated().sum() == 0, "Whoops, duplicates somehow snuck back in."

    print(f"Final Shape: {df.shape}")
    
    # All clean! Saving it out for the next stage.
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")

if __name__ == "__main__":
    clean_data()
