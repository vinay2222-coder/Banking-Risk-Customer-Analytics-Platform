import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def customer_segmentation(df):
    print("\n--- Running K-Means Customer Segmentation ---")
    # Let's pick the features that actually define customer financial behavior.
    # We want a mix of demographics (Age, Income) and actual banking activity.
    features = ['Age', 'Income', 'Account_Balance', 'Transaction_Count', 'Avg_Monthly_Spend']
    X = df[features].copy()
    
    # Just in case some missing values snuck through earlier cleaning steps, 
    # we'll plug them with the median so the clustering model doesn't complain.
    X = X.fillna(X.median())
    
    # K-Means is sensitive to scale (e.g., Income in thousands vs Age in tens), 
    # so we need to standardize everything to have a mean of 0 and variance of 1.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Let's group our customers into 4 distinct segments. 
    # Setting n_init to 10 ensures we try a few different starting points to find the best groupings.
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Customer_Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Now let's see what these groups actually look like on average.
    cluster_summary = df.groupby('Customer_Cluster')[features].mean().round(2)
    cluster_summary['Count'] = df['Customer_Cluster'].value_counts()
    
    print("Cluster Summary:")
    print(cluster_summary)
    
    # IDEALLY: Here is where we'd assign cool business names to these clusters (like "High Rollers" or "Budget Conscious")
    # based on their average spend and balance. Since this is an automated run, we'll keep them numbered for now.
    
    # Awesome, let's save this enriched dataset so we can visualize it in Power BI later!
    df.to_csv('data/processed/segmented_customers.csv', index=False)
    print("Segmented customer data saved to 'data/processed/segmented_customers.csv'")
    return df

def forecast_revenue_trends(df):
    print("\n--- Revenue Trend Forecasting (Moving Average Proxy) ---")
    # First, let's treat the 'Last_Transaction_Date' as our timeline for revenue events.
    df['Last_Transaction_Date'] = pd.to_datetime(df['Last_Transaction_Date'])
    
    # We need to roll up the spend data to the daily level to see the big picture.
    daily_spend = df.groupby('Last_Transaction_Date')['Avg_Monthly_Spend'].sum().reset_index()
    daily_spend = daily_spend.sort_values('Last_Transaction_Date')
    daily_spend.set_index('Last_Transaction_Date', inplace=True)
    
    # Daily data is noisy, so let's smooth it out using 7-day and 30-day moving averages. 
    # This will help us spot actual long-term trends instead of daily hiccups.
    daily_spend['7_Day_MA'] = daily_spend['Avg_Monthly_Spend'].rolling(window=7).mean()
    daily_spend['30_Day_MA'] = daily_spend['Avg_Monthly_Spend'].rolling(window=30).mean()
    
    print(daily_spend.tail(10))
    
    # Just a heads up: In a real production setup, we'd probably bring in heavy hitters 
    # like Prophet or ARIMA for this forecasting, but MAs are perfect for a quick trend check!
    
    plt.figure(figsize=(12, 6))
    plt.plot(daily_spend.index, daily_spend['Avg_Monthly_Spend'], label='Daily Spend', alpha=0.3)
    plt.plot(daily_spend.index, daily_spend['7_Day_MA'], label='7-Day MA', color='orange')
    plt.plot(daily_spend.index, daily_spend['30_Day_MA'], label='30-Day MA', color='red')
    plt.title('Daily Transaction Volume & Revenue Trend Forecast')
    plt.xlabel('Date')
    plt.ylabel('Total Spend Volume')
    plt.legend()
    plt.savefig('docs/eda_plots/revenue_forecast_trend.png')
    plt.close()
    
    print("Forecasting plot saved to 'docs/eda_plots/revenue_forecast_trend.png'")

if __name__ == "__main__":
    import os
    if os.path.exists('data/processed/cleaned_banking_data.csv'):
        df = pd.read_csv('data/processed/cleaned_banking_data.csv')
        os.makedirs('docs/eda_plots', exist_ok=True)
        customer_segmentation(df)
        forecast_revenue_trends(df)
    else:
        print("Error: Cleaned data not found. Run data_cleaning.py first.")
