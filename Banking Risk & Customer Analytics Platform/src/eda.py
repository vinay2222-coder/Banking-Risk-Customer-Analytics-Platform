import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Let's make our charts look decent with a nice dark grid theme.
sns.set_theme(style="darkgrid", palette="muted")

def perform_eda(input_path='data/processed/cleaned_banking_data.csv', output_dir='docs/eda_plots/'):
    print("Loading cleaned data for EDA...")
    df = pd.read_csv(input_path)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Customer Demographics - Age Distribution
    # First question: Who are our customers? Let's check out the age spread.
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
    plt.title('Age Distribution of Customers')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}age_distribution.png')
    plt.close()

    # 2. Risk Segment by Default Risk
    # Does our arbitrary risk segmentation actually correlate with defaults? Let's find out.
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Risk_Segment', hue='Default_Risk', palette='Set2')
    plt.title('Default Risk by Risk Segment')
    plt.xlabel('Risk Segment')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}risk_segment_default.png')
    plt.close()

    # 3. Credit Score vs Interest Rate
    # Classic banking reality check: Do people with worse credit get hit with higher interest rates?
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df[df['Has_Loan'] == 1], x='Credit_Score', y='Interest_Rate', alpha=0.3)
    plt.title('Credit Score vs Interest Rate')
    plt.xlabel('Credit Score')
    plt.ylabel('Interest Rate (%)')
    plt.savefig(f'{output_dir}credit_score_vs_interest.png')
    plt.close()

    # 4. Correlation Matrix Heatmap
    # The big picture: How does everything relate to everything else?
    plt.figure(figsize=(14, 10))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.savefig(f'{output_dir}correlation_matrix.png')
    plt.close()

    # 5. Fraud by Transaction Count
    # Are fraudsters making a ton of small transactions or a few big ones? Let's look at volume.
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Fraud_Flag', y='Transaction_Count', palette='Set1')
    plt.title('Transaction Count Distribution for Fraud vs Non-Fraud')
    plt.xlabel('Fraud Flag (1 = Fraud)')
    plt.ylabel('Transaction Count')
    plt.savefig(f'{output_dir}fraud_by_transaction_count.png')
    plt.close()

    # 6. Churn Analysis by Account Balance
    # Why do people leave? Are they draining their accounts before they churn? 
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df[df['Customer_Churn'] == 0], x='Account_Balance', label='Retained', fill=True)
    sns.kdeplot(data=df[df['Customer_Churn'] == 1], x='Account_Balance', label='Churned', fill=True)
    plt.title('Account Balance Distribution: Churned vs Retained')
    plt.xlabel('Account Balance')
    plt.xlim(0, 100000)
    plt.legend()
    plt.savefig(f'{output_dir}churn_by_balance.png')
    plt.close()

    print(f"EDA completed. Plots saved to {output_dir}")

if __name__ == "__main__":
    perform_eda()
