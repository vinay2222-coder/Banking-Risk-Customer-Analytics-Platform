import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(y_test, y_pred, model_name):
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def train_models():
    print("Loading processed data for ML modeling...")
    df = pd.read_csv('data/processed/cleaned_banking_data.csv')

    # Prepare features for Loan Default Prediction
    print("\n--- Task 1: Loan Default Prediction ---")
    # We only care about people who actually took out loans. Can't default if you don't owe anything!
    loan_df = df[df['Has_Loan'] == 1].copy()
    
    # What variables actually matter for paying back a loan? Age, income, and the loan terms.
    features = ['Age', 'Income', 'Credit_Score', 'Loan_Amount', 'Interest_Rate', 'EMI', 'Debt_to_Income_Ratio']
    X = loan_df[features]
    y = loan_df['Default_Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling is crucial so our Logistic Regression doesn't freak out over the difference between $50k income and 0.2 DTI ratio.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Logistic Regression
    # The industry standard baseline. It's simple, interpretable, and regulators love it.
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    evaluate_model(y_test, lr_preds, "Logistic Regression (Default Risk)")
    
    # 2. Random Forest
    # Let's see if a more complex tree-based model can squeeze out better performance.
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)
    evaluate_model(y_test, rf_preds, "Random Forest (Default Risk)")

    print("\n--- Task 2: Fraud Detection ---")
    # For fraud, we're looking at account activity, not long-term loan terms.
    fraud_features = ['Transaction_Count', 'Avg_Monthly_Spend', 'Account_Balance']
    X_fraud = df[fraud_features]
    y_fraud = df['Fraud_Flag']
    
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)
    
    # 3. XGBoost for Fraud 
    # Fraud is incredibly rare (highly imbalanced data), so we use XGBoost and heavily penalize it for missing a fraud case (scale_pos_weight).
    xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=99) # Approx ratio of non-fraud to fraud
    xgb_model.fit(X_train_f, y_train_f)
    xgb_preds = xgb_model.predict(X_test_f)
    evaluate_model(y_test_f, xgb_preds, "XGBoost (Fraud Detection)")

    print("\n--- Task 3: Customer Churn Prediction ---")
    # Churn is influenced by a mix of demographics and banking engagement.
    churn_features = ['Age', 'Income', 'Account_Balance', 'Transaction_Count', 'Avg_Monthly_Spend', 'Has_Loan', 'Credit_Score']
    X_churn = df[churn_features]
    y_churn = df['Customer_Churn']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42)
    
    # Random Forest usually handles these mixed data types well for churn.
    rf_churn = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_churn.fit(X_train_c, y_train_c)
    churn_preds = rf_churn.predict(X_test_c)
    evaluate_model(y_test_c, churn_preds, "Random Forest (Churn Prediction)")

    # Feature Importance for Churn
    # The business always wants to know *why* people are leaving, not just *who*. Let's extract the top drivers.
    feature_imp = pd.Series(rf_churn.feature_importances_, index=churn_features).sort_values(ascending=False)
    print("\nFeature Importances for Churn Prediction:")
    print(feature_imp)

if __name__ == "__main__":
    train_models()
