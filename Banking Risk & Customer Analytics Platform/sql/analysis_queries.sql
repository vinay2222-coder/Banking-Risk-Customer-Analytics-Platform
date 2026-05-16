-- 1. Top 10 Riskiest Customers (High Loan Amount, Low Credit Score, High Debt-to-Income)
SELECT 
    c.Customer_ID, 
    c.Age, 
    a.Credit_Score, 
    l.Loan_Amount, 
    c.Income,
    ROUND((l.EMI * 12) / NULLIF(c.Income, 0), 2) AS DTI_Ratio
FROM Customers c
JOIN Accounts a ON c.Customer_ID = a.Customer_ID
JOIN Loans l ON c.Customer_ID = l.Customer_ID
WHERE l.Has_Loan = TRUE AND a.Credit_Score < 600
ORDER BY DTI_Ratio DESC, l.Loan_Amount DESC
LIMIT 10;

-- 2. Branch-wise Profit and Default Ratio
WITH BranchMetrics AS (
    SELECT 
        c.Branch,
        COUNT(c.Customer_ID) AS Total_Customers,
        SUM(l.Loan_Amount) AS Total_Loan_Disbursed,
        SUM(CASE WHEN l.Default_Risk = TRUE THEN 1 ELSE 0 END) AS Defaulted_Loans,
        COUNT(l.Loan_ID) AS Total_Loans
    FROM Customers c
    LEFT JOIN Loans l ON c.Customer_ID = l.Customer_ID AND l.Has_Loan = TRUE
    GROUP BY c.Branch
)
SELECT 
    Branch,
    Total_Customers,
    Total_Loan_Disbursed,
    Defaulted_Loans,
    Total_Loans,
    ROUND((Defaulted_Loans * 100.0) / NULLIF(Total_Loans, 0), 2) AS Default_Rate_Pct
FROM BranchMetrics
ORDER BY Default_Rate_Pct DESC;

-- 3. Customer Segmentation based on Spending & Credit
SELECT 
    Card_Type,
    CASE 
        WHEN Credit_Score >= 750 THEN 'Excellent'
        WHEN Credit_Score >= 650 THEN 'Good'
        ELSE 'Poor'
    END AS Credit_Tier,
    COUNT(c.Customer_ID) AS Customer_Count,
    ROUND(AVG(t.Avg_Monthly_Spend), 2) AS Avg_Monthly_Spend
FROM Customers c
JOIN Accounts a ON c.Customer_ID = a.Customer_ID
JOIN Transactions t ON c.Customer_ID = t.Customer_ID
GROUP BY Card_Type, Credit_Tier
ORDER BY Avg_Monthly_Spend DESC;

-- 4. Fraud Detection Report: Anomalous Transactions using Window Functions
WITH TransactionPercentiles AS (
    SELECT 
        t.Customer_ID,
        t.Transaction_Count,
        t.Avg_Monthly_Spend,
        t.Fraud_Flag,
        PERCENT_RANK() OVER(ORDER BY t.Avg_Monthly_Spend) AS Spend_Percentile,
        PERCENT_RANK() OVER(ORDER BY t.Transaction_Count) AS Tx_Percentile
    FROM Transactions t
)
SELECT 
    Customer_ID,
    Transaction_Count,
    Avg_Monthly_Spend,
    Fraud_Flag
FROM TransactionPercentiles
WHERE Fraud_Flag = TRUE OR (Spend_Percentile > 0.99 AND Tx_Percentile > 0.99)
ORDER BY Avg_Monthly_Spend DESC;

-- 5. Customer Churn Analysis
SELECT 
    a.Account_Type,
    COUNT(c.Customer_ID) AS Total_Customers,
    SUM(CASE WHEN cs.Customer_Churn = TRUE THEN 1 ELSE 0 END) AS Churned_Customers,
    ROUND((SUM(CASE WHEN cs.Customer_Churn = TRUE THEN 1 ELSE 0 END) * 100.0) / COUNT(c.Customer_ID), 2) AS Churn_Rate_Pct
FROM Customers c
JOIN Accounts a ON c.Customer_ID = a.Customer_ID
JOIN Customer_Status cs ON c.Customer_ID = cs.Customer_ID
GROUP BY a.Account_Type
ORDER BY Churn_Rate_Pct DESC;

-- 6. Customer Lifetime Value (CLV) Calculation (Simplified Proxy)
SELECT 
    c.Customer_ID,
    c.Age,
    c.Occupation,
    a.Account_Balance,
    t.Avg_Monthly_Spend,
    (t.Avg_Monthly_Spend * 12 * 5) AS Estimated_5Yr_CLV
FROM Customers c
JOIN Accounts a ON c.Customer_ID = a.Customer_ID
JOIN Transactions t ON c.Customer_ID = t.Customer_ID
JOIN Customer_Status cs ON c.Customer_ID = cs.Customer_ID
WHERE cs.Customer_Churn = FALSE
ORDER BY Estimated_5Yr_CLV DESC
LIMIT 50;
