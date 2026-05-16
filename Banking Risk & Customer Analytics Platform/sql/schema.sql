CREATE TABLE IF NOT EXISTS Customers (
    Customer_ID VARCHAR(20) PRIMARY KEY,
    Age INT,
    Gender VARCHAR(10),
    Income DECIMAL(15,2),
    Occupation VARCHAR(50),
    Region VARCHAR(50),
    Branch VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS Accounts (
    Account_ID SERIAL PRIMARY KEY,
    Customer_ID VARCHAR(20),
    Account_Type VARCHAR(20),
    Account_Balance DECIMAL(15,2),
    Credit_Score INT,
    Card_Type VARCHAR(20),
    FOREIGN KEY (Customer_ID) REFERENCES Customers(Customer_ID)
);

CREATE TABLE IF NOT EXISTS Loans (
    Loan_ID SERIAL PRIMARY KEY,
    Customer_ID VARCHAR(20),
    Has_Loan BOOLEAN,
    Loan_Amount DECIMAL(15,2),
    Interest_Rate DECIMAL(5,2),
    EMI DECIMAL(10,2),
    Loan_Status VARCHAR(20),
    Default_Risk BOOLEAN,
    FOREIGN KEY (Customer_ID) REFERENCES Customers(Customer_ID)
);

CREATE TABLE IF NOT EXISTS Transactions (
    Transaction_ID SERIAL PRIMARY KEY,
    Customer_ID VARCHAR(20),
    Transaction_Count INT,
    Avg_Monthly_Spend DECIMAL(15,2),
    Last_Transaction_Date DATE,
    Fraud_Flag BOOLEAN,
    FOREIGN KEY (Customer_ID) REFERENCES Customers(Customer_ID)
);

CREATE TABLE IF NOT EXISTS Customer_Status (
    Status_ID SERIAL PRIMARY KEY,
    Customer_ID VARCHAR(20),
    Customer_Churn BOOLEAN,
    FOREIGN KEY (Customer_ID) REFERENCES Customers(Customer_ID)
);
