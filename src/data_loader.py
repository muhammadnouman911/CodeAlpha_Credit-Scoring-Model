import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def generate_synthetic_data(n_samples=1000):
    """
    Generates a synthetic credit dataset with features:
    - Annual Income
    - Total Debt
    - Payment History (0: Poor, 1: Average, 2: Good)
    - Age
    - Credit Utilization (ratio of debt to income)
    - Target: Creditworthy (0: No, 1: Yes)
    """
    np.random.seed(42)
    
    income = np.random.normal(50000, 15000, n_samples).clip(20000, 150000)
    debt = np.random.normal(15000, 10000, n_samples).clip(0, 100000)
    age = np.random.randint(18, 70, n_samples)
    payment_history = np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.3, 0.5])
    
    # Simple logic for creditworthiness
    # Higher income, lower debt, and better payment history increase chances
    score = (income / 10000) - (debt / 5000) + (payment_history * 2) + (age / 10)
    # Add some noise
    score += np.random.normal(0, 2, n_samples)
    
    creditworthy = (score > np.median(score)).astype(int)
    
    df = pd.DataFrame({
        'AnnualIncome': income,
        'TotalDebt': debt,
        'Age': age,
        'PaymentHistory': payment_history,
        'Creditworthy': creditworthy
    })
    
    return df

def save_data(df, path='data/credit_data.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Data saved to {path}")

def load_credit_data(path='data/credit_data.csv'):
    if not os.path.exists(path):
        print("Data file not found. Generating synthetic data...")
        df = generate_synthetic_data()
        save_data(df, path)
    else:
        df = pd.read_csv(path)
    
    X = df.drop('Creditworthy', axis=1)
    y = df['Creditworthy']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    df = generate_synthetic_data()
    save_data(df)
    print(df.head())
