import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def perform_eda(df, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Correlation Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.savefig(f'{output_dir}/correlation_matrix.png')
    plt.close()
    
    # Distribution of Income vs Creditworthiness
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='AnnualIncome', hue='Creditworthy', kde=True, element='step')
    plt.title('Income Distribution by Creditworthiness')
    plt.savefig(f'{output_dir}/income_distribution.png')
    plt.close()
    
    # Debt vs Income Scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='AnnualIncome', y='TotalDebt', hue='Creditworthy', alpha=0.6)
    plt.title('Annual Income vs Total Debt')
    plt.savefig(f'{output_dir}/income_vs_debt.png')
    plt.close()
    
    print(f"EDA plots saved to {output_dir}")

if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    df = generate_synthetic_data()
    perform_eda(df)
