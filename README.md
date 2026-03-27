# Credit Scoring Model

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)

## Objective
The goal of this project is to build a classification model to predict an individual's creditworthiness based on their financial history. This is a critical task for financial institutions to minimize default risk.

## Features
The model uses several key financial indicators:
- **Annual Income**: The total yearly earnings of the individual.
- **Total Debt**: Current outstanding liabilities.
- **Payment History**: Historical record of on-time vs. late payments.
- **Age**: Demographic information.
- **Creditworthy**: The target variable (1 for Yes, 0 for No).

## Project Structure
```text
.
├── src/
│   ├── data_loader.py    # Synthetic data generation and loading
│   ├── eda.py            # Exploratory Data Analysis
│   ├── preprocessing.py  # Feature engineering and scaling
│   └── model_trainer.py  # Model training and evaluation logic
├── data/                 # Dataset folder (generated)
├── outputs/              # Evaluation plots and visualizations
├── main.py               # Main entry point
└── requirements.txt      # Project dependencies
```

## Implementation Details

### 1. Data Generation
Since a real-world credit dataset is sensitive, this project includes a synthetic data generator that creates realistic financial profiles with noise to simulate real-world conditions.

### 2. Exploratory Data Analysis (EDA)
The pipeline automatically generates:
- Correlation matrices to identify feature importance.
- Distribution plots for income and debt relative to creditworthiness.

### 3. Machine Learning Models
We implement and compare three robust classification algorithms:
- **Logistic Regression**: Linear model for binary classification.
- **Decision Tree**: Non-linear model for easier interpretation.
- **Random Forest**: Ensemble method for improved accuracy and reduced variance.

### 4. Evaluation Metrics
Models are assessed using:
- **Precision, Recall, and F1-Score**
- **ROC-AUC Score**: To measure the ability of the model to distinguish between classes.
- **Confusion Matrices**: To visualize true vs. false positives/negatives.

## How to Get Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/muhammadnouman911/CodeAlpha_Credit-Scoring-Model.git
   cd CodeAlpha_Credit-Scoring-Model
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline
Run the following command to execute the entire end-to-end process:
```bash
python main.py
```
After execution, results will be available in the `outputs/` directory.

---
*Created as part of the CodeAlpha Internship project.*
