import os
import pandas as pd
from src.data_loader import load_credit_data, generate_synthetic_data, save_data
from src.eda import perform_eda
from src.preprocessing import get_preprocessing_pipeline
from src.model_trainer import train_models, plot_evaluation

def main():
    print("--- Credit Scoring Model Project ---")
    
    # 1. Data Loading / Generation
    data_path = 'data/credit_data.csv'
    if not os.path.exists(data_path):
        print("Generating data...")
        df = generate_synthetic_data(1500)
        save_data(df, data_path)
    else:
        df = pd.read_csv(data_path)
        print("Loaded existing data.")

    # 2. EDA
    print("\nStarting EDA...")
    perform_eda(df)
    
    # 3. Data Split
    X_train, X_test, y_train, y_test = load_credit_data(data_path)
    
    # 4. Preprocessing
    print("\nPreprocessing data...")
    preprocessor = get_preprocessing_pipeline()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    
    # 5. Model Training & Evaluation
    print("\nTraining models...")
    results = train_models(X_train_p, y_train, X_test_p, y_test)
    
    # 6. Plotting Results
    print("\nGenerating evaluation plots...")
    plot_evaluation(results, X_test_p, y_test)
    
    print("\n--- Project Completed Successfully ---")
    print("Check the 'outputs/' folder for visualizations.")

if __name__ == "__main__":
    main()
