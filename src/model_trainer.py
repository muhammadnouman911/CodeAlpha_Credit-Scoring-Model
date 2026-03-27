from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_prob[:, 1])
        
        results[name] = {
            'model': model,
            'report': report,
            'auc': auc,
            'y_prob': y_prob
        }
        
        print(f"{name} AUC: {auc:.4f}")
    
    return results

def plot_evaluation(results, X_test, y_test, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    
    for name, data in results.items():
        # ROC Curve
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, data['y_prob'][:, 1])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {data["auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc='lower right')
        plt.savefig(f'{output_dir}/roc_{name.lower().replace(" ", "_")}.png')
        plt.close()
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, data['model'].predict(X_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Creditworthy', 'Creditworthy'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix - {name}')
        plt.savefig(f'{output_dir}/cm_{name.lower().replace(" ", "_")}.png')
        plt.close()

    print(f"Evaluation plots saved to {output_dir}")

if __name__ == "__main__":
    from data_loader import load_credit_data
    from preprocessing import get_preprocessing_pipeline
    
    X_train, X_test, y_train, y_test = load_credit_data()
    preprocessor = get_preprocessing_pipeline()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    
    results = train_models(X_train_p, y_train, X_test_p, y_test)
    plot_evaluation(results, X_test_p, y_test)
