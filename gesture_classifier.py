import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load and preprocess data
def load_data(filepath):
    data = pd.read_csv(filepath)
    # Assuming last column is the label
    X = data.iloc[:, :-1]  
    y = data.iloc[:, -1]
    return X, y

# Normalize features
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Store results
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        results[name]['CV_Score'] = cv_scores.mean()
    
    return results

def plot_metrics_comparison(results):
    # Convert results to DataFrame for easier plotting
    df_results = pd.DataFrame(results).T
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
    
    for ax, metric, color in zip(axes.ravel(), metrics, colors):
        sns.barplot(x=df_results.index, y=df_results[metric], ax=ax, color=color)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Plot cross-validation scores
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[results[model]['CV_Score'] for model in results.keys()])
    plt.title('Cross-validation Scores Distribution')
    plt.xticks(range(len(results)), results.keys(), rotation=45)
    plt.ylabel('Score')
    plt.tight_layout()
    plt.show()

def save_model(model, scaler, model_name):
    joblib.dump(model, f'./models/{model_name}_model.pkl')
    joblib.dump(scaler, f'./models/{model_name}_scaler.pkl')

def main():
    # Load and preprocess data
    X, y = load_data('./datasets/dataset.csv')
    X
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Preprocess data
    X_train_scaled, scaler = preprocess_data(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate all models
    results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Save the KNN model and its scaler
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    save_model(knn_model, scaler, 'knn')
    
    # Print detailed results
    print("\nDetailed Results:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Plot comparisons
    plot_metrics_comparison(results)

if __name__ == "__main__":
    main()
