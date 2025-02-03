import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
from typing import Dict, Any, List

load_dotenv()

class MLModel:
    def __init__(self, model, param_grid: Dict[str, Any]):
        self.model = model
        self.param_grid = param_grid
        self.results = []
        self.best_model = None

def load_data() -> pd.DataFrame:
    file_path = os.getenv('FILE_PATH_NORMAL')
    if not file_path:
        raise ValueError("La ruta del archivo no está definida en el archivo .env")
    return pd.read_excel(file_path)

def preprocess_data(dataset: pd.DataFrame) -> tuple:
    X = dataset.iloc[:, 2:]  # Características
    y = dataset['E1']        # Variable objetivo
    return X, y

def train_model(X_train, y_train, model: MLModel, cv_strategy) -> MLModel:
    grid_search = GridSearchCV(
        estimator=model.model,
        param_grid=model.param_grid,
        cv=cv_strategy,
        scoring='accuracy',
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    model.best_model = grid_search.best_estimator_
    return model

def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    y_pred = model.best_model.predict(X_test)
    y_prob = model.best_model.predict_proba(X_test)[:, 1]
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, average='macro', zero_division=0),
    }
    if len(y_test.unique()) == 2:
        metrics['AUC-ROC'] = roc_auc_score(y_test, y_prob)
    else:
        metrics['AUC-ROC'] = roc_auc_score(
            y_test, 
            model.best_model.predict_proba(X_test), 
            multi_class='ovr', average='macro'
        )

    conf_matrix = confusion_matrix(y_test, y_pred)
    specificity = []
    for i in range(conf_matrix.shape[0]):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    
    metrics['Specificity'] = sum(specificity) / len(specificity)
    return metrics

def save_results(results: List[Dict], filename: str) -> None:
    results_df = pd.DataFrame(results)
    results_df.to_excel(filename, index=False)
    print(f"\nResultados exportados a {filename}")

def main():
    seeds = [42, 7, 21, 34, 50, 19, 73, 88, 91, 123, 3, 56, 60, 99, 101]
    results = []
    dataset = load_data()
    X, y = preprocess_data(dataset)
    models = {
        'DecisionTree': MLModel(
            model=DecisionTreeClassifier(),
            param_grid={
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': [None, 'sqrt', 'log2']
            }
        )
    }
    for seed in seeds:
        print(f"\n{'='*40}\nEjecución con semilla: {seed}\n{'='*40}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=seed, 
            stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        for model_name, model in models.items():
            print(f"\nEntrenando {model_name}...")
            try:
                trained_model = train_model(
                    X_train_scaled, 
                    y_train, 
                    model, 
                    cv_strategy=StratifiedKFold(
                        n_splits=5, 
                        shuffle=True, 
                        random_state=seed
                    )
                )
                metrics = evaluate_model(trained_model, X_test_scaled, y_test)
                results.append({
                    'Semilla': seed,
                    'Modelo': model_name,
                    **metrics,
                    'Mejores Parámetros': trained_model.best_model.get_params()
                })
            except Exception as e:
                print(f"Error entrenando {model_name}: {str(e)}")
                continue
    print("\nResumen de resultados:")
    results_df = pd.DataFrame(results)
    print(results_df.groupby('Modelo').mean(numeric_only=True))
    save_results(results, "resultado_DT_2.xlsx")

if __name__ == "__main__":
    main()