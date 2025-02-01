import os  
from dotenv import load_dotenv  
import pandas as pd  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt  

load_dotenv()  
file_path = os.getenv('FILE_PATH_NORMAL')  

if not file_path:  
    raise ValueError("La ruta del archivo no está definida en el archivo .env")  


dataset = pd.read_excel(file_path)  

X = dataset.iloc[:, 2:]  # Características desde la columna 3 en adelante  
y = dataset['E1']        # Etiqueta E1  

seeds = [42, 7, 21, 34, 50, 19, 73, 88, 91, 123, 3, 56, 60, 99, 101]
results = []

for seed in seeds:
    print(f"\nEjecución con semilla: {seed}")
    
    X_train, X_test, y_train, y_test = train_test_split(  
        X, y, test_size=0.2, random_state=seed, stratify=y  
    )  

    scaler = StandardScaler()  
    X_train_scaled = scaler.fit_transform(X_train)  
    X_test_scaled = scaler.transform(X_test)  
    print(X_train_scaled)
    dt_model = DecisionTreeClassifier(random_state=seed)  

    param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'],  # Función para medir la calidad del split
        'max_depth': [5, 10, 20, None],               # Profundidad máxima del árbol
        'min_samples_split': [2, 5, 10],              # Mínimo de muestras para dividir un nodo
        'min_samples_leaf': [1, 2, 4],                # Mínimo de muestras en hojas
        'max_features': [None, 'sqrt', 'log2']        # Máximo número de características consideradas para dividir
    }  

    grid_search = GridSearchCV(estimator=dt_model, 
                               param_grid=param_grid, 
                               cv=5, 
                               scoring='accuracy', 
                               verbose=0)  

    grid_search.fit(X_train_scaled, y_train)  

    best_model = grid_search.best_estimator_  
    y_pred = best_model.predict(X_test_scaled)  
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]  # Probabilidades de la clase positiva  

    accuracy = accuracy_score(y_test, y_pred)  
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)  
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)  
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)  
    
    if len(y.unique()) == 2:  # Si es un problema binario  
        auc_score = roc_auc_score(y_test, y_prob)
    else:
        auc_score = roc_auc_score(y_test, 
                                  best_model.predict_proba(X_test_scaled), 
                                  multi_class='ovr', 
                                  average='macro')
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    specificity = []
    for i in range(conf_matrix.shape[0]):  # Para cada clase
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    specificity_macro = sum(specificity) / len(specificity)

    # Guardar los resultados junto con los hiperparámetros
    results.append({
        'Semilla': seed,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_score,
        'Specificity': specificity_macro,
        'Best Params': grid_search.best_params_  # Guardar los hiperparámetros
    })

results_df = pd.DataFrame(results)

print("\nResultados promedio y desviación estándar:")
print(results_df.describe())

results_df.to_excel("resultados_dt_NORMAL-reescalado.xlsx", index=False)
print("\nResultados exportados a 'resultados_multiples_ejecuciones_dt_FULL.xlsx'")
