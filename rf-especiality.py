import os  
from dotenv import load_dotenv  
import pandas as pd  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt  

# Cargar variables desde el archivo .env
load_dotenv()  
file_path = os.getenv('FILE_PATH_NORMAL')  

if not file_path:  
    raise ValueError("La ruta del archivo no está definida en el archivo .env")  

# Cargar el dataset
dataset = pd.read_excel(file_path)  

# Separar las características (X) y la etiqueta (y)
X = dataset.iloc[:, 2:]  # Características desde la columna 3 en adelante  
y = dataset['E1']        # Etiqueta E1  
#informe :
# Lista de semillas para múltiples ejecuciones
seeds = [42, 7, 21, 34, 50, 19, 73, 88, 91, 123, 3, 56, 60, 99, 101]

# Resultados acumulados
results = []

# Bucle de ejecuciones
for seed in seeds:
    print(f"\nEjecución con semilla: {seed}")
    
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(  
        X, y, test_size=0.2, random_state=seed, stratify=y  
    )  

    # Escalar las características (opcional para Random Forest)
    scaler = StandardScaler()  
    X_train_scaled = scaler.fit_transform(X_train)  
    X_test_scaled = scaler.transform(X_test)  
    #redaccion del informe

    # Configurar el modelo base
    rf_model = RandomForestClassifier(random_state=seed)  
    # Definir el espacio de búsqueda de hiperparámetros
    param_grid = {
        'n_estimators': [50, 100, 200],       # Número de árboles
        'max_depth': [5, 10, None],          # Profundidad máxima
        'min_samples_split': [2, 5, 10],     # Mínimo de muestras para dividir un nodo
        'min_samples_leaf': [1, 2, 4],       # Mínimo de muestras en hojas
        'max_features': ['sqrt', 'log2']     # Número de características consideradas para dividir
    }  
    

    # Configurar la búsqueda de hiperparámetros con validación cruzada
    grid_search = GridSearchCV(estimator=rf_model, 
                               param_grid=param_grid, 
                               cv=5, 
                               scoring='accuracy', 
                               verbose=2)  

    # Ajustar el modelo a los datos de entrenamiento
    grid_search.fit(X_train_scaled, y_train)  

    # Evaluar el mejor modelo en el conjunto de prueba
    best_model = grid_search.best_estimator_  
    y_pred = best_model.predict(X_test_scaled)  
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]  # Probabilidades de la clase positiva  

    # Calcular métricas principales
    accuracy = accuracy_score(y_test, y_pred)  
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)  
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)  
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)  
    
    # Calcular AUC-ROC (binario o multiclase)
    if len(y.unique()) == 2:  # Si es un problema binario  
        auc_score = roc_auc_score(y_test, y_prob)
    else:
        auc_score = roc_auc_score(y_test, 
                                  best_model.predict_proba(X_test_scaled), 
                                  multi_class='ovr', 
                                  average='macro')
    
    # Calcular Especificidad
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



# Convertir resultados a DataFrame
results_df = pd.DataFrame(results)

# Mostrar estadísticas descriptivas
print("\nResultados promedio y desviación estándar:")
print(results_df.describe())

# Exportar resultados a Excel
results_df.to_excel("resultados_rf_10ET.xlsx", index=False)
print("\nResultados exportados a 'resultados_multiples_ejecuciones_hiperparametros.xlsx'")
