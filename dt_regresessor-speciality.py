import os  
from dotenv import load_dotenv  
import pandas as pd  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.tree import DecisionTreeClassifier  # Usamos DecisionTreeClassifier en lugar de Regressor
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix  # Métricas de clasificación
import matplotlib.pyplot as plt  
from sklearn.metrics import confusion_matrix

# Cargar las variables de entorno desde .env
load_dotenv()  
file_path = os.getenv('FILE_PATH_NORMAL')  

if not file_path:  
    raise ValueError("La ruta del archivo no está definida en el archivo .env")  

# Leer el dataset
dataset = pd.read_excel(file_path)  

# Selección de características y etiqueta
X = dataset.iloc[:, 2:].drop(columns=['M3','M12'], errors='ignore')  # Características desde la columna 3 en adelante  
y = dataset['E1']        # Etiqueta E1 (continua)

# Convertir la variable objetivo a binaria si es necesario (esto depende de tu caso)
# Asegúrate de que la variable objetivo esté binarizada si usas métricas de clasificación
y = (y > y.median()).astype(int)  # Aquí transformamos E1 en una variable binaria

# Definir las semillas para la ejecución
seeds = [42, 7, 21, 34, 50, 19, 73, 88, 91, 123, 3, 56, 60, 99, 101]
results = []

# Loop sobre las semillas
for seed in seeds:
    print(f"\nEjecución con semilla: {seed}")
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(  
        X, y, test_size=0.2, random_state=seed, stratify=y  
    )  

    # Escalar las características
    scaler = StandardScaler()  
    X_train_scaled = scaler.fit_transform(X_train)  
    X_test_scaled = scaler.transform(X_test)  

    # Crear el modelo de clasificación con el árbol de decisión
    dt_model = DecisionTreeClassifier(random_state=seed)

    # Definir la grilla de parámetros para la búsqueda en cuadrícula
    param_grid = {
        'criterion': ['gini', 'entropy'],  
        'max_depth': [5, 10, 20, None],               
        'min_samples_split': [2, 5, 10],              
        'min_samples_leaf': [1, 2, 4],                
        'max_features': [None, 'sqrt', 'log2']        
    }  

    # Realizar la búsqueda en cuadrícula
    grid_search = GridSearchCV(estimator=dt_model, 
                               param_grid=param_grid, 
                               cv=5, 
                               scoring='accuracy',  
                               verbose=0)  

    # Ajustar el modelo con los datos de entrenamiento
    grid_search.fit(X_train_scaled, y_train)  

    # Obtener el mejor modelo
    best_model = grid_search.best_estimator_  
    y_pred = best_model.predict(X_test_scaled)  

   # Evaluación de las métricas de clasificación
    accuracy = accuracy_score(y_test, y_pred)  # Exactitud
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)  # Precisión
    recall = recall_score(y_test, y_pred, average='macro')  # Recall
    f1 = f1_score(y_test, y_pred, average='macro')  # F1-Score
    auc_score = roc_auc_score(y_test, y_pred) if len(set(y_test)) > 1 else 0  # AUC-ROC

    # Calcular la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Asegurarse de que la matriz de confusión tiene la forma correcta (2x2)
    if cm.shape == (2, 2):
        specificity_macro = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if cm[1, 1] + cm[1, 0] > 0 else 0  # Specificity
    else:
        specificity_macro = 0  # Si no es una matriz 2x2, asignar 0
    
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

# Convertir los resultados en un DataFrame para análisiss
results_df = pd.DataFrame(results)

# Imprimir resumen estadístico
print("\nResultados promedio y desviación estándar:")
print(results_df.describe())

# Exportar los resultados a un archivo Excel
results_df.to_excel("resultados_dt_NORMAL_clasificacion.xlsx", index=False)
print("\nResultados exportados a 'resultados_dt_NORMAL_clasificacion.xlsx'")
