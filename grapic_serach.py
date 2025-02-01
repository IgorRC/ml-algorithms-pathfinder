import os  
import numpy as np
import pandas as pd  
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib
matplotlib.use('Agg')  # Usa un backend sin interfaz gráfica
import matplotlib.pyplot as plt

# Configuración inicial
seed = 42
print(f"\nEjecución con semilla: {seed}")

load_dotenv()  
file_path = os.getenv('FILE_PATH_NORMAL')  

if not file_path:  
    raise ValueError("La ruta del archivo no está definida en el archivo .env")  

# Cargar datos desde un archivo
dataset = pd.read_excel(file_path)  

# Preparar las características (X) y etiquetas (y)
X = dataset.iloc[:, 2:]  # Características desde la columna 3 en adelante  
y = dataset['E1']        # Etiqueta E1  

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.2, random_state=seed, stratify=y  
)  

# Escalar las características
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

# Modelo base
dt_model = DecisionTreeClassifier(random_state=seed)

# Definir la cuadrícula de hiperparámetros
param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],  # Función para medir la calidad del split
    'max_depth': [5, 10, 20, None],               # Profundidad máxima del árbol
    'min_samples_split': [2, 5, 10],              # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 4],                # Mínimo de muestras en hojas
    'max_features': [None, 'sqrt', 'log2']        # Máximo número de características consideradas para dividir
}  

# Realizar GridSearchCV
grid_search = GridSearchCV(estimator=dt_model, 
                           param_grid=param_grid, 
                           cv=5, 
                           scoring='accuracy', 
                           verbose=0)  

grid_search.fit(X_train_scaled, y_train)  

# Modelo óptimo
best_model = grid_search.best_estimator_  
print(f"\nMejores hiperparámetros encontrados: {grid_search.best_params_}")

# Extraer resultados del GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)

# Graficar el desempeño promedio para cada combinación de hiperparámetros
plt.figure(figsize=(12, 6))
plt.plot(results.index, results['mean_test_score'], marker='o', linestyle='-', label='Mean Accuracy')
plt.xlabel('Combinaciones de Hiperparámetros')
plt.ylabel('Precisión Promedio')
plt.title('GridSearchCV - Desempeño vs. Hiperparámetros')
plt.legend()
plt.grid()

# Guardar la figura en un archivo en lugar de mostrarla
plt.savefig('gridsearch_performance.png')
print("La gráfica fue guardada como 'gridsearch_performance.png'.")
