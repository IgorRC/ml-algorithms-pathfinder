import os  
from dotenv import load_dotenv  
import pandas as pd  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import (confusion_matrix, 
                             accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             classification_report, 
                             roc_auc_score, 
                             roc_curve,
                             auc)
import matplotlib.pyplot as plt  

# Cargar variables desde el archivo .env
load_dotenv()  
file_path = os.getenv('FILE_PATH')  

if not file_path:  
    raise ValueError("La ruta del archivo no está definida en el archivo .env")  

# Cargar el dataset
dataset = pd.read_excel(file_path)  

# Separar las características (X) y la etiqueta (y)
X = dataset.iloc[:, 2:]  # Características desde la columna 3 en adelante  
y = dataset['E1']        # Etiqueta E1  

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.2, random_state=42, stratify=y  
)  

# Escalar las características (opcional para Random Forest, pero se incluye por consistencia)
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

# Configurar el modelo base
rf_model = RandomForestClassifier(random_state=42)  

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

# Mostrar los mejores parámetros y la mejor puntuación
print("Mejores Hiperparámetros:", grid_search.best_params_)  
print("Mejor Puntuación:", grid_search.best_score_)  

# Evaluar el mejor modelo en el conjunto de prueba
best_model = grid_search.best_estimator_  
y_pred = best_model.predict(X_test_scaled)  
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]  # Probabilidades de la clase positiva  

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)  
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)  
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)  
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)  

# Calcular AUC-ROC (binario o multiclase)
if len(y.unique()) == 2:  # Si es un problema binario  
    auc_score = roc_auc_score(y_test, y_prob)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Curva ROC
else:
    auc_score = roc_auc_score(y_test, 
                              best_model.predict_proba(X_test_scaled), 
                              multi_class='ovr', 
                              average='macro')
    print(f"AUC-ROC (promedio macro): {auc_score:.2f}")     

# Imprimir resultados
print(f"Accuracy: {accuracy:.2f}")  
print(f"Precision: {precision:.2f}")  
print(f"Recall (Sensibilidad): {recall:.2f}")  
print(f"F1 Score: {f1:.2f}")  
print(f"ROC-AUC Score: {auc_score:.2f}")
print("Reporte de Clasificación:")  
print(classification_report(y_test, y_pred))  

# Visualización de la curva ROC (para problemas binarios)
if len(y.unique()) == 2:  
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Línea diagonal
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
