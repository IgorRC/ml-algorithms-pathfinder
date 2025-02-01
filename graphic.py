import os  
import numpy as np
import pandas as pd  
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Usa un backend sin interfaz gráfica
import matplotlib.pyplot as plt
import tkinter  # Si no lo necesitas, lo puedes eliminar
from scipy.stats import zscore

load_dotenv()  
file_path = os.getenv('FILE_PATH_NORMAL')  

if not file_path:  
    raise ValueError("La ruta del archivo no está definida en el archivo .env")  

# Cargar el dataset
dataset = pd.read_excel(file_path)
print(dataset)
# Calcular la matriz de correlación
correlation_matrix = dataset.corr()
print(correlation_matrix)

# Crear la figura de la gráfica para la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlación')
# Guardar la gráfica como un archivo de imagen
plt.savefig('matriz_correlacion.png')  # Esto guarda la gráfica como un archivo PNG
plt.close()  # Cierra la gráfica después de guardarla, para liberar recursos

# 1. Gráfico de Estadísticas Descriptivas (Histogramas para cada columna numérica)
plt.figure(figsize=(10, 8))
dataset.hist(bins=20, figsize=(10, 8))  # Histograma de las variables numéricas
plt.tight_layout()
plt.savefig('estadisticas_descriptivas.png')  # Guardar el histograma como imagen
plt.close()  # Cerrar la gráfica después de guardarla

# 2. Gráfico de Valores Nulos (Mapa de calor de valores nulos)
plt.figure(figsize=(10, 8))
sns.heatmap(dataset.isnull(), cbar=False, cmap='viridis')
plt.title('Mapa de Valores Nulos')
# Guardar el gráfico como una imagen
plt.savefig('valores_nulos.png')  # Guardar el mapa de calor de valores nulos
plt.close()  # Cerrar la gráfica después de guardarla

# Identificar columnas numéricas
numeric_columns = dataset.select_dtypes(include=[np.number]).columns

# Seleccionar las columnas específicas para el análisis
selected_columns = ['M7','M8']

# Verificar que las columnas existan en el dataset
for col in selected_columns:
    if col not in dataset.columns:
        raise ValueError(f"La columna {col} no existe en el dataset.")

# Crear un diagrama de caja (boxplot) para las columnas seleccionadas
plt.figure(figsize=(12, 6))
boxplot = sns.boxplot(data=dataset[selected_columns], palette="Set2")

# Agregar la media y la mediana a cada caja
for i, column in enumerate(selected_columns):
    column_data = dataset[column].dropna()  # Eliminar valores nulos
    mean = np.mean(column_data)
    median = np.median(column_data)

    # Añadir la media como un círculo rojo
    plt.scatter(i, mean, color='red', s=100, label='Media' if i == 0 else "")
    # Añadir la mediana como un triángulo azul
    plt.scatter(i, median, color='blue', marker='^', s=100, label='Mediana' if i == 0 else "")

# Añadir leyenda
plt.legend(loc='upper right')

# Títulos y etiquetas
plt.title('Diagrama de Caja (Boxplot) con Media y Mediana')
plt.xlabel('Ítems')
plt.ylabel('Valores')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Guardar el gráfico como una imagen
plt.savefig('boxplot_items_likert_with_mean_median.png')
plt.close()

print("El diagrama de caja con media y mediana ha sido guardado como 'boxplot_items_likert_with_mean_median.png'.")

# Datos del ítem M7
data_M7 = [3, 2, 3, 3, 2, 4, 3, 6]  # Ejemplo con el outlier incluido

# Identificar el outlier usando Z-scores
z_scores = zscore(data_M7)
outliers = np.where(np.abs(z_scores) > 2)  # Condición común para identificar outliers
print(f"Índices de los outliers: {outliers}")

# Recalcular métricas sin el outlier
data_cleaned = [x for i, x in enumerate(data_M7) if i not in outliers[0]]
mean_with_outlier = np.mean(data_M7)
mean_without_outlier = np.mean(data_cleaned)

print(f"Media con outlier: {mean_with_outlier}")
print(f"Media sin outlier: {mean_without_outlier}")