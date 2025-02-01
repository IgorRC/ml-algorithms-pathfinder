import os  
from dotenv import load_dotenv  
import pandas as pd  

# Cargar variables de entorno
load_dotenv()  
file_path = os.getenv('FILE_PATH')  

# Verificar que la ruta esté definida
if not file_path:  
    raise ValueError("La ruta del archivo no está definida en el archivo .env")  

# Leer el archivo Excel
dataset = pd.read_excel(file_path)  

# Reescalar los ítems M7 y M8 de 1-6 a 1-5
dataset['M7'] = (1 + ((dataset['M7'] - 1) * (5 - 1)) / (6 - 1)).round().astype(int)
dataset['M8'] = (1 + ((dataset['M8'] - 1) * (5 - 1)) / (6 - 1)).round().astype(int)

# Guardar el DataFrame reescalado en un nuevo archivo Excel
output_file_path = 'dataset_reescalado.xlsx'
dataset.to_excel(output_file_path, index=False)

# Imprimir mensaje de éxito
print(f"Archivo reescalado guardado en: {output_file_path}")
