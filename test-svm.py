import os
from dotenv import load_dotenv  
import pandas as pd

load_dotenv() 
file_path = os.getenv('FILE_PATH_EXEL_XLSX')

dataset = pd.read_excel(file_path)
#data_describe=dataset.describe()
print(dataset.info())
print(dataset.head())
print(dataset.nunique())    