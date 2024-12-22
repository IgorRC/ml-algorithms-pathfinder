import os  
from dotenv import load_dotenv  
import pandas as pd  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.tree import DecisionTreeClassifier  
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

load_dotenv()  
file_path = os.getenv('FILE_PATH')  

if not file_path:  
    raise ValueError("La ruta del archivo no está definida en el archivo .env")  


dataset = pd.read_excel(file_path)  


X = dataset.iloc[:, 2:]   
y = dataset['E1']      


X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.2, random_state=42, stratify=y  
)  


scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  


dt_model = DecisionTreeClassifier(random_state=42)  


param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],  
    'max_depth': [5, 10, 20, None],               
    'min_samples_split': [2, 5, 10],             
    'min_samples_leaf': [1, 2, 4],                
    'max_features': [None, 'sqrt', 'log2']        
}  


grid_search = GridSearchCV(estimator=dt_model, 
                           param_grid=param_grid, 
                           cv=5, 
                           scoring='accuracy', 
                           verbose=2)  


grid_search.fit(X_train_scaled, y_train)  


print("Mejores Hiperparámetros:", grid_search.best_params_)  
print("Mejor Puntuación:", grid_search.best_score_)  


best_model = grid_search.best_estimator_  
y_pred = best_model.predict(X_test_scaled)  
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]  


accuracy = accuracy_score(y_test, y_pred)  
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)  
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)  
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)  


if len(y.unique()) == 2:  
    auc_score = roc_auc_score(y_test, y_prob)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Curva ROC
else:
    auc_score = roc_auc_score(y_test, 
                              best_model.predict_proba(X_test_scaled), 
                              multi_class='ovr', 
                              average='macro')
    print(f"AUC-ROC (promedio macro): {auc_score:.2f}")     


print(f"Accuracy: {accuracy:.2f}")  
print(f"Precision: {precision:.2f}")  
print(f"Recall (Sensibilidad): {recall:.2f}")  
print(f"F1 Score: {f1:.2f}")  
print(f"ROC-AUC Score: {auc_score:.2f}")
print("Reporte de Clasificación:")  
print(classification_report(y_test, y_pred))  


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
