"""Descripción del código
1. Propósito:
Filtrar datos, entrenar un modelo y analizar nuevas muestras para identificar si una muestra es atípica (ATP = 1).
2. Pasos principales:
Filtrar datos:

Se filtran registros donde ATP = 0.
Se guarda el archivo filtrado como FilteredFirstWork.csv.
Entrenamiento del modelo:

Se entrena un modelo Random Forest usando las columnas VALUE_* como características y ATP como etiqueta.
El modelo se guarda en atp_model.pkl.
Análisis de nuevas muestras:

Usa la función analizar_nuevas_muestras() para predecir ATP en un archivo de entrada (FilteredFirstWork.csv).
Genera un archivo de salida con predicciones (FilteredPredictions.csv).
Archivos generados:
FilteredFirstWork.csv: Datos filtrados (ATP = 0) con todas las columnas.
atp_model.pkl: Modelo entrenado.
FilteredPredictions.csv: Predicciones de ATP para nuevas muestras.  """


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Cargar datos del archivo FirstWork.csv
file_path = r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\CodeSqlAtipicos\FirstWork.csv"
data = pd.read_csv(file_path)

# 2. Filtrar registros donde ATP = 0
data_atp_0 = data[data["ATP"] == 0]

# 3. Seleccionar columnas que comiencen con VALUE_
value_columns = [col for col in data.columns if col.startswith("VALUE_")]
filtered_data = data_atp_0[value_columns]

# 4. Guardar el nuevo DataFrame filtrado
filtered_csv_path = r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\CodeSqlAtipicos\FilteredFirstWork.csv"
filtered_data.to_csv(filtered_csv_path, index=False)
print(f"Nuevo DataFrame filtrado guardado en: {filtered_csv_path}")
print(f"Número de registros donde ATP = 0: {len(filtered_data)}")

# 5. Entrenar el modelo con los datos originales
X = data[value_columns]  # Características
y = data["ATP"]  # Variable objetivo

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Guardar el modelo entrenado
model_path = "atp_model.pkl"
joblib.dump(model, model_path)

# 6. Función para analizar nuevas muestras
def analizar_nuevas_muestras(input_csv, output_csv, model_path="atp_model.pkl"):
    """
    Analiza un archivo CSV con nuevas muestras y genera un archivo de salida con predicciones ATP.
    """
    # Cargar el modelo entrenado
    model = joblib.load(model_path)
    
    # Cargar el archivo de entrada
    new_data = pd.read_csv(input_csv)
    
    # Verificar si las columnas `VALUE_*` están presentes
    value_columns = [col for col in new_data.columns if col.startswith("VALUE_")]
    if not value_columns:
        raise ValueError("El archivo de entrada no contiene columnas `VALUE_*` necesarias para las predicciones.")
    
    # Seleccionar las columnas `VALUE_*`
    X_new = new_data[value_columns]
    
    # Realizar predicciones
    new_data["ATP"] = model.predict(X_new)
    
    # Guardar el archivo con las predicciones
    new_data.to_csv(output_csv, index=False)
    print(f"Análisis completado. Archivo de salida guardado en: {output_csv}")

# 7. Analizar nuevas muestras usando el DataFrame filtrado
input_csv = filtered_csv_path
output_csv = r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\CodeSqlAtipicos\FilteredPredictions.csv"

analizar_nuevas_muestras(input_csv, output_csv)
