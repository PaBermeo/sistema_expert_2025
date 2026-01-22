""" Descripción del código
1. Propósito:
Entrenar dos modelos de clasificación supervisada (Random Forest) para predecir si una muestra es atípica (ATP = 1):
Usando solo las columnas VALUE_*.
Combinando las columnas VALUE_* y BIN_VALUE_*.
2. Pasos principales:
Cargar datos:

Lee el archivo FirstWork.csv con información de muestras.
Selecciona las columnas VALUE_* (valores numéricos) y BIN_VALUE_* (indicadores binarios).
Define ATP como la variable objetivo.
Entrenamiento de modelos:

Modelo 1: Entrenado solo con VALUE_*.
Modelo 2: Entrenado con VALUE_* y BIN_VALUE_*.
Ambos modelos usan Random Forest con 100 árboles.
Evaluación de modelos:

Genera reportes de clasificación y matrices de confusión para evaluar el desempeño de cada modelo.
Análisis de importancia de características:

Exporta las características más importantes para ambos modelos:
Feature_Importances_Value.csv: Importancia de las columnas VALUE_*.
Feature_Importances_Combined.csv: Importancia de las columnas VALUE_* + BIN_VALUE_*.
Archivos generados:
Feature_Importances_Value.csv:

Importancia de las características usadas en el modelo entrenado solo con VALUE_*.
Feature_Importances_Combined.csv:

Importancia de las características usadas en el modelo combinado (VALUE_* + BIN_VALUE_*).
Puntos importantes:
Entrada:

Archivo FirstWork.csv con columnas VALUE_*, BIN_VALUE_* y ATP.
Salida:

Resultados de evaluación (clasificación y confusión).
Archivos CSV con las características más importantes.
Limitación:

El modelo combinado es más preciso porque utiliza BIN_VALUE_*, pero depende de estas columnas, que podrían no estar disponibles en futuros datos.. """

""" 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
# este modelo tiene en cuenta las columnas VALUE_* y BIN_VALUE_* 
# para predecir la variable objetivo ATP, no es muy preciso

file_path = r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\CodeSqlAtipicos\FirstWork.csv"

data = pd.read_csv(file_path)


# Filtrar columnas para cada enfoque
value_columns = [col for col in data.columns if col.startswith("VALUE_")]
bin_columns = [col for col in data.columns if col.startswith("BIN_VALUE_")]
target = "ATP"  # Variable objetivo

# **Enfoque 1: Usar solo `VALUE_*` como características**
X_value = data[value_columns]
y = data[target]

# **Enfoque 2: Combinar `VALUE_*` y `BIN_VALUE_*`**
X_combined = data[value_columns + bin_columns]

# Dividir los datos en entrenamiento y prueba para ambos enfoques
X_train_value, X_test_value, y_train, y_test = train_test_split(X_value, y, test_size=0.2, random_state=42)
X_train_combined, X_test_combined, _, _ = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Entrenar el modelo para ambos enfoques
model_value = RandomForestClassifier(n_estimators=100, random_state=42)
model_value.fit(X_train_value, y_train)

model_combined = RandomForestClassifier(n_estimators=100, random_state=42)
model_combined.fit(X_train_combined, y_train)

# Realizar predicciones
y_pred_value = model_value.predict(X_test_value)
y_pred_combined = model_combined.predict(X_test_combined)

# Evaluar el modelo para cada enfoque
print("Resultados para `VALUE_*`:")
print(confusion_matrix(y_test, y_pred_value))
print(classification_report(y_test, y_pred_value))

print("\nResultados para combinación `VALUE_*` + `BIN_VALUE_*`:")
print(confusion_matrix(y_test, y_pred_combined))
print(classification_report(y_test, y_pred_combined))

# Exportar las características más importantes de ambos modelos
feature_importances_value = pd.DataFrame({
    'Feature': X_value.columns,
    'Importance': model_value.feature_importances_
}).sort_values(by='Importance', ascending=False)

feature_importances_combined = pd.DataFrame({
    'Feature': X_combined.columns,
    'Importance': model_combined.feature_importances_
}).sort_values(by='Importance', ascending=False)

feature_importances_value.to_csv("Feature_Importances_Value.csv", index=False)
feature_importances_combined.to_csv("Feature_Importances_Combined.csv", index=False)
 """
 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Cargar datos del archivo principal FirstWork.csv
file_path = r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\CodeSqlAtipicos\FirstWork.csv"
data = pd.read_csv(file_path)

# Seleccionar columnas `VALUE_*` y `BIN_VALUE_*`
value_columns = [col for col in data.columns if col.startswith("VALUE_")]
bin_columns = [col for col in data.columns if col.startswith("BIN_VALUE_")]
target = "ATP"  # Variable objetivo

# Dividir los datos en entrenamiento y prueba
X_value = data[value_columns]
X_combined = data[value_columns + bin_columns]
y = data[target]

X_train_value, X_test_value, y_train, y_test = train_test_split(X_value, y, test_size=0.2, random_state=42)
X_train_combined, X_test_combined, _, _ = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Entrenar ambos modelos
model_value = RandomForestClassifier(n_estimators=100, random_state=42)
model_value.fit(X_train_value, y_train)

model_combined = RandomForestClassifier(n_estimators=100, random_state=42)
model_combined.fit(X_train_combined, y_train)

# Guardar los modelos entrenados
joblib.dump(model_value, "model_value.pkl")
joblib.dump(model_combined, "model_combined.pkl")

# Función para analizar nuevos datos con ambos modelos
def analizar_con_modelos(input_csv, output_csv_value, output_csv_combined):
    """
    Analiza un archivo CSV usando dos modelos:
    - `model_value.pkl`: Entrenado solo con columnas VALUE_*.
    - `model_combined.pkl`: Entrenado con VALUE_* + BIN_VALUE_*.
    Genera dos archivos de salida con las predicciones.
    """
    # Cargar modelos entrenados
    model_value = joblib.load("model_value.pkl")
    model_combined = joblib.load("model_combined.pkl")
    
    # Cargar el archivo de entrada
    new_data = pd.read_csv(input_csv)
    
    # Verificar columnas necesarias
    value_columns = [col for col in new_data.columns if col.startswith("VALUE_")]
    bin_columns = [col for col in new_data.columns if col.startswith("BIN_VALUE_")]
    
    if not value_columns:
        raise ValueError("El archivo de entrada no contiene columnas `VALUE_*` necesarias para las predicciones.")
    
    # Predicciones con el modelo `VALUE_*`
    X_new_value = new_data[value_columns]
    new_data["ATP_Value"] = model_value.predict(X_new_value)
    
    # Predicciones con el modelo `VALUE_* + BIN_VALUE_*`
    if bin_columns:
        X_new_combined = new_data[value_columns + bin_columns]
        new_data["ATP_Combined"] = model_combined.predict(X_new_combined)
    else:
        print("Advertencia: El archivo no contiene columnas `BIN_VALUE_*`. Solo se realizará predicción con `VALUE_*`.")
    
    # Guardar los resultados
    new_data.to_csv(output_csv_value if bin_columns else output_csv_combined, index=False)
    print(f"Análisis completado. Archivo de salida guardado en: {output_csv_value if bin_columns else output_csv_combined}")

# Analizar el archivo FilteredFirstWork.csv
input_csv = r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\CodeSqlAtipicos\FilteredFirstWork.csv"
output_csv_value = r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\CodeSqlAtipicos\Predictions_Value.csv"
output_csv_combined = r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\CodeSqlAtipicos\Predictions_Combined.csv"

analizar_con_modelos(input_csv, output_csv_value, output_csv_combined)
