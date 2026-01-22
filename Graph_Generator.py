
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# ---------------------------
# Funciones Auxiliares
# ---------------------------
def sanitize_filename(filename):
    """Remueve caracteres no permitidos para formar nombres de archivo válidos."""
    return re.sub(r'[\\/:"*?<>|]', '_', filename)

def load_csv(file_path):
    """Carga un CSV y retorna el DataFrame (finaliza si no se encuentra el archivo)."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Archivo no encontrado en la ruta: {file_path}")
        exit()

def filtrar_columnas(df, columnas_necesarias):
    """Retiene únicamente las columnas indicadas en el DataFrame."""
    return df[columnas_necesarias]

# ---------------------------
# Funciones para las Gráficas
# ---------------------------
# Se omiten ciertos parámetros en las gráficas (por ejemplo, pueden ser redundantes o no de interés)
parametros_excluir = ["clase textural", "relacion de adsorcion de sodio (ras)"]

def grafica_distribucion_valores(df, parametro_col, valor_col, status_col, output_folder):
    """
    Para cada valor único en 'parametro_col' (excepto los excluidos), genera un histograma 
    de 'valor_col' con una línea de densidad (KDE) que estima la distribución subyacente.
    """
    distribucion_folder = os.path.join(output_folder, "distribucion_valores")
    os.makedirs(distribucion_folder, exist_ok=True)
    
    for parametro in df[parametro_col].unique():
        if parametro.lower() in parametros_excluir:
            continue
        
        subset = df[df[parametro_col] == parametro]
        plt.figure(figsize=(10, 6))
        sns.histplot(data=subset, x=valor_col, hue=status_col, kde=True,
                     bins=30, palette="Set1", alpha=0.7)
        plt.title(f"Distribución del parámetro: {parametro}")
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        file_name = sanitize_filename(f"distribucion_{parametro}.png")
        plt.savefig(os.path.join(distribucion_folder, file_name))
        plt.close()

def grafica_comparacion_estados(df, status_cols, output_folder):
    """
    Crea gráficos de barras que comparan el número de muestras para cada estado definido
    en las columnas de 'status_cols'.
    """
    comparacion_folder = os.path.join(output_folder, "comparacion_estados")
    os.makedirs(comparacion_folder, exist_ok=True)
    
    for status_col in status_cols:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=status_col, palette="Set2")
        plt.title(f"Comparación de Estados: {status_col}")
        plt.xlabel("Estado")
        plt.ylabel("Número de muestras")
        file_name = sanitize_filename(f"comparacion_{status_col}.png")
        plt.savefig(os.path.join(comparacion_folder, file_name))
        plt.close()

def grafica_distribucion_departamentos(df, departamento_col, output_folder):
    """
    Genera un gráfico de barras que muestra la distribución de muestras por departamento,
    ordenándolos de mayor a menor según su frecuencia.
    """
    departamento_folder = os.path.join(output_folder, "distribucion_departamentos")
    os.makedirs(departamento_folder, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x=departamento_col, palette="Set3",
                  order=df[departamento_col].value_counts().index)
    plt.title("Distribución por Departamento")
    plt.xlabel("Departamento")
    plt.ylabel("Número de muestras")
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(departamento_folder, "distribucion_departamentos.png"))
    plt.close()

# ---------------------------
# Función Principal
# ---------------------------
def main():
    # Rutas de los archivos CSV
    file_path_activo = r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\CodeSqlAtipicos\Graph\Graph_FirstexportFinalFirst.csv"
    file_path_commit = r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\CodeSqlAtipicos\Graph\Graph_Firstc_exportFinalFirst.csv"
        
    # Cargar y combinar ambos CSV
    df_activo = load_csv(file_path_activo)
    df_commit = load_csv(file_path_commit)
    df_compilado = pd.concat([df_activo, df_commit], ignore_index=True)
    
    # Definir carpeta de salida
    output_folder = os.path.join(os.path.dirname(file_path_activo), "graph")
    os.makedirs(output_folder, exist_ok=True)
    
    # Filtrar columnas necesarias y exportar el DataFrame compilado
    columnas_necesarias = ["ID_NUMERIC", "TF_DEPARTAMENTO", "TF_MUNICIPIO",
                           "TF_CENTRO_POBLADO", "ANALYSIS", "SAMPLE", "STATUS_y",
                           "NAME", "VALUE"]
    df_filtrado = filtrar_columnas(df_compilado, columnas_necesarias)
    csv_export_path = os.path.join(output_folder, "compilado_commit_activo.csv")
    df_filtrado.to_csv(csv_export_path, index=False)
    
    # Filtrar por parámetros deseados en la columna "NAME"
    parametros_deseados = [
        "Porcentaje de arena (% A)",
        "Porcentaje de arcilla (% Ar)",
        "Porcentaje de limo (% L)",
        "Clase textural",
        "pH (1:2,5)",
        "Conductividad eléctrica (CE) (1:5)",
        "Carbono Orgánico (CO)",
        "Materia Orgánica (MO)",
        "Fosforo (P) Disponible (Bray II)",
        "Azufre (S) disponible",
        "Capacidad Interc Catiónico Efect (CICE)",
        "Boro (B) Disponible",
        "Acidez (Al+H)",
        "Aluminio (Al) Intercambiable",
        "Calcio (Ca) disponible",
        "Magnesio (Mg) Disponible",
        "Potasio (K) Disponible",
        "Sodio (Na) Disponible",
        "Hierro (Fe) olsen Disponible",
        "Cobre (Cu) olsen Disponible",
        "Manganeso (Mn) olsen Disponible",
        "Zinc (Zn) olsen Disponible",
        "Saturación de Calcio",
        "Saturación de Magnesio",
        "Saturación de Potasio",
        "Saturación de Sodio",
        "Saturación de Aluminio"
    ]
    df_filtrado_param = df_filtrado[df_filtrado["NAME"].isin(parametros_deseados)]
    csv_param_path = os.path.join(output_folder, "filtrado_parametros.csv")
    df_filtrado_param.to_csv(csv_param_path, index=False)
    
    # Configuración para las gráficas (usando el DataFrame filtrado por parámetros)
    parametro_col = "NAME"
    valor_col = "VALUE"
    status_cols = ["STATUS_x", "STATUS_y"]
    departamento_col = "TF_DEPARTAMENTO"
    
    grafica_distribucion_valores(df_filtrado_param, parametro_col, valor_col, "STATUS_y", output_folder)
    grafica_comparacion_estados(df_filtrado_param, status_cols, output_folder)
    grafica_distribucion_departamentos(df_filtrado_param, departamento_col, output_folder)
    
if __name__ == "__main__":
    main()
