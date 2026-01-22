import pandas as pd

# Rutas de los archivos de entrada
input_file = r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\CodeSQLAtípicos\grupo 4_rows_44706_value_columns_18.csv"
input_file_1 = r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\CodeSQLAtípicos\Firstc_df_filtered_R_pivoted.csv"
input_file_2 = r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\CodeSQLAtípicos\Firstdf_filtered_R_pivoted.csv"
output_file = r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\CodeSQLAtípicos\valores_para_revision.xlsx"

# Cargar los archivos CSV sin eliminar columnas
try:
    df_principal = pd.read_csv(input_file, encoding='utf-8')
    df_1 = pd.read_csv(input_file_1, encoding='utf-8')
    df_2 = pd.read_csv(input_file_2, encoding='utf-8')
except Exception as e:
    print(f"Error al leer los archivos CSV: {e}")
    exit()

# Concatenar los archivos df_1 y df_2 en un solo DataFrame df_nuevo
df_nuevo = pd.concat([df_1, df_2], ignore_index=True)

# Unir df_nuevo con df_principal en base a ID_NUMERIC_, manteniendo solo los IDs de df_principal
df_final = df_principal.merge(df_nuevo, on="ID_NUMERIC_", how="left")

# Filtrar y mantener solo las filas donde ATP_x == 0
df_final = df_final[df_final["ATP_x"] == 0]

# Renombrar las columnas eliminando el sufijo _x y VALUE_
df_final.columns = [col.replace("_x", "").replace("VALUE_", "") for col in df_final.columns]

# Reemplazar _y por ATP en los nombres de las columnas
df_final.columns = [col.replace("_y", "_ATP") for col in df_final.columns]

# Definir las primeras cinco columnas en el orden correcto
fixed_columns = [
    "ID_NUMERIC_", "A_RECEP_FECHA_RECIB_", "TF_DEPARTAMENTO_NOMBRE", 
    "TF_MUNICIPIO_NOMBRE", "TF_CENTRO_POBLADO_NOMBRE"
]

# Obtener las demás columnas y ordenarlas alfabéticamente
remaining_columns = [col for col in df_final.columns if col not in fixed_columns]
remaining_columns_sorted = sorted(remaining_columns)

# Definir el orden final de columnas
df_final = df_final[fixed_columns + remaining_columns_sorted]

# Eliminar columnas innecesarias al final del proceso
cols_to_drop = [
    'ATP', 'A_RECEP_FECHA_RECIB__ATP', 'TF_CENTRO_POBLADO_', 'TF_CENTRO_POBLADO__ATP',
    'TF_CULTIVO_', 'TF_CULTIVO__ATP', 'TF_DEPARTAMENTO_', 'TF_DEPARTAMENTO__ATP',
    'TF_LABORATORIO_', 'TF_LABORATORIO__ATP', 'TF_MUNICIPIO_', 'TF_MUNICIPIO__ATP'
]

# También eliminar todas las columnas que comiencen con "BIN_"
cols_to_drop += [col for col in df_final.columns if col.startswith("BIN_")]

df_final = df_final.drop(columns=cols_to_drop, errors='ignore')

# Imprimir las columnas finales después de eliminar las no deseadas
print("Columnas finales después de eliminación:", df_final.columns.tolist())

# Guardar el DataFrame final en un archivo Excel
try:
    df_final.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Archivo guardado exitosamente en: {output_file}")
except Exception as e:
    print(f"Error al guardar el archivo Excel: {e}")
