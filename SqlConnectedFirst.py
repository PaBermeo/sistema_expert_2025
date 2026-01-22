

import pandas as pd
import datetime
import pyodbc

def establecer_conexion():
    # Función para establecer la conexión a la base de datos
    conn = pyodbc.connect(
        r"DRIVER={SQL Server};SERVER=COMOSPSPSQL01\SQL2019LS;DATABASE=LimsSampleManager;UID=UserSMInfo;PWD=BlIhFEJcAXpf6aFDZBLS"
    )
    return conn


def ejecutar_consulta(conn, consulta):
    # Función para ejecutar una consulta en la base de datos
    cursor = conn.cursor()
    cursor.execute(consulta)
    rows = cursor.fetchall()
    return rows


def cerrar_conexion(conn):
    # Función para cerrar la conexión a la base de datos
    conn.close()


def crear_dataframe(rows, columnas):
    # Función para crear un dataframe a partir de filas y columnas
    df = pd.DataFrame([tuple(t) for t in rows], columns=columnas)
    return df


def filtrar_dataframe(df, columna, valores):
    # Función para filtrar un dataframe por los valores de una columna
    df_filtrado = df[df[columna].isin(valores)]
    return df_filtrado


def guardar_dataframe(df, ruta):
    # Función para guardar un dataframe en un archivo CSV
    df.to_csv(ruta, index=False)


def mainActive():
    # Establecer conexión a la base de datos
    conn = establecer_conexion()

    # Consulta para obtener los datos de la tabla SAMPLE
    consulta_sample = "SELECT ID_NUMERIC, ID_TEXT, A_RECEP_FECHA_RECIB, STATUS, TF_DEPARTAMENTO, TF_MUNICIPIO, TF_CENTRO_POBLADO , TF_CULTIVO, TF_LABORATORIO FROM SAMPLE"
    rows_sample = ejecutar_consulta(conn, consulta_sample)

    # Consulta para obtener los datos de la tabla TEST
    consulta_test = "SELECT TEST_NUMBER, ANALYSIS, SAMPLE, STATUS FROM TEST"
    rows_test = ejecutar_consulta(conn, consulta_test)

    # Consulta para obtener los datos de la tabla RESULT
    consulta_result = "SELECT TEST_NUMBER, NAME, VALUE, TEXT, REP_CONTROL FROM RESULT"
    rows_result = ejecutar_consulta(conn, consulta_result)

    # Consulta para obtener los datos de la tabla TF_DEPARTAMENTO
    consulta_departamento = "SELECT [IDENTITY], NOMBRE FROM TF_DEPARTAMENTO"
    rows_departamento = ejecutar_consulta(conn, consulta_departamento)

    # Consulta para obtener los datos de la tabla TF_MUNICIPIO
    consulta_municipio = "SELECT [IDENTITY], NOMBRE FROM TF_MUNICIPIO"
    rows_municipio = ejecutar_consulta(conn, consulta_municipio)

    # Consulta para obtener los datos de la tabla TF_POBLADO
    consulta_poblado = "SELECT [IDENTITY], NOMBRE FROM TF_POBLADO"
    rows_poblado = ejecutar_consulta(conn, consulta_poblado)

    # Cerrar la conexión a la base de datos
    cerrar_conexion(conn)

    # Definir las columnas del dataframe para la tabla SAMPLE
    columnas_sample = [
        "ID_NUMERIC",
        "ID_TEXT",
        "A_RECEP_FECHA_RECIB",
        "STATUS",
        "TF_DEPARTAMENTO",
        "TF_MUNICIPIO",
        "TF_CENTRO_POBLADO",
        "TF_CULTIVO",
        "TF_LABORATORIO",
        
    ]

    # Crear el dataframe para la tabla SAMPLE
    df_sample = pd.DataFrame([tuple(t) for t in rows_sample], columns=columnas_sample)

    # Filtrar el dataframe de la tabla SAMPLE por el valor de la columna TF_LABORATORIO
    df_sample = filtrar_dataframe(df_sample, "TF_LABORATORIO", ["LQA"])

    # Filtrar el dataframe de la tabla SAMPLE por el valor de la columna STATUS
    df_sample = filtrar_dataframe(df_sample, "STATUS", ["A"])

    # Convertir la columna ID_NUMERIC del dataframe de la tabla SAMPLE a tipo numérico
    df_sample["ID_NUMERIC"] = pd.to_numeric(df_sample["ID_NUMERIC"])

    # Definir las columnas del dataframe para la tabla TEST
    columnas_test = ["TEST_NUMBER", "ANALYSIS", "SAMPLE", "STATUS"]

    # Crear el dataframe para la tabla TEST
    df_test = pd.DataFrame([tuple(t) for t in rows_test], columns=columnas_test)

    # Filtrar el dataframe de la tabla TEST por el valor de la columna STATUS
    df_test = filtrar_dataframe(df_test, "STATUS", ["A", "R"])

    # Filtrar el dataframe de la tabla TEST por el valor de la columna STATUS igual a "R"
    df_test_atipicas = filtrar_dataframe(df_test, "STATUS", ["R"])

    # Filtrar el dataframe de la tabla TEST por el valor de la columna STATUS igual a "A"
    df_test_autorizados = filtrar_dataframe(df_test, "STATUS", ["A"])

    # Convertir la columna SAMPLE del dataframe de la tabla TEST a tipo numérico
    df_test["SAMPLE"] = pd.to_numeric(df_test["SAMPLE"])

    # Definir las columnas del dataframe para la tabla RESULT
    columnas_result = ["TEST_NUMBER", "NAME", "VALUE", "TEXT", "REP_CONTROL"]

    # Crear el dataframe para la tabla RESULT
    df_result = pd.DataFrame([tuple(t) for t in rows_result], columns=columnas_result)

    # Filtrar el dataframe de la tabla RESULT por el valor de la columna REP_CONTROL igual a "RPT"
    df_result_rpt = filtrar_dataframe(df_result, "REP_CONTROL", ["RPT"])

    # Guardar el dataframe de la tabla SAMPLE en un archivo CSV
    df_sample.to_csv(
        r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\first\FirstexportSampleFirst.csv",
        index=False,
    )

    # Guardar el dataframe de la tabla TEST en un archivo CSV
    df_test.to_csv(
        r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\first\FirstexportTestFirst.csv",
        index=False,
    )

    # Guardar el dataframe de la tabla RESULT en un archivo CSV
    df_result.to_csv(
        r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\first\FirstexportResultFirst.csv",
        index=False,
    )

    # Combinar los dataframes de las tablas SAMPLE y TEST en base a la columna ID_NUMERIC
    merged_df = pd.merge(
        df_sample, df_test, left_on="ID_NUMERIC", right_on="SAMPLE", how="left"
    )

    # Guardar el dataframe combinado en un archivo CSV
    merged_df.to_csv(
        r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\first\FirstexportFirst.csv",
        index=False,
    )

    # Combinar el dataframe combinado con el dataframe de la tabla RESULT filtrado por REP_CONTROL igual a "RPT"
    final_merged_df = pd.merge(merged_df, df_result_rpt, on="TEST_NUMBER", how="inner")

    # Imprimir el dataframe final combinado
    print(final_merged_df)

    # Filtrar solo los parámetros deseados antes de guardar el archivo final
    """ parametros_deseados = [
        "Porcentaje de arena (% A)", "Porcentaje de arcilla (% Ar)", "Porcentaje de limo (% L)", 
        "Clase textural", "pH (1:2,5)", "Conductividad eléctrica (CE) (1:5)", "Carbono Orgánico (CO)",
        "Materia Orgánica (MO)", "Fosforo (P) Disponible (Bray II)", "Azufre (S) disponible",
        "Capacidad Interc Catiónico Efect (CICE)", "Boro (B) Disponible", "Acidez (Al+H)",
        "Aluminio (Al) Intercambiable", "Calcio (Ca) disponible", "Magnesio (Mg) Disponible",
        "Potasio (K) Disponible", "Sodio (Na) Disponible", "Hierro (Fe) olsen Disponible",
        "Cobre (Cu) olsen Disponible", "Manganeso (Mn) olsen Disponible", "Zinc (Zn) olsen Disponible",
        "Saturación de Calcio", "Saturación de Magnesio", "Saturación de Potasio",
        "Saturación de Sodio", "Saturación de Aluminio"
    ] oordonez 12-02-2024  contiene los parametros de textura"""
    parametros_deseados = [
        "pH (1:2,5)", "Conductividad eléctrica (CE) (1:5)", "Carbono Orgánico (CO)",
        "Materia Orgánica (MO)", "Fosforo (P) Disponible (Bray II)", "Azufre (S) disponible",
        "Capacidad Interc Catiónico Efect (CICE)", "Boro (B) Disponible", "Acidez (Al+H)",
        "Aluminio (Al) Intercambiable", "Calcio (Ca) disponible", "Magnesio (Mg) Disponible",
        "Potasio (K) Disponible", "Sodio (Na) Disponible", "Hierro (Fe) olsen Disponible",
        "Cobre (Cu) olsen Disponible", "Manganeso (Mn) olsen Disponible", "Zinc (Zn) olsen Disponible",
        "Saturación de Calcio", "Saturación de Magnesio", "Saturación de Potasio",
        "Saturación de Sodio", "Saturación de Aluminio"
    ]
    
    # Filtrar los datos para mantener solo los NAME que están en parametros_deseados
    final_merged_df = final_merged_df[final_merged_df["NAME"].isin(parametros_deseados)]
    

    # Guardar el dataframe final combinado en un archivo CSV
    final_merged_df.to_csv(
        r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\first\FirstexportFinalFirst.csv",
        index=False,
    )

    # Leer el archivo CSV del dataframe final combinado
    df = pd.read_csv(
        r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\first\FirstexportFinalFirst.csv"
    )

    # Filtrar el dataframe por el valor de la columna STATUS_y igual a "R"
    df_filtered_R = df[df["STATUS_y"] == "R"].copy()

    # Filtrar el dataframe por el valor de la columna STATUS_y igual a "A"
    df_filtered_A = df[df["STATUS_y"] == "A"].copy()

    # Crear una copia del dataframe filtrado por el valor de la columna STATUS_y igual a "A"
    df_filtered_A_copia = df_filtered_A.copy()

    # Guardar la copia del dataframe filtrado en un archivo CSV
    df_filtered_A_copia.to_csv("Firstdf_filtered_A_copia.csv", index=False)

    # Imprimir el dataframe
    print(df)

    # Realizar una tabla pivote del dataframe filtrado por el valor de la columna STATUS_y igual a "R"
    df_filtered_R_pivoted = df_filtered_R.pivot_table(
        index=[
            "ID_NUMERIC",
            "A_RECEP_FECHA_RECIB",
            "TF_DEPARTAMENTO",
            "TF_MUNICIPIO",
            "TF_CENTRO_POBLADO",
            "TF_CULTIVO",
            "TF_LABORATORIO",
        ],
        columns=["NAME"],
        values=["VALUE"],
        aggfunc="first", # # Use "first" to select the first value when multiple entries exist for the same index/column combination
    ).reset_index()

    # Renombrar las columnas del dataframe pivote
    df_filtered_R_pivoted.columns = [
        "_".join(col).strip() for col in df_filtered_R_pivoted.columns.values
    ]

    # Agregar una columna "ATP" con valor 0 al dataframe pivote
    df_filtered_R_pivoted["ATP"] = 1

    # Guardar el dataframe pivote en un archivo CSV
    df_filtered_R_pivoted.to_csv("Firstdf_filtered_R_pivoted.csv", index=False)

    # Leer el archivo CSV del dataframe pivote filtrado por el valor de la columna STATUS_y igual a "R"
    df_filtered_R_pivoted = pd.read_csv("Firstdf_filtered_R_pivoted.csv")

    # Duplicar las columnas a partir de la cuarta columna en el dataframe df_filtered_R
    df_copy = df_filtered_R_pivoted.iloc[:, 5:].copy()
    df_copy.columns = [f"BIN_{col}" for col in df_copy.columns]

    # Reemplazar todos los valores no nulos por 1 y los nulos por 0
    df_copy = df_copy.notna().astype(int)

    # Concatenar el DataFrame completo con las columnas duplicadas
    df_filtered_R_pivoted = pd.concat([df_filtered_R_pivoted, df_copy], axis=1)
    df_filtered_R_pivoted.to_csv(r"FirstArcBinario.csv", index=False)

    # Realizar una tabla pivote del dataframe filtrado por el valor de la columna STATUS_y igual a "A"
    df_filtered_A_pivoted = df_filtered_A.pivot_table(
        index=[
            "ID_NUMERIC",
            "A_RECEP_FECHA_RECIB",
            "TF_DEPARTAMENTO",
            "TF_MUNICIPIO",
            "TF_CENTRO_POBLADO",
            "TF_CULTIVO",
            "TF_LABORATORIO",
        ],
        columns=["NAME"],
        values=["VALUE"],
        aggfunc="first",
    ).reset_index()

    # Renombrar las columnas del dataframe pivote
    df_filtered_A_pivoted.columns = [
        "_".join(col).strip() for col in df_filtered_A_pivoted.columns.values
    ]

    # Agregar una columna "ATP" con valor 1 al dataframe pivote
    df_filtered_A_pivoted["ATP"] = 0

    # Guardar el dataframe pivote en un archivo CSV
    df_filtered_A_pivoted.to_csv("Firstdf_filtered_A_pivoted.csv", index=False)

    # Leer el archivo CSV del dataframe pivote filtrado por el valor de la columna STATUS_y igual a "R"
    df_filtered_R_pivoted = pd.read_csv("FirstArcBinario.csv")

    # Leer el archivo CSV del dataframe pivote filtrado por el valor de la columna STATUS_y igual a "A"
    df_filtered_A_pivoted = pd.read_csv("Firstdf_filtered_A_pivoted.csv")

    # Filtrar el dataframe pivote filtrado por el valor de la columna ID_NUMERIC_ presente en el dataframe pivote filtrado por el valor de la columna STATUS_y igual a "R"
    df_filtered_A_pivotedMod = df_filtered_A_pivoted[
        df_filtered_A_pivoted["ID_NUMERIC_"].isin(df_filtered_R_pivoted["ID_NUMERIC_"])
    ]

    # Combinar los dataframes filtrados por el valor de la columna STATUS_y igual a "R"
    df_R_Final = (
        df_filtered_R_pivoted.set_index("ID_NUMERIC_")
        .combine_first(df_filtered_A_pivotedMod.set_index("ID_NUMERIC_"))
        .reset_index()
    )

    # Guardar el dataframe final combinado en un archivo CSV
    df_R_Final.to_csv("Firstdf_R_Final.csv", index=False)

    # Concatenar los dataframes df_R_Final y df_filtered_A_pivoted
    df_resultado = pd.concat(
        [df_R_Final, df_filtered_A_pivoted], axis=0, ignore_index=True
    )

    # Obtener las columnas de df_filtered_A_pivoted
    columnas_A = df_filtered_A_pivoted.columns.tolist()

    # Obtener las columnas de df_resultado
    columnas_resultado = df_resultado.columns.tolist()

    # Crear una nueva lista de columnas que comienza con las columnas de df_filtered_A_pivoted
    # seguido de las columnas en df_resultado que no están en df_filtered_A_pivoted
    columnas_nuevas = columnas_A + [
        col for col in columnas_resultado if col not in columnas_A
    ]

    # Reordenar las columnas de df_resultado
    df_resultado = df_resultado[columnas_nuevas]
    # Filtrar por la columna "ATP" Autorizadas
    df_filtered = df_resultado[df_resultado["ATP"] == 0]
    df_resultado = df_resultado[df_resultado["ATP"] == 1]

    # Reemplazar los valores NaN por 0 en las columnas que comienzan con "BIN"
    bin_columns = [col for col in df_filtered.columns if col.startswith("BIN")]
    df_filtered[bin_columns] = df_filtered[bin_columns].fillna(0)

    # Añadir las filas de df_filtered a df_resultado
    df_resultado = pd.concat([df_resultado, df_filtered], axis=0, ignore_index=True)
    # Imprimir el dataframe resultado
    print(df_resultado)

    # Guardar el dataframe resultado en un archivo CSV
    df_resultado.to_csv("Firstdf_resultado.csv", index=False)

    # Leer el archivo CSV del dataframe resultado
    df = pd.read_csv("Firstdf_resultado.csv")

    # Reemplazar ".0" en las columnas específicas
    cols_especificas = ["TF_DEPARTAMENTO_", "TF_MUNICIPIO_", "TF_CENTRO_POBLADO_"]
    df[cols_especificas] = (
        df[cols_especificas].astype(str).replace(r"\.0", "", regex=True)
    )

    # Convertir las columnas específicas a enteros
    df[cols_especificas] = df[cols_especificas].astype(int, errors="ignore")

    # Definir las columnas del dataframe para la tabla RESULT
    columnas_departamento = ["IDENTITY", "NOMBRE"]

    # Crear el dataframe para la tabla DEPARTAMENTO
    df_departamento = pd.DataFrame(
        [tuple(t) for t in rows_departamento], columns=columnas_departamento
    )

    # Definir las columnas del dataframe para la tabla RESULT
    columnas_municipio = ["IDENTITY", "NOMBRE"]

    # Crear el dataframe para la tabla MUNICIPIO
    df_municipio = pd.DataFrame(
        [tuple(t) for t in rows_municipio], columns=columnas_municipio
    )

    # Guardar el dataframe resultado en un archivo CSV
    df.to_csv("Firstmunicipio.csv", index=False)

    # Definir las columnas del dataframe para la tabla RESULT
    columnas_poblado = ["IDENTITY", "NOMBRE"]

    # Crear el dataframe para la tabla POBLADO
    df_poblado = pd.DataFrame(
        [tuple(t) for t in rows_poblado], columns=columnas_poblado
    )

    # Duplicar y reemplazar los identitys por los nombres en la columna "DEPARTAMENTO"
    df["TF_DEPARTAMENTO_NOMBRE"] = df["TF_DEPARTAMENTO_"].map(
        df_departamento.set_index("IDENTITY")["NOMBRE"]
    )

    # Duplicar y reemplazar los identitys por los nombres en la columna "MUNICIPIO"
    df["TF_MUNICIPIO_NOMBRE"] = df["TF_MUNICIPIO_"].map(
        df_municipio.set_index("IDENTITY")["NOMBRE"]
    )

    # Duplicar y reemplazar los identitys por los nombres en la columna "CENTRO_POBLADO"
    df["TF_CENTRO_POBLADO_NOMBRE"] = df["TF_CENTRO_POBLADO_"].map(
        df_poblado.set_index("IDENTITY")["NOMBRE"]
    )
    # Guardar el dataframe resultado en un archivo CSV
    df.to_csv("Firstdf_resultadoV2.csv", index=False)

    return df

def mainCommit():
    # Establecer conexión a la base de datos
    conn = establecer_conexion()

    # Consulta para obtener los datos de la tabla C_SAMPLE
    consulta_sample = "SELECT ID_NUMERIC, ID_TEXT, A_RECEP_FECHA_RECIB, STATUS, TF_DEPARTAMENTO, TF_MUNICIPIO, TF_CENTRO_POBLADO , TF_CULTIVO, TF_LABORATORIO  FROM C_SAMPLE"
    rows_sample = ejecutar_consulta(conn, consulta_sample)

    # Consulta para obtener los datos de la tabla C_TEST
    consulta_test = "SELECT TEST_NUMBER, ANALYSIS, SAMPLE, STATUS FROM C_TEST"
    rows_test = ejecutar_consulta(conn, consulta_test)

    # Consulta para obtener los datos de la tabla C_RESULT
    consulta_result = "SELECT TEST_NUMBER, NAME, VALUE, TEXT, REP_CONTROL FROM C_RESULT"
    rows_result = ejecutar_consulta(conn, consulta_result)

    # Consulta para obtener los datos de la tabla TF_DEPARTAMENTO
    consulta_departamento = "SELECT [IDENTITY], NOMBRE FROM TF_DEPARTAMENTO"
    rows_departamento = ejecutar_consulta(conn, consulta_departamento)

    # Consulta para obtener los datos de la tabla TF_MUNICIPIO
    consulta_municipio = "SELECT [IDENTITY], NOMBRE FROM TF_MUNICIPIO"
    rows_municipio = ejecutar_consulta(conn, consulta_municipio)

    # Consulta para obtener los datos de la tabla TF_POBLADO
    consulta_poblado = "SELECT [IDENTITY], NOMBRE FROM TF_POBLADO"
    rows_poblado = ejecutar_consulta(conn, consulta_poblado)

    # Cerrar la conexión a la base de datos
    cerrar_conexion(conn)

# ================================================================================================================================================

    # Definir las columnas del dataframe para la tabla SAMPLE
    columnas_sample = [
        "ID_NUMERIC",
        "ID_TEXT",
        "A_RECEP_FECHA_RECIB",
        "STATUS",
        "TF_DEPARTAMENTO",
        "TF_MUNICIPIO",
        "TF_CENTRO_POBLADO",
        "TF_CULTIVO",
        "TF_LABORATORIO",
    ]
   
    # Crear el dataframe para la tabla SAMPLE
    df_sample = pd.DataFrame([tuple(t) for t in rows_sample], columns=columnas_sample)

    # Filtrar el dataframe de la tabla SAMPLE por el valor de la columna TF_LABORATORIO
    df_sample = filtrar_dataframe(df_sample, "TF_LABORATORIO", ["LQA"])

    # Filtrar el dataframe de la tabla SAMPLE por el valor de la columna STATUS
    df_sample = filtrar_dataframe(df_sample, "STATUS", ["A"])

    # Convertir la columna ID_NUMERIC del dataframe de la tabla SAMPLE a tipo numérico
    df_sample["ID_NUMERIC"] = pd.to_numeric(df_sample["ID_NUMERIC"])

#=================================================================================================================================================

    # Definir las columnas del dataframe para la tabla TEST
    columnas_test = ["TEST_NUMBER", "ANALYSIS", "SAMPLE", "STATUS"]

    # Crear el dataframe para la tabla TEST
    df_test = pd.DataFrame([tuple(t) for t in rows_test], columns=columnas_test)

    # Filtrar el dataframe de la tabla TEST por el valor de la columna STATUS
    df_test = filtrar_dataframe(df_test, "STATUS", ["A", "R"])

    # Filtrar el dataframe de la tabla TEST por el valor de la columna STATUS igual a "R"
    df_test_atipicas = filtrar_dataframe(df_test, "STATUS", ["R"])

    # Filtrar el dataframe de la tabla TEST por el valor de la columna STATUS igual a "A"
    df_test_autorizados = filtrar_dataframe(df_test, "STATUS", ["A"])

    # Convertir la columna SAMPLE del dataframe de la tabla TEST a tipo numérico
    df_test["SAMPLE"] = pd.to_numeric(df_test["SAMPLE"])

#=================================================================================================================================================

    # Definir las columnas del dataframe para la tabla RESULT
    columnas_result = ["TEST_NUMBER", "NAME", "VALUE", "TEXT", "REP_CONTROL"]

    # Crear el dataframe para la tabla RESULT
    df_result = pd.DataFrame([tuple(t) for t in rows_result], columns=columnas_result)

    # Filtrar el dataframe de la tabla RESULT por el valor de la columna REP_CONTROL igual a "RPT"
    df_result_rpt = filtrar_dataframe(df_result, "REP_CONTROL", ["RPT"])

#=================================================================================================================================================

    # Guardar el dataframe de la tabla SAMPLE en un archivo CSV
    df_sample.to_csv(
        r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\first\Firstc_exportSampleFirst.csv",
        index=False,
    )

    """
ID_NUMERIC    ID_TEXT          A_RECEP_FECHA_RECIB      STATUS   TF_DEPARTAMENTO   TF_MUNICIPIO   TF_CENTRO_POBLADO   TF_LABORATORIO
580916        LQAS23-013321    2023-10-26 15:07:19.867  A        85               85440          85440000            LQA
580917        LQAS23-013322    2023-10-26 15:07:19.883  A        85               85440          85440000            LQA
580920        LQAS23-013325    2023-10-26 15:07:19.930  A        85               85440          85440000            LQA
"""

    # Guardar el dataframe de la tabla TEST en un archivo CSV
    df_test.to_csv(
        r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\first\Firstc_exportTestFirst.csv",
        index=False,
    )
    """
TEST_NUMBER   ANALYSIS    SAMPLE   STATUS
1784259       CR-AUT-L    297229   A
1889960       PW_S        310347   A
1889974       CD_DISP_S   310348   A
"""


    # Guardar el dataframe de la tabla RESULT en un archivo CSV
    df_result.to_csv(
        r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\first\Firstc_exportResultFirst.csv",
        index=False,
    )
    """
TEST_NUMBER   NAME                                      VALUE   TEXT     REP_CONTROL
2086061       Determinación Punto Crioscópico Automa.   -0.548  -0.5480  RPT
2086062       Determinación Grasa Automatizada          3.105   3.105    RPT
2086064       Determinación lactosa                     4.498   4.498    RPT
"""


#=================================================================================================================================================

    # Combinar los dataframes de las tablas SAMPLE y TEST en base a la columna ID_NUMERIC uniendo izquierda
    merged_df = pd.merge(
        df_sample, df_test, left_on="ID_NUMERIC", right_on="SAMPLE", how="left"
    )

    # Guardar el dataframe combinado en un archivo CSV
    merged_df.to_csv(
        r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\first\Firstc_exportFirst.csv",
        index=False,
    )
    """
ID_NUMERIC   ID_TEXT        A_RECEP_FECHA_RECIB      STATUS_x   TF_DEPARTAMENTO   TF_MUNICIPIO   TF_CENTRO_POBLADO   TF_LABORATORIO   TEST_NUMBER   ANALYSIS     SAMPLE     STATUS_y
580916       LQAS23-013321  2023-10-26 15:07:19.867  A          85                85440          85440000            LQA              4122168       ACIYALU      580916.0   A
580916       LQAS23-013321  2023-10-26 15:07:19.867  A          85                85440          85440000            LQA              4122174       B_CA2PO4_S   580916.0   A
580916       LQAS23-013321  2023-10-26 15:07:19.867  A          85                85440          85440000            LQA              4122172       BASESINT     580916.0   A
"""

    # Combinar el dataframe combinado con el dataframe de la tabla RESULT filtrado por REP_CONTROL igual a "RPT" conserva todos los registros de ambos df
    final_merged_df = pd.merge(merged_df, df_result_rpt, on="TEST_NUMBER", how="inner")

    # Imprimir el dataframe final combinado
    print(final_merged_df)
    
        # Filtrar solo los parámetros deseados antes de guardar el archivo final
    """ parametros_deseados = [
        "Porcentaje de arena (% A)", "Porcentaje de arcilla (% Ar)", "Porcentaje de limo (% L)", 
        "Clase textural", "pH (1:2,5)", "Conductividad eléctrica (CE) (1:5)", "Carbono Orgánico (CO)",
        "Materia Orgánica (MO)", "Fosforo (P) Disponible (Bray II)", "Azufre (S) disponible",
        "Capacidad Interc Catiónico Efect (CICE)", "Boro (B) Disponible", "Acidez (Al+H)",
        "Aluminio (Al) Intercambiable", "Calcio (Ca) disponible", "Magnesio (Mg) Disponible",
        "Potasio (K) Disponible", "Sodio (Na) Disponible", "Hierro (Fe) olsen Disponible",
        "Cobre (Cu) olsen Disponible", "Manganeso (Mn) olsen Disponible", "Zinc (Zn) olsen Disponible",
        "Saturación de Calcio", "Saturación de Magnesio", "Saturación de Potasio",
        "Saturación de Sodio", "Saturación de Aluminio"
    ] oordonez 12-02-2024  contiene los parametros de textura"""
    parametros_deseados = [
        "pH (1:2,5)", "Conductividad eléctrica (CE) (1:5)", "Carbono Orgánico (CO)",
        "Materia Orgánica (MO)", "Fosforo (P) Disponible (Bray II)", "Azufre (S) disponible",
        "Capacidad Interc Catiónico Efect (CICE)", "Boro (B) Disponible", "Acidez (Al+H)",
        "Aluminio (Al) Intercambiable", "Calcio (Ca) disponible", "Magnesio (Mg) Disponible",
        "Potasio (K) Disponible", "Sodio (Na) Disponible", "Hierro (Fe) olsen Disponible",
        "Cobre (Cu) olsen Disponible", "Manganeso (Mn) olsen Disponible", "Zinc (Zn) olsen Disponible",
        "Saturación de Calcio", "Saturación de Magnesio", "Saturación de Potasio",
        "Saturación de Sodio", "Saturación de Aluminio"
    ]
    
    
    # Aplicar el filtro antes de guardar el archivo
    final_merged_df = final_merged_df[final_merged_df["NAME"].isin(parametros_deseados)]
    
    # Guardar el dataframe final combinado en un archivo CSV 
    final_merged_df.to_csv(
        r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\first\Firstc_exportFinalFirst.csv",
        index=False,
    )


    # Guardar el dataframe final combinado en un archivo CSV 
    final_merged_df.to_csv(
        r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\first\Firstc_exportFinalFirst.csv",
        index=False,
    )
    """
ID_NUMERIC   ID_TEXT        A_RECEP_FECHA_RECIB      STATUS_x   TF_DEPARTAMENTO   TF_MUNICIPIO   TF_CENTRO_POBLADO   TF_LABORATORIO   TEST_NUMBER   ANALYSIS     SAMPLE     STATUS_y   NAME                          VALUE   TEXT    REP_CONTROL
580916       LQAS23-013321  2023-10-26 15:07:19.867  A          85                85440          85440000            LQA              4122168       ACIYALU      580916.0   A         Acidez (Al+H)                0.27    0.27    RPT
580916       LQAS23-013321  2023-10-26 15:07:19.867  A          85                85440          85440000            LQA              4122168       ACIYALU      580916.0   A         Aluminio (Al) Intercambiable  0.13    0.13    RPT
580916       LQAS23-013321  2023-10-26 15:07:19.867  X         85                85440          85440000            LQA              4122174       B_CA2PO4_S   580916.0   A         Boro (B) Disponible           0.28    0.28    RPT
"""


    # Leer el archivo CSV del dataframe final combinado
    df = pd.read_csv(
        r"D:\OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA\SampleManager\Desarrollos\SQLAtipicos\first\Firstc_exportFinalFirst.csv"
    )

#=================================================================================================================================================

    # Filtrar el dataframe por el valor de la columna STATUS_y igual a "R"
    df_filtered_R = df[df["STATUS_y"] == "R"].copy()

    # Filtrar el dataframe por el valor de la columna STATUS_y igual a "A"
    df_filtered_A = df[df["STATUS_y"] == "A"].copy()

    # Crear una copia del dataframe filtrado por el valor de la columna STATUS_y igual a "A"
    df_filtered_A_copia = df_filtered_A.copy()

    # Guardar la copia del dataframe filtrado en un archivo CSV
    df_filtered_A_copia.to_csv("Firstc_df_filtered_A_copia.csv", index=False)

    """
ID_TEXT                A_RECEP_FECHA_RECIB          STATUS_x   TF_DEPARTAMENTO   TF_MUNICIPIO   TF_CENTRO_POBLADO   TF_LABORATORIO   TEST_NUMBER   ANALYSIS       SAMPLE     STATUS_y   NAME                         VALUE   TEXT    REP_CONTROL
LQAS23-013321          2023-10-26 15:07:19.867     A           85                85440          85440000.0          LQA              4122168       ACIYALU        580916.0   A         Acidez (Al+H)                0.27    0.27    RPT
LQAS23-013321          2023-10-26 15:07:19.867     A           85                85440          85440000.0          LQA              4122168       Aluminio (Al)  580916.0   A         Aluminio (Al) Intercambiable  0.13    0.13    RPT
LQAS23-013321          2023-10-26 15:07:19.867     A           85                85440          85440000.0          LQA              4122174       B_CA2PO4_S     580916.0   A         Boro (B) Disponible           0.28    0.28    RPT
"""

    # Realizar una tabla pivote del dataframe filtrado por el valor de la columna STATUS_y igual a "R"
    df_filtered_R_pivoted = df_filtered_R.pivot_table(
        index=[
            "ID_NUMERIC",
            "A_RECEP_FECHA_RECIB",
            "TF_DEPARTAMENTO",
            "TF_MUNICIPIO",
            "TF_CENTRO_POBLADO",
            "TF_CULTIVO",
            "TF_LABORATORIO",
        ],
        columns=["NAME"],
        values=["VALUE"],
        aggfunc="first",
    ).reset_index()

    # Renombrar las columnas del dataframe pivote
    df_filtered_R_pivoted.columns = [
        "_".join(col).strip() for col in df_filtered_R_pivoted.columns.values
    ]

    # Agregar una columna "ATP" con valor 0 al dataframe pivote
    df_filtered_R_pivoted["ATP"] = 1

    # Guardar el dataframe pivote en un archivo CSV
    df_filtered_R_pivoted.to_csv("Firstc_df_filtered_R_pivoted.csv", index=False)

    """
ID_NUMERIC_   A_RECEP_FECHA_RECIB_       TF_DEPARTAMENTO_   TF_MUNICIPIO_   TF_CENTRO_POBLADO_   TF_LABORATORIO_   VALUE_Acidez (Al+H)   VALUE_Aluminio (Al) Intercambiable   VALUE_Arsénico (As) mg/kg   VALUE_Arsénico (As) mg/kg CONTROL   VALUE_Azufre (S) disponible   VALUE_Boro (B) Disponible   VALUE_Cadmio (Cd) disponible   VALUE_Cadmio (Cd) pseudototal   VALUE_Calcio (Ca) Soluble   VALUE_Calcio (Ca) disponible   VALUE_Capacidad Interc Catiónico Aceta (CICA)   VALUE_Capacidad Interc Catiónico Efect (CICE)   VALUE_Carbono Orgánico (CO)   VALUE_Carbono total (CT)   VALUE_Clase textural   VALUE_Cloruros (Cl-)   VALUE_Cobre (Cu) Disponible (Doble ácido)   VALUE_Cobre (Cu) olsen Disponible   VALUE_Conductividad eléctrica (CE) (1:5)   VALUE_Conductividad eléctrica (pasta saturada)   VALUE_Cromo (Cr) Pseudototal   VALUE_Fosforo (P) Disponible (Bray II)   VALUE_Hierro (Fe) Disponible (Doble ácido)   VALUE_Hierro (Fe) olsen Disponible   VALUE_Incertidumbre Calcio (Ca) disponible +/-   VALUE_Incertidumbre Conductividad eléctrica +/-   VALUE_Incertidumbre Magnesio (Mg) disp +/-   VALUE_Incertidumbre Potasio (K) disponible +/-   VALUE_Incertidumbre Sodio (Na) disponible +/-   VALUE_Magnesio (Mg) Disponible   VALUE_Magnesio (Mg) soluble   VALUE_Manganeso (Mn) Disponible (Doble ácido)   VALUE_Manganeso (Mn) olsen Disponible   VALUE_Materia Orgánica (MO)   VALUE_Mercurio (Hg) pseudototal   VALUE_Nitrógeno total (NT)   VALUE_Plomo (Pb) pseudototal   VALUE_Porcentaje de arcilla (% Ar)   VALUE_Porcentaje de arena (% A)   VALUE_Porcentaje de limo (% L)   VALUE_Potasio (K) Disponible   VALUE_Sodio (Na) Disponible   VALUE_Sodio (Na) Soluble   VALUE_Sulfatos (SO4)   VALUE_Zinc (Zn) Disponible (Doble ácido)   VALUE_Zinc (Zn) olsen Disponible   VALUE_pH (1:1)   "VALUE_pH (1:2,5)"   VALUE_pH (pasta saturada)   ATP
819           2019-04-26 11:17:11.130    25.0              25754.0         25754000             LQA               2.38                                         1
851           2019-04-26 11:32:14.710    54.0              54385.0         54385000             LQA               5.1                                          1.22
865           2019-04-26 13:57:51.897    54.0              54820           54820000             LQA               10.81                                        1
"""
    # Leer el archivo CSV del dataframe pivote filtrado por el valor de la columna STATUS_y igual a "R"
    df_filtered_R_pivoted = pd.read_csv("Firstc_df_filtered_R_pivoted.csv")

    # Duplicar las columnas a partir de la cuarta columna en el dataframe df_filtered_R
    df_copy = df_filtered_R_pivoted.iloc[:, 5:].copy()
    df_copy.columns = [f"BIN_{col}" for col in df_copy.columns]

    # Reemplazar todos los valores no nulos por 1 y los nulos por 0
    df_copy = df_copy.notna().astype(int)

    # Concatenar el DataFrame completo con las columnas duplicadas
    df_filtered_R_pivoted = pd.concat([df_filtered_R_pivoted, df_copy], axis=1)
    df_filtered_R_pivoted.to_csv(r"Firstc_ArcBinario.csv", index=False)
    """
ID_NUMERIC_   A_RECEP_FECHA_RECIB_       TF_DEPARTAMENTO_   TF_MUNICIPIO_   TF_CENTRO_POBLADO_   TF_LABORATORIO_   VALUE_Acidez (Al+H)   VALUE_Aluminio (Al) Intercambiable   VALUE_Arsénico (As) mg/kg   VALUE_Arsénico (As) mg/kg CONTROL   VALUE_Azufre (S) disponible   VALUE_Boro (B) Disponible   VALUE_Cadmio (Cd) disponible   VALUE_Cadmio (Cd) pseudototal   VALUE_Calcio (Ca) Soluble   VALUE_Calcio (Ca) disponible   VALUE_Capacidad Interc CatiÃ³nico Aceta (CICA)   VALUE_Capacidad Interc CatiÃ³nico Efect (CICE)   VALUE_Carbono Orgánico (CO)   VALUE_Carbono total (CT)   VALUE_Clase textural   VALUE_Cloruros (Cl-)   VALUE_Cobre (Cu) Disponible (Doble ácido)   VALUE_Cobre (Cu) olsen Disponible   VALUE_Conductividad eléctrica (CE) (1:5)   VALUE_Conductividad eléctrica (pasta saturada)   VALUE_Cromo (Cr) Pseudototal   VALUE_Fosforo (P) Disponible (Bray II)   VALUE_Hierro (Fe) Disponible (Doble ácido)   VALUE_Hierro (Fe) olsen Disponible   VALUE_Incertidumbre Calcio (Ca) disponible +/-   VALUE_Incertidumbre Conductividad eléctrica +/-   VALUE_Incertidumbre Magnesio (Mg) disp +/-   VALUE_Incertidumbre Potasio (K) disponible +/-   VALUE_Incertidumbre Sodio (Na) disponible +/-   VALUE_Magnesio (Mg) Disponible   VALUE_Magnesio (Mg) soluble   VALUE_Manganeso (Mn) Disponible (Doble ácido)   VALUE_Manganeso (Mn) olsen Disponible   VALUE_Materia Orgánica (MO)   VALUE_Mercurio (Hg) pseudototal   VALUE_Nitrógeno total (NT)   VALUE_Plomo (Pb) pseudototal   VALUE_Porcentaje de arcilla (% Ar)   VALUE_Porcentaje de arena (% A)   VALUE_Porcentaje de limo (% L)   VALUE_Potasio (K) Disponible   VALUE_Sodio (Na) Disponible   VALUE_Sodio (Na) Soluble   VALUE_Sulfatos (SO4)   VALUE_Zinc (Zn) Disponible (Doble ácido)   VALUE_Zinc (Zn) olsen Disponible   VALUE_pH (1:1)   "VALUE_pH (1:2,5)"   VALUE_pH (pasta saturada)   ATP   BIN_TF_LABORATORIO_   BIN_VALUE_Acidez (Al+H)   BIN_VALUE_Aluminio (Al) Intercambiable   BIN_VALUE_Arsénico (As) mg/kg   BIN_VALUE_Arsénico (As) mg/kg CONTROL   BIN_VALUE_Azufre (S) disponible   BIN_VALUE_Boro (B) Disponible   BIN_VALUE_Cadmio (Cd) disponible   BIN_VALUE_Cadmio (Cd) pseudototal   BIN_VALUE_Calcio (Ca) Soluble   BIN_VALUE_Calcio (Ca) disponible   BIN_VALUE_Capacidad Interc Catiónico Aceta (CICA)   BIN_VALUE_Capacidad Interc Catiónico Efect (CICE)   BIN_VALUE_Carbono Orgánico (CO)   BIN_VALUE_Carbono total (CT)   BIN_VALUE_Clase textural   BIN_VALUE_Cloruros (Cl-)   BIN_VALUE_Cobre (Cu) Disponible (Doble ácido)   BIN_VALUE_Cobre (Cu) olsen Disponible   BIN_VALUE_Conductividad eléctrica (CE) (1:5)   BIN_VALUE_Conductividad eléctrica (pasta saturada)   BIN_VALUE_Cromo (Cr) Pseudototal   BIN_VALUE_Fosforo (P) Disponible (Bray II)   BIN_VALUE_Hierro (Fe) Disponible (Doble ácido)   BIN_VALUE_Hierro (Fe) olsen Disponible   BIN_VALUE_Incertidumbre Calcio (Ca) disponible +/-   BIN_VALUE_Incertidumbre Conductividad eléctrica +/-   BIN_VALUE_Incertidumbre Magnesio (Mg) disp +/-   BIN_VALUE_Incertidumbre Potasio (K) disponible +/-   BIN_VALUE_Incertidumbre Sodio (Na) disponible +/-   BIN_VALUE_Magnesio (Mg) Disponible   BIN_VALUE_Magnesio (Mg) soluble   BIN_VALUE_Manganeso (Mn) Disponible (Doble ácido)   BIN_VALUE_Manganeso (Mn) olsen Disponible   BIN_VALUE_Materia Orgánica (MO)   BIN_VALUE_Mercurio (Hg) pseudototal   BIN_VALUE_Nitrógeno total (NT)   BIN_VALUE_Plomo (Pb) pseudototal   BIN_VALUE_Porcentaje de arcilla (% Ar)   BIN_VALUE_Porcentaje de arena (% A)   BIN_VALUE_Porcentaje de limo (% L)   BIN_VALUE_Potasio (K) Disponible   BIN_VALUE_Sodio (Na) Disponible   BIN_VALUE_Sodio (Na) Soluble   BIN_VALUE_Sulfatos (SO4)   BIN_VALUE_Zinc (Zn) Disponible (Doble ácido)   BIN_VALUE_Zinc (Zn) olsen Disponible   BIN_VALUE_pH (1:1)   "BIN_VALUE_pH (1:2,5)"   BIN_VALUE_pH (pasta saturada)   BIN_ATP
819           2019-04-26 11:17:11.130    25.0              25754.0         25754000             LQA               2.38                                         1       1                     0                                0                                         0                           0                              0                                  1                                   0                                 0                                         0                                          0                                         0                                                 0                                                    0                                                   0                                                    0                                                      0                                                    0                                                        0                                                       0                                                       0                                                       0                                                     0                                                    0                                                    0                                                    0                                                    0                                                0                                                0                                               0                                                0                                             0                                                    0                                                   0                                                    0                                                  0                                                 0                                                0                                                0                                                0                                                0                                                0                                                0                                                0                                                 1
851           2019-04-26 11:32:14.710    54.0              54385.0         54385000             LQA               5.1                                          1.22   1                     0                                0                                         0                           0                              0                                  0                                   0                                 0                                         1                                          0                                         0                                                 0                                                    0                                                   0                                                    0                                                      0                                                    0                                                        0                                                       1.85                                                    0                                                     0                                                    0                                                    1                                                    1                                                0                                                0                                               0                                                0                                             0                                                    0                                                   0                                                    0                                                  0                                                 0                                                0                                                0                                                0                                                0                                                0                                                0                                                0                                                 1
865           2019-04-26 13:57:51.897    54.0              54820           54820000             LQA               10.81                                        1       0                     0                                0                                         0                           0                              0                                  0                                   0                                 0                                         0                                          0                                         0                                                 0                                                    0                                                   0                                                    0                                                      0                                                    0                                                        0                                                       0                                                       0                                                       0                                                     0                                                    0                                                    0                                                    0                                                    0                                                0                                                0                                               0                                                0                                             0                                                    0                                                   0                                                    0                                                  0                                                 0                                                0                                                0                                                0                                                0                                                0                                                0                                                0                                                 1
"""


    # Realizar una tabla pivote del dataframe filtrado por el valor de la columna STATUS_y igual a "A"
    df_filtered_A_pivoted = df_filtered_A.pivot_table(
        index=[
            "ID_NUMERIC",
            "A_RECEP_FECHA_RECIB",
            "TF_DEPARTAMENTO",
            "TF_MUNICIPIO",
            "TF_CENTRO_POBLADO",
            "TF_CULTIVO",
            "TF_LABORATORIO",
        ],
        columns=["NAME"],
        values=["VALUE"],
        aggfunc="first",
    ).reset_index()

    # Renombrar las columnas del dataframe pivote
    df_filtered_A_pivoted.columns = [
        "_".join(col).strip() for col in df_filtered_A_pivoted.columns.values
    ]

    # Agregar una columna "ATP" con valor 1 al dataframe pivote
    df_filtered_A_pivoted["ATP"] = 0

    # Guardar el dataframe pivote en un archivo CSV
    df_filtered_A_pivoted.to_csv("Firstc_df_filtered_A_pivoted.csv", index=False)

    """
ID_NUMERIC_   A_RECEP_FECHA_RECIB_       TF_DEPARTAMENTO_   TF_MUNICIPIO_   TF_CENTRO_POBLADO_   TF_LABORATORIO_   VALUE_Acidez (Al+H)   VALUE_Aluminio (Al) Intercambiable   VALUE_Arsénico (As) mg/kg   VALUE_Arsénico (As) mg/kg CONTROL   VALUE_Azufre (S) disponible   VALUE_Bicarbonatos (HCO3)-   VALUE_Boro (B) Disponible   VALUE_Cadmio (Cd) disponible   VALUE_Cadmio (Cd) pseudototal   VALUE_Calcio (Ca) Soluble   VALUE_Calcio (Ca) disponible   VALUE_Capacidad Interc Catiónico Aceta (CICA)   VALUE_Capacidad Interc Catiónico Efect (CICE)   VALUE_Carbonatos (CO3-2)   VALUE_Carbono Orgánico (CO)   VALUE_Carbono orgánico oxidable   VALUE_Carbono total (CT)   VALUE_Clase textural   VALUE_Cloruros (Cl-)   VALUE_Cobre (Cu) Disponible (Doble ácido)   VALUE_Cobre (Cu) olsen Disponible   VALUE_Cobre (Cu) pseudototal   VALUE_Conductividad eléctrica (CE) (1:5)   VALUE_Conductividad eléctrica (pasta saturada)   VALUE_Cromo (Cr) Pseudototal   VALUE_Fosforo (P) Disponible (Bray II)   VALUE_Hierro (Fe) Disponible (Doble ácido)   VALUE_Hierro (Fe) olsen Disponible   VALUE_Humedad gravimétrica 105 ºC (%)   VALUE_Incertidumbre Calcio (Ca) disponible +/-   VALUE_Incertidumbre Carbono Orgánico (CO) +/-   VALUE_Incertidumbre Cobre (Cu) +/-   VALUE_Incertidumbre Conductividad eléctrica +/-   VALUE_Incertidumbre Fosforo (P) (Bray II) +/-   VALUE_Incertidumbre Fosforo (P) Disponible +/-   VALUE_Incertidumbre Hierro (Fe) +/-   VALUE_Incertidumbre Magnesio (Mg) disp +/-   VALUE_Incertidumbre Manganeso (Mn) +/-   VALUE_Incertidumbre Potasio (K) disponible +/-   VALUE_Incertidumbre Sodio (Na) disponible +/-   VALUE_Incertidumbre Zinc (Zn) +/-   VALUE_Incertidumbre de Conductividad Eléctrica (CE) (1:5) +/-   VALUE_Incertidumbre de Cobre (Cu) olsen +/-   VALUE_Incertidumbre de Hierro (Fe) olsen +/-   VALUE_Incertidumbre de Manganeso (Mn) olsen +/-   VALUE_Incertidumbre de Zinc (Zn) olsen +/-   "VALUE_Incertidumbre de pH (1:2,5) +/-"   "VALUE_Incertidumbre pH (1:2,5) +/-"   VALUE_Magnesio (Mg) Disponible   VALUE_Magnesio (Mg) soluble   VALUE_Manganeso (Mn) Disponible (Doble ácido)   VALUE_Manganeso (Mn) olsen Disponible   VALUE_Materia Orgánica (MO)   VALUE_Mercurio (Hg) pseudototal   VALUE_Nitrógeno total (NT)   VALUE_Plomo (Pb) pseudototal   VALUE_Porcentaje de arcilla (% Ar)   VALUE_Porcentaje de arena (% A)   VALUE_Porcentaje de limo (% L)   VALUE_Porcentaje de sodio intercambiable (PSI)   VALUE_Potasio (K) Disponible   VALUE_Potasio (K) soluble   VALUE_Relacion de adsorcion de sodio (RAS)   VALUE_Saturación de agua   VALUE_Sodio (Na) Disponible   VALUE_Sodio (Na) Soluble   VALUE_Sulfatos (SO4)   VALUE_Zinc (Zn) Disponible (Doble ácido)   VALUE_Zinc (Zn) olsen Disponible   VALUE_Zinc (Zn) pseudototal   VALUE_incertidumbre Magnesio (Mg) disponible +/-   VALUE_incertidumbre Potasio (K) Disponible +/-   VALUE_incertidumbre Sodio (Na) Disponible +/-   VALUE_pH (1:1)   "VALUE_pH (1:2,5)"   VALUE_pH (1:5)   VALUE_pH (pasta saturada)   ATP
1             2019-04-15 19:29:23.810    5.0               5154            5154000.0            LQA               0.0                 0.0                                     6.09                        0.09                              1.37                        2.09                          0.74                                  0.16                           1.1                              262.15               2.99                                                                                                                      0.59                                      15.28                           1.9                            0.07                          0.06                             0.35                                4.53               0
2             2019-04-16 08:45:49.447    5.0               5154            5154000.0            LQA               0.0                 0.0                                     6.6                         0.12                              2.0                         3.18                          0.97                                  0.16                           2.33                             167.56               5.26                                                                                                                      1.02                                      27.89                           2.39                           0.09                          0.07                             0.64                                4.61               0
3             2019-04-16 08:45:49.510    5.0               5154            5154000.0            LQA               6.18                5.343                                   8.83                        0.07                              0.8                         7.48                          0.72                                  0.17                           0.95                             327.73               2.56                                                                                                                      0.32                                      12.46                           1.82                           0.12                          0.06                             0.17                                4.66               0
"""

    # Leer el archivo CSV del dataframe pivote filtrado por el valor de la columna STATUS_y igual a "R"
    df_filtered_R_pivoted = pd.read_csv("Firstc_ArcBinario.csv")

    # Leer el archivo CSV del dataframe pivote filtrado por el valor de la columna STATUS_y igual a "A"
    df_filtered_A_pivoted = pd.read_csv("Firstc_df_filtered_A_pivoted.csv")

    # Filtrar el dataframe pivote filtrado por el valor de la columna ID_NUMERIC_ presente en el dataframe pivote filtrado por el valor de la columna STATUS_y igual a "R"
    df_filtered_A_pivotedMod = df_filtered_A_pivoted[
        df_filtered_A_pivoted["ID_NUMERIC_"].isin(df_filtered_R_pivoted["ID_NUMERIC_"])
    ]

    # Combinar los dataframes filtrados por el valor de la columna STATUS_y igual a "R"
    df_R_Final = (
        df_filtered_R_pivoted.set_index("ID_NUMERIC_")
        .combine_first(df_filtered_A_pivotedMod.set_index("ID_NUMERIC_"))
        .reset_index()
    )

    # Guardar el dataframe final combinado en un archivo CSV
    df_R_Final.to_csv("Firstc_df_R_Final.csv", index=False)

    """
ID_NUMERIC_   ATP   A_RECEP_FECHA_RECIB_       BIN_ATP   BIN_TF_LABORATORIO_   BIN_VALUE_Acidez (Al+H)   BIN_VALUE_Aluminio (Al) Intercambiable   BIN_VALUE_Arsénico (As) mg/kg   BIN_VALUE_Arsénico (As) mg/kg CONTROL   BIN_VALUE_Azufre (S) disponible   BIN_VALUE_Boro (B) Disponible   BIN_VALUE_Cadmio (Cd) disponible   BIN_VALUE_Cadmio (Cd) pseudototal   BIN_VALUE_Calcio (Ca) Soluble   BIN_VALUE_Calcio (Ca) disponible   BIN_VALUE_Capacidad Interc Catiónico Aceta (CICA)   BIN_VALUE_Capacidad Interc Catiónico Efect (CICE)   BIN_VALUE_Carbono Orgánico (CO)   BIN_VALUE_Carbono total (CT)   BIN_VALUE_Clase textural   BIN_VALUE_Cloruros (Cl-)   BIN_VALUE_Cobre (Cu) Disponible (Doble ácido)   BIN_VALUE_Cobre (Cu) olsen Disponible   BIN_VALUE_Conductividad eléctrica (CE) (1:5)   BIN_VALUE_Conductividad eléctrica (pasta saturada)   BIN_VALUE_Cromo (Cr) Pseudototal   BIN_VALUE_Fosforo (P) Disponible (Bray II)   BIN_VALUE_Hierro (Fe) Disponible (Doble ácido)   BIN_VALUE_Hierro (Fe) olsen Disponible   BIN_VALUE_Incertidumbre Calcio (Ca) disponible +/-   BIN_VALUE_Incertidumbre Conductividad eléctrica +/-   BIN_VALUE_Incertidumbre Magnesio (Mg) disp +/-   BIN_VALUE_Incertidumbre Potasio (K) disponible +/-   BIN_VALUE_Incertidumbre Sodio (Na) disponible +/-   BIN_VALUE_Magnesio (Mg) Disponible   BIN_VALUE_Magnesio (Mg) soluble   BIN_VALUE_Manganeso (Mn) Disponible (Doble ácido)   BIN_VALUE_Manganeso (Mn) olsen Disponible   BIN_VALUE_Materia Orgánica (MO)   BIN_VALUE_Mercurio (Hg) pseudototal   BIN_VALUE_Nitrógeno total (NT)   BIN_VALUE_Plomo (Pb) pseudototal   BIN_VALUE_Porcentaje de arcilla (% Ar)   BIN_VALUE_Porcentaje de arena (% A)   BIN_VALUE_Porcentaje de limo (% L)   BIN_VALUE_Potasio (K) Disponible   BIN_VALUE_Sodio (Na) Disponible   BIN_VALUE_Sodio (Na) Soluble   BIN_VALUE_Sulfatos (SO4)   BIN_VALUE_Zinc (Zn) Disponible (Doble ácido)   BIN_VALUE_Zinc (Zn) olsen Disponible   BIN_VALUE_pH (1:1)   "BIN_VALUE_pH (1:2,5)"   BIN_VALUE_pH (pasta saturada)   TF_CENTRO_POBLADO_   TF_DEPARTAMENTO_   TF_LABORATORIO_   TF_MUNICIPIO_   VALUE_Acidez (Al+H)   VALUE_Aluminio (Al) Intercambiable   VALUE_Arsénico (As) mg/kg   VALUE_Arsénico (As) mg/kg CONTROL   VALUE_Azufre (S) disponible   VALUE_Bicarbonatos (HCO3)-   VALUE_Boro (B) Disponible   VALUE_Cadmio (Cd) disponible   VALUE_Cadmio (Cd) pseudototal   VALUE_Calcio (Ca) Soluble   VALUE_Calcio (Ca) disponible   VALUE_Capacidad Interc Catiónico Aceta (CICA)   VALUE_Capacidad Interc Catiónico Efect (CICE)   VALUE_Carbonatos (CO3-2)   VALUE_Carbono Orgánico (CO)   VALUE_Carbono orgánico oxidable   VALUE_Carbono total (CT)   VALUE_Clase textural   VALUE_Cloruros (Cl-)   VALUE_Cobre (Cu) Disponible (Doble ácido)   VALUE_Cobre (Cu) olsen Disponible   VALUE_Cobre (Cu) pseudototal   VALUE_Conductividad eléctrica (CE) (1:5)   VALUE_Conductividad eléctrica (pasta saturada)   VALUE_Cromo (Cr) Pseudototal   VALUE_Fosforo (P) Disponible (Bray II)   VALUE_Hierro (Fe) Disponible (Doble ácido)   VALUE_Hierro (Fe) olsen Disponible   VALUE_Humedad gravimétrica 105 ºC (%)   VALUE_Incertidumbre Calcio (Ca) disponible +/-   VALUE_Incertidumbre Carbono Orgánico (CO) +/-   VALUE_Incertidumbre Cobre (Cu) +/-   VALUE_Incertidumbre Conductividad eléctrica +/-   VALUE_Incertidumbre Fosforo (P) (Bray II) +/-   VALUE_Incertidumbre Fosforo (P) Disponible +/-   VALUE_Incertidumbre Hierro (Fe) +/-   VALUE_Incertidumbre Magnesio (Mg) disp +/-   VALUE_Incertidumbre Manganeso (Mn) +/-   VALUE_Incertidumbre Potasio (K) disponible +/-   VALUE_Incertidumbre Sodio (Na) disponible +/-   VALUE_Incertidumbre Zinc (Zn) +/-   VALUE_Incertidumbre de Conductividad Eléctrica (CE) (1:5) +/-   VALUE_Incertidumbre de Cobre (Cu) olsen +/-   VALUE_Incertidumbre de Hierro (Fe) olsen +/-   VALUE_Incertidumbre de Manganeso (Mn) olsen +/-   VALUE_Incertidumbre de Zinc (Zn) olsen +/-   "VALUE_Incertidumbre de pH (1:2,5) +/-"   "VALUE_Incertidumbre pH (1:2,5) +/-"   VALUE_Magnesio (Mg) Disponible   VALUE_Magnesio (Mg) soluble   VALUE_Manganeso (Mn) Disponible (Doble ácido)   VALUE_Manganeso (Mn) olsen Disponible   VALUE_Materia Orgánica (MO)   VALUE_Mercurio (Hg) pseudototal   VALUE_Nitrógeno total (NT)   VALUE_Plomo (Pb) pseudototal   VALUE_Porcentaje de arcilla (% Ar)   VALUE_Porcentaje de arena (% A)   VALUE_Porcentaje de limo (% L)   VALUE_Porcentaje de sodio intercambiable (PSI)   VALUE_Potasio (K) Disponible   VALUE_Potasio (K) soluble   VALUE_Relacion de adsorcion de sodio (RAS)   VALUE_Saturación de agua   VALUE_Sodio (Na) Disponible   VALUE_Sodio (Na) Soluble   VALUE_Sulfatos (SO4)   VALUE_Zinc (Zn) Disponible (Doble ácido)   VALUE_Zinc (Zn) olsen Disponible   VALUE_Zinc (Zn) pseudototal   VALUE_incertidumbre Magnesio (Mg) disponible +/-   VALUE_incertidumbre Potasio (K) Disponible +/-   VALUE_incertidumbre Sodio (Na) Disponible +/-   VALUE_pH (1:1)   "VALUE_pH (1:2,5)"   VALUE_pH (1:5)   VALUE_pH (pasta saturada)
819           1     2019-04-26 11:17:11.130    1        1                       0                         0                                   0                                0                             0                            0                           0                                   1                            0                                   0                                 0                                   0                          0                          0                        0                         0                                         0                                   0                                         0                                   0                                             0                                                    0                                                        0                                                       0                                                       0                                                       0                                                     0                                                    0                                                    0                                                    0                                                0                                                0                                               0                                                0                                             0                                                    0                                                   0                                                    0                                                  0                                                 0                                                0                                                0                                                0                                                0                                                0                                                0                                                0                                                 0                                              25754000         25.0              LQA           25754.0          3.12                                                                                                                                                                                                                                                                                                                2.38                                                                                   28.68                                                                                                                                                                                                                                           10.39
851           1     2019-04-26 11:32:14.710    1        1                       0                         0                                   0                                0                             0                            0                           0                                   0                            1                                   0                                 0                                   0                          0                          0                        0                         0                                         0                                   0                                         0                                   0                                             0                                                    0                                                        0                                                       0                                                       0                                                       1                                                     0                                                    0                                                    0                                                    0                                                1                                                1                                               0                                                0                                             0                                                    0                                                   0                                                    0                                                  0                                                 1                                                1                                                0                                                0                                                0                                                0                                                0                                                0                                                 0                                              54385000         54.0              LQA           54385.0          0.0           0.0                                             9.5                                                                                     0.18                                                                                                                                                                                                                                                                                                                5.1                                                                                    8.85                                                                                                                                                     9.11                    71.94                                                                                                      710.79                                                                                                                                                                                                                                          1.22                    5.01            6.24                    19.61            61.98            18.41                    1.85                          0.68                                                                                                                                   5.97                                                                                                                                                      6.22
865           1     2019-04-26 13:57:51.897    1        1                       0                         0                                   0                                0                             0                            0                           0                                   0                            0                                   0                                 0                                   0                          0                          0                        0                         0                                         0                                   0                                         0                                   0                                             0                                                    0                                                        0                                                       0                                                       0                                                       0                                                     0                                                    0                                                    0                                                    0                                                0                                                0                                               0                                                0                                             0                                                    0                                                   0                                                    0                                                  0                                                 0                                                0                                                0                                                0                                                0                                                0                                                0                                                0                                                 0                                              54820000         54.0              LQA           54820            0.0           0.0                                            18.78                                                                                    0.77                                                                                                                                                                                                                                                                                                               21.26                                                                                   29.52                                                                                                                                                     0.61                    16.92                                                                                                       31.72                                                                                                                                                                                                                                           7.71                    9.84            10.81                   27.39            39.27            33.34                    0.44                          0.11                                                                                                                                   8.2                                                                                                                                                       6.52
"""

    # Concatenar los dataframes df_R_Final y df_filtered_A_pivoted
    df_resultado = pd.concat(
        [df_R_Final, df_filtered_A_pivoted], axis=0, ignore_index=True
    )

    # Obtener las columnas de df_filtered_A_pivoted
    columnas_A = df_filtered_A_pivoted.columns.tolist()

    # Obtener las columnas de df_resultado
    columnas_resultado = df_resultado.columns.tolist()

    # Crear una nueva lista de columnas que comienza con las columnas de df_filtered_A_pivoted
    # seguido de las columnas en df_resultado que no están en df_filtered_A_pivoted
    columnas_nuevas = columnas_A + [
        col for col in columnas_resultado if col not in columnas_A
    ]

    # Reordenar las columnas de df_resultado
    df_resultado = df_resultado[columnas_nuevas]
    # Filtrar por la columna "ATP" Autorizadas
    df_filtered = df_resultado[df_resultado["ATP"] == 0]
    df_resultado = df_resultado[df_resultado["ATP"] == 1]

    # Reemplazar los valores NaN por 0 en las columnas que comienzan con "BIN"
    bin_columns = [col for col in df_filtered.columns if col.startswith("BIN")]
    df_filtered[bin_columns] = df_filtered[bin_columns].fillna(0)

    # Añadir las filas de df_filtered a df_resultado
    df_resultado = pd.concat([df_resultado, df_filtered], axis=0, ignore_index=True)
    # Imprimir el dataframe resultado
    print(df_resultado)

    # Guardar el dataframe resultado en un archivo CSV
    df_resultado.to_csv("Firstc_df_resultado.csv", index=False)

    """
ID_NUMERIC_   A_RECEP_FECHA_RECIB_       TF_DEPARTAMENTO_   TF_MUNICIPIO_   TF_CENTRO_POBLADO_   TF_LABORATORIO_   VALUE_Acidez (Al+H)   VALUE_Aluminio (Al) Intercambiable   VALUE_Arsénico (As) mg/kg   VALUE_Arsénico (As) mg/kg CONTROL   VALUE_Azufre (S) disponible   VALUE_Bicarbonatos (HCO3)-   VALUE_Boro (B) Disponible   VALUE_Cadmio (Cd) disponible   VALUE_Cadmio (Cd) pseudototal   VALUE_Calcio (Ca) Soluble   VALUE_Calcio (Ca) disponible   VALUE_Capacidad Interc Catiónico Aceta (CICA)   VALUE_Capacidad Interc Catiónico Efect (CICE)   VALUE_Carbonatos (CO3-2)   VALUE_Carbono Orgánico (CO)   VALUE_Carbono orgánico oxidable   VALUE_Carbono total (CT)   VALUE_Clase textural   VALUE_Cloruros (Cl-)   VALUE_Cobre (Cu) Disponible (Doble ácido)   VALUE_Cobre (Cu) olsen Disponible   VALUE_Cobre (Cu) pseudototal   VALUE_Conductividad eléctrica (CE) (1:5)   VALUE_Conductividad eléctrica (pasta saturada)   VALUE_Cromo (Cr) Pseudototal   VALUE_Fosforo (P) Disponible (Bray II)   VALUE_Hierro (Fe) Disponible (Doble ácido)   VALUE_Hierro (Fe) olsen Disponible   VALUE_Humedad gravimétrica 105 ºC (%)   VALUE_Incertidumbre Calcio (Ca) disponible +/-   VALUE_Incertidumbre Carbono Orgánico (CO) +/-   VALUE_Incertidumbre Cobre (Cu) +/-   VALUE_Incertidumbre Conductividad eléctrica +/-   VALUE_Incertidumbre Fosforo (P) (Bray II) +/-   VALUE_Incertidumbre Fosforo (P) Disponible +/-   VALUE_Incertidumbre Hierro (Fe) +/-   VALUE_Incertidumbre Magnesio (Mg) disp +/-   VALUE_Incertidumbre Manganeso (Mn) +/-   VALUE_Incertidumbre Potasio (K) disponible +/-   VALUE_Incertidumbre Sodio (Na) disponible +/-   VALUE_Incertidumbre Zinc (Zn) +/-   VALUE_Incertidumbre de Conductividad Eléctrica (CE) (1:5) +/-   VALUE_Incertidumbre de Cobre (Cu) olsen +/-   VALUE_Incertidumbre de Hierro (Fe) olsen +/-   VALUE_Incertidumbre de Manganeso (Mn) olsen +/-   VALUE_Incertidumbre de Zinc (Zn) olsen +/-   "VALUE_Incertidumbre de pH (1:2,5) +/-"   "VALUE_Incertidumbre pH (1:2,5) +/-"   VALUE_Magnesio (Mg) Disponible   VALUE_Magnesio (Mg) soluble   VALUE_Manganeso (Mn) Disponible (Doble ácido)   VALUE_Manganeso (Mn) olsen Disponible   VALUE_Materia Orgánica (MO)   VALUE_Mercurio (Hg) pseudototal   VALUE_Nitrógeno total (NT)   VALUE_Plomo (Pb) pseudototal   VALUE_Porcentaje de arcilla (% Ar)   VALUE_Porcentaje de arena (% A)   VALUE_Porcentaje de limo (% L)   VALUE_Porcentaje de sodio intercambiable (PSI)   VALUE_Potasio (K) Disponible   VALUE_Potasio (K) soluble   VALUE_Relacion de adsorcion de sodio (RAS)   VALUE_Saturación de agua   VALUE_Sodio (Na) Disponible   VALUE_Sodio (Na) Soluble   VALUE_Sulfatos (SO4)   VALUE_Zinc (Zn) Disponible (Doble ácido)   VALUE_Zinc (Zn) olsen Disponible   VALUE_Zinc (Zn) pseudototal   VALUE_incertidumbre Magnesio (Mg) disponible +/-   VALUE_incertidumbre Potasio (K) Disponible +/-   VALUE_incertidumbre Sodio (Na) Disponible +/-   VALUE_pH (1:1)   "VALUE_pH (1:2,5)"   VALUE_pH (1:5)   VALUE_pH (pasta saturada)   ATP   BIN_ATP   BIN_TF_LABORATORIO_   BIN_VALUE_Acidez (Al+H)   BIN_VALUE_Aluminio (Al) Intercambiable   BIN_VALUE_Arsénico (As) mg/kg   BIN_VALUE_Arsénico (As) mg/kg CONTROL   BIN_VALUE_Azufre (S) disponible   BIN_VALUE_Boro (B) Disponible   BIN_VALUE_Cadmio (Cd) disponible   BIN_VALUE_Cadmio (Cd) pseudototal   BIN_VALUE_Calcio (Ca) Soluble   BIN_VALUE_Calcio (Ca) disponible   BIN_VALUE_Capacidad Interc Catiónico Aceta (CICA)   BIN_VALUE_Capacidad Interc Catiónico Efect (CICE)   BIN_VALUE_Carbono Orgánico (CO)   BIN_VALUE_Carbono total (CT)   BIN_VALUE_Clase textural   BIN_VALUE_Cloruros (Cl-)   BIN_VALUE_Cobre (Cu) Disponible (Doble ácido)   BIN_VALUE_Cobre (Cu) olsen Disponible   BIN_VALUE_Conductividad eléctrica (CE) (1:5)   BIN_VALUE_Conductividad eléctrica (pasta saturada)   BIN_VALUE_Cromo (Cr) Pseudototal   BIN_VALUE_Fosforo (P) Disponible (Bray II)   BIN_VALUE_Hierro (Fe) Disponible (Doble ácido)   BIN_VALUE_Hierro (Fe) olsen Disponible   BIN_VALUE_Incertidumbre Calcio (Ca) disponible +/-   BIN_VALUE_Incertidumbre Conductividad eléctrica +/-   BIN_VALUE_Incertidumbre Magnesio (Mg) disp +/-   BIN_VALUE_Incertidumbre Potasio (K) disponible +/-   BIN_VALUE_Incertidumbre Sodio (Na) disponible +/-   BIN_VALUE_Magnesio (Mg) Disponible   BIN_VALUE_Magnesio (Mg) soluble   BIN_VALUE_Manganeso (Mn) Disponible (Doble ácido)   BIN_VALUE_Manganeso (Mn) olsen Disponible   BIN_VALUE_Materia Orgánica (MO)   BIN_VALUE_Mercurio (Hg) pseudototal   BIN_VALUE_Nitrógeno total (NT)   BIN_VALUE_Plomo (Pb) pseudototal   BIN_VALUE_Porcentaje de arcilla (% Ar)   BIN_VALUE_Porcentaje de arena (% A)   BIN_VALUE_Porcentaje de limo (% L)   BIN_VALUE_Potasio (K) Disponible   BIN_VALUE_Sodio (Na) Disponible   BIN_VALUE_Sodio (Na) Soluble   BIN_VALUE_Sulfatos (SO4)   BIN_VALUE_Zinc (Zn) Disponible (Doble ácido)   BIN_VALUE_Zinc (Zn) olsen Disponible   BIN_VALUE_pH (1:1)   "BIN_VALUE_pH (1:2,5)"   BIN_VALUE_pH (pasta saturada)
819           2019-04-26 11:17:11.130    25.0              25754.0         25754000             LQA               3.12                                                                                                                                                                                                                                                     2.38                                                                                                                                                                                                                                                28.68                                                                                                                                                                                                                                           10.39                                                                                                                                                                                                                                                                            1       1.0       1.0                   0.0                                  0.0                                               0.0                                             0.0                                        0.0                                        0.0                                         0.0                                        1.0                                              0.0                                         0.0                                         0.0                                               0.0                                              0.0                                               0.0                                              0.0                                               0.0                                          0.0                                         0.0                                               0.0                                          0.0                                               0.0                                            0.0                                            0.0                                               0.0                                            0.0                                           0.0                                               0.0                                            0.0                                              0.0                                            0.0                                            0.0                                            0.0                                            0.0                                            0.0                                              0.0                                            0.0                                              0.0                                              0.0                                           0.0                                              0.0
851           2019-04-26 11:32:14.710    54.0              54385.0         54385000             LQA               0.0                 0.0                        9.5                       0.18                                                                                                                                                      5.1                           8.85                                                                                                                                                                                                                                                     9.11                    71.94                                                                                                                                                                                                                  710.79                                                                                                                                                                                                                                           1.22                    5.01            6.24                    19.61            61.98            18.41                    1.85                          0.68                                                                                                                                   5.97                                                                                                                                                      6.22                    1       1.0       1.0                   0.0                                  0.0                                               0.0                                             0.0                                        0.0                                        0.0                                         0.0                                        0.0                                              1.0                                         0.0                                         0.0                                               0.0                                              0.0                                               0.0                                              0.0                                               0.0                                          0.0                                         0.0                                               0.0                                          0.0                                               0.0                                            0.0                                            0.0                                               0.0                                            0.0                                           0.0                                               0.0                                            1.0                                              0.0                                            0.0                                            0.0                                            0.0                                            0.0                                              0.0                                            0.0                                              1.0                                              1.0                                           0.0                                              0.0
865           2019-04-26 13:57:51.897    54.0              54820           54820000             LQA               0.0                 0.0                       18.78                      0.77                                                                                                                                                     21.26                          29.52                                                                                                                                                                                                                                                    0.61                    16.92                                                                                                                                                                                                                  31.72                                                                                                                                                                                                                                           7.71                    9.84            10.81                   27.39            39.27            33.34                    0.44                          0.11                                                                                                                                   8.2                                                                                                                                                       6.52                    1       1.0       1.0                   0.0                                  0.0                                               0.0                                             0.0                                        0.0                                        0.0                                         0.0                                        0.0                                              0.0                                         0.0                                         0.0                                               0.0                                              0.0                                               0.0                                              0.0                                               0.0                                          0.0                                         0.0                                               0.0                                          0.0                                               0.0                                            0.0                                            0.0                                               0.0                                            0.0                                           0.0                                               0.0                                            0.0                                              0.0                                            0.0                                            0.0                                            0.0                                            0.0                                              1.0                                            0.0                                              0.0                                              0.0                                           0.0                                              0.0
"""

    # Leer el archivo CSV del dataframe resultado
    df = pd.read_csv("Firstc_df_resultado.csv")

    # Reemplazar ".0" en las columnas específicas
    cols_especificas = ["TF_DEPARTAMENTO_", "TF_MUNICIPIO_", "TF_CENTRO_POBLADO_"]
    df[cols_especificas] = (
        df[cols_especificas].astype(str).replace(r"\.0", "", regex=True)
    )

    # Convertir las columnas específicas a enteros
    df[cols_especificas] = df[cols_especificas].astype(int, errors="ignore")

    # Definir las columnas del dataframe para la tabla RESULT
    columnas_departamento = ["IDENTITY", "NOMBRE"]

    # Crear el dataframe para la tabla DEPARTAMENTO
    df_departamento = pd.DataFrame(
        [tuple(t) for t in rows_departamento], columns=columnas_departamento
    )

    # Definir las columnas del dataframe para la tabla RESULT
    columnas_municipio = ["IDENTITY", "NOMBRE"]

    # Crear el dataframe para la tabla MUNICIPIO
    df_municipio = pd.DataFrame(
        [tuple(t) for t in rows_municipio], columns=columnas_municipio
    )

    # Definir las columnas del dataframe para la tabla RESULT
    columnas_poblado = ["IDENTITY", "NOMBRE"]

    # Crear el dataframe para la tabla POBLADO
    df_poblado = pd.DataFrame(
        [tuple(t) for t in rows_poblado], columns=columnas_poblado
    )

    # Duplicar y reemplazar los identitys por los nombres en la columna "DEPARTAMENTO"
    df["TF_DEPARTAMENTO_NOMBRE"] = df["TF_DEPARTAMENTO_"].map(
        df_departamento.set_index("IDENTITY")["NOMBRE"]
    )

    # Duplicar y reemplazar los identitys por los nombres en la columna "MUNICIPIO"
    df["TF_MUNICIPIO_NOMBRE"] = df["TF_MUNICIPIO_"].map(
        df_municipio.set_index("IDENTITY")["NOMBRE"]
    )

    # Duplicar y reemplazar los identitys por los nombres en la columna "CENTRO_POBLADO"
    df["TF_CENTRO_POBLADO_NOMBRE"] = df["TF_CENTRO_POBLADO_"].map(
        df_poblado.set_index("IDENTITY")["NOMBRE"]
    )

    # Guardar el dataframe resultado en un archivo CSV
    df.to_csv("Firstc_df_resultadoV2.csv", index=False)

    """
ID_NUMERIC_   A_RECEP_FECHA_RECIB_       TF_DEPARTAMENTO_   TF_MUNICIPIO_   TF_CENTRO_POBLADO_   TF_LABORATORIO_   VALUE_Acidez (Al+H)   VALUE_Aluminio (Al) Intercambiable   VALUE_Arsénico (As) mg/kg   VALUE_Arsénico (As) mg/kg CONTROL   VALUE_Azufre (S) disponible   VALUE_Bicarbonatos (HCO3)-   VALUE_Boro (B) Disponible   VALUE_Cadmio (Cd) disponible   VALUE_Cadmio (Cd) pseudototal   VALUE_Calcio (Ca) Soluble   VALUE_Calcio (Ca) disponible   VALUE_Capacidad Interc Catiónico Aceta (CICA)   VALUE_Capacidad Interc Catiónico Efect (CICE)   VALUE_Carbonatos (CO3-2)   VALUE_Carbono Orgánico (CO)   VALUE_Carbono orgánico oxidable   VALUE_Carbono total (CT)   VALUE_Clase textural   VALUE_Cloruros (Cl-)   VALUE_Cobre (Cu) Disponible (Doble ácido)   VALUE_Cobre (Cu) olsen Disponible   VALUE_Cobre (Cu) pseudototal   VALUE_Conductividad eléctrica (CE) (1:5)   VALUE_Conductividad eléctrica (pasta saturada)   VALUE_Cromo (Cr) Pseudototal   VALUE_Fosforo (P) Disponible (Bray II)   VALUE_Hierro (Fe) Disponible (Doble ácido)   VALUE_Hierro (Fe) olsen Disponible   VALUE_Humedad gravimétrica 105 ºC (%)   VALUE_Incertidumbre Calcio (Ca) disponible +/-   VALUE_Incertidumbre Carbono Orgánico (CO) +/-   VALUE_Incertidumbre Cobre (Cu) +/-   VALUE_Incertidumbre Conductividad eléctrica +/-   VALUE_Incertidumbre Fosforo (P) (Bray II) +/-   VALUE_Incertidumbre Fosforo (P) Disponible +/-   VALUE_Incertidumbre Hierro (Fe) +/-   VALUE_Incertidumbre Magnesio (Mg) disp +/-   VALUE_Incertidumbre Manganeso (Mn) +/-   VALUE_Incertidumbre Potasio (K) disponible +/-   VALUE_Incertidumbre Sodio (Na) disponible +/-   VALUE_Incertidumbre Zinc (Zn) +/-   VALUE_Incertidumbre de Conductividad Eléctrica (CE) (1:5) +/-   VALUE_Incertidumbre de Cobre (Cu) olsen +/-   VALUE_Incertidumbre de Hierro (Fe) olsen +/-   VALUE_Incertidumbre de Manganeso (Mn) olsen +/-   VALUE_Incertidumbre de Zinc (Zn) olsen +/-   "VALUE_Incertidumbre de pH (1:2,5) +/-"   "VALUE_Incertidumbre pH (1:2,5) +/-"   VALUE_Magnesio (Mg) Disponible   VALUE_Magnesio (Mg) soluble   VALUE_Manganeso (Mn) Disponible (Doble ácido)   VALUE_Manganeso (Mn) olsen Disponible   VALUE_Materia Orgánica (MO)   VALUE_Mercurio (Hg) pseudototal   VALUE_Nitrógeno total (NT)   VALUE_Plomo (Pb) pseudototal   VALUE_Porcentaje de arcilla (% Ar)   VALUE_Porcentaje de arena (% A)   VALUE_Porcentaje de limo (% L)   VALUE_Porcentaje de sodio intercambiable (PSI)   VALUE_Potasio (K) Disponible   VALUE_Potasio (K) soluble   VALUE_Relacion de adsorcion de sodio (RAS)   VALUE_Saturación de agua   VALUE_Sodio (Na) Disponible   VALUE_Sodio (Na) Soluble   VALUE_Sulfatos (SO4)   VALUE_Zinc (Zn) Disponible (Doble ácido)   VALUE_Zinc (Zn) olsen Disponible   VALUE_Zinc (Zn) pseudototal   VALUE_incertidumbre Magnesio (Mg) disponible +/-   VALUE_incertidumbre Potasio (K) Disponible +/-   VALUE_incertidumbre Sodio (Na) Disponible +/-   VALUE_pH (1:1)   "VALUE_pH (1:2,5)"   VALUE_pH (1:5)   VALUE_pH (pasta saturada)   ATP   BIN_ATP   BIN_TF_LABORATORIO_   BIN_VALUE_Acidez (Al+H)   BIN_VALUE_Aluminio (Al) Intercambiable   BIN_VALUE_Arsénico (As) mg/kg   BIN_VALUE_Arsénico (As) mg/kg CONTROL   BIN_VALUE_Azufre (S) disponible   BIN_VALUE_Boro (B) Disponible   BIN_VALUE_Cadmio (Cd) disponible   BIN_VALUE_Cadmio (Cd) pseudototal   BIN_VALUE_Calcio (Ca) Soluble   BIN_VALUE_Calcio (Ca) disponible   BIN_VALUE_Capacidad Interc Catiónico Aceta (CICA)   BIN_VALUE_Capacidad Interc Catiónico Efect (CICE)   BIN_VALUE_Carbono Orgánico (CO)   BIN_VALUE_Carbono total (CT)   BIN_VALUE_Clase textural   BIN_VALUE_Cloruros (Cl-)   BIN_VALUE_Cobre (Cu) Disponible (Doble ácido)   BIN_VALUE_Cobre (Cu) olsen Disponible   BIN_VALUE_Conductividad eléctrica (CE) (1:5)   BIN_VALUE_Conductividad eléctrica (pasta saturada)   BIN_VALUE_Cromo (Cr) Pseudototal   BIN_VALUE_Fosforo (P) Disponible (Bray II)   BIN_VALUE_Hierro (Fe) Disponible (Doble ácido)   BIN_VALUE_Hierro (Fe) olsen Disponible   BIN_VALUE_Incertidumbre Calcio (Ca) disponible +/-   BIN_VALUE_Incertidumbre Conductividad eléctrica +/-   BIN_VALUE_Incertidumbre Magnesio (Mg) disp +/-   BIN_VALUE_Incertidumbre Potasio (K) disponible +/-   BIN_VALUE_Incertidumbre Sodio (Na) disponible +/-   BIN_VALUE_Magnesio (Mg) Disponible   BIN_VALUE_Magnesio (Mg) soluble   BIN_VALUE_Manganeso (Mn) Disponible (Doble ácido)   BIN_VALUE_Manganeso (Mn) olsen Disponible   BIN_VALUE_Materia Orgánica (MO)   BIN_VALUE_Mercurio (Hg) pseudototal   BIN_VALUE_Nitrógeno total (NT)   BIN_VALUE_Plomo (Pb) pseudototal   BIN_VALUE_Porcentaje de arcilla (% Ar)   BIN_VALUE_Porcentaje de arena (% A)   BIN_VALUE_Porcentaje de limo (% L)   BIN_VALUE_Potasio (K) Disponible   BIN_VALUE_Sodio (Na) Disponible   BIN_VALUE_Sodio (Na) Soluble   BIN_VALUE_Sulfatos (SO4)   BIN_VALUE_Zinc (Zn) Disponible (Doble ácido)   BIN_VALUE_Zinc (Zn) olsen Disponible   BIN_VALUE_pH (1:1)   "BIN_VALUE_pH (1:2,5)"   BIN_VALUE_pH (pasta saturada)   TF_DEPARTAMENTO_NOMBRE   TF_MUNICIPIO_NOMBRE   TF_CENTRO_POBLADO_NOMBRE
819           2019-04-26 11:17:11.130    25                 25754           25754000            LQA               3.12                                                                                                                                                                                                                                                     2.38                                                                                                                                                                                                                                                28.68                                                                                                                                                                                                                                           10.39                                                                                                                                                                                                                                                                            1       1.0       1.0                   0.0                                  0.0                                               0.0                                             0.0                                        0.0                                        0.0                                         0.0                                        1.0                                              0.0                                         0.0                                         0.0                                               0.0                                              0.0                                               0.0                                              0.0                                               0.0                                          0.0                                         0.0                                               0.0                                          0.0                                               0.0                                            0.0                                            0.0                                               0.0                                            0.0                                           0.0                                               0.0                                            0.0                                              0.0                                            0.0                                            0.0                                            0.0                                            0.0                                              0.0                                            0.0                                              0.0                                              0.0                                           0.0                                              0.0                                           CUNDINAMARCA          SOACHA           SOACHA
851           2019-04-26 11:32:14.710    54                 54385           54385000            LQA               0.0                 0.0                        9.5                       0.18                                                                                                                                                      5.1                           8.85                                                                                                                                                                                                                                                     9.11                    71.94                                                                                                                                                                                                                  710.79                                                                                                                                                                                                                                           1.22                    5.01            6.24                    19.61            61.98            18.41                    1.85                          0.68                                                                                                                                   5.97                                                                                                                                                      6.22                    1       1.0       1.0                   0.0                                  0.0                                                0.0                                             0.0                                        0.0                                        0.0                                         0.0                                        0.0                                              1.0                                         0.0                                         0.0                                               0.0                                              0.0                                               0.0                                              0.0                                               0.0                                          0.0                                         0.0                                               0.0                                          0.0                                               0.0                                            0.0                                            0.0                                               0.0                                            0.0                                           0.0                                               0.0                                            1.0                                              0.0                                            0.0                                            0.0                                            0.0                                            0.0                                              0.0                                            0.0                                              1.0                                              1.0                                           0.0                                              0.0                                           NORTE DE SANTANDER     LA ESPERANZA     LA ESPERANZA
865           2019-04-26 13:57:51.897    54                 54820           54820000            LQA               0.0                 0.0                       18.78                      0.77                                                                                                                                                     21.26                          29.52                                                                                                                                                                                                                                                    0.61                    16.92                                                                                                                                                                                                                  31.72                                                                                                                                                                                                                                           7.71                    9.84            10.81                   27.39            39.27            33.34                    0.44                          0.11                                                                                                                                   8.2                                                                                                                                                       6.52                    1       1.0       1.0                   0.0                                  0.0                                                0.0                                             0.0                                        0.0                                        0.0                                         0.0                                        0.0                                              0.0                                         0.0                                         0.0                                               0.0                                              0.0                                               0.0                                              0.0                                               0.0                                          0.0                                         0.0                                               0.0                                          0.0                                               0.0                                            0.0                                            0.0                                               0.0                                            0.0                                           0.0                                               0.0                                            0.0                                              0.0                                            0.0                                            0.0                                            0.0                                            0.0                                              1.0                                            0.0                                              0.0                                              0.0                                           0.0                                              0.0                                           NORTE DE SANTANDER     TOLEDO           TOLEDO
"""

    return df


if __name__ == "__main__":
    # Obtener la hora de inicio de la ejecución
    start_time = datetime.datetime.now()
    df_active = mainActive()
    df_commit = mainCommit()

    print("df_active")
    print(df_active)
    print("df_commit")
    print(df_commit)

    # Imprimir la cantidad de filas y columnas de cada dataframe
    print("Cantidad de filas en df_active:", len(df_active))
    print("Cantidad de columnas en df_active:", len(df_active.columns))
    print("Cantidad de filas en df_commit:", len(df_commit))
    print("Cantidad de columnas en df_commit:", len(df_commit.columns))

    # Combinar los dataframes
    df_combined = pd.concat([df_active, df_commit])

    # Divide el DataFrame en dos partes
    df1 = df_combined.iloc[:, :6]
    df2 = df_combined.iloc[:, 6:]

    # Ordena las columnas del segundo DataFrame
    df2 = df2.reindex(sorted(df2.columns), axis=1)

    # Concatena los dos DataFrames
    df_combined = pd.concat([df1, df2], axis=1)

    # Imprimir la cantidad de filas y columnas del dataframe combinado
    print("Cantidad de filas en df_combined:", len(df_combined))
    print("Cantidad de columnas en df_combined:", len(df_combined.columns))

    # Crear una columna "non_null_columns" que contiene las columnas no nulas de cada fila
    df_combined["non_null_columns"] = df_combined.apply(
        lambda row: "".join(row.dropna().index.values), axis=1
    )

    # Ordenar el dataframe por la columna "non_null_columns"
    df_combined = df_combined.sort_values(by="non_null_columns")

    # Crear una columna "contador" que cuenta los cambios en la columna "non_null_columns"
    df_combined["contador"] = (
        df_combined["non_null_columns"] != df_combined["non_null_columns"].shift()
    ).cumsum()

    # Crear una columna "grupo" que contiene el nombre del grupo basado en la columna "contador"
    df_combined["grupo"] = "grupo " + df_combined["contador"].astype(str)

    # Iterar sobre cada grupo del dataframe
    for name, group_df in df_combined.groupby("grupo"):
        # Eliminar las columnas "non_null_columns", "contador" y "grupo" del grupo
        group_df = group_df.drop(columns=["non_null_columns", "contador", "grupo"])

        # Eliminar las columnas con valores nulos en el grupo
        group_df = group_df.dropna(axis=1, how="all")
        # Obtener las columnas que comienzan con "VALUE"
        value_columns = [col for col in group_df.columns if col.startswith("VALUE")]

        # Obtener la cantidad de columnas que comienzan con "VALUE"
        num_value_columns = len(value_columns)
        # Guardar el grupo en un archivo CSV con el nombre del grupo, la cantidad de filas y la cantidad de columnas que comienzan con "VALUE"
        group_df.to_csv(
            f"{name}_rows_{len(group_df)}_value_columns_{num_value_columns}.csv",
            index=False,
        )

    # Obtener la hora de finalización de la ejecución
    end_time = datetime.datetime.now()

    # Calcular la duración de la ejecución
    duration = end_time - start_time

    # Imprimir la duración de la ejecución
    print("Duración de la ejecución:", duration)

    # Guardar el dataframe resultado en un archivo CSV
    df_combined.to_csv("Firstall_df_resultado.csv", index=False)



