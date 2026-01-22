import pandas as pd
import datetime
import pyodbc

def establecer_conexion():
    conn = pyodbc.connect(
        r"DRIVER={SQL Server};SERVER=COMOSPSPSQL01\SQL2019LS;DATABASE=LimsSampleManager;UID=UserSMInfo;PWD=BlIhFEJcAXpf6aFDZBLS"
    )
    return conn

def ejecutar_consulta(conn, consulta):
    cursor = conn.cursor()
    cursor.execute(consulta)
    rows = cursor.fetchall()
    return rows

def cerrar_conexion(conn):
    conn.close()

def filtrar_dataframe(df, columna, valores):
    return df[df[columna].isin(valores)]

def mainCommit():
    # Conexión y consultas
    conn = establecer_conexion()
    consulta_sample = "SELECT ID_NUMERIC, ID_TEXT, A_RECEP_FECHA_RECIB, STATUS, TF_DEPARTAMENTO, TF_MUNICIPIO, TF_CENTRO_POBLADO , TF_CULTIVO, TF_LABORATORIO FROM C_SAMPLE"
    consulta_test = "SELECT TEST_NUMBER, ANALYSIS, SAMPLE, STATUS FROM C_TEST"
    consulta_result = "SELECT TEST_NUMBER, NAME, VALUE, TEXT, REP_CONTROL FROM C_RESULT"
    consulta_departamento = "SELECT [IDENTITY], NOMBRE FROM TF_DEPARTAMENTO"
    consulta_municipio = "SELECT [IDENTITY], NOMBRE FROM TF_MUNICIPIO"
    consulta_poblado = "SELECT [IDENTITY], NOMBRE FROM TF_POBLADO"

    rows_sample = ejecutar_consulta(conn, consulta_sample)
    rows_test = ejecutar_consulta(conn, consulta_test)
    rows_result = ejecutar_consulta(conn, consulta_result)
    rows_departamento = ejecutar_consulta(conn, consulta_departamento)
    rows_municipio = ejecutar_consulta(conn, consulta_municipio)
    rows_poblado = ejecutar_consulta(conn, consulta_poblado)
    cerrar_conexion(conn)

    # DataFrames
    columnas_sample = [
        "ID_NUMERIC", "ID_TEXT", "A_RECEP_FECHA_RECIB", "STATUS",
        "TF_DEPARTAMENTO", "TF_MUNICIPIO", "TF_CENTRO_POBLADO",
        "TF_CULTIVO", "TF_LABORATORIO"
    ]
    df_sample = pd.DataFrame([tuple(t) for t in rows_sample], columns=columnas_sample)
    df_sample = filtrar_dataframe(df_sample, "TF_LABORATORIO", ["LQA"])
    df_sample = filtrar_dataframe(df_sample, "STATUS", ["C"])
    df_sample["ID_NUMERIC"] = pd.to_numeric(df_sample["ID_NUMERIC"])

    columnas_test = ["TEST_NUMBER", "ANALYSIS", "SAMPLE", "STATUS"]
    df_test = pd.DataFrame([tuple(t) for t in rows_test], columns=columnas_test)
    df_test = filtrar_dataframe(df_test, "STATUS", ["C"])
    df_test["SAMPLE"] = pd.to_numeric(df_test["SAMPLE"])

    columnas_result = ["TEST_NUMBER", "NAME", "VALUE", "TEXT", "REP_CONTROL"]
    df_result = pd.DataFrame([tuple(t) for t in rows_result], columns=columnas_result)
    df_result_rpt = filtrar_dataframe(df_result, "REP_CONTROL", ["RPT"])

    # Guardar CSVs base
    df_sample.to_csv("Completed_exportSample.csv", index=False)
    df_test.to_csv("Completed_exportTest.csv", index=False)
    df_result.to_csv("Completed_exportResult.csv", index=False)

    # Merge y filtrado
    merged_df = pd.merge(
        df_sample, df_test, left_on="ID_NUMERIC", right_on="SAMPLE", how="left"
    )
    merged_df.to_csv("Completed_exportMerge.csv", index=False)

    final_merged_df = pd.merge(merged_df, df_result_rpt, on="TEST_NUMBER", how="inner")

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
    final_merged_df = final_merged_df[final_merged_df["NAME"].isin(parametros_deseados)]
    final_merged_df.to_csv("Completed_exportFinal.csv", index=False)

    # Pivot
    df_pivoted = final_merged_df.pivot_table(
        index=[
            "ID_NUMERIC", "A_RECEP_FECHA_RECIB", "TF_DEPARTAMENTO",
            "TF_MUNICIPIO", "TF_CENTRO_POBLADO", "TF_CULTIVO", "TF_LABORATORIO"
        ],
        columns=["NAME"],
        values=["VALUE"],
        aggfunc="first"
    ).reset_index()

    df_pivoted.columns = [
        "_".join(col).strip() if col[1] else col[0] for col in df_pivoted.columns.values
    ]
    df_pivoted.to_csv("Completed_df_pivoted.csv", index=False)

    # Reemplazo de IDs por nombres
    cols_especificas = ["TF_DEPARTAMENTO_", "TF_MUNICIPIO_", "TF_CENTRO_POBLADO_"]
    for col in cols_especificas:
        if col in df_pivoted.columns:
            df_pivoted[col] = df_pivoted[col].astype(str).replace(r"\.0", "", regex=True)
            df_pivoted[col] = df_pivoted[col].astype(int, errors="ignore")

    df_departamento = pd.DataFrame([tuple(t) for t in rows_departamento], columns=["IDENTITY", "NOMBRE"])
    df_municipio = pd.DataFrame([tuple(t) for t in rows_municipio], columns=["IDENTITY", "NOMBRE"])
    df_poblado = pd.DataFrame([tuple(t) for t in rows_poblado], columns=["IDENTITY", "NOMBRE"])

    if "TF_DEPARTAMENTO_" in df_pivoted.columns:
        df_pivoted["TF_DEPARTAMENTO_NOMBRE"] = df_pivoted["TF_DEPARTAMENTO_"].map(
            df_departamento.set_index("IDENTITY")["NOMBRE"]
        )
    if "TF_MUNICIPIO_" in df_pivoted.columns:
        df_pivoted["TF_MUNICIPIO_NOMBRE"] = df_pivoted["TF_MUNICIPIO_"].map(
            df_municipio.set_index("IDENTITY")["NOMBRE"]
        )
    if "TF_CENTRO_POBLADO_" in df_pivoted.columns:
        df_pivoted["TF_CENTRO_POBLADO_NOMBRE"] = df_pivoted["TF_CENTRO_POBLADO_"].map(
            df_poblado.set_index("IDENTITY")["NOMBRE"]
        )

    df_pivoted.to_csv("Completed_df_pivotedV2.csv", index=False)
    return df_pivoted

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    df_commit = mainCommit()
    print("df_commit")
    print(df_commit)
    print("Cantidad de filas en df_commit:", len(df_commit))
    print("Cantidad de columnas en df_commit:", len(df_commit.columns))
    end_time = datetime.datetime.now()
    print("Duración de la ejecución:", end_time - start_time)
    df_commit.to_csv("Completed_final.csv", index=False)