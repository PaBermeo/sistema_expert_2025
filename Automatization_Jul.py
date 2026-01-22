import pandas as pd

# # Cargar datos desde un archivo CSV
df = pd.read_csv("predictors.csv")  # Cambia el nombre si tu archivo tiene otro

# Crear una lista para almacenar observaciones
observaciones = []

# Iterar sobre cada fila para verificar las condiciones
for index, row in df.iterrows():
    obs = []

    # Asignación de variables para legibilidad
    pH = row["VALUE_pH (1:2,5)"]
    CE = row["VALUE_Conductividad eléctrica (CE) (1:5)"]
    Al_H = row["VALUE_Acidez (Al+H)"]
    Al = row["VALUE_Aluminio (Al) Intercambiable"]
    S = row["VALUE_Azufre (S) disponible"]
    P = row["VALUE_Fosforo (P) Disponible (Bray II)"]
    Ca = row["VALUE_Calcio (Ca) disponible"]
    Mg = row["VALUE_Magnesio (Mg) Disponible"]
    K = row["VALUE_Potasio (K) Disponible"]
    Na = row["VALUE_Sodio (Na) Disponible"]
    B = row["VALUE_Boro (B) Disponible"]
    Fe = row["VALUE_Hierro (Fe) olsen Disponible"]

    # Reglas generales
    if pH < 3 or pH > 11:
        obs.append("pH fuera de rango (<3 o >11)")
    if CE == 0 or CE > 40:
        obs.append("CE fuera de rango (=0 o >40)")
    if Al_H > 10:
        obs.append("Al+H > 10")
    if Al > 10:
        obs.append("Al > 10")
    if Al > Al_H:
        obs.append("Al > Al+H")
    if S == 0 or S > 600:
        obs.append("S fuera de rango (=0 o >600)")
    if P == 0 or P > 600:
        obs.append("P fuera de rango (=0 o >600)")
    if Ca == 0 or Ca > 30:
        obs.append("Ca fuera de rango (=0 o >30)")
    if Mg == 0 or Mg > 10:
        obs.append("Mg fuera de rango (=0 o >10)")
    if K == 0 or K > 10:
        obs.append("K fuera de rango (=0 o >10)")
    if Mg > Ca:
        obs.append("Mg > Ca")

    # Reglas condicionales según pH
    if pH < 5:
        if Al_H == 0:
            obs.append("pH<5 y Al+H=0")
        if Al == 0:
            obs.append("pH<5 y Al=0")
        if Ca > 6:
            obs.append("pH<5 y Ca>6")
        if Mg > 2.5:
            obs.append("pH<5 y Mg>2.5")
        if K > 1:
            obs.append("pH<5 y K>1")
        if CE > 2:
            obs.append("pH<5 y CE>2")
        if Na > 1:
            obs.append("pH<5 y Na>1")
        if B > 1:
            obs.append("pH<5 y B>1")
        if Fe < 50:
            obs.append("pH<5 y Fe<50")
    if pH > 7:
        if Al_H > 0:
            obs.append("pH>7 y Al+H>0")
        if Al > 0:
            obs.append("pH>7 y Al>0")
        if Ca < 6:
            obs.append("pH>7 y Ca<6")
        if Mg < 2.5:
            obs.append("pH>7 y Mg<2.5")
        if K < 1:
            obs.append("pH>7 y K<1")
        if Fe > 50:
            obs.append("pH>7 y Fe>50")
    if pH > 8:
        if Na <= 1:
            obs.append("pH>8 y Na<=1")
    if CE > 4:
        if Na <= 1:
            obs.append("CE>4 y Na<=1")
        if S <= 20:
            obs.append("CE>4 y S<=20")
    if 3 < pH < 4:
        if CE <= 2:
            obs.append("3<pH<4 y CE<=2")
        if S <= 20:
            obs.append("3<pH<4 y S<=20")
        if Na <= 1:
            obs.append("3<pH<4 y Na<=1")

    # Agregar observaciones a la lista
    observaciones.append("; ".join(obs) if obs else "")

# Añadir columna de observaciones al DataFrame
df["observaciones"] = observaciones

# Filtrar solo las filas con observaciones
df_atipicos = df[df["observaciones"] != ""]

#print(df_atipicos)


# # Guardar el archivo CSV con los datos atípicos
# df_atipicos.to_csv("valores_atipicos_suelos.csv", index=False, encoding="utf-8-sig")

# print("Archivo 'valores_atipicos_suelos.csv' generado con éxito.")