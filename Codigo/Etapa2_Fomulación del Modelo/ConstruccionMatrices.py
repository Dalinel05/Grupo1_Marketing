from pathlib import Path
import numpy as np
import pandas as pd

carpeta_script = Path(_file_).resolve().parent
proyecto_root = carpeta_script.parent.parent
carpeta_datos = proyecto_root / "Data"

ruta_entrada = carpeta_datos / "data_entrenamiento_zscore.csv"

if not ruta_entrada.exists():
    raise FileNotFoundError(f"No se encontró el archivo:\n{ruta_entrada}")

datos = pd.read_csv(ruta_entrada)

variable_dependiente = "revenue"

if variable_dependiente not in datos.columns:
    raise ValueError(f"No se encontró la columna: {variable_dependiente}")

y = datos[variable_dependiente].to_numpy().reshape(-1, 1)

variables_modelo_completo = ["cost","displays","clicks","post_click_conversions","post_click_sales_amount"]

for columna in variables_modelo_completo:
    if columna not in datos.columns:
        raise ValueError(f"No se encontró la columna: {columna}")
    
X_variables_completo = datos[variables_modelo_completo].to_numpy()
columna_unos = np.ones((X_variables_completo.shape[0], 1))
X_completo = np.hstack((columna_unos, X_variables_completo))

variables_modelo_reducido = ["cost","displays","clicks","post_click_conversions"]

for columna in variables_modelo_reducido:
    if columna not in datos.columns:
        raise ValueError(f"No se encontró la columna: {columna}")
X_variables_reducido = datos[variables_modelo_reducido].to_numpy()
columna_unos = np.ones((X_variables_reducido.shape[0], 1))
X_reducido = np.hstack((columna_unos, X_variables_reducido))

print("Etapa 2 completada correctamente.")
print("Cantidad de registros:", y.shape[0])
print("Vector y:", y.shape)
print("Matriz X_completo:", X_completo.shape)
print("Matriz X_reducido:", X_reducido.shape)
