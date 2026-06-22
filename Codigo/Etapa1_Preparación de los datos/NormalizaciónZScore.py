from pathlib import Path
import numpy as np
import pandas as pd

"""
Normalización mediante Z-Score.

Lee los archivos data_entrenamiento.csv y data_prueba.csv
generados en la etapa de separación del dataset.

La media y la desviación estándar se calculan únicamente
con el conjunto de entrenamiento. Luego, esos mismos valores
se aplican al conjunto de entrenamiento y al conjunto de prueba.

Fórmula:
z = (x - media) / desviacion_estandar
"""
carpeta_script = Path(__file__).resolve().parent
proyecto_root = carpeta_script.parent.parent
carpeta_datos = proyecto_root / "Data"

ruta_entrenamiento = carpeta_datos / "data_entrenamiento.csv"
ruta_prueba = carpeta_datos / "data_prueba.csv"

ruta_entrenamiento_zscore = carpeta_datos / "data_entrenamiento_zscore.csv"
ruta_prueba_zscore = carpeta_datos / "data_prueba_zscore.csv"
ruta_parametros = carpeta_datos / "parametros_zscore.csv"

columnas_proyecto = ["cost","displays","clicks","post_click_conversions","post_click_sales_amount","revenue"]

if not ruta_entrenamiento.exists():
    raise FileNotFoundError(f"No se encontró el archivo:\n{ruta_entrenamiento}")

if not ruta_prueba.exists():
    raise FileNotFoundError(f"No se encontró el archivo:\n{ruta_prueba}")

df_entrenamiento = pd.read_csv(ruta_entrenamiento)
df_prueba = pd.read_csv(ruta_prueba)

if df_entrenamiento.empty:
    raise ValueError("El dataset de entrenamiento no contiene registros.")

if df_prueba.empty:
    raise ValueError("El dataset de prueba no contiene registros.")

columnas_faltantes_entrenamiento = [
    col for col in columnas_proyecto
    if col not in df_entrenamiento.columns
]

columnas_faltantes_prueba = [
    col for col in columnas_proyecto
    if col not in df_prueba.columns
]

if columnas_faltantes_entrenamiento:
    raise ValueError(f"Faltan columnas en entrenamiento: {columnas_faltantes_entrenamiento}")

if columnas_faltantes_prueba:
    raise ValueError(f"Faltan columnas en prueba: {columnas_faltantes_prueba}")

df_entrenamiento_zscore = df_entrenamiento.copy()
df_prueba_zscore = df_prueba.copy()

parametros = []

for columna in columnas_proyecto:
    media = df_entrenamiento[columna].mean()
    desviacion = df_entrenamiento[columna].std(ddof=0)

    if desviacion == 0:
        desviacion = 1

    df_entrenamiento_zscore[columna] = (
        df_entrenamiento_zscore[columna] - media
    ) / desviacion

    df_prueba_zscore[columna] = (
        df_prueba_zscore[columna] - media
    ) / desviacion

    parametros.append({
        "variable": columna,
        "media": media,
        "desviacion": desviacion
    })

assert not df_entrenamiento_zscore.isna().any().any()
assert not df_prueba_zscore.isna().any().any()
assert np.isfinite(df_entrenamiento_zscore.to_numpy()).all()
assert np.isfinite(df_prueba_zscore.to_numpy()).all()

try:
    df_entrenamiento_zscore.to_csv(ruta_entrenamiento_zscore, index=False)
    df_prueba_zscore.to_csv(ruta_prueba_zscore, index=False)

    df_parametros = pd.DataFrame(parametros)
    df_parametros.to_csv(ruta_parametros, index=False)

except PermissionError as error:
    raise PermissionError(
        "No se pudieron guardar los archivos. Cierra los CSV si están abiertos."
    ) from error

print(
    f"Normalización Z-score completada.\n"
    f"Filas de entrenamiento procesadas: {len(df_entrenamiento_zscore)}\n"
    f"Filas de prueba procesadas: {len(df_prueba_zscore)}\n"
    f"Archivo de entrenamiento generado: {ruta_entrenamiento_zscore}\n"
    f"Archivo de prueba generado: {ruta_prueba_zscore}\n"
    f"Archivo de parámetros generado: {ruta_parametros}"
)