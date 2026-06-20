from pathlib import Path
import numpy as np
import pandas as pd

"""
    Este programa realiza la limpieza y validación de un conjunto de datos
    publicitarios almacenado en un archivo CSV. El proceso incluye la
    verificación de columnas requeridas, eliminación de registros
    duplicados, conversión de datos a formato numérico, eliminación de
    valores nulos, infinitos y negativos, validación de relaciones
    lógicas entre variables y generación de un nuevo archivo con los
    datos limpios.

    Parámetros:
    ruta_entrada : Path
        Ruta del archivo CSV que contiene el conjunto de datos original.

    ruta_salida : Path
        Ruta donde se almacenará el archivo CSV resultante después del
        proceso de limpieza.

    columnas_proyecto : list[str]
        Lista de columnas que serán utilizadas durante el proceso de
        limpieza y validación.

    Retorna:
    No retorna ningún valor.
    Genera el archivo 'data_limpio.csv' con los registros válidos y
    muestra en consola un resumen del proceso realizado.

    """

carpeta_script = Path(__file__).resolve().parent
proyecto_root = carpeta_script.parent.parent
carpeta_datos = proyecto_root / "Data"

ruta_entrada = carpeta_datos / "dataset_original.csv"
ruta_salida = carpeta_datos / "data_limpio.csv"

columnas_proyecto = [
    "cost",
    "displays",
    "clicks",
    "post_click_conversions",
    "post_click_sales_amount",
    "revenue"
]

if not ruta_entrada.exists():
    raise FileNotFoundError(f"No se encontró el archivo:\n{ruta_entrada}")

df_original = pd.read_csv(
    ruta_entrada,
    na_values=["", " ", "NA", "N/A", "NULL", "null", "None"],
    low_memory=False,
)

filas_iniciales = len(df_original)

columnas_faltantes = [col for col in columnas_proyecto if col not in df_original.columns]
if columnas_faltantes:
    raise ValueError(f"Faltan las columnas: {columnas_faltantes}")

duplicados_eliminados = int(df_original.duplicated(keep="first").sum())
df_sin_duplicados = df_original.drop_duplicates(keep="first").reset_index(drop=True)
df_limpio = df_sin_duplicados[columnas_proyecto].copy()

for columna in columnas_proyecto:
    df_limpio[columna] = pd.to_numeric(df_limpio[columna], errors="coerce")

df_limpio = df_limpio.replace([np.inf, -np.inf], np.nan).dropna()
df_limpio = df_limpio[(df_limpio >= 0).all(axis=1)]

condiciones_validas = (
    (df_limpio["clicks"] <= df_limpio["displays"]) &
    (df_limpio["post_click_conversions"] <= df_limpio["clicks"]) &
    ~((df_limpio["clicks"] == 0) & (df_limpio["post_click_conversions"] > 0)) &
    ~((df_limpio["post_click_conversions"] == 0) & (df_limpio["post_click_sales_amount"] > 0)) &
    ~((df_limpio["displays"] == 0) & (df_limpio["revenue"] > 0))
)

df_limpio = df_limpio[condiciones_validas].reset_index(drop=True)

assert not df_limpio.isna().any().any()
assert np.isfinite(df_limpio.to_numpy()).all()
assert not (df_limpio < 0).any().any()
assert not (df_limpio["clicks"] > df_limpio["displays"]).any()
assert not (df_limpio["post_click_conversions"] > df_limpio["clicks"]).any()

try:
    df_limpio.to_csv(ruta_salida, index=False)
except PermissionError as error:
    raise PermissionError(
        "No se pudo guardar data_limpio.csv. Cierra el archivo si está abierto."
    ) from error

print(
    f"Limpieza terminada exitosamente.\n"
    f"Filas iniciales: {filas_iniciales}\n"
    f"Duplicados exactos eliminados: {duplicados_eliminados}\n"
    f"Filas finales: {len(df_limpio)}\n"
    f"Archivo generado: {ruta_salida}"
)