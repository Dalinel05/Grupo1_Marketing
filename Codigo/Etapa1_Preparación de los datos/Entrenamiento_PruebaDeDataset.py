#Librerias utilizadas
from pathlib import Path
import pandas as pd

carpeta_script = Path(__file__).resolve().parent
proyecto_root = carpeta_script.parent.parent
carpeta_datos = proyecto_root / "Data"

ruta_entrada = carpeta_datos / "data_limpio.csv"
ruta_entrenamiento = carpeta_datos / "data_entrenamiento.csv"
ruta_prueba = carpeta_datos / "data_prueba.csv"

porcentaje_de_entrenamiento = 0.80
mezcla_aleatoria = 42 #mezcla todas las filas del dataset antes de separar entrenamiento y prueba

if not ruta_entrada.exists():
    raise FileNotFoundError(f"No se encontró el archivo:\n{ruta_entrada}")

datos_limpios = pd.read_csv(ruta_entrada)

if datos_limpios.empty:
    raise ValueError("El dataset no contiene registros")

datos_limpios = datos_limpios.sample(frac=1, random_state=mezcla_aleatoria).reset_index(drop=True)

cantidad_entrenamiento = int(len(datos_limpios) * porcentaje_de_entrenamiento)
datos_limpios_entrenamiento = datos_limpios.iloc[:cantidad_entrenamiento].copy()
datos_limpios_prueba = datos_limpios.iloc[cantidad_entrenamiento:].copy()

try:
    datos_limpios_entrenamiento.to_csv(ruta_entrenamiento,index=False)
    datos_limpios_prueba.to_csv(ruta_prueba,index=False)
except PermissionError as error:
    raise PermissionError("No se guardaron los archivos") from error


print(
    f"Registros totales: {len(datos_limpios)}\n"
    f"Entrenamiento: {len(datos_limpios_entrenamiento)} "
    f"({len(datos_limpios_entrenamiento) / len(datos_limpios):.0%})\n"
    f"Prueba: {len(datos_limpios_prueba)} "
    f"({len(datos_limpios_prueba) / len(datos_limpios):.0%})")