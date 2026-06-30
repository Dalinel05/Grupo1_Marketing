"""
    Este script calcula los coeficientes de regresión (betas) para dos modelos 
    (uno completo y uno reducido) utilizando el método de mínimos cuadrados. 
    Para encontrar estos valores, construye las ecuaciones normales y resuelve 
    el sistema matemático paso a paso utilizando el algoritmo de eliminación 
    de Gauss con pivoteo parcial.

    Parámetros :
    No recibe parámetros directamente como función, pero lee automáticamente 
    los siguientes archivos generados en la Etapa 2 desde la carpeta 'Data':
    - X_completo.npy : Matriz con todas las variables de entrenamiento.
    - X_reducido.npy : Matriz con las variables reducidas de entrenamiento.
    - y_entrenamiento.npy : Vector con la variable objetivo (ingresos).

    Retorna:
    No retorna valores al usuario en la terminal más que un mensaje de éxito. 
    Físicamente, genera y guarda dos archivos en la carpeta 'Data':
    - beta_completo.npy : Los pesos calculados para el modelo completo.
    - beta_reducido.npy : Los pesos calculados para el modelo reducido.
    Estos archivos quedan listos para ser consumidos en la Etapa 4.
    """

from pathlib import Path
import numpy as np

def eliminacion_gauss(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = A.shape[0]
    matriz_aumentada = np.hstack((A, b))
    for k in range(n - 1):
        fila_pivote = k + np.argmax(np.abs(matriz_aumentada[k:, k]))
        if np.isclose(matriz_aumentada[fila_pivote, k], 0):
            raise ValueError("No se puede resolver el sistema porque el pivote es cero")
        if fila_pivote != k:
            matriz_aumentada[[k, fila_pivote]] = matriz_aumentada[[fila_pivote, k]]
        for i in range(k + 1, n):
            multiplicador = matriz_aumentada[i, k] / matriz_aumentada[k, k]
            matriz_aumentada[i, k:] = matriz_aumentada[i, k:] - multiplicador * matriz_aumentada[k, k:]
    beta = np.zeros((n, 1))
    for i in range(n - 1, -1, -1):
        suma = np.dot(matriz_aumentada[i, i + 1:n], beta[i + 1:n, 0])
        beta[i, 0] = (matriz_aumentada[i, -1] - suma) / matriz_aumentada[i, i]
    return beta

def calcular_beta(X, y):
    A = X.T @ X
    b = X.T @ y
    beta = eliminacion_gauss(A, b)
    return beta, A, b


carpeta_script = Path(__file__).resolve().parent
proyecto_root = carpeta_script.parent.parent
carpeta_datos = proyecto_root / "Data"

ruta_y = carpeta_datos / "y_entrenamiento.npy"
ruta_X_completo = carpeta_datos / "X_completo.npy"
ruta_X_reducido = carpeta_datos / "X_reducido.npy"
for ruta in [ruta_y, ruta_X_completo, ruta_X_reducido]:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo:\n{ruta}")

y = np.load(ruta_y)
X_completo = np.load(ruta_X_completo)
X_reducido = np.load(ruta_X_reducido)
beta_completo, A_completo, b_completo = calcular_beta(X_completo, y)
beta_reducido, A_reducido, b_reducido = calcular_beta(X_reducido, y)

np.save(carpeta_datos / "beta_completo.npy", beta_completo)
np.save(carpeta_datos / "beta_reducido.npy", beta_reducido)
print("Etapa 3 completada correctamente y coeficientes guardados en archivos .npy")