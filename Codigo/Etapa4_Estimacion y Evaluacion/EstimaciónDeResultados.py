"""
Etapa 4: Estimación y Evaluación del Desempeño

Este script aplica los coeficientes de regresión (betas) calculados en la 
Etapa 3 sobre el conjunto de prueba para evaluar la capacidad de generalización 
de los modelos (completo y reducido). 

Calcula de forma manual las métricas de error RMSE (Raíz del Error Cuadrático 
Medio) y el coeficiente de determinación R², y genera gráficos comparativos de 
las predicciones en unidades monetarias.

Parámetros :
No recibe parámetros directamente como función, pero lee automáticamente los 
siguientes archivos desde la carpeta 'Data':
- data_prueba_zscore.csv : Conjunto de datos de prueba normalizado con Z-Score.
- beta_completo.npy      : Pesos del modelo calculados con todas las variables.
- beta_reducido.npy      : Pesos del modelo calculados sin la variable redundante.

Retorna:
No retorna valores físicos en archivos, pero despliega en pantalla:
1. Una tabla comparativa con los valores de RMSE y R² para ambos modelos.
2. Una ventana gráfica con la dispersión 'Real vs Predicho' y barras de desempeño.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# 1. FUNCIONES MATEMÁTICAS DE EVALUACIÓN

def predecir(X, beta):
    """Realiza la estimación del ingreso mediante el producto matricial X@beta."""
    return X @ beta


def calcular_rmse(y_real, y_pred):
    """Calcula de forma manual la Raíz del Error Cuadrático Medio (RMSE)."""
    error = y_real - y_pred
    mse = np.mean(error ** 2)
    return np.sqrt(mse)


def calcular_r2(y_real, y_pred):
    """Calcula de forma manual el coeficiente de determinación R²."""
    media_y = np.mean(y_real)
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - media_y) ** 2)
    if ss_tot == 0: 
        return 0.0
    return 1 - (ss_res / ss_tot)


# 2. CONFIGURACIÓN DE RUTAS Y SISTEMA DE ARCHIVOS

# Localización de directorios del proyecto
carpeta_script = Path(__file__).resolve().parent
proyecto_root = carpeta_script.parent.parent
carpeta_datos = proyecto_root / "Data"

# Definición de rutas de entrada
ruta_prueba_csv = carpeta_datos / "data_prueba_zscore.csv"
ruta_beta_completo = carpeta_datos / "beta_completo.npy"
ruta_beta_reducido = carpeta_datos / "beta_reducido.npy"

# Verificación de la existencia de los archivos requeridos
for ruta in [ruta_prueba_csv, ruta_beta_completo, ruta_beta_reducido]:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo:\n{ruta}")


# 3. CARGA DE DATOS Y COEFICIENTES

beta_completo = np.load(ruta_beta_completo)
beta_reducido = np.load(ruta_beta_reducido)
datos_prueba = pd.read_csv(ruta_prueba_csv)

# Extracción de la variable objetivo en su escala monetaria original
y_prueba = datos_prueba["revenue"].to_numpy().reshape(-1, 1)


# 4. CONSTRUCCIÓN DE MATRICES DE PRUEBA (CON COLUMNA DE UNOS)

# Formulación para el Modelo Completo (5 variables + intercepto)
vars_modelo_completo = ["cost", "displays", "clicks", "post_click_conversions", "post_click_sales_amount"]
X_var_completo = datos_prueba[vars_modelo_completo].to_numpy()
X_completo_prueba = np.hstack((np.ones((X_var_completo.shape[0], 1)), X_var_completo))

# Formulación para el Modelo Reducido (4 variables + intercepto)
vars_modelo_reducido = ["cost", "displays", "clicks", "post_click_conversions"]
X_var_reducido = datos_prueba[vars_modelo_reducido].to_numpy()
X_reducido_prueba = np.hstack((np.ones((X_var_reducido.shape[0], 1)), X_var_reducido))


# 5. EJECUCIÓN DE ESTIMACIONES Y CÁLCULO DE MÉTRICAS

# Cálculo de ingresos estimados para ambos modelos
y_pred_completo = predecir(X_completo_prueba, beta_completo)
y_pred_reducido = predecir(X_reducido_prueba, beta_reducido)

# Evaluación del Modelo Completo
rmse_completo = calcular_rmse(y_prueba, y_pred_completo)
r2_completo = calcular_r2(y_prueba, y_pred_completo)

# Evaluación del Modelo Reducido
rmse_reducido = calcular_rmse(y_prueba, y_pred_reducido)
r2_reducido = calcular_r2(y_prueba, y_pred_reducido)


# 6. DESPLEGUE DE RESULTADOS EN CONSOLA

print("=" * 60)
print(" ETAPA 4: ESTIMACIÓN Y EVALUACIÓN DE RESULTADOS")
print("=" * 60)
print(f"RMSE Completo: {rmse_completo:.4f} | R² Completo: {r2_completo:.4f}")
print(f"RMSE Reducido: {rmse_reducido:.4f} | R² Reducido: {r2_reducido:.4f}")
print("=" * 60)


# 7. VISUALIZACIÓN GRÁFICA DE DESEMPEÑO

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Dispersión del Ingreso Real vs Predicho (Modelo Completo)
ax1.scatter(y_prueba, y_pred_completo, alpha=0.5, color='blue', label='Predicciones')
ax1.plot([y_prueba.min(), y_prueba.max()], [y_prueba.min(), y_prueba.max()], 'r--', lw=2, label='Línea Ideal (Predicción Perfecta)')
ax1.set_title(f'Modelo Completo\nReal vs Predicho ($R^2$={r2_completo:.3f})', fontsize=12)
ax1.set_xlabel('Ingresos Reales (Unidades Monetarias)')
ax1.set_ylabel('Ingresos Predichos (Unidades Monetarias)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Gráfico 2: Comparación métrica de barras entre modelos
modelos = ['Modelo Completo', 'Modelo Reducido']
valores_r2 = [r2_completo, r2_reducido]
valores_rmse = [rmse_completo, rmse_reducido]

x = np.arange(len(modelos))
ancho = 0.35

barras_r2 = ax2.bar(x - ancho/2, valores_r2, ancho, label='R² (Más alto es mejor)', color='green')
barras_rmse = ax2.bar(x + ancho/2, valores_rmse, ancho, label='RMSE (Más bajo es mejor)', color='orange')

ax2.set_title('Comparación de Desempeño', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(modelos)
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()