from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Funciones
def predecir(matriz_x, coeficientes_beta):
    return matriz_x @ coeficientes_beta

def calcular_rmse(valores_reales, valores_predichos):
    errores = valores_reales - valores_predichos
    return np.sqrt(np.mean(errores ** 2))

def calcular_r2(valores_reales, valores_predichos):
    media_real = np.mean(valores_reales)
    suma_residuos = np.sum((valores_reales - valores_predichos) ** 2)
    suma_total = np.sum((valores_reales - media_real) ** 2)
    if suma_total == 0:
        return 0.0
    return 1 - (suma_residuos / suma_total)

def desnormalizar_zscore(valores_zscore, media, desviacion):
    return valores_zscore * desviacion + media

def verificar_columnas(datos, columnas_requeridas):
    for columna in columnas_requeridas:
        if columna not in datos.columns:
            raise ValueError(f"No se encontró la columna: {columna}")

def obtener_parametros_revenue(ruta_parametros):
    parametros = pd.read_csv(ruta_parametros)
    verificar_columnas(parametros, ["variable", "media", "desviacion"])
    fila_revenue = parametros[parametros["variable"] == "revenue"]
    if fila_revenue.empty:
        raise ValueError("No se encontraron los parámetros de revenue.")
    media_revenue = fila_revenue["media"].values[0]
    desviacion_revenue = fila_revenue["desviacion"].values[0]
    if desviacion_revenue == 0:
        raise ValueError("La desviación estándar de revenue es cero.")
    return media_revenue, desviacion_revenue

def construir_matriz_modelo(datos, variables_modelo):
    verificar_columnas(datos, variables_modelo)
    matriz_variables = datos[variables_modelo].to_numpy()
    columna_unos = np.ones((matriz_variables.shape[0], 1))
    return np.hstack((columna_unos, matriz_variables))

def graficar_real_vs_predicho(valores_reales, valores_predichos, titulo, r2):
    plt.figure(figsize=(7, 6))
    plt.scatter(valores_reales.flatten(), valores_predichos.flatten(),alpha=0.5,label="Predicciones" )
    minimo = valores_reales.min()
    maximo = valores_reales.max()
    plt.plot([minimo, maximo],[minimo, maximo],linestyle="--",linewidth=2,label="Línea ideal")
    plt.title(f"{titulo}\nR² = {r2:.3f}")
    plt.xlabel("Revenue real en dólares")
    plt.ylabel("Revenue predicho en dólares")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def graficar_barras(modelos, valores, titulo, etiqueta_y, formato_valor):
    plt.figure(figsize=(7, 5))
    plt.bar(modelos, valores)
    plt.title(titulo)
    plt.ylabel(etiqueta_y)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for i, valor in enumerate(valores):
        plt.text(i, valor, formato_valor.format(valor), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()

# Rutas
carpeta_script = Path(__file__).resolve().parent
proyecto_root = carpeta_script.parent.parent
carpeta_datos = proyecto_root / "Data"
ruta_prueba = carpeta_datos / "data_prueba_zscore.csv"
ruta_beta_completo = carpeta_datos / "beta_completo.npy"
ruta_beta_reducido = carpeta_datos / "beta_reducido.npy"
ruta_parametros = carpeta_datos / "parametros_zscore.csv"

#Carga de datos
datos_prueba = pd.read_csv(ruta_prueba)
beta_completo = np.load(ruta_beta_completo)
beta_reducido = np.load(ruta_beta_reducido)
media_revenue, desviacion_revenue = obtener_parametros_revenue(ruta_parametros)

#Matrices de prueba
variable_dependiente = "revenue"
variables_modelo_completo = ["cost","displays","clicks","post_click_conversions","post_click_sales_amount"]
variables_modelo_reducido = ["cost","displays","clicks","post_click_conversions"]

verificar_columnas(datos_prueba, [variable_dependiente] + variables_modelo_completo)

revenue_real_zscore = datos_prueba[variable_dependiente].to_numpy().reshape(-1, 1)

matriz_completa_prueba = construir_matriz_modelo(datos_prueba,variables_modelo_completo)

matriz_reducida_prueba = construir_matriz_modelo(datos_prueba,variables_modelo_reducido)

#Verificación

if matriz_completa_prueba.shape[1] != beta_completo.shape[0]:
    raise ValueError("Las dimensiones del modelo completo no coinciden.")

if matriz_reducida_prueba.shape[1] != beta_reducido.shape[0]:
    raise ValueError("Las dimensiones del modelo reducido no coinciden.")

# Predicciones

revenue_pred_completo_zscore = predecir(matriz_completa_prueba, beta_completo)
revenue_pred_reducido_zscore = predecir(matriz_reducida_prueba, beta_reducido)

# Métricas en Z-score

rmse_completo_zscore = calcular_rmse(revenue_real_zscore,revenue_pred_completo_zscore)
rmse_reducido_zscore = calcular_rmse(revenue_real_zscore,revenue_pred_reducido_zscore)
r2_completo = calcular_r2(revenue_real_zscore,revenue_pred_completo_zscore)
r2_reducido = calcular_r2(revenue_real_zscore,revenue_pred_reducido_zscore)

# Revenue en dólares

revenue_real_dolares = desnormalizar_zscore(revenue_real_zscore,media_revenue,desviacion_revenue)
revenue_pred_completo_dolares = desnormalizar_zscore(revenue_pred_completo_zscore,media_revenue,desviacion_revenue)
revenue_pred_reducido_dolares = desnormalizar_zscore(revenue_pred_reducido_zscore,media_revenue,desviacion_revenue)

# Métricas en dólares

rmse_completo_dolares = calcular_rmse(revenue_real_dolares,revenue_pred_completo_dolares)
rmse_reducido_dolares = calcular_rmse(revenue_real_dolares,revenue_pred_reducido_dolares)

# Guardar métricas

metricas_modelos = pd.DataFrame({"modelo": ["Completo", "Reducido"],"RMSE_Zscore": [rmse_completo_zscore, rmse_reducido_zscore],
                                 "RMSE_Dolares": [rmse_completo_dolares, rmse_reducido_dolares],"R2": [r2_completo, r2_reducido]})
metricas_modelos.to_csv(carpeta_datos / "metricas_modelos.csv",index=False)

# Resultados
print(" ETAPA 4: ESTIMACIÓN Y EVALUACIÓN")
print("Resultados en escala Z-score:")
print(f"Modelo completo -> RMSE: {rmse_completo_zscore:.4f} | R²: {r2_completo:.4f}")
print(f"Modelo reducido -> RMSE: {rmse_reducido_zscore:.4f} | R²: {r2_reducido:.4f}")

print("\nResultados en dólares:")
print(f"Modelo completo -> RMSE: ${rmse_completo_dolares:.2f} | R²: {r2_completo:.4f}")
print(f"Modelo reducido -> RMSE: ${rmse_reducido_dolares:.2f} | R²: {r2_reducido:.4f}")

print("\nMétricas guardadas en: metricas_modelos.csv")

# Graficas

modelos = ["Completo", "Reducido"]
graficar_real_vs_predicho(revenue_real_dolares,revenue_pred_completo_dolares,"Modelo completo: Real vs Predicho", r2_completo)
graficar_real_vs_predicho(revenue_real_dolares,revenue_pred_reducido_dolares,"Modelo reducido: Real vs Predicho",r2_reducido)
graficar_barras(modelos, [r2_completo, r2_reducido], "Comparación de R²", "R²", "{:.3f}")
graficar_barras(modelos,[rmse_completo_dolares, rmse_reducido_dolares],"Comparación de RMSE en dólares","RMSE ($)","${:.2f}")