# Grupo 1 - Marketing

## Análisis numérico de la relación entre los factores de desempeño publicitario y los ingresos generados por campañas digitales

## 1. Descripción general del proyecto

Este proyecto analiza la relación entre distintos factores del desempeño publicitario y los ingresos generados por campañas de publicidad digital. Para ello, se utilizarán variables relacionadas con la inversión, las visualizaciones, los clics, las conversiones y las ventas posteriores al clic.
A partir de estos datos se construirá un modelo lineal multivariable mediante mínimos cuadrados, con el propósito de identificar qué factores presentan una mayor relación con los ingresos y estimar su valor. De manera complementaria, se analizará una versión reducida del modelo para comprobar si la variable post_click_sales_amount aporta información adicional o genera redundancia.

## 2. Descripción del dataset

El dataset utilizado corresponde a *Online Advertising Digital Marketing Data*, disponibleen la plataforma Kaggle. Según la fuente, contiene registros del rendimiento publicitario de una empresa identificada de manera anónima, correspondientes a un periodo de tres meses del año 2020. El conjunto de datos cuenta con 15 408 registros.

Incluye información relacionada con la inversión realizada, las visualizaciones de los anuncios, los clics obtenidos, las conversiones posteriores al clic, el monto de ventas y los ingresos generados. También contiene variables sobre la campaña, el nivel de interacción, el formato y la ubicación del anuncio.

Estas variables permiten representar cuantitativamente las diferentes etapas de una campaña publicitaria, desde la inversión y exposición del anuncio hasta la interacción de los usuarios, las conversiones y la obtención de resultados económicos.
resultados económicos.

## 3. Variables escogidas del dataset
Para el desarrollo del proyecto se seleccionaron variables numéricas que representan las principales etapas de una campaña publicitaria, desde la inversión inicial hasta la generación de resultados económicos.

* **`cost`**: representa la inversión realizada en la campaña. Se seleccionó para analizar si un mayor gasto publicitario se relaciona con un incremento en los ingresos.

* **`displays`**: indica la cantidad de veces que el anuncio fue mostrado. Se incluyó porque permite medir el nivel de exposición alcanzado por la campaña.

* **`clicks`**: representa el número de clics obtenidos. Esta variable permite analizar la interacción y el interés de los usuarios después de visualizar el anuncio.

* **`post_click_conversions`**: corresponde a la cantidad de conversiones registradas después de un clic. Se seleccionó porque representa una acción efectiva del usuario y se encuentra próxima a la generación de resultados económicos.

* **`post_click_sales_amount`**: representa el monto de ventas generado después del clic. Se incluyó para analizar su relación con los ingresos y determinar si aporta información adicional al modelo.

* **`revenue`**: indica los ingresos generados por la actividad publicitaria. Será utilizada como variable dependiente, ya que el objetivo del proyecto es analizar y estimar su comportamiento a partir de los demás factores.

Estas variables permiten representar el proceso publicitario mediante la relación entre inversión, exposición, interacción, conversión y resultados económicos.

## 4. Etapas de resolución del problema
### Etapa 1. Preparación de los datos

Se seleccionarán y limpiarán las variables necesarias para el análisis. Luego, el dataset se dividirá en datos de entrenamiento y prueba, y las variables independientes serán normalizadas mediante el método Min-Max.

### Etapa 2. Formulación del modelo

Se construirá un modelo lineal multivariable que relacione los factores publicitarios con los ingresos. También se formulará una versión reducida sin `post_click_sales_amount` para analizar el aporte de esta variable.

### Etapa 3. Ajuste mediante mínimos cuadrados

Los coeficientes de los modelos se calcularán mediante mínimos cuadrados lineales multivariables. Para ello, se formarán las ecuaciones normales y los sistemas resultantes se resolverán mediante eliminación de Gauss con pivoteo parcial.

### Etapa 4. Estimación y evaluación

Los modelos se utilizarán para estimar los ingresos del conjunto de prueba. Su desempeño se evaluará mediante la raíz del error cuadrático medio, RMSE, y el coeficiente de determinación, (R^2).

### Etapa 5. Análisis de los resultados

Se interpretarán los coeficientes obtenidos para identificar los factores con mayor relación con los ingresos. Además, se compararán ambos modelos para determinar si `post_click_sales_amount` aporta información relevante o genera redundancia.

---
## Integrantes 
* Martínez Dayana 
* Montenegro Letizia
* Salazar Karolina 

