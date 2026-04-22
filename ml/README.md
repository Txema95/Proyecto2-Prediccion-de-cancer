# README `ml/main.py`

## Objetivo

El archivo `ml/main.py` implementa una aplicacion en `Streamlit` para entrenar, evaluar y comparar tres modelos de clasificacion para la deteccion de cancer de colon:

- `regresion_logistica`
- `random_forest`
- `xgboost`
- `svm`

Los tres modelos trabajan sobre el mismo dataset procesado y utilizan el mismo conjunto final de variables de entrada, incluyendo las features derivadas seleccionadas durante la fase de `feature engineering`.

## Ejecucion

Desde la raiz del proyecto:

```bash
uv run streamlit run ml/main.py
```

Para ejecutar el front del simulador clinico (flujo de datos + imagenes):

```bash
uv run streamlit run frontend/app.py
```

## Flujo general de `main.py`

La aplicacion sigue este flujo:

1. Localiza automaticamente la raiz del proyecto con `buscar_raiz_proyecto()`.
2. Carga el dataset `data/processed/cancer_final_clean_v2.csv`.
3. Genera dos variables derivadas en `preparar_datos_modelo()`:
  - `n_sintomas`
  - `riesgo_familiar_x_edad`
4. Separa variables predictoras, variable objetivo (`cancer_diagnosis`) e identificadores.
5. Divide los datos en:
  - entrenamiento
  - validacion
  - test
6. Entrena los tres algoritmos con su pipeline correspondiente.
7. Calcula metricas de validacion y test.
8. Guarda artefactos y muestra resultados en la interfaz.

## Variables derivadas incluidas

Las features nuevas que se incorporan al dataset son:

- `n_sintomas`: suma de `sof`, `tenesmus` y `rectorrhagia`.
- `riesgo_familiar_x_edad`: producto entre `digestive_family_risk_level` y `age`.

En la interfaz existe un boton, `Mostrar features generadas`, que enseña estas variables y una breve justificacion de su uso.

## Modelos entrenados

`main.py` entrena estos tres clasificadores:

- `LogisticRegression`
- `RandomForestClassifier`
- `XGBClassifier`
- `SVM`

Cada modelo usa un preprocesado numerico adaptado a su naturaleza:

- La regresion logistica aplica imputacion de medianas y escalado.
- Random forest y XGBoost aplican imputacion de medianas sin escalado.

## Particionado de datos

La funcion `hacer_particiones()` realiza una division estratificada:

- `15%` para test.
- Del `85%` restante, una parte para validacion.

Esto permite:

- ajustar y entrenar inicialmente sobre entrenamiento,
- revisar rendimiento intermedio en validacion,
- volver a entrenar con `train + validation`,
- medir el resultado final sobre test.

## Metricas calculadas

Para cada modelo se calculan:

- `accuracy`
- `recall`
- `f1`
- `roc_auc`
- `pr_auc`

Estas metricas se guardan en JSON y se usan para construir la tabla comparativa dentro de la app.

## Desbalanceo de clases (enfoque: detectar positivos)

El dataset presenta un desbalanceo moderado (aprox. `3.87 : 1` negativos vs positivos). Dado que el objetivo del simulador es **detectar positivos**, el pipeline aplica estrategias de balanceo para priorizar `recall`:

- `LogisticRegression`: `class_weight="balanced"`
- `RandomForestClassifier`: `class_weight="balanced_subsample"`
- `XGBClassifier`: `scale_pos_weight` calculado automaticamente a partir de la proporcion de clases del conjunto de entrenamiento
- `SVM`: `class_weight="balanced"`, Kernel `RBF`

**Nota técnica sobre SVM:** 

Al igual que la regresión logística, el modelo SVM requiere que las variables estén en la misma escala para funcionar correctamente, por lo que su pipeline incluye un StandardScaler. Además, se configura con probability=True para permitir el cálculo de métricas de área bajo la curva (ROC-AUC y PR-AUC).

Ademas, el particionado es estratificado para conservar la proporcion de clases en `train/val/test`.

### Fase de pruebas (antes vs despues del balanceo en XGBoost)

Se comparo `xgboost` antes y despues de introducir `scale_pos_weight`, manteniendo el resto del pipeline constante.

Valores usados:

- `scale_pos_weight` en entrenamiento: `3.86685`
- `scale_pos_weight` en entrenamiento + validacion: `3.867722`

**Antes del balanceo**

- Validacion: `accuracy=0.9480`, `recall=0.8462`, `f1=0.8699`, `roc_auc=0.9823`, `pr_auc=0.9494`
- Test: `accuracy=0.9375`, `recall=0.8237`, `f1=0.8440`, `roc_auc=0.9731`, `pr_auc=0.9123`

**Despues del balanceo**

- Validacion: `accuracy=0.9388`, `recall=0.9006`, `f1=0.8580`, `roc_auc=0.9812`, `pr_auc=0.9467`
- Test: `accuracy=0.9211`, `recall=0.8974`, `f1=0.8235`, `roc_auc=0.9725`, `pr_auc=0.9121`

Interpretacion: el balanceo mejora el `recall` de forma clara (p. ej. en test `0.8237 -> 0.8974`) con caidas pequenas en `accuracy`/`f1` y cambios minimos en `pr_auc`/`roc_auc`. Por eso se mantiene, al estar alineado con el objetivo de detectar positivos.


En el caso de SVM, el uso de class_weight="balanced" ajusta automáticamente los pesos de las clases inversamente proporcionales a sus frecuencias en los datos de entrada, penalizando más los errores en la clase positiva (cáncer detectado).


## Artefactos generados

Para cada algoritmo se guardan artefactos en su carpeta correspondiente:

- `modelo.joblib`
- `metricas_validacion.json`
- `metricas_test.json`
- `reporte_test.txt`
- `matriz_confusion_test.png`

Ademas, se guardan artefactos comunes en `ml/comun/v1`:

- `columnas_modelo.json`
- `ids_test.csv`

## Que muestra la interfaz

La aplicacion enseña:

- Un boton para visualizar las features generadas.
- Una tabla comparativa con las metricas de test de los tres modelos.
- Un mensaje destacando el mejor modelo segun `PR-AUC`.
- Las matrices de confusion de cada algoritmo en test.

## Estructura funcional del codigo

Las funciones principales de `main.py` son:

- `buscar_raiz_proyecto()`: localiza la raiz donde esta el dataset procesado.
- `obtener_rutas()`: define carpetas de entrada y salida de artefactos.
- `preparar_datos_modelo()`: carga el CSV y construye las variables derivadas.
- `crear_modelo()`: monta el pipeline segun el algoritmo.
- `hacer_particiones()`: divide el dataset en entrenamiento, validacion y test.
- `entrenar_y_evaluar_modelo()`: entrena, calcula metricas y guarda artefactos.
- `construir_tabla_comparacion()`: resume resultados de los tres modelos.
- `main()`: construye la interfaz de `Streamlit`.

## Nota sobre `ml/feature_engineering.py`

El archivo `ml/feature_engineering.py` se utilizo como fase de experimentacion para decidir que variables derivadas merecia la pena incorporar al flujo final.

### Como se hicieron las pruebas

El proceso seguido fue un `ablation study` paso a paso:

- Se definio un conjunto de features candidatas derivadas a partir de variables clinicas y de sintomas.
- Se evaluo cada candidata con validacion cruzada estratificada de `5 folds`.
- En cada iteracion se probaba anadir una feature nueva sobre el conjunto ya seleccionado.
- La decision se tomaba comparando la mejora marginal en varias metricas:
  - `pr_auc`
  - `recall`
  - `f1`
  - `roc_auc`

Ademas, el script imponia dos condiciones para aceptar una nueva variable:

- una mejora minima en `PR-AUC` (`GANANCIA_MINIMA_PR_AUC = 0.0005`)
- que el `recall` no empeorase mas alla de la tolerancia fijada (`TOLERANCIA_CAIDA_RECALL = 0.0020`)

### Por que nos quedamos con las features de XGBoost

Durante las pruebas, el comportamiento mas consistente y util aparecio en `xgboost`. Las dos variables que mostraron mejor equilibrio entre mejora de capacidad predictiva y estabilidad fueron:

- `n_sintomas`
- `riesgo_familiar_x_edad`

Se eligieron porque aportaban senal clinicamente interpretable y mejoraban el rendimiento sin complicar en exceso el espacio de variables. A partir de esa conclusion, se decidio reutilizar este mismo conjunto final en los tres algoritmos para mantener un pipeline comun y una comparacion justa entre modelos.

### Por que no se anadieron mas features

No se incorporaron mas variables derivadas por tres razones:

- varias candidatas no aportaban una mejora suficiente en `PR-AUC`,
- algunas podian degradar el `recall` o no ofrecer una ganancia estable,
- anadir mas interacciones aumenta la complejidad del modelo y el riesgo de introducir ruido o sobreajuste sin beneficio claro.

Por tanto, se priorizo un conjunto pequeno, interpretable y con mejora demostrada frente a una expansion de features sin evidencia solida de valor adicional.