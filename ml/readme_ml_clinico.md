# README: ML tabular (diagnóstico clínico) — `ml/main.py`

## Objetivo

El archivo `ml/main.py` implementa una aplicación en **Streamlit** para entrenar, evaluar y comparar **cuatro** modelos de clasificación para la detección de cáncer de colon:

- `regresion_logistica`
- `random_forest`
- `xgboost`
- `svm`

Todos comparten el mismo dataset procesado y el mismo conjunto final de variables de entrada, incluidas las features derivadas seleccionadas durante la fase de *feature engineering*.

## Ejecución

Desde la raíz del proyecto:

```bash
uv run streamlit run ml/main.py
```

## Flujo general de `main.py`

La aplicación sigue este flujo:

1. Localiza automáticamente la raíz del proyecto con `buscar_raiz_proyecto()`.
2. Carga el dataset `data/processed/cancer_final_clean_v2.csv`.
3. Genera dos variables derivadas en `preparar_datos_modelo()`:
  - `n_sintomas`
  - `riesgo_familiar_x_edad`
4. Separa variables predictoras, variable objetivo (`cancer_diagnosis`) e identificadores.
5. Divide los datos en entrenamiento, validación y test.
6. Entrena cada algoritmo con su pipeline correspondiente.
7. Calcula métricas de validación y test.
8. Guarda artefactos y muestra resultados en la interfaz.

## Variables derivadas incluidas

Las features nuevas que se incorporan al dataset son:

- `n_sintomas`: suma de `sof`, `tenesmus` y `rectorrhagia`.
- `riesgo_familiar_x_edad`: producto entre `digestive_family_risk_level` y `age`.

En la interfaz existe un botón, **Mostrar features generadas**, que muestra estas variables y una breve justificación de su uso.

## Modelos entrenados

`main.py` entrena estos cuatro clasificadores:

- `LogisticRegression`
- `RandomForestClassifier`
- `XGBClassifier`
- `SVC` (SVM)

Cada modelo usa un preprocesado numérico adaptado a su naturaleza:

- Regresión logística y SVM: imputación de medianas y **StandardScaler**.
- Random forest y XGBoost: imputación de medianas **sin** escalado.

## Particionado de datos

La función `hacer_particiones()` realiza una división estratificada:

- `15%` para test.
- Del `85%` restante, una parte para validación.

Esto permite entrenar primero sobre entrenamiento, revisar rendimiento en validación, reentrenar con `train + validation` y medir el resultado final sobre test.

## Métricas calculadas

Para cada modelo se calculan:

- `accuracy`
- `recall`
- `f1`
- `roc_auc`
- `pr_auc`

Estas métricas se guardan en JSON y se usan para construir la tabla comparativa dentro de la app.

## Desbalanceo de clases (enfoque: detectar positivos)

El dataset presenta un desbalanceo moderado (aprox. `3.87 : 1` negativos vs positivos). Dado que el objetivo del simulador es **detectar positivos**, el pipeline aplica estrategias de balanceo para priorizar `recall`:

- `LogisticRegression`: `class_weight="balanced"`
- `RandomForestClassifier`: `class_weight="balanced_subsample"`
- `XGBClassifier`: `scale_pos_weight` calculado automáticamente a partir de la proporción de clases del conjunto de entrenamiento
- `SVM`: `class_weight="balanced"`, kernel RBF

**Nota técnica sobre SVM:** al igual que la regresión logística, el modelo SVM se beneficia de variables en la misma escala; el pipeline incluye `StandardScaler`. Se configura con `probability=True` para permitir ROC-AUC y PR-AUC.

Además, el particionado es estratificado para conservar la proporción de clases en `train/val/test`.

### Fase de pruebas (antes vs después del balanceo en XGBoost)

Se comparó `xgboost` antes y después de introducir `scale_pos_weight`, manteniendo el resto del pipeline constante.

Valores usados:

- `scale_pos_weight` en entrenamiento: `3.86685`
- `scale_pos_weight` en entrenamiento + validación: `3.867722`

**Antes del balanceo**

- Validación: `accuracy=0.9480`, `recall=0.8462`, `f1=0.8699`, `roc_auc=0.9823`, `pr_auc=0.9494`
- Test: `accuracy=0.9375`, `recall=0.8237`, `f1=0.8440`, `roc_auc=0.9731`, `pr_auc=0.9123`

**Después del balanceo**

- Validación: `accuracy=0.9388`, `recall=0.9006`, `f1=0.8580`, `roc_auc=0.9812`, `pr_auc=0.9467`
- Test: `accuracy=0.9211`, `recall=0.8974`, `f1=0.8235`, `roc_auc=0.9725`, `pr_auc=0.9121`

Interpretación: el balanceo mejora el `recall` de forma clara (p. ej. en test `0.8237 -> 0.8974`) con caídas pequeñas en `accuracy`/`f1` y cambios mínimos en `pr_auc`/`roc_auc`. Por eso se mantiene, alineado con el objetivo de detectar positivos.

En SVM, `class_weight="balanced"` ajusta los pesos de las clases de forma inversa a su frecuencia, penalizando más los errores en la clase positiva.

## Artefactos generados

Para cada algoritmo se guardan artefactos en su carpeta correspondiente:

- `modelo.joblib`
- `metricas_validacion.json`
- `metricas_test.json`
- `reporte_test.txt`
- `matriz_confusion_test.png`

Además, se guardan artefactos comunes en `ml/comun/v1`:

- `columnas_modelo.json`
- `ids_test.csv`

## Qué muestra la interfaz

La aplicación muestra:

- Un botón para visualizar las features generadas.
- Una tabla comparativa con las métricas de test de los cuatro modelos.
- Un mensaje destacando el mejor modelo según `PR-AUC`.
- Las matrices de confusión de cada algoritmo en test.

## Estructura funcional del código

Las funciones principales de `main.py` son:

- `buscar_raiz_proyecto()`: localiza la raíz donde está el dataset procesado.
- `obtener_rutas()`: define carpetas de entrada y salida de artefactos.
- `preparar_datos_modelo()`: carga el CSV y construye las variables derivadas.
- `crear_modelo()`: monta el pipeline según el algoritmo.
- `hacer_particiones()`: divide el dataset en entrenamiento, validación y test.
- `entrenar_y_evaluar_modelo()`: entrena, calcula métricas y guarda artefactos.
- `construir_tabla_comparacion()`: resume resultados de los modelos.
- `main()`: construye la interfaz de Streamlit.

## Nota sobre `ml/feature_engineering.py`

El archivo `ml/feature_engineering.py` se utilizó como fase de experimentación para decidir qué variables derivadas merecían incorporarse al flujo final.

### Cómo se hicieron las pruebas

El proceso seguido fue un *ablation study* paso a paso:

- Se definió un conjunto de features candidatas derivadas a partir de variables clínicas y de síntomas.
- Se evaluó cada candidata con validación cruzada estratificada de **5 folds**.
- En cada iteración se probó añadir una feature nueva sobre el conjunto ya seleccionado.
- La decisión se tomó comparando la mejora marginal en varias métricas: `pr_auc`, `recall`, `f1`, `roc_auc`.

Además, el script imponía dos condiciones para aceptar una nueva variable:

- mejora mínima en PR-AUC (`GANANCIA_MINIMA_PR_AUC = 0.0005`)
- que el `recall` no empeorase más allá de la tolerancia (`TOLERANCIA_CAIDA_RECALL = 0.0020`)

### Por qué se eligieron `n_sintomas` y `riesgo_familiar_x_edad`

Aportan señal clínicamente interpretable y mejoraban el rendimiento sin complicar en exceso el espacio de variables. Se reutilizó el mismo conjunto en todos los algoritmos para mantener un pipeline común y una comparación justa.

### Por qué no se añadieron más features

Se priorizó un conjunto pequeño e interpretable: varias candidatas no mejoraban PR-AUC de forma suficiente, otras inestabilizaban el recall, y más interacciones aumentan riesgo de ruido o sobreajuste sin beneficio claro demostrado.

---

Para el **módulo de clasificación de imágenes** (pólipo vs normal), ver **`dl/vision_baseline/README.md`**.
