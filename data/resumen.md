# Resumen: análisis y preparación de datos

Documento vivo del flujo de datos del simulador (cáncer de colon). Los scripts viven en `data/scripts/cleaning/` (limpieza) y `data/scripts/analysis/` (tipos, EDA).

El CSV **`cancer_final.csv`** procede del Excel **`data/raw/rectesestadistica.xlsx`** (hoja `Full1`, cuestionario en catalán); las etiquetas de alcohol, tabaco, hábito intestinal y antecedentes familiares en texto → códigos en el CSV están detalladas en **`data/raw/README.md`**.

---

## 1. Transformación de variables

El dataset en bruto mezcla texto (Excel) y, en la exportación, ordinales numéricos. La versión principal de trabajo es **`data/processed/cancer_final_clean_v2.csv`**, generada con `cancer_final_clean_v2.py`.

### Ordinales (se mantienen como enteros)

- **alcohol** (0–4), **tobacco** (0–2), **intestinal_habit** (0–5): en el Excel son categorías en catalán; en el CSV, enteros. Ver README para el significado clínico de las etiquetas originales.

### Binarias (0 / 1)

Tras la limpieza v2:

- **sex**: hombre = 0, mujer = 1  
- **sof**, **diabetes**, **tenesmus**, **previous_rt**, **rectorrhagia**, **cancer_diagnosis**: no = 0, sí = 1

> **sof** en el diccionario de datos se define como presencia de sangre en las heces.

---

## 2. Columna `digestive_family_risk_level`

La columna textual `**digestive_family_history`** tenía valores heterogéneos (incl. ruido tipo `#NOMBRE?`, `UNESCO`, etc.). En **v2** esa columna **no se exporta**; solo el nivel numérico, derivado de una **recodificación interna** (y corrección de mojibake UTF-8 leído como Latin-1, p. ej. `anÃ§a` → `ança`).


| Valor | Etiqueta        | Lógica (categorías internas tras limpiar)                                                     |
| ----- | --------------- | --------------------------------------------------------------------------------------------- |
| **0** | No_Risk         | Sin antecedente relevante: `no`                                                               |
| **1** | Unknown / Noise | Ruido o dato no fiable: categoría interna `unknown`                                           |
| **2** | Medium_Risk     | `yes` genérico, `yes(gastric)` → mapeados a antecedente no colon-específico                   |
| **3** | High_Risk       | Antecedente colon: `yes(colon)` y equivalentes internos `yes_colon` (incl. `colon` corregido) |


> **Nota:** en el script, variantes como `yes metav`, `yes mutations`, `yes stresses` se agrupan en la categoría interna `**yes`** → nivel **2**, no en el máximo.

---

## 3. Exploración de datos (EDA)

Análisis automatizado: `data/scripts/analysis/eda.py` → figuras en `data/processed/eda/`.

### 3.1. Correlación con el objetivo

La correlación de Pearson con `**cancer_diagnosis`** sitúa sobre todo **rectorragia** y **sof** por encima del resto; la **edad** y **intestinal_habit** tienen correlación moderada; **alcohol** es más débil en este conjunto. El mapa completo:

Mapa de calor de correlaciones

### 3.2. Perfil positivo vs negativo (resumen cuantitativo)

- **Edad media:** ~61 años sin cáncer vs ~68 años con cáncer.  
- **Alcohol / tabaco:** medias algo mayores en el grupo positivo.  
- **Síntomas:** la **rectorragia** y **sof** marcan fuerte diferencia de proporciones entre grupos (coherente con correlaciones altas).

### 3.3. `digestive_family_risk_level` vs diagnóstico

Barras apiladas por nivel de riesgo familiar (ver proporción de verde “sin cáncer” vs rojo “con cáncer”):

Riesgo digestivo apilado por diagnóstico

> El nivel **1** tiene pocas observaciones; conviene interpretarlo con cautela. La jerarquía 0→3 no tiene por qué ser monótona en tasas empíricas en todos los niveles; revisar la tabla de proporciones que imprime `eda.py`.

### 3.4. Tríada de síntomas (sof + tenesmus + rectorrhagia)

Variable sintética = suma de los tres (0 a 3). La tasa empírica de cáncer **sube fuerte** al pasar de 0 a 3 síntomas (útil para modelos basados en reglas o árboles):

Tasa de cáncer por número de síntomas

### 3.5. Desequilibrio de clases

En **~10 131** filas: aprox. **8 050** negativos y **2 081** positivos (orden **~3,9 : 1**). No es un 1:20 extremo, pero conviene usar **métricas adecuadas** (AUC-PR, F1, recall) y valorar **pesos de clase** o remuestreo si el modelo sesga hacia la mayoritaria.

### 3.6. Edad por sexo

Distribución de edad con `sex` por color (0 = hombre, 1 = mujer). Con diagnóstico positivo, las medias de edad por sexo son **muy parecidas** (~68 años); el gráfico ayuda a ver forma de las distribuciones, no solo la media.

Histograma de edad por sexo

---

## 4. Otros datasets / scripts


| Fichero                                              | Origen                                                             |
| ---------------------------------------------------- | ------------------------------------------------------------------ |
| `data/raw/cancer_final.csv`                          | Entrada bruta (`;`, encoding Latin-1 al leer)                      |
| `data/processed/cancer_final_clean.csv`              | Versión v1: one-hot digestivo + binarias (`cancer_final_clean.py`) |
| `data/raw/README.md`                                 | Diccionario de columnas del raw                                    |
| `data/scripts/analysis/cancer_final_column_types.py` | Conteos y tipos por columna del raw                                |


---

## 5. Notas para el modelo y el simulador

1. **Correlaciones muy altas** (p. ej. sangrados) pueden ser realidad del dataset o **filosofía del caso de uso**: en despistaje temprano sin síntomas graves, el modelo no debe depender solo de variables que en la práctica llegan tarde.
2. **Recall (sensibilidad)** suele priorizarse frente a un accuracy “vanidoso” en detección.
3. Algoritmos como **Random Forest** o **gradient boosting** encajan bien con muchas binarias y relaciones no lineales (p. ej. tríada).
4. **Variables derivadas:** opción **Senior** (edad > 50 o umbral clínico acordado) puede ayudar a explicabilidad sin sustituir la edad continua en el modelo.

---

*Última revisión alineada con el pipeline en `data/scripts/cleaning/cancer_final_clean_v2.py` y EDA en `data/scripts/analysis/eda.py`.*