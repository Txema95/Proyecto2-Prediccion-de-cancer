# README: visión — Kvasir multiclase (`ml/vision_baseline_kvasir/`)

Documentación del **prototipo solo Kvasir v2**: clasificación en **4 clases** a partir de imágenes preprocesadas, distinto del baseline binario CVC+Kvasir en `ml/vision_baseline/` (ver `ml/readme_imagenes.md` y `data/resumen_diagnostico_imagen.md`).

**Aviso:** el modelo es un **apoyo a investigación o demostración**; no sustituye el criterio clínico.

---

## Flujo de datos (resumen)

1. **Raw:** `data/raw/kvasir-dataset-v2/` con al menos las carpetas  
   `normal-cecum`, `polyps`, `dyed-lifted-polyps`, `ulcerative-colitis`.
2. **EDA (pasos 1–4):** `data/scripts/analysis/image_analysis/ejecutar_analisis.py`  
   → salida en `data/processed/kvasir_image_eda/` (inventario, muestreo, hashes MD5/dHash, EDA de visión).
3. **Preprocesado mínimo:** `data/scripts/cleaning/kvasir_preprocesado_minimo.py`  
   → `data/processed/kvasir_min_clean/` (`imagenes/<clase>/`, `manifest_clean.csv`).  
   Detalle: `data/scripts/cleaning/README.md`.
4. **Manifest de entrenamiento (deduplicado por MD5 del EDA):**  
   `ml/vision_baseline_kvasir/generar_manifest.py`  
   → `data/processed/kvasir_min_clean/manifest_kvasir_multiclase.csv` y `resumen_manifest_dedup.json`.
5. **Splits train/val/test (por `group_id`, estratificado por clase):**  
   `ml/vision_baseline_kvasir/crear_splits.py`  
   → `data/processed/kvasir_min_clean/splits_kvasir_multiclase.csv`.

Tras el paso 4 se eliminan **8** filas duplicadas (mismo MD5 en bruto) respecto a 4000 muestreadas: el manifest queda con **3992** imágenes. Los conteos aproximados en splits (semilla 42) son: train **2795**, val **599**, test **598**.

> Nota: `data/processed/**` está en `.gitignore`; en otro clon hay que **re-ejecutar** el pipeline o copiar esas carpetas a mano.

---

## Módulo `ml/vision_baseline_kvasir/`

| Archivo | Descripción |
|---------|-------------|
| `constantes.py` | Orden fijo de las 4 clases e índices 0–3. |
| `paths.py` | Resolución de la raíz del repositorio. |
| `manifest_entrenamiento.py` | Lógica de unión con `paso3_hashes_por_archivo.csv` y un representante por MD5. |
| `particion.py` | Asignación train/val/test respetando `group_id` y equilibrio aproximado por etiqueta. |
| `generar_manifest.py` | Script CLI: genera el manifest multiclase. |
| `crear_splits.py` | Script CLI: genera `splits_kvasir_multiclase.csv`. |
| `dataset_torch.py` | `DatasetKvasirMulticlase` y transforms estilo ImageNet: train (resize + **rotación, flip, ColorJitter**) vs. val/test (solo resize + normalizar). |
| `modelo_baseline.py` | ResNet-18 con cabecera de 4 clases y `evaluar_cargador` (inferencia por lotes). |
| `entrenar.py` | Entrenamiento (Adam, `CrossEntropyLoss`, augmentations en train, **early stopping** por F1 macro en val, checkpoint; ver sección *Decisiones de entrenamiento*). |
| `evaluar.py` | Carga `mejor_pesos.pt` y evalúa en `train` / `val` / `test`. |
| `analisis_evaluacion.py` | Evaluación ampliada en un split: matriz de confusión, **ROC** one-vs-rest, probabilidades por muestra, pares de confusión, entropía (aciertos vs errores) → `runs/.../analisis_{split}/resumen_analisis.json` y gráficos. Complementa `evaluar.py`. |

Clases y etiqueta entera (columna `label` en el CSV):

| Índice | Clase (carpeta Kvasir) |
|--------|-------------------------|
| 0 | `normal-cecum` |
| 1 | `polyps` |
| 2 | `dyed-lifted-polyps` |
| 3 | `ulcerative-colitis` |

---

## Comandos (desde la raíz del repo)

```bash
# 1) Manifest deduplicado
uv run python ml/vision_baseline_kvasir/generar_manifest.py

# 2) Splits
uv run python ml/vision_baseline_kvasir/crear_splits.py

# 3) Entrenamiento (por defecto: early stopping, tope 30 epocas; se puede fijar otro tope o desactivar)
uv run python ml/vision_baseline_kvasir/entrenar.py
# Equivalente a correr 10 epocas completas sin early stopping (comportamiento antiguo aprox.):
# uv run python ml/vision_baseline_kvasir/entrenar.py --sin-early-stopping --epocas 10

# 4) Re-evaluar el último run (test por defecto)
uv run python ml/vision_baseline_kvasir/evaluar.py --ultimo-run

# 5) Analisis ampliado (mismas imagenes, metricas y figuras adicionales)
uv run python ml/vision_baseline_kvasir/analisis_evaluacion.py --ultimo-run --split test
```

Argumentos frecuentes en `entrenar.py`: `--batch`, `--paciencia-early`, `--min-delta-f1-val`, `--sin-early-stopping`, `--epocas` (tope máximo), `--workers` (0 por defecto en `entrenar.py`), `--log-cada-lotes`, `--output-dir`, `--dispositivo auto|cpu|cuda`.

---

## Artefactos por entrenamiento

Cada ejecución crea `ml/vision_baseline_kvasir/runs/resnet18_YYYYMMDD_HHMMSS/` con, entre otros:

- `config.json` — híperparámetros, rutas y, al finalizar, `epocas_ejecutadas`, `detenido_por_early_stopping`, `mejor_f1_val_checkpoint` si aplica.
- `historial.json` — por época: `perdida_train`, `f1_val_macro`, `acc_val`.
- `mejor_pesos.pt` — mejores pesos según **F1 macro en validación** al final de cada época.
- `metricas_test.json` y `reporte_clasificacion_test.txt` — resultados al cerrar el script de entrenamiento (test).
- `evaluar.py` regenera `metricas_test.json` (resumen) y `reporte_clasificacion_test.txt` / `metricas_{split}.json` según el split elegido.

---

## Modelo (baseline)

- **Arquitectura:** ResNet-18 preentrenada en ImageNet, cabecera `Linear` a **4** salidas.
- **Pérdida:** entropía cruzada multiclase.
- **Optimizador:** Adam (por defecto `lr=1e-4`, `weight_decay=1e-4`).
- **Imagen de entrada al modelo:** 224×224, normalización ImageNet; en train ver *Decisiones de entrenamiento*; val/test: solo `Resize` + normalización.
- **Checkpoint:** se guarda cuando mejora el F1 macro en **validación**; al terminar (o al parar por early stopping) se evalúa en **test** con el mejor checkpoint de val.

---

## Decisiones de entrenamiento (augmentation, early stopping, tope de épocas)

Estas elecciones están implementadas en `dataset_torch.py` (solo **train**) y en `entrenar.py` (bucle y argumentos en línea de comandos).

**Data augmentation (más exigente que el baseline mínimo)**  
En el conjunto de entrenamiento, además de espejo horizontal, se aplica **rotación aleatoria pequeña** (±15°) y un **`ColorJitter` algo más intenso** (brillo/contraste, saturación y matiz) antes de `ToTensor` y la normalización ImageNet. Racional:

- la red ya no ve siempre el mismo píxel en la misma posición, lo que **dificulta la memorización** de apariencias fijas;
- se simula parte de la variación típica en endoscopía (orientación, iluminación) sin tocar val/test, que **siguen** siendo deterministas y comparables entre runs;
- mantiene coherencia con el criterio clínico de desplegar luego con imágenes ligeramente distintas a las de entrenamiento.

**Early stopping (monitor: F1 macro en validación)**  
El F1 en validación no sube de forma monótona; en `historial.json` aparecen bajones puntuales (p. ej. entre épocas) seguidos de recuperación. Un **criterio de paciencia** (por defecto **4** épocas consecutivas **sin** mejora de F1 frente al mejor, respetando un `min_delta` mínimo) detiene el entrenamiento **sin** desperdiciar tiempo si el modelo deja de generalizar al conjunto de validación, y mantiene el criterio del **checkpoint** alineado con el mejor F1 de val, no con la última época. Así se reduce riesgo de continuar aprendiendo ruido o patrones de train.

**Tope de épocas = 30 (frente a 10 como valor por defecto antiguo)**  
Con más augmentación, el ajuste puede requerir **más ciclos** en el piso de train para converger, pero en la práctica el tope a menudo **no se alcanza**: el early stopping corta antes. Subir el máximo evita quedarse articialmente en “solo 10 épocas” cuando aún habría mejora en val; a la vez, si val se estanca, se para solo.

> Para replicar un entrenamiento al estilo antiguo (fijo 10 épocas, sin early stopping): `uv run python ml/vision_baseline_kvasir/entrenar.py --sin-early-stopping --epocas 10`.

**Análisis con `analisis_evaluacion.py`**  
Más allá de accuracy y F1 de `evaluar.py`, el script añade **AUC-ROC** one-vs-rest, **entropia** promedio de la softmax (más baja = más seguro en términos de distribución), y los **principales pares de confusión** (qué pares de clases se confunden más a menudo). Eso matiza un solo escalar: dos modelos con accuracy parecida pueden distinguirse en AUC, en inseguridad en aciertos o en qué pares de clases se equivocan.

---

## Resultados y comparación de runs (mismo split de test, 598 imágenes)

Comparación entre el **modelo “anterior”** (sin augmentation fuerte, 10 épocas fijas, `workers=2` en el `config` registrado) y el **último entrenamiento documentado** con las mejoras (augmentation + early stopping, tope 30 épocas, `workers=0`).

| | Run `resnet18_20260422_121534` (referencia **antigua**) | Run `resnet18_20260422_190751` (**reciente**) |
|--|--|--|
| Entrenamiento | 10/10 épocas; train: flip + `ColorJitter` leve (baseline) | 10/30 **épocas efectivas**; early stopping activado; train: **rotación ±15°**, `ColorJitter` más intenso |
| Mejor F1 en val (checkpoint) | ≈0,977 (ép. 9–10 en `historial.json`) | ≈0,975 (**ép. 6**; val no mejora ≥`min_delta` 4 épocas seguidas) |
| **Test — accuracy** | 0,9615 | **0,9666** |
| **Test — F1 macro** | 0,9615 | **0,9666** |
| AUC-ROC macro OvR (`analisis_evaluacion.py`, `resumen_analisis.json`) | 0,9969 | **0,9984** |
| Muestras erróneas (test) | 23 | **20** |
| Entropía media de la salida (menor = más pico) | 0,0695 | 0,0633 |

F1 **por clase en test** (mismo soporte 598; fuente: `reporte_clasificacion_test.txt`):

| Clase | F1 (121534) | F1 (190751) |
|--------|------------:|------------:|
| `normal-cecum` | 0,959 | 0,956 |
| `polyps` | 0,946 | 0,957 |
| `dyed-lifted-polyps` | 0,987 | 0,987 |
| `ulcerative-colitis` | 0,954 | 0,967 |

**Lectura breve**  
- La mejora global es **acotada** (unos **0,5** puntos porcentuales en accuracy/F1 y **3** errores menos de 598), con **misma partición y semilla 42** en el CSV de splits; el mayor salto de calidad vía `analisis_evaluacion` está en el **AUC-ROC** y en algo menos de **entropía media** en el reciente, coherente con **predicciones algo más puntuales** en términos de probabilidad.  
- Sigue el patrón de confusión entre vías y lesiones: en el run reciente, los pares frecuentes (véase `resumen_analisis.json` → `pares_confusion_principales`) siguen juntando **`normal-cecum`** con **`ulcerative-colitis`** o **`polyps`**, lo que indica dificultad estructural del task, no solo ajuste de épocas.  
- No se debe **extrapolar a nuevos centros o protocolos** sin validación externa: la ganancia se mide en **este** test fijo.

**Run de referencia (histórico, baseline suave de documentación detallada por época en val):** `ml/vision_baseline_kvasir/runs/resnet18_20260422_121534/`

**Configuración guardada (`config.json` del 121534):** 10 épocas, batch 32, `lr=1e-4`, `weight_decay=1e-4`, semilla 42, imágenes 224×224, dispositivo **cpu**, `workers=2` (en ejecuciones manuales posteriores se recomienda `workers=0` si el DataLoader se comporta de forma rara).

**Run con augmentation + early stopping (`resnet18_20260422_190751`, semilla 42, mismo CSV de splits):** tope 30 épocas, paciencia 4, `min_delta_f1_val=1e-4`, se **detuvo en la época 10** al cumplirse 4 val seguidas sin mejora frente al mejor F1 de val (alcanzado en **época 6**, ≈0,975 en `config.json` como `mejor_f1_val_checkpoint`).

### Run 121534 — validación (por época; fuente: `historial.json`)

| Época | Pérdida media (train) | F1 macro (val) | Accuracy (val) |
|------:|----------------------:|-----------------:|-----------------:|
| 1 | 0,295 | 0,957 | 0,957 |
| 2 | 0,091 | 0,968 | 0,968 |
| 3 | 0,063 | 0,967 | 0,967 |
| 4 | 0,040 | 0,964 | 0,963 |
| 5 | 0,037 | 0,975 | 0,975 |
| 6 | 0,023 | 0,975 | 0,975 |
| 7 | 0,017 | 0,975 | 0,975 |
| 8 | 0,021 | 0,975 | 0,975 |
| 9 | 0,018 | 0,977 | 0,977 |
| 10 | 0,021 | 0,977 | 0,977 |

### Run 190751 — validación (por época; fuente: `historial.json`)

| Época | Pérdida media (train) | F1 macro (val) | Accuracy (val) |
|------:|----------------------:|-----------------:|-----------------:|
| 1 | 0,339 | 0,943 | 0,943 |
| 2 | 0,131 | 0,948 | 0,948 |
| 3 | 0,087 | 0,962 | 0,962 |
| 4 | 0,068 | 0,968 | 0,968 |
| 5 | 0,048 | 0,967 | 0,967 |
| 6 | 0,048 | 0,975 | 0,975 |
| 7 | 0,041 | 0,968 | 0,968 |
| 8 | 0,053 | 0,970 | 0,970 |
| 9 | 0,030 | 0,963 | 0,963 |
| 10 | 0,020 | 0,970 | 0,970 |

(Al terminar la época 10 se activó el criterio de parada: desde la ép. 6 no había mejora suficiente 4 ciclos seguidos frente a ese máximo de val en el criterio del script.)

### Run 121534 — test (598 imágenes; fuente: `reporte_clasificacion_test.txt` / `evaluar.py`)

| Métrica | Valor |
|--------|------:|
| Accuracy | **0,9615** |
| F1 macro | **0,9615** |
| F1 micro (coincide con accuracy en test) | **0,9615** |

| Clase | Precision | Recall | F1 | Soporte (test) |
|-------|----------:|-------:|---:|---------------:|
| `normal-cecum` | 0,979 | 0,940 | 0,959 | 150 |
| `polyps` | 0,946 | 0,946 | 0,946 | 149 |
| `dyed-lifted-polyps` | 0,987 | 0,987 | 0,987 | 150 |
| `ulcerative-colitis` | 0,936 | 0,973 | 0,954 | 149 |

### Run 190751 — test (598 imágenes; misma estructura)

| Métrica | Valor |
|--------|------:|
| Accuracy | **0,9666** |
| F1 macro | **0,9666** |
| F1 micro | **0,9666** |

| Clase | Precision | Recall | F1 | Soporte (test) |
|-------|----------:|-------:|---:|---------------:|
| `normal-cecum` | 0,966 | 0,947 | 0,956 | 150 |
| `polyps` | 0,947 | 0,966 | 0,957 | 149 |
| `dyed-lifted-polyps` | 0,993 | 0,980 | 0,987 | 150 |
| `ulcerative-colitis` | 0,960 | 0,973 | 0,967 | 149 |

**Comentario breve (ambos runs):** `dyed-lifted-polyps` mantiene métricas altas (pistas cromáticas/tinción). `polyps` y `normal-cecum` suelen acumular buena parte del error, coherente con el solapamiento visual. Las cifras de la tabla de **comparación** y de `resumen_analisis.json` son **indicativas** de estos datos; no deben entenderse como desempeño clínico sin un estudio aparte.

---

## Enlaces con el resto del proyecto

- Datasets en bruto y estructura de carpetas: `data/scripts/analysis/image_analysis/configuracion.py` (clases esperadas).
- Pipeline clínico/tabular (otro módulo): `ml/readme_ml_clinico.md`.
