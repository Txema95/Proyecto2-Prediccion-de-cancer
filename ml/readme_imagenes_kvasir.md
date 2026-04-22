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
| `dataset_torch.py` | `DatasetKvasirMulticlase` y transforms estilo ImageNet. |
| `modelo_baseline.py` | ResNet-18 con cabecera de 4 clases y `evaluar_cargador` (inferencia por lotes). |
| `entrenar.py` | Entrenamiento (Adam, `CrossEntropyLoss`, checkpoint por mayor F1 macro en validación). |
| `evaluar.py` | Carga `mejor_pesos.pt` y evalúa en `train` / `val` / `test`. |

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

# 3) Entrenamiento
uv run python ml/vision_baseline_kvasir/entrenar.py --epocas 10

# 4) Re-evaluar el último run (test por defecto)
uv run python ml/vision_baseline_kvasir/evaluar.py --ultimo-run
```

Argumentos frecuentes: `--batch`, `--workers` (0 por defecto en `entrenar.py`), `--log-cada-lotes` (progreso por lotes en train), `--output-dir` para otra carpeta de runs, `--dispositivo auto|cpu|cuda`.

---

## Artefactos por entrenamiento

Cada ejecución crea `ml/vision_baseline_kvasir/runs/resnet18_YYYYMMDD_HHMMSS/` con, entre otros:

- `config.json` — híperparámetros y rutas.
- `historial.json` — por época: `perdida_train`, `f1_val_macro`, `acc_val`.
- `mejor_pesos.pt` — mejores pesos según **F1 macro en validación** al final de cada época.
- `metricas_test.json` y `reporte_clasificacion_test.txt` — resultados al cerrar el script de entrenamiento (test).
- `evaluar.py` regenera `metricas_test.json` (resumen) y `reporte_clasificacion_test.txt` / `metricas_{split}.json` según el split elegido.

---

## Modelo (baseline)

- **Arquitectura:** ResNet-18 preentrenada en ImageNet, cabecera `Linear` a **4** salidas.
- **Pérdida:** entropía cruzada multiclase.
- **Optimizador:** Adam (por defecto `lr=1e-4`, `weight_decay=1e-4`).
- **Imagen de entrada al modelo:** 224×224, normalización ImageNet; en train: flip horizontal, `ColorJitter` suave.
- **Checkpoint:** se guarda cuando mejora el F1 macro en **validación**; al terminar las épocas se evalúa en **test** con ese checkpoint.

---

## Resultados de referencia (entrenamiento registrado)

**Run de referencia:** `ml/vision_baseline_kvasir/runs/resnet18_20260422_121534/`

**Configuración guardada (`config.json`):** 10 épocas, batch 32, `lr=1e-4`, `weight_decay=1e-4`, semilla 42, imágenes 224×224, dispositivo **cpu**, `workers=2` (en ejecuciones manuales posteriores se recomienda `workers=0` si el DataLoader se comporta de forma rara).

### Validación (por época; fuente: `historial.json`)

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

### Conjunto de prueba (598 imágenes; fuente: `reporte_clasificacion_test.txt` tras entrenar / `evaluar.py`)

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

**Comentario breve:** la clase `dyed-lifted-polyps` alcanza métricas muy altas (suele asociarse a pistas cromáticas/artefacto de tinción fáciles de separar). `polyps` presenta el F1 por clase más bajo, coherente con solapamiento visual con otras categorías. Estas cifras son **indicativas** del run y de los datos; conviene no extrapolar a pacientes reales sin validación clínica e independiente.

---

## Enlaces con el resto del proyecto

- Datasets en bruto y estructura de carpetas: `data/scripts/analysis/image_analysis/configuracion.py` (clases esperadas).
- Pipeline clínico/tabular (otro módulo): `ml/readme_ml_clinico.md`.
