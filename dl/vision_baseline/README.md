# README: ML por imagen — `dl/vision_baseline/`

Baseline de **clasificación binaria** (pólipo vs mucosa normal) con **PyTorch** y **torchvision**. Los datos tabulares de splits deben generarse antes con los scripts de `data/scripts/preparation/` (ver **`data/resumen_diagnostico_imagen.md`**).

## Requisitos

- Entorno con dependencias del proyecto (`torch`, `torchvision`, etc.; el repo usa `uv`).
- Fichero **`data/processed/splits.csv`** con columnas: `filepath`, `label`, `source`, `group_id`, `image_id`, `split`.
- Imágenes accesibles en rutas `raiz_proyecto / filepath` según ese CSV.

## Archivos del módulo

### `dataset.py`

Módulo de biblioteca (no se ejecuta por sí solo). Define:

- **`RegistroSplit`**: dataclass con una fila del split (`filepath`, `label`, `source`, `group_id`, `image_id`, `split`).
- **`cargar_registros_desde_csv`**: lee `splits.csv`, filtra por `split` ∈ {`train`, `val`, `test`} y valida columnas.
- **`DatasetColonoscopiaBinario`**: `torch.utils.data.Dataset` que carga cada imagen en RGB, aplica `transform` opcional y devuelve `(tensor_o_PIL, etiqueta)` con etiqueta float binaria.

`train.py` y `evaluate.py` importan esta clase (`from dl.vision_baseline.dataset import ...` o `from dataset import ...` si se ejecuta desde la subcarpeta).

### `train.py`

Entrena **ResNet50** o **MobileNet v2** (cabecera binaria con una neurona de salida, `BCEWithLogitsLoss`).

**Rutas por defecto** (relativas a la raíz del repo, resueltas desde la ubicación del script):

- `--splits-csv`: `data/processed/splits.csv`
- `--output-dir`: `dl/vision_baseline/runs/`

Cada ejecución crea una carpeta `dl/vision_baseline/runs/<modelo>_YYYYMMDD_HHMMSS/` con, entre otros:

- `config.json`
- `best_checkpoint.pt` (mejor época según `--metrica-checkpoint`: `val_recall` o `val_f1`)
- `historial_entrenamiento.json`
- `resumen_final.json` (incluye métricas en test al finalizar)

**Ejemplo**

```bash
uv run python dl/vision_baseline/train.py
uv run python dl/vision_baseline/train.py --modelo mobilenet_v2 --epocas 20 --batch-size 32 --metrica-checkpoint val_f1
```

Parámetros útiles: `--learning-rate`, `--weight-decay`, `--seed`, `--workers` (workers del `DataLoader`; `0` evita multiproceso en la carga).

Al inicio se usan pesos **ImageNet** (`ResNet50_Weights.DEFAULT` / `MobileNet_V2_Weights.DEFAULT`); la primera corrida puede descargar pesos.

### `evaluate.py`

Carga un checkpoint (`best_checkpoint.pt`) y evalúa el split **test** con el mismo backbone que indiques (`--modelo` debe coincidir con el entrenamiento).

**Obligatorio uno de:**

- `--ultimo-run`: elige el `best_checkpoint.pt` más reciente por fecha de modificación bajo `dl/vision_baseline/runs/`.
- `--checkpoint RUTA`: ruta relativa a la raíz del proyecto o absoluta al `.pt`.

**Ejemplos**

```bash
uv run python dl/vision_baseline/evaluate.py --ultimo-run --modelo resnet50
uv run python dl/vision_baseline/evaluate.py --checkpoint dl/vision_baseline/runs/resnet50_20260420_165518/best_checkpoint.pt --modelo resnet50
```

Escribe en la **carpeta del run** (junto al checkpoint):

- `metricas_test_detalladas.json`
- `reporte_test.txt`
- `matriz_confusion_test.png`

Opciones: `--splits-csv`, `--batch-size`, `--workers`, `--umbral` (por defecto `0.5` para binarizar probabilidades).

## Flujo recomendado

1. Preparar imágenes y CSVs con `data/scripts/preparation/` → `splits.csv`.
2. `uv run python dl/vision_baseline/train.py` [opciones].
3. `uv run python dl/vision_baseline/evaluate.py --ultimo-run --modelo <mismo_que_train>`.

## Datos tabulares (cuestionario / variables clínicas)

Ver **`ml/readme_ml_clinico.md`** (`ml/main.py` y *feature engineering*).
