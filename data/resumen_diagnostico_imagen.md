# Resumen: preparación de datos para diagnóstico asistido por imagen

Este documento describe los scripts en **`data/scripts/preparation/`**, que construyen el conjunto binario **pólipo (positivo) / mucosa normal (negativo)** y los ficheros tabulares que consume el módulo de visión (`ml/vision_baseline/`).

**Orden recomendado de ejecución**

1. `prepare_processed_data.py`  
2. `generate_manifest.py`  
3. `split_dataset.py`  

Todos asumen la **raíz del repositorio** como referencia (suben tres niveles desde la carpeta del script hasta localizar `data/` y el resto del proyecto).

---

## Fuentes de los datos

| Conjunto | Uso en el proyecto | Enlace |
| -------- | ------------------ | ------ |
| **CVC-ClinicDB** | Imágenes de **pólipos** (clase positiva) y `metadata.csv` con rutas y `sequence_id`. | [Kaggle — balraj98/cvcclinicdb](https://www.kaggle.com/datasets/balraj98/cvcclinicdb) |
| **Kvasir** | Imágenes de **mucosa normal** (clase negativa), versión **2** del dataset. | [Simula — Kvasir Dataset](https://datasets.simula.no/kvasir/) → descargar **Kvasir v2** (`kvasir-dataset-v2`) |

Las licencias y condiciones de uso de cada repositorio aplican al descargar y almacenar los datos; conviene citarlas en la documentación del trabajo académico.

---

## Cómo colocar los datos en el repositorio

Los scripts leen rutas **fijas** bajo `data/raw/`. Sin esa estructura, fallarán con `FileNotFoundError` o no encontrarán categorías.

### CVC-ClinicDB → `data/raw/polipos/`

1. Descarga el dataset desde Kaggle (enlace arriba).
2. Debe existir el fichero **`data/raw/polipos/metadata.csv`** con al menos las columnas que usa el código: **`png_image_path`** (ruta relativa a la carpeta de trabajo de las imágenes CVC) y **`sequence_id`** (para agrupar secuencias en el manifiesto y en los splits).
3. Todas las imágenes referenciadas en `png_image_path` deben existir en disco de forma que **`data/raw/polipos/` + ruta relativa del CSV** sea un fichero válido. En la práctica suele bastar con copiar el `metadata.csv` del release y la carpeta de PNG (o equivalente) dentro de `data/raw/polipos/`, respetando las rutas que indica el CSV. Si el ZIP trae otra jerarquía, **reorganiza o ajusta** hasta que las rutas del CSV coincidan con la ubicación real bajo `polipos/`.

### Kvasir v2 → `data/raw/kvasir-dataset-v2/`

1. Descarga **Kvasir Dataset v2** desde la página de Simula (enlace arriba).
2. Descomprime el contenido de forma que exista exactamente esta carpeta en la raíz del proyecto:

   `data/raw/kvasir-dataset-v2/`

3. Dentro deben estar (como directorios con imágenes) al menos estas tres subcarpetas usadas por el script de preparación:

   - `normal-cecum`
   - `normal-pylorus`
   - `normal-z-line`

   Si al descomprimir aparece un directorio intermedio (p. ej. solo `kvasir-dataset-v2/kvasir-dataset-v2/...`), **mueve o renombra** hasta que las rutas relativas coincidan con lo anterior; el código no busca nombres alternativos.

### Resumen de rutas obligatorias

```
data/raw/
├── polipos/
│   ├── metadata.csv          # CVC: png_image_path, sequence_id, ...
│   └── ...                  # imágenes según png_image_path
└── kvasir-dataset-v2/
    ├── normal-cecum/        # imágenes .png/.jpg/...
    ├── normal-pylorus/
    └── normal-z-line/
```

*(El resto de carpetas de Kvasir v2 pueden coexistir; el script solo lee las tres “normal-*”.)*

Tras comprobar esta estructura, ejecuta los scripts de preparación en el orden indicado al inicio del documento.

---

## 1. `prepare_processed_data.py`

**Objetivo:** poblar `data/processed/polipo/` y `data/processed/sano/` a partir de datos en bruto.

**Entradas**

- **Positivos (CVC-ClinicDB):** `data/raw/polipos/metadata.csv` (columna `png_image_path`) e imágenes bajo `data/raw/polipos/`.
- **Negativos (Kvasir):** `data/raw/kvasir-dataset-v2/` en las carpetas `normal-cecum`, `normal-pylorus` y `normal-z-line`.

**Salidas**

- `data/processed/polipo/`: copia de **todas** las imágenes de pólipo referenciadas en la metadata CVC (nombre de archivo plano).
- `data/processed/sano/`: subconjunto muestreado de imágenes “normales” de Kvasir, con prefijo `categoria__nombre` para evitar colisiones de nombre entre carpetas.

**Parámetros destacados**

| Argumento | Por defecto | Descripción |
|-----------|-------------|-------------|
| `--kvasir-target` | `612` | Número total de imágenes sanas a copiar. Se reparte de forma equitativa entre las tres categorías normales. |
| `--seed` | `42` | Semilla del muestreo aleatorio de Kvasir (reproducibilidad). |
| `--overwrite` | (flag) | Si existen `polipo/` y `sano/`, los elimina y los vuelve a crear. |

**Notas**

- Sin `--overwrite`, las carpetas de destino deben gestionarse manualmente si se quiere una recreación limpia.
- Los positivos se copian solo con `origen.name`; se asume que los nombres en CVC son únicos.

**Ejemplo**

```bash
uv run python data/scripts/preparation/prepare_processed_data.py --overwrite
```

---

## 2. `generate_manifest.py`

**Objetivo:** generar un inventario único de todas las imágenes procesadas con metadatos estables para ML y trazabilidad.

**Entradas**

- Carpetas `data/processed/polipo/` y `data/processed/sano/` (tras el paso anterior).
- `data/raw/polipos/metadata.csv` para enlazar **nombre de archivo → `sequence_id`** en imágenes CVC.

**Salida**

- Por defecto: **`data/processed/manifest.csv`**.

**Columnas** (en inglés por compatibilidad con pipelines)

| Columna | Significado |
|---------|-------------|
| `filepath` | Ruta del fichero relativa a la raíz del proyecto (formato POSIX). |
| `label` | `1` = pólipo, `0` = sano. |
| `source` | Origen lógico (`cvc_clinicdb`, `kvasir_<categoria>`, etc.). |
| `group_id` | Identificador de agrupación para splits: secuencia CVC (`cvc_seq_<id>`) o grupo derivado para Kvasir. |
| `image_id` | Identificador único legible por imagen. |

**Ejemplo**

```bash
uv run python data/scripts/preparation/generate_manifest.py
```

Opcional: `--output ruta/personalizada.csv`.

---

## 3. `split_dataset.py`

**Objetivo:** partir el manifiesto en **train / val / test** asignando splits por **`group_id`**, no por imagen suelta, para reducir **fuga de información** entre conjuntos (p. ej. varias imágenes de la misma secuencia clínica quedan en un solo split).

**Entradas**

- Por defecto: `data/processed/manifest.csv`.

**Salida**

- Por defecto: **`data/processed/splits.csv`**: mismas columnas que el manifiesto más la columna `split` (`train`, `val`, `test`).

**Parámetros destacados**

| Argumento | Por defecto | Descripción |
|-----------|-------------|-------------|
| `--train-ratio` | `0.70` | Proporción objetivo en train. |
| `--val-ratio` | `0.15` | Proporción objetivo en validación. |
| `--test-ratio` | `0.15` | Proporción objetivo en test. |
| `--seed` | `42` | Semilla para el orden aleatorio de grupos. |

Las tres proporciones deben **sumar 1.0**. El algoritmo reparte grupos por etiqueta (`label` 0 y 1 por separado) y valida que no haya **intersección de `group_id`** entre train, val y test.

**Ejemplo**

```bash
uv run python data/scripts/preparation/split_dataset.py
```

---

## Enlace con el resto del proyecto

- Tras generar **`splits.csv`**, el entrenamiento de visión puede ejecutarse con `ml/vision_baseline/train.py`, que lee rutas e imágenes según las columnas del split.
- Para datos tabulares del simulador (cuestionario / variables clínicas), ver **`data/resumen_diagnostico_clinico.md`**.
