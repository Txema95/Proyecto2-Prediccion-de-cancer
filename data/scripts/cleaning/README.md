# Cleaning de imagenes Kvasir (preprocesado minimo comun)

Este directorio contiene el script `kvasir_preprocesado_minimo.py`, que aplica un preprocesado comun y reproducible antes del entrenamiento del modelo multiclase.

## Objetivo del preprocesado minimo

Estandarizar la entrada del modelo sin introducir reglas agresivas que puedan borrar señal clinica:

- recorte de bordes negros (viñeteado),
- normalizacion geometrica (recorte centrado cuadrado),
- redimensionado final a tamano fijo (por defecto `512x512`),
- salida en una carpeta limpia con manifest y resumen.

Problemas mas complejos (artefactos verdes, metadatos quemados, sesgo cromatico, oclusiones) se abordaran en el diseno del modelo y en estrategias de entrenamiento.

## Ejecucion

Desde la raiz del repositorio:

```bash
uv run python data/scripts/cleaning/kvasir_preprocesado_minimo.py
```

Opciones utiles:

- `--dataset-root`: ruta del dataset de entrada.
- `--manifest-in`: manifest base del EDA (`paso2_manifest_muestreo.csv`).
- `--output-root`: carpeta de salida (default `data/processed/kvasir_min_clean`).
- `--size`: tamano final cuadrado.
- `--umbral-negro`: umbral de deteccion para borde negro.
- `--padding-fraccion`: padding interno tras recorte.
- `--max-imagenes`: modo prueba rapida.

## Resultados del ultimo run (ejecutado por el usuario)

Fuente: `data/processed/kvasir_min_clean/resumen_cleaning.json`

- Filas de entrada del manifest: **4000**
- Imagenes procesadas: **4000**
- Errores: **0**
- Tamano final: **512x512**
- Umbral de negro: **12**
- Padding fraccion: **0.02**

Detalle por clase:

- `normal-cecum`: `n=1000`, recorte aplicado `1000`, media pixeles recortados `69220.847`
- `polyps`: `n=1000`, recorte aplicado `977`, media pixeles recortados `73503.243`
- `dyed-lifted-polyps`: `n=1000`, recorte aplicado `996`, media pixeles recortados `74371.783`
- `ulcerative-colitis`: `n=1000`, recorte aplicado `978`, media pixeles recortados `73460.412`

Archivos generados:

- `data/processed/kvasir_min_clean/imagenes/<clase>/*.jpg`
- `data/processed/kvasir_min_clean/manifest_clean.csv`
- `data/processed/kvasir_min_clean/resumen_cleaning.json`

## Conclusiones sobre estos resultados

1. **Pipeline estable**: el cleaning completo termina sin errores sobre las 4000 imagenes.
2. **Balance preservado**: se mantiene el equilibrio 1000/1000/1000/1000 por clase.
3. **Viñeteado tratado de forma consistente**: el recorte de borde negro se aplica en casi todas las imagenes (y en el 100% de `normal-cecum`), lo que reduce ruido de esquinas y homogeniza el campo visual.
4. **Entrada homogenea para entrenar**: todas las imagenes quedan en `512x512`, lo que simplifica dataloaders, batches y comparabilidad entre clases.
5. **Cleaning minimo correcto, no final**: el script estandariza geometria y bordes, pero no reemplaza las defensas del modelo contra sesgos (color azul, overlays, reflejos, blur, etc.).

## Siguientes pasos recomendados

1. **Construir manifest final de entrenamiento (sobre limpio)**  
   Generar un CSV final con columnas tipo `filepath,label,source,group_id,image_id`, apuntando a `kvasir_min_clean/imagenes/...`.

2. **Deduplicar para evitar fuga**  
   Usar `data/processed/kvasir_image_eda/paso3_duplicados_resumen.json` y `paso3_hashes_por_archivo.csv`.
   - Dato clave actual: **8 grupos MD5** con **16 archivos**.
   - Regla minima: dejar 1 representante por grupo MD5 (o agrupar por `group_id` para que no crucen splits).

3. **Crear splits train/val/test sin fuga**  
   Estratificar por clase y respetar `group_id` (duplicados y grupos relacionados no deben cruzar splits).

4. **Entrenar baseline sobre dataset limpio**  
   Entrenar primer modelo de referencia y reportar:
   - `accuracy`,
   - `macro-F1`,
   - matriz de confusion,
   - recall por clase (especial atencion a `polyps` y `ulcerative-colitis`).

5. **Mitigaciones en el modelo (fase siguiente)**  
   Incorporar augmentations y controles para los riesgos identificados:
   - sesgo cromatico (`dyed-lifted-polyps`),
   - artefactos de captura/metadatos,
   - oclusiones y baja calidad.

## Nota de trazabilidad

Para mantener comparabilidad de experimentos, guardar junto a cada entrenamiento:

- version del script de cleaning,
- parametros usados (`size`, `umbral-negro`, `padding-fraccion`),
- hash o timestamp de `manifest_clean.csv`.
