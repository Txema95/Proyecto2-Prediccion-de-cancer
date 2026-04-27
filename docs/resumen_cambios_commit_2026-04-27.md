# Resumen de cambios para commit (2026-04-27)

Este documento resume los cambios funcionales y de artefactos generados en la rama actual antes de crear el commit.

## 1) ML tabular: incorporacion de CatBoost y evaluacion mas robusta

- Se anade `CatBoostClassifier` al pipeline de entrenamiento en `ml/main.py`.
- El flujo pasa a entrenar y comparar **5 baselines**: `regresion_logistica`, `random_forest`, `xgboost`, `svm` y `catboost`.
- Se crea la ruta de artefactos `ml/catboost/v1/artefactos` y se guardan:
  - `modelo.joblib`
  - `metricas_cv.json`
  - `metricas_validacion.json`
  - `metricas_test.json`
  - `reporte_test.txt`
  - `matriz_confusion_test.png`
- Se anade validacion cruzada estratificada configurable (`folds`) para todos los modelos y se reporta media/desviacion estandar.
- Se consolida el uso de umbral de decision en evaluacion:
  - configurable desde variable de entorno,
  - ajustable desde la UI,
  - opcionalmente optimizable en validacion para maximizar sensibilidad con precision minima.

## 2) Backend: modelo tabular por defecto y umbral configurable

- `backend/app/services/prediccion_tabular.py` ahora carga por defecto el modelo tabular desde:
  - `ml/catboost/v1/artefactos/modelo.joblib`
  en lugar de `xgboost`.
- `backend/app/core/config.py` incorpora parseo y validacion de `SIMULATOR_DECISION_THRESHOLD` para asegurar un valor entre `0.0` y `1.0`.

## 3) Frontend: alineacion con nuevo baseline tabular

- Se actualizan componentes de frontend para trabajar con la nueva configuracion de modelo tabular/umbral:
  - `frontend/config.py`
  - `frontend/servicio_modelo.py`
  - `frontend/servicio_vision_kvasir.py`
  - `frontend/state.py`
  - `frontend/views/carga_imagenes.py`

## 4) Baselines de vision: ajustes de entrenamiento/evaluacion

- Cambios en scripts de vision para mejorar seleccion de dispositivo y flujo de ejecucion:
  - `dl/vision_baseline/train.py`
  - `dl/vision_baseline_kvasir/entrenar.py`
  - `dl/vision_baseline_kvasir/evaluar.py`
- Se incluyen nuevos runs y artefactos de entrenamiento/evaluacion en carpetas `dl/vision_baseline/runs/` y `dl/vision_baseline_kvasir/runs/`.

## 5) Dependencias

- Se anade `catboost` al proyecto:
  - `pyproject.toml`
  - `uv.lock`

## 6) Documentacion actualizada

- Se actualizan README de ML tabular para reflejar:
  - inclusion de `CatBoost`,
  - comparativa de 5 modelos,
  - metricas CV,
  - ranking por robustez de generalizacion.
- Archivos actualizados:
  - `ml/README.md`
  - `ml/readme_ml_clinico.md`

## 7) Nota sobre artefactos y ruido de commit

En el estado actual tambien aparecen archivos de ejecucion/entrenamiento (por ejemplo `catboost_info/`, `runs/`, imagenes y metricas generadas). Segun estrategia del equipo, se recomienda validar si todos deben versionarse o si conviene filtrar parte en `.gitignore`.
