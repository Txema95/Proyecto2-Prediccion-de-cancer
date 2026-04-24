# Frontend (Streamlit)

Interfaz web del proyecto: simulación de apoyo al diagnóstico, consulta de modelos tabulares y de visión, con un tema de color orientado a entornos clínicos (ver `estilos_clinicos.py` y `.streamlit/config.toml`).

## Cómo arrancar

- Solo la interfaz: desde la raíz del repositorio, `uv run streamlit run frontend/app.py` (o con el directorio de trabajo donde exista el CSV de datos procesados).
- API + interfaz en la misma consola: `uv run python main.py` (raíz del repo).

La predicción del simulador (pestaña **Diagnosticar**) llama al backend HTTP; la URL base por defecto es `http://127.0.0.1:8000` y se puede cambiar con la variable de entorno `SIMULATOR_API_BASE_URL` (ver `config.py`).

## Flujo general: tres pestañas

El punto de entrada es `app.py`. La cabecera fija el título del simulador y bajo ella hay **tres pestañas** (etiquetas definidas en `config.py` como `TAB_CONSULTA`, `TAB_DATOS`, `TAB_IMAGEN`):

1. **Diagnosticar**  
   Asistente por pasos que pide datos clínicos, imágenes opcionales, una revisión del caso y, al final, muestra el resultado (probabilidades vía API y lógica local de imagen cuando aplica). El detalle del recorrido está en la sección siguiente.

2. **Diagnóstico con datos (tabular)**  
   Panel informativo y, si se desea, reentrenamiento de los modelos clásicos (regresión logística, random forest, XGBoost, SVM) a partir del CSV procesado. Lee y escribe artefactos bajo `ml/` en el repositorio. Implementado en `views/explorador_ml.py` (lógica reutilizada desde `ml/main.py`).

3. **Diagnóstico con imagen**  
   Revisión de entrenamientos de visión (carpetas de run bajo `dl/vision_baseline_kvasir/runs/`), métricas y análisis si existen. El simulador usa un checkpoint reciente para inferencia local; ver `servicio_vision_kvasir.py`. Implementado en `views/explorador_dl.py`.

En cada recarga de la aplicación, Streamlit ejecuta el código de **las tres pestañas**; el usuario solo ve el contenido de la pestaña activa, pero conviene no dejar tareas pesadas en vistas que no lo requieran.

## Flujo dentro de la pestaña “Diagnosticar”

`views/portal_simulador.py` dirige el asistente según `state.PASO_ACTUAL` (0 a 3). El aviso de uso académico y la barra de progreso se pintan en `layout.py` (`pintar_encabezado`, `pintar_progreso`). Los nombres de los pasos para el usuario coinciden con `PASOS` en `config.py`:

| Paso | Contenido | Módulo |
|------|------------|--------|
| 0 | Datos clínicos (formulario a partir del dataset de referencia) | `views/datos_clinicos.py` |
| 1 | Carga de imágenes (opcional) | `views/carga_imagenes.py` |
| 2 | Revisión del caso antes de predecir | `views/revision_caso.py` |
| 3 | Resultado (llamada al API, visión local, métricas en pantalla) | `views/resultado.py` |

El formulario reutiliza `formulario_clinico.py` y `labels.py` donde aplica. La comunicación con el API está en `servicio_api.py`; la orquestación de predicción y el CSV en `servicio_modelo.py`. El estado compartido vive en `state.py` (claves como `datos_formulario`, `imagenes`, `paso_actual`, etc.).

## Archivos de apoyo frecuentes

| Ruta | Rol |
|------|-----|
| `app.py` | `main`: configuración de página, tema, pestañas y enlace a las tres vistas. |
| `config.py` | Títulos, textos, URL del API, mapas de etiquetas. |
| `state.py` | Inicialización y claves de `st.session_state`. |
| `paths.py` | Raíz del repo y `sys.path` para importar `ml` / `dl`. |
| `estilos_clinicos.py` | Inyección de CSS global. |

Para una visión de todo el repositorio (API, datos, arranque unificado), ver el `README` de la raíz o la documentación en `docs/`.
