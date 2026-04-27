"""Constantes del simulador (textos, claves de datos, mapas de etiquetas)."""

import os

OBJETIVO = "cancer_diagnosis"
COLUMNA_ID = "id"
MAX_MB_POR_IMAGEN = 8

PASOS = ["Datos clinicos", "Imagenes", "Revision", "Resultado"]

PAGE_TITLE = "Cancer de colon — simulador y analisis (ML / DL)"
LAYOUT = "wide"

# Pestañas de la cabecera
TAB_CONSULTA = "Diagnosticar"
TAB_DATOS = "Diagnóstico con datos (tabular)"
TAB_IMAGEN = "Diagnóstico con imagen"
CABECERA_TITULO = "Simulador de apoyo al diagnóstico de cáncer de colon"
TAB_LABELS: tuple[str, str, str] = (TAB_CONSULTA, TAB_DATOS, TAB_IMAGEN)


def _parsear_umbral_decision() -> float:
    valor_bruto = os.environ.get("SIMULATOR_DECISION_THRESHOLD", "0.5").strip()
    try:
        valor = float(valor_bruto)
    except ValueError as exc:
        raise ValueError("SIMULATOR_DECISION_THRESHOLD debe ser numerico") from exc
    if not 0.0 <= valor <= 1.0:
        raise ValueError("SIMULATOR_DECISION_THRESHOLD debe estar entre 0.0 y 1.0")
    return valor


API_BASE_URL = os.environ.get("SIMULATOR_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DECISION_THRESHOLD = _parsear_umbral_decision()

NOMBRES_VISUALES_VARIABLES = {
    "age": "Edad (años)",
    "sex": "Sexo",
    "sof": "Sangre en heces (SOF)",
    "alcohol": "Consumo de alcohol",
    "tobacco": "Consumo de tabaco",
    "diabetes": "Diabetes",
    "tenesmus": "Tenesmo",
    "previous_rt": "Radioterapia previa",
    "rectorrhagia": "Rectorragia",
    "intestinal_habit": "Habito intestinal",
    "digestive_family_risk_level": "Riesgo familiar digestivo",
    "n_sintomas": "Numero de sintomas",
    "riesgo_familiar_x_edad": "Riesgo familiar por edad",
}

ETIQUETAS_POR_COLUMNA = {
    "sex": {0: "Hombre", 1: "Mujer"},
    "sof": {0: "No", 1: "Si"},
    "diabetes": {0: "No", 1: "Si"},
    "tenesmus": {0: "No", 1: "Si"},
    "previous_rt": {0: "No", 1: "Si"},
    "rectorrhagia": {0: "No", 1: "Si"},
    "cancer_diagnosis": {0: "No", 1: "Si"},
    "digestive_family_risk_level": {
        0: "Sin riesgo familiar relevante",
        1: "Dato incierto / ruido",
        2: "Riesgo medio (antecedente digestivo no especifico)",
        3: "Riesgo alto (antecedente de colon)",
    },
    "tobacco": {0: "No fumador", 1: "Fumador", 2: "Ex fumador"},
    "alcohol": {
        0: "Sin consumo",
        1: "Ocasional",
        2: "Moderado",
        3: "Consumo habitual",
        4: "Ex bebedor / antecedente severo",
    },
    "intestinal_habit": {
        0: "Sin alteracion relevante",
        1: "Normal",
        2: "Alternante",
        3: "Alteracion generica",
        4: "Diarrea",
        5: "Estrenimiento / otros sintomas",
    },
}
