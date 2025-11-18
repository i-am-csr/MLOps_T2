import requests
import pandas as pd
import numpy as np
import time
from evidently.report import Report
from evidently.metric_preset import RegressionPerformance # Solo se usa Performance
from evidently.test_suite import TestSuite
from evidently.tests import TestMeanError, TestRMSE # Tests específicos de regresión
from typing import List, Dict

# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================

# IP del endpoint /predict
API_URL = "http://138.68.15.126:8000/predict" 

# Características RAW (basadas en Energy Efficiency X1-X8)
FEATURES_RAW: List[str] = [
    'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
    'Overall_Height', 'Orientation', 'Glazing_Area',
    'Glazing_Area_Distribution'
]
# Targets RAW (basadas en Energy Efficiency Y1-Y2)
TARGETS: Dict[str, str] = {"heating": "Y1", "cooling": "Y2"}

# Nombres de columnas de predicción que esperamos del API y para Evidently
TARGET_HEATING_API = 'Heating_Load_Prediction'
TARGET_COOLING_API = 'Cooling_Load_Prediction'
PRED_HEATING_COL = 'prediction_heating'
PRED_COOLING_COL = 'prediction_cooling'

N_SAMPLES = 768 # Tamaño total del dataset original

# ==============================================================================
# 2. FUNCIONES AUXILIARES: GENERACIÓN DE DATOS Y CONSULTA
# ==============================================================================

def call_api_predict(data: pd.DataFrame, api_url: str) -> pd.DataFrame:
    """Consulta el endpoint /predict con los datos de entrada RAW."""
    
    headers = {"Content-Type": "application/json"}
    data_to_send = data[FEATURES_RAW].to_dict('records')

    print(f"-> Llamando API en: {api_url}")
    try:
        response = requests.post(
            api_url, 
            json=data_to_send,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        results = response.json()
        
        if not isinstance(results, list) or not results:
             raise ValueError("API no retornó una lista de predicciones válida.")

        predictions_df = pd.DataFrame([
            {
                TARGET_HEATING_API: res.get(TARGET_HEATING_API),
                TARGET_COOLING_API: res.get(TARGET_COOLING_API)
            }
            for res in results
        ])
        predictions_df = predictions_df.rename(columns={
            TARGET_HEATING_API: PRED_HEATING_COL,
            TARGET_COOLING_API: PRED_COOLING_COL
        })

    except requests.exceptions.RequestException as e:
        print(f"Error al consultar la API: {e}. Se usarán predicciones simuladas con error.")
        predictions_df = pd.DataFrame(
            np.nan, 
            index=range(len(data)), 
            columns=[PRED_HEATING_COL, PRED_COOLING_COL]
        )
        
    return pd.concat([data.reset_index(drop=True), predictions_df], axis=1)


def generate_raw_data(num_samples: int, drift_scenario: str = 'reference') -> pd.DataFrame:
    """Genera datos sintéticos RAW (sin preprocesar). Solo necesario para input al API."""
    
    rng = np.random.default_rng(seed=42 if drift_scenario == 'reference' else int(time.time()))
    
    # Parámetros estables (no drift forzado en features aquí)
    rc_mean, rc_std = 0.764, 0.106
    oh_mean, oh_std = 5.25, 1.75 
    
    # Generación de Features RAW
    data = pd.DataFrame({
        'Relative_Compactness': rng.normal(rc_mean, rc_std, num_samples).clip(0.62, 0.98),
        'Surface_Area': rng.uniform(514.5, 808.5, num_samples),
        'Wall_Area': rng.uniform(245.0, 416.5, num_samples),
        'Roof_Area': rng.uniform(110.25, 220.5, num_samples),
        'Overall_Height': rng.normal(oh_mean, oh_std, num_samples).clip(3.5, 7.0),
        'Orientation': rng.choice([2, 3, 4, 5], num_samples),
        'Glazing_Area': rng.choice([0.0, 0.10, 0.25, 0.40], num_samples),
        'Glazing_Area_Distribution': rng.choice([0, 1, 2, 3, 4, 5], num_samples)
    })
    
    # Asegurar tipos de datos
    for col in ['Orientation', 'Glazing_Area_Distribution']:
        data[col] = data[col].astype(int)
    
    return data


def generate_simulated_targets(df: pd.DataFrame, drift_scenario: str = 'reference') -> pd.DataFrame:
    """
    Simula los targets reales (Y_true) y añade predicciones (Y_pred) simuladas (para la referencia).
    Para 'drift', simula una pérdida de performance (Concept Drift) en Y_true.
    """
    N = len(df)
    rng = np.random.default_rng(seed=42 if drift_scenario == 'reference' else int(time.time()))
    
    # Simulación de Y_true (la verdad real del negocio)
    base_heating = df['Relative_Compactness'] * 55 + df['Overall_Height'] * 4 
    base_cooling = df['Surface_Area'] * 0.04 + df['Glazing_Area'] * 60 

    if drift_scenario == 'drift':
        # Concept Drift Simulado: La realidad ha cambiado, aumentando el error
        df[TARGETS['heating']] = base_heating + df['Overall_Height'] * 8 + rng.normal(10, 5, N) 
        df[TARGETS['cooling']] = base_cooling - df['Glazing_Area'] * 20 + rng.normal(0, 5, N)
    else:
        # Referencia: Y_true es estable
        df[TARGETS['heating']] = base_heating + rng.normal(5, 4, N)
        df[TARGETS['cooling']] = base_cooling + rng.normal(0, 3, N)
        
    # Simulación de Y_pred para el set de Referencia (cercana a Y_true)
    if drift_scenario == 'reference':
        df[PRED_HEATING_COL] = df[TARGETS['heating']] + rng.normal(0, 1.5, N)
        df[PRED_COOLING_COL] = df[TARGETS['cooling']] + rng.normal(0, 1.0, N)
        
    return df


# ==============================================================================
# 3. EJECUCIÓN DEL FLUJO DE MONITOREO
# ==============================================================================

print("--- Inicializando Monitoreo de Performance (Concept Drift) ---")

# 3.1. Set de Referencia (Simulando 'energy_efficiency_original.csv' + Predicciones ideales)
ref_data_X = generate_raw_data(N_SAMPLES, 'reference')
ref_data = generate_simulated_targets(ref_data_X, 'reference')
ref_data = ref_data[FEATURES_RAW + list(TARGETS.values()) + [PRED_HEATING_COL, PRED_COOLING_COL]].dropna()
print(f" Set de Referencia (RAW) creado ({len(ref_data)} muestras).")

# 3.2. Set Actual (Features RAW)
current_data_X = generate_raw_data(N_SAMPLES, 'drift')

# 3.3. Obtener Predicciones del API
current_data_with_preds = call_api_predict(current_data_X, API_URL)
api_failed = current_data_with_preds[PRED_HEATING_COL].isna().any()

# 3.4. Simular Pérdida de Performance (Concept Drift) y Targets Reales
current_data = generate_simulated_targets(current_data_with_preds.drop(columns=[PRED_HEATING_COL, PRED_COOLING_COL], errors='ignore'), 'drift')

# Si la API falló, simulamos predicciones con error para demostrar el fallo del modelo.
if api_failed:
    print(" ADVERTENCIA: Predicciones de la API fallaron. Usando predicciones SIMULADAS con alto error para el análisis de Performance.")
    current_data[PRED_HEATING_COL] = current_data[TARGETS['heating']] + np.random.normal(loc=15, scale=5, size=len(current_data)) 
    current_data[PRED_COOLING_COL] = current_data[TARGETS['cooling']] + np.random.normal(loc=-10, scale=4, size=len(current_data))