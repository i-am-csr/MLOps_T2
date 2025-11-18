import pandas as pd
import numpy as np
import requests
import json
import logging
from typing import Dict, Any, List, Tuple

# ⚠️ Asegúrate de tener estas librerías instaladas:
# pip install pandas numpy requests scikit-learn alibi-detect

from sklearn.metrics import mean_squared_error, mean_absolute_error
from alibi_detect.cd import RegressorDrift
from alibi_detect.utils.data import create_batches
from alibi_detect.utils.thresholds import fit_threshold
from alibi_detect.utils.visualize import plot_performance_drift

# Configuración de Logging
logging.basicConfig(level=logging.INFO)v
logger = logging.getLogger(__name__)

# --- Constantes y Configuración ---

# ⚠️ IP/Puerto del endpoint de tu API
API_URL = "http://127.0.0.1:8000/predict" # <-- IP/ENDPOINT A CONSULTAR

# ⚠️ RUTA AL ARCHIVO BASELINE
BASELINE_FILE_PATH = "/Users/alanrods/Documents/GitHub/MLOps_T2/data/raw/energy_efficiency_original.csv"

# Nombres de las columnas
EXPLICIT_COLUMNS = [
    'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
    'Overall_Height', 'Orientation', 'Glazing_Area',
    'Glazing_Area_Distribution', 'Heating_Load', 'Cooling_Load'
]
FEATURE_COLS = EXPLICIT_COLUMNS[:8]

# Parámetros de Alibi-Detect
P_VAL_THRESHOLD = 0.01  # Umbral de significancia para el test de drift
BATCH_SIZE = 64
WINDOW_SIZE = 1000 # Parámetro necesario para inicializar el detector

# --- 1. Carga y Preparación del Baseline (Reference Data) ---

def load_and_prepare_baseline() -> pd.DataFrame:
    """
    Carga el CSV original, renombra las columnas, y simula predicciones 
    para crear el set de referencia (baseline) de ALTA CALIDAD.
    """
    logger.info(f"--- 1. Cargando Baseline desde: {BASELINE_FILE_PATH} ---")
    
    try:
        # Carga el archivo original. Asume que tiene 10 columnas (8 features + Y1 + Y2)
        reference_data = pd.read_csv(BASELINE_FILE_PATH, header=None, skiprows=1)
        reference_data.columns = EXPLICIT_COLUMNS
        
    except FileNotFoundError:
        logger.error(f"⚠️ ERROR: No se encontró el archivo en la ruta: {BASELINE_FILE_PATH}.")
        raise FileNotFoundError(f"Verifica la ruta: {BASELINE_FILE_PATH}")
    
    # Simulación de Predicciones del Modelo sobre el Baseline
    # El modelo debe tener un buen rendimiento en el set de referencia (error bajo).
    logger.info("   - Simulando predicciones de ALTA CALIDAD para el Baseline (error bajo).")
    
    # Predicción de Heating_Load (Y1)
    noise_heating = np.random.normal(0, 1.0, len(reference_data))
    reference_data['Pred_Heating_Load'] = (reference_data['Heating_Load'] + noise_heating).clip(lower=0)
    
    # Predicción de Cooling_Load (Y2)
    noise_cooling = np.random.normal(0, 0.8, len(reference_data))
    reference_data['Pred_Cooling_Load'] = (reference_data['Cooling_Load'] + noise_cooling).clip(lower=0)

    # Calcular métricas base
    rmse_h = mean_squared_error(reference_data['Heating_Load'], reference_data['Pred_Heating_Load'], squared=False)
    rmse_c = mean_squared_error(reference_data['Cooling_Load'], reference_data['Pred_Cooling_Load'], squared=False)
    logger.info(f"   - RMSE Base (Heating): {rmse_h:.3f} | RMSE Base (Cooling): {rmse_c:.3f}")

    return reference_data

# --- 2. Generación de Datos Actuales y Predicciones Degradadas ---

def generate_current_data_and_predictions(num_samples: int) -> pd.DataFrame:
    """
    Genera datos nuevos (simulando producción con drift) y simula predicciones 
    degradadas (performance loss).
    """
    logger.info(f"\n--- 2. Generando Set Actual ({num_samples} muestras) y Predicciones Degradadas ---")
    np.random.seed(44) 
    
    data = pd.DataFrame()
    
    # Generar features (con drift en Overall_Height)
    data['Relative_Compactness'] = np.random.uniform(0.6, 0.9, num_samples)
    data['Surface_Area'] = np.random.uniform(500, 850, num_samples)
    data['Wall_Area'] = np.random.uniform(245, 430, num_samples)
    data['Roof_Area'] = np.random.uniform(100, 250, num_samples)
    # Drift: Aumentamos la frecuencia de construcciones altas (7.0)
    data['Overall_Height'] = np.random.choice([3.5, 7.0], num_samples, p=[0.2, 0.8])
    data['Orientation'] = np.random.choice([2, 3, 4, 5], num_samples)
    data['Glazing_Area'] = np.random.choice([0.0, 0.1, 0.25, 0.4], num_samples, p=[0.1, 0.3, 0.3, 0.3])
    data['Glazing_Area_Distribution'] = np.random.choice([0, 1, 2, 3, 4, 5], num_samples)

    # Cálculo del Ground Truth (Y1 y Y2) basado en las nuevas features
    data['Heating_Load'] = (
        10 + 20 * data['Overall_Height'] + 0.1 * data['Wall_Area'] +
        50 * data['Relative_Compactness'] + np.random.normal(0, 5, num_samples)
    ).clip(lower=0)
    
    data['Cooling_Load'] = (
        5 + 15 * data['Overall_Height'] + 0.05 * data['Surface_Area'] +
        25 * data['Glazing_Area'] + np.random.normal(0, 4, num_samples)
    ).clip(lower=0)

    # --- SIMULACIÓN DE CONSULTA AL API Y PÉRDIDA DE RENDIMIENTO ---
    # En un entorno real, aquí harías la llamada a requests.post(API_URL, json=payload)
    
    # Predicción degradada (Heating_Load)
    # Se introduce un sesgo (-5) y un error mayor (ruido de 5.0)
    noise_drift_h = np.random.normal(0, 5.0, num_samples)
    data['Pred_Heating_Load'] = (data['Heating_Load'] - 5 + noise_drift_h).clip(lower=0)
    
    # Predicción degradada (Cooling_Load)
    # Se introduce un sesgo (-3) y un error mayor (ruido de 4.0)
    noise_drift_c = np.random.normal(0, 4.0, num_samples)
    data['Pred_Cooling_Load'] = (data['Cooling_Load'] - 3 + noise_drift_c).clip(lower=0)

    logger.info("   - Se simuló el drift y se introdujo un sesgo/error mayor en las predicciones (Performance Loss).")

    return data[EXPLICIT_COLUMNS + ['Pred_Heating_Load', 'Pred_Cooling_Load']]

# --- 3. Detección de Regressor Drift con Alibi-Detect ---

def run_alibi_drift_detection(ref_df: pd.DataFrame, curr_df: pd.DataFrame, target: str) -> None:
    """
    Entrena e inicializa el detector RegressorDrift y verifica el drift.
    """
    logger.info(f"\n--- 4. Monitoreando Pérdida de Rendimiento para {target} ---")
    
    # Definir columnas para el detector
    target_col = target
    prediction_col = f'Pred_{target}'
    
    # Preparar datos para Alibi-Detect
    # Alibi-Detect requiere X (features), Y (ground truth) y Y_pred (predictions)
    X_ref = ref_df[FEATURE_COLS].values
    Y_ref = ref_df[target_col].values
    Y_pred_ref = ref_df[prediction_col].values
    
    X_curr = curr_df[FEATURE_COLS].values
    Y_curr = curr_df[target_col].values
    Y_pred_curr = curr_df[prediction_col].values
    
    # Inicializar RegressorDrift
    # Utilizamos el test Kolmogorov-Smirnov (KS) para comparar la distribución del error.
    # El modelo es Dummy (LinearRegression) para calcular el error.
    cd = RegressorDrift(
        x_ref=X_ref,
        y_ref=Y_ref,
        model=None,  # No se requiere un modelo para calcular el error si se pasa y_pred_ref
        preds_ref=Y_pred_ref,
        p_val=P_VAL_THRESHOLD,
        preprocess_at_init=False,
        set_threshold=0.0, # El umbral se establecerá con fit_threshold
        preprocess_fns=None,
        feature_names=FEATURE_COLS,
        n_features=len(FEATURE_COLS)
    )

    # 4.1 Entrenar el detector (calcular el umbral de deriva)
    # Esto compara el error de referencia con el error de los datos actuales
    # Alibi-Detect calcula el error y aplica el test KS sobre la distribución del error.
    
    # La función fit_threshold ya no se usa directamente en la API actual. 
    # En su lugar, el threshold se establece internamente o se usa predict para detectar.
    
    # 4.2 Detección de Drift
    detection_result = cd.predict(
        X_curr,
        y_curr=Y_curr,
        preds_curr=Y_pred_curr,
        drift_type='performance', # Especifica la detección de drift de rendimiento
        return_p_val=True,
        return_distance=True
    )

    # 4.3 Análisis y Alerta
    is_drift = detection_result['data']['is_drift'][0]
    p_value = detection_result['data']['p_val'][0]
    distance = detection_result['data']['distance'][0]
    
    logger.info(f"   - Resultado de la Detección:")
    logger.info(f"     - Drift Detectado: {is_drift} (Umbral p-val: {P_VAL_THRESHOLD})")
    logger.info(f"     - P-value (Error Distribution): {p_value:.5f}")
    logger.info(f"     - Distancia (Drift Magnitude): {distance:.5f}")
    
    # 4.4 Comparación de Métricas (Para cuantificar la pérdida)
    rmse_ref = mean_squared_error(Y_ref, Y_pred_ref, squared=False)
    rmse_curr = mean_squared_error(Y_curr, Y_pred_curr, squared=False)
    mae_ref = mean_absolute_error(Y_ref, Y_pred_ref)
    mae_curr = mean_absolute_error(Y_curr, Y_pred_curr)

    logger.info(f"   - Comparación de Métricas:")
    logger.info(f"     - RMSE Baseline: {rmse_ref:.3f} | RMSE Actual: {rmse_curr:.3f}")
    logger.info(f"     - MAE Baseline: {mae_ref:.3f} | MAE Actual: {mae_curr:.3f}")
    
    if is_drift:
        logger.warning("\n--- 🚨 ALERTA: PERFORMANCE DRIFT DETECTADO 🚨 ---")
        logger.warning(f"La distribución del error en {target} ha cambiado significativamente. Acción: Revisión y Retrain.")
    elif rmse_curr > 1.2 * rmse_ref:
         logger.warning("\n--- ⚠️ ADVERTENCIA: DEGRADACIÓN NO ESTADÍSTICA ⚠️ ---")
         logger.warning(f"El RMSE aumentó más del 20% ({rmse_curr/rmse_ref:.2f}x) aunque el test estadístico no falló. ¡Monitorear de cerca!")
    else:
        logger.info("\n--- ✅ RENDIMIENTO ESTABLE ---")


if __name__ == '__main__':
    try:
        # 1. Cargar y preparar el set de Referencia (Baseline)
        reference_df = load_and_prepare_baseline()

        # 2. Generar datos Actuales y obtener/simular predicciones degradadas
        current_df = generate_current_data_and_predictions(num_samples=768)

        # 3. Monitorear Heating_Load (Y1)
        run_alibi_drift_detection(reference_df, current_df, TARGET_HEATING)
        
        # 4. Monitorear Cooling_Load (Y2)
        run_alibi_drift_detection(reference_df, current_df, 'Cooling_Load')

    except Exception as e:
        logger.error(f"Ocurrió un error en el script principal: {e}")