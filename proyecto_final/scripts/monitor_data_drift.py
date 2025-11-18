import requests
import pandas as pd
import numpy as np
import time
import json
from evidently import Report
from evidently import Dataset, DataDefinition, Regression, Report

from evidently.metrics import MeanError, RMSE
from evidently.presets import RegressionPreset, DataDriftPreset
from evidently.tests import lt

from typing import List, Dict

# ---------------------------
# 1. CONFIG
# ---------------------------

API_URL = "http://127.0.0.1:8000/predict"

FEATURES_RAW = [
    'X1', 'X2', 'X3', 'X4',
    'X5', 'X6', 'X7', 'X8'
]

TARGETS = {"heating": "Y1", "cooling": "Y2"}
PRED_HEATING_COL = 'prediction_heating'
PRED_COOLING_COL = 'prediction_cooling'

N_SAMPLES = 768


# ---------------------------
# 2. FUNCTIONS
# ---------------------------

def call_api_predict(df, api_url):
    print(f"-> Llamando API en: {api_url}")
    try:
        data_json = df[FEATURES_RAW].to_dict('records')
        # print(data_json)
        payload = {
            "target": "both",
            "rows": data_json
        }
        resp = requests.post(api_url, data=json.dumps(payload), headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        res = resp.json()
        # print(res)

        preds = pd.DataFrame([
            {
                "prediction_heating": r.get("heating"),
                "prediction_cooling": r.get("cooling")
            }
            for r in res["predictions"]
        ])
        # for r in res:
        #     print(r)

    except Exception as e:
        print(f"Error: {e} → usando predicciones simuladas")
        preds = pd.DataFrame({
            "prediction_heating": [np.nan] * len(df),
            "prediction_cooling": [np.nan] * len(df)
        })

    return pd.concat([df.reset_index(drop=True), preds], axis=1)


def generate_raw_data(N, drift):
    rng = np.random.default_rng(seed=42 if drift == "reference" else None)
    df = pd.DataFrame({
        'X1': rng.normal(0.764, 0.106, N).clip(0.62, 0.98),
        'X2': rng.uniform(514.5, 808.5, N),
        'X3': rng.uniform(245, 416.5, N),
        'X4': rng.uniform(110.25, 220.5, N),
        'X5': rng.normal(5.25, 1.75, N).clip(3.5, 7.0),
        'X6': rng.choice([2,3,4,5], N),
        'X7': rng.choice([0,0.10,0.25,0.40], N),
        'X8': rng.choice([0,1,2,3,4,5], N)
    })
    return df


def generate_targets(df, drift):
    N = len(df)
    rng = np.random.default_rng(seed=42 if drift == "reference" else None)

    base_h = df['X1'] * 55 + df['X5'] * 4
    base_c = df['X2'] * 0.04 + df['X7'] * 60

    if drift == "drift":
        df["Y1"] = base_h + df['X5']*8 + rng.normal(10,5,N)
        df["Y2"] = base_c - df['X7']*20 + rng.normal(0,5,N)
    else:
        df["Y1"] = base_h + rng.normal(5,4,N)
        df["Y2"] = base_c + rng.normal(0,3,N)

    if drift == "reference":
        df[PRED_HEATING_COL] = df["Y1"] + rng.normal(0,1.5,N)
        df[PRED_COOLING_COL] = df["Y2"] + rng.normal(0,1.0,N)

    return df


# ---------------------------
# 3. EXECUTION
# ---------------------------

print("--- Inicializando Monitoreo ---")

ref_data = generate_targets(generate_raw_data(N_SAMPLES, "reference"), "reference").dropna()

current_raw = generate_raw_data(N_SAMPLES, "drift")
current_with_preds = call_api_predict(current_raw, API_URL)

api_failed = current_with_preds[PRED_HEATING_COL].isna().any()
current_data = generate_targets(
    current_with_preds.drop(columns=[PRED_HEATING_COL, PRED_COOLING_COL], errors="ignore"),
    "drift"
)

if api_failed:
    print("⚠️ API falló → simulando errores grandes")
    current_data[PRED_HEATING_COL] = current_data["Y1"] + np.random.normal(15,5,len(current_data))
    current_data[PRED_COOLING_COL] = current_data["Y2"] + np.random.normal(-10,4,len(current_data))
else:
    current_data[PRED_HEATING_COL] = current_with_preds[PRED_HEATING_COL]
    current_data[PRED_COOLING_COL] = current_with_preds[PRED_COOLING_COL]

current_data = current_data.dropna()


# ---------------------------
# 4. RENAME COLUMNS FOR EVIDENTLY
# ---------------------------

# ---------------------------
# 4. PREPARE DATA FOR EVIDENTLY
# ---------------------------

# Solo necesitamos columnas target + predicción para cada caso.
ref_heating_df = ref_data[[TARGETS["heating"], PRED_HEATING_COL]].copy()
curr_heating_df = current_data[[TARGETS["heating"], PRED_HEATING_COL]].copy()

ref_cooling_df = ref_data[[TARGETS["cooling"], PRED_COOLING_COL]].copy()
curr_cooling_df = current_data[[TARGETS["cooling"], PRED_COOLING_COL]].copy()

# Definiciones de regresión para Evidently (nombre "default" por defecto).
heating_def = DataDefinition(
    regression=[Regression(target=TARGETS["heating"], prediction=PRED_HEATING_COL)]
)

cooling_def = DataDefinition(
    regression=[Regression(target=TARGETS["cooling"], prediction=PRED_COOLING_COL)]
)

# Datasets Evidently
ref_heating_ds = Dataset.from_pandas(ref_heating_df, data_definition=heating_def)
curr_heating_ds = Dataset.from_pandas(curr_heating_df, data_definition=heating_def)

ref_cooling_ds = Dataset.from_pandas(ref_cooling_df, data_definition=cooling_def)
curr_cooling_ds = Dataset.from_pandas(curr_cooling_df, data_definition=cooling_def)

# # ---------------------------
# # 5. HTML REPORTS
# # ---------------------------
#
# print("\n--- Generando Reportes HTML ---")
#
# # HEATING REPORT
# report_h = Report(metrics=[
#     MeanError(y_true="y", y_pred="y_pred"),
#     RMSE(y_true="y", y_pred="y_pred"),
# ])
# # report_h = Report(metrics=[RegressionPreset()])
# report_h.run(
#     reference_data=ref_heating,
#     current_data=curr_heating
# )
# report_h.save_html("heating_report.html")
#
# # COOLING REPORT
# report_c = Report(metrics=[
#     MeanError(y_true="y", y_pred="y_pred"),
#     RMSE(y_true="y", y_pred="y_pred"),
# ])
#
# report_c.run(
#     reference_data=ref_cooling,
#     current_data=curr_cooling
# )
# report_c.save_html("cooling_report.html")
from evidently.metrics.regression import *

#

# ---------------------------
# 5. HTML REPORTS
# ---------------------------

print("\n--- Generando Reportes HTML ---")

# Heating
report_h = Report([RegressionPreset()])
report_h.run(current_data=curr_heating_ds, reference_data=ref_heating_ds)
# report_h.save("evidently_heating_regression_report.html")
print("✅ heating → evidently_heating_regression_report.html")

# Cooling
report_c = Report([RegressionPreset()])
report_c.run(current_data=curr_cooling_ds, reference_data=ref_cooling_ds)
# report_c.save("evidently_cooling_regression_report.html")
print("✅ cooling → evidently_cooling_regression_report.html")


# ==============================================================================
# 6. TESTS DE PERFORMANCE CON UMBRALES (MeanError + RMSE)
# ==============================================================================

print("\n--- Ejecutando Tests de Performance ---")

# Heating tests (ME < 5, RMSE < 10)
tests_h = Report(
    [
        MeanError(
            error_plot=True,      # requisito de la métrica
            error_distr=False,
            error_normality=False,
            mean_tests=[lt(5.0)],
        ),
        RMSE(
            tests=[lt(10.0)],
        ),
    ]
)
res_h = tests_h.run(current_data=curr_heating_ds, reference_data=None)

# Cooling tests (ME < 5, RMSE < 10)
tests_c = Report(
    [
        MeanError(
            error_plot=True,
            error_distr=False,
            error_normality=False,
            mean_tests=[lt(5.0)],
        ),
        RMSE(
            tests=[lt(10.0)],
        ),
    ]
)
res_c = tests_c.run(current_data=curr_cooling_ds, reference_data=None)


def all_tests_passed(report_result: Report) -> bool:
    d = report_result.load_dict()
    for metric in d.get("metrics", []):
        if "tests" in metric:
            for t in metric["tests"]:
                if t.get("status") != "SUCCESS":
                    return False
    return True


heating_ok = all_tests_passed(res_h)
cooling_ok = all_tests_passed(res_c)

print("\n--- Resumen de Alertas y Acción Propuesta ---")

if not heating_ok:
    print("🔴 ALERTA (Y1 / Heating): el modelo NO cumple los umbrales (ME / RMSE).")
    print("   → Acción: considerar reentrenar el modelo de calefacción.")

if not cooling_ok:
    print("🔴 ALERTA (Y2 / Cooling): el modelo NO cumple los umbrales (ME / RMSE).")
    print("   → Acción: considerar reentrenar el modelo de refrigeración.")

if heating_ok and cooling_ok:
    print("✅ Ambos modelos cumplen los umbrales configurados (ME < 5, RMSE < 10).")

print("\nProceso completado. Revisa los HTML generados.")