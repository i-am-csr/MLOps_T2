import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import requests
import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

# Performance monitoring thresholds
PERFORMANCE_THRESHOLDS = {
    'rmse_degradation_threshold': 0.20,  # 20% degradation threshold
    'r2_degradation_threshold': 0.15,    # 15% degradation threshold
    'mae_degradation_threshold': 0.25    # 25% degradation threshold
}

# Data drift thresholds
DRIFT_THRESHOLDS = {
    'drift_share_threshold': 0.30,       # 30% of features with drift is critical
    'individual_feature_threshold': 0.05  # p-value threshold for individual features
}


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


def calculate_performance_metrics(y_true, y_pred):
    """Calculate regression performance metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def analyze_performance_drift(ref_metrics, curr_metrics, target_name):
    """Analyze performance degradation between reference and current data"""

    degradation = {}
    alerts = []

    for metric in ['rmse', 'mae', 'r2']:
        if metric in ['rmse', 'mae']:  # Higher is worse
            degradation[f'{metric}_change'] = ((curr_metrics[metric] - ref_metrics[metric]) / ref_metrics[metric]) * 100
            threshold_key = f'{metric}_degradation_threshold'

            if degradation[f'{metric}_change'] > PERFORMANCE_THRESHOLDS[threshold_key] * 100:
                alerts.append(f"🚨 {metric.upper()} degradation of {degradation[f'{metric}_change']:.2f}% exceeds threshold ({PERFORMANCE_THRESHOLDS[threshold_key]*100:.1f}%) for {target_name}")

        else:  # r2: Higher is better
            degradation[f'{metric}_change'] = ((ref_metrics[metric] - curr_metrics[metric]) / ref_metrics[metric]) * 100
            threshold_key = f'{metric}_degradation_threshold'

            if degradation[f'{metric}_change'] > PERFORMANCE_THRESHOLDS[threshold_key] * 100:
                alerts.append(f"🚨 {metric.upper()} degradation of {degradation[f'{metric}_change']:.2f}% exceeds threshold ({PERFORMANCE_THRESHOLDS[threshold_key]*100:.1f}%) for {target_name}")

    return degradation, alerts


def detect_data_drift(reference_data, current_data):
    """Detect data drift using statistical tests"""

    try:
        from scipy.stats import ks_2samp

        drift_results = {
            'feature_drift': {},
            'overall_drift': False,
            'drift_count': 0
        }

        # Perform Kolmogorov-Smirnov test for each feature
        for feature in FEATURES_RAW:
            try:
                ref_values = reference_data[feature].values
                curr_values = current_data[feature].values

                # KS test for distribution comparison
                ks_stat, p_value = ks_2samp(ref_values, curr_values)

                # Consider drift if p-value < threshold
                is_drift = p_value < DRIFT_THRESHOLDS['individual_feature_threshold']

                drift_results['feature_drift'][feature] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': is_drift
                }

                if is_drift:
                    drift_results['drift_count'] += 1

            except Exception as e:
                print(f"Error testing feature {feature}: {str(e)}")

        # Overall drift if more than threshold percentage of features drift
        drift_share = drift_results['drift_count'] / len(FEATURES_RAW)
        drift_results['overall_drift'] = drift_share > DRIFT_THRESHOLDS['drift_share_threshold']
        drift_results['drift_share'] = drift_share

        return drift_results, True

    except ImportError:
        print("scipy not available, using simplified drift detection")
        # Simplified drift detection using mean and std comparison
        drift_results = {'feature_drift': {}, 'overall_drift': False, 'drift_count': 0}

        for feature in FEATURES_RAW:
            ref_mean = reference_data[feature].mean()
            ref_std = reference_data[feature].std()
            curr_mean = current_data[feature].mean()
            curr_std = current_data[feature].std()

            # Simple drift detection: significant change in mean or std
            mean_change = abs(curr_mean - ref_mean) / ref_std if ref_std > 0 else 0
            std_change = abs(curr_std - ref_std) / ref_std if ref_std > 0 else 0

            is_drift = (mean_change > 2.0) or (std_change > 1.0)  # 2 sigma threshold

            drift_results['feature_drift'][feature] = {
                'mean_change': mean_change,
                'std_change': std_change,
                'drift_detected': is_drift
            }

            if is_drift:
                drift_results['drift_count'] += 1

        drift_share = drift_results['drift_count'] / len(FEATURES_RAW)
        drift_results['overall_drift'] = drift_share > DRIFT_THRESHOLDS['drift_share_threshold']
        drift_results['drift_share'] = drift_share

        return drift_results, True

    except Exception as e:
        print(f"Error en drift detection: {str(e)}")
        return {}, False


def generate_drift_alerts(drift_results, drift_success):
    """Generate alerts based on drift detection results"""

    alerts = []

    if not drift_success:
        alerts.append("⚠️ Error en detección de drift - revisa configuración")
        return alerts

    # Check overall drift
    if drift_results.get('overall_drift', False):
        drift_share = drift_results.get('drift_share', 0)
        alerts.append(f"🚨 Data drift detected: {drift_share:.2%} of features show drift (threshold: {DRIFT_THRESHOLDS['drift_share_threshold']:.1%})")

    # Check individual features
    feature_drift = drift_results.get('feature_drift', {})
    drifted_features = []

    for feature, result in feature_drift.items():
        if result.get('drift_detected', False):
            drifted_features.append(feature)

            # Add detailed information based on detection method
            if 'p_value' in result:
                alerts.append(f"⚠️ Feature '{feature}' shows significant drift (KS test p-value: {result['p_value']:.4f})")
            elif 'mean_change' in result:
                alerts.append(f"⚠️ Feature '{feature}' shows significant drift (mean change: {result['mean_change']:.2f}σ, std change: {result['std_change']:.2f}σ)")

    if drifted_features:
        alerts.append(f"📊 Total features with drift: {len(drifted_features)} out of {len(FEATURES_RAW)}")
    else:
        alerts.append("✅ No significant feature drift detected")

    return alerts


def create_visualizations(ref_data, curr_data, ref_metrics_h, curr_metrics_h, ref_metrics_c, curr_metrics_c):
    """Create visualizations for monitoring report"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Performance comparison - Heating
    metrics_comparison_h = pd.DataFrame({
        'Reference': [ref_metrics_h['rmse'], ref_metrics_h['mae'], ref_metrics_h['r2']],
        'Current': [curr_metrics_h['rmse'], curr_metrics_h['mae'], curr_metrics_h['r2']]
    }, index=['RMSE', 'MAE', 'R²'])

    metrics_comparison_h.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Performance Metrics - Heating (Y1)')
    axes[0, 0].set_ylabel('Metric Value')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Performance comparison - Cooling
    metrics_comparison_c = pd.DataFrame({
        'Reference': [ref_metrics_c['rmse'], ref_metrics_c['mae'], ref_metrics_c['r2']],
        'Current': [curr_metrics_c['rmse'], curr_metrics_c['mae'], curr_metrics_c['r2']]
    }, index=['RMSE', 'MAE', 'R²'])

    metrics_comparison_c.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Performance Metrics - Cooling (Y2)')
    axes[0, 1].set_ylabel('Metric Value')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Feature distribution comparison
    feature_to_plot = 'X1'
    axes[0, 2].hist(ref_data[feature_to_plot], alpha=0.5, label='Reference', bins=30)
    axes[0, 2].hist(curr_data[feature_to_plot], alpha=0.5, label='Current', bins=30)
    axes[0, 2].set_title(f'Distribution Comparison - {feature_to_plot}')
    axes[0, 2].set_xlabel(feature_to_plot)
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()

    # Prediction vs Actual scatter plots
    axes[1, 0].scatter(ref_data[TARGETS['heating']], ref_data[PRED_HEATING_COL], alpha=0.5, label='Reference')
    axes[1, 0].scatter(curr_data[TARGETS['heating']], curr_data[PRED_HEATING_COL], alpha=0.5, label='Current')
    axes[1, 0].plot([ref_data[TARGETS['heating']].min(), ref_data[TARGETS['heating']].max()],
                   [ref_data[TARGETS['heating']].min(), ref_data[TARGETS['heating']].max()], 'r--')
    axes[1, 0].set_xlabel('Actual Heating Load (Y1)')
    axes[1, 0].set_ylabel('Predicted Heating Load')
    axes[1, 0].set_title('Heating: Predicted vs Actual')
    axes[1, 0].legend()

    axes[1, 1].scatter(ref_data[TARGETS['cooling']], ref_data[PRED_COOLING_COL], alpha=0.5, label='Reference')
    axes[1, 1].scatter(curr_data[TARGETS['cooling']], curr_data[PRED_COOLING_COL], alpha=0.5, label='Current')
    axes[1, 1].plot([ref_data[TARGETS['cooling']].min(), ref_data[TARGETS['cooling']].max()],
                   [ref_data[TARGETS['cooling']].min(), ref_data[TARGETS['cooling']].max()], 'r--')
    axes[1, 1].set_xlabel('Actual Cooling Load (Y2)')
    axes[1, 1].set_ylabel('Predicted Cooling Load')
    axes[1, 1].set_title('Cooling: Predicted vs Actual')
    axes[1, 1].legend()

    # Residuals plot
    residuals_h_ref = ref_data[TARGETS['heating']] - ref_data[PRED_HEATING_COL]
    residuals_h_curr = curr_data[TARGETS['heating']] - curr_data[PRED_HEATING_COL]

    axes[1, 2].scatter(ref_data[PRED_HEATING_COL], residuals_h_ref, alpha=0.5, label='Reference')
    axes[1, 2].scatter(curr_data[PRED_HEATING_COL], residuals_h_curr, alpha=0.5, label='Current')
    axes[1, 2].axhline(y=0, color='r', linestyle='--')
    axes[1, 2].set_xlabel('Predicted Heating Load')
    axes[1, 2].set_ylabel('Residuals')
    axes[1, 2].set_title('Residuals Plot - Heating')
    axes[1, 2].legend()

    plt.tight_layout()
    return fig


def generate_recommendations(all_alerts, performance_degradation_h, performance_degradation_c):
    """Generate actionable recommendations based on detected issues"""

    recommendations = []

    if len(all_alerts) == 0:
        recommendations.append("✅ No critical issues detected. Continue monitoring.")
        return recommendations

    # Performance-based recommendations
    if any("RMSE degradation" in alert for alert in all_alerts):
        recommendations.append("📊 RETRAIN REQUIRED: Significant RMSE degradation detected")
        recommendations.append("   • Consider retraining the model with recent data")
        recommendations.append("   • Investigate feature engineering improvements")

    if any("R² degradation" in alert for alert in all_alerts):
        recommendations.append("📊 MODEL PERFORMANCE: R² degradation indicates poor model fit")
        recommendations.append("   • Review feature selection and importance")
        recommendations.append("   • Consider ensemble methods or model architecture changes")

    # Drift-based recommendations
    if any("Data drift detected" in alert for alert in all_alerts):
        recommendations.append("🔄 FEATURE PIPELINE REVIEW: Significant data drift detected")
        recommendations.append("   • Review data preprocessing and feature engineering pipeline")
        recommendations.append("   • Check data collection process for changes")
        recommendations.append("   • Implement feature drift monitoring in production")

    if any("Feature" in alert and "drift" in alert for alert in all_alerts):
        recommendations.append("🔍 FEATURE INVESTIGATION: Individual features showing drift")
        recommendations.append("   • Analyze specific features mentioned in alerts")
        recommendations.append("   • Consider feature transformation or normalization")
        recommendations.append("   • Validate data quality and collection methods")

    # General recommendations
    recommendations.append("\n🎯 IMMEDIATE ACTIONS:")
    recommendations.append("   1. Stop using current model predictions for critical decisions")
    recommendations.append("   2. Investigate root cause of performance degradation")
    recommendations.append("   3. Prepare emergency fallback procedures")
    recommendations.append("   4. Schedule urgent model retraining session")

    return recommendations


# ---------------------------
# 3. EXECUTION - Data Drift Detection & Performance Monitoring
# ---------------------------

def main():
    print("🚀 === INICIALIZANDO MONITOREO DE DRIFT Y PERFORMANCE ===")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"📅 Timestamp: {timestamp}\n")

    # Generate reference (baseline) data
    print("📊 Generando datos de referencia (baseline)...")
    ref_data_raw = pd.read_csv("data/raw/energy_efficiency_modified.csv")
    print(f"   📥 Cargados {len(ref_data_raw)} datos de referencia desde CSV")

    # Clean and convert data types
    # Remove extra columns and ensure numeric types
    expected_cols = FEATURES_RAW + [TARGETS["heating"], TARGETS["cooling"]]
    ref_data_raw = ref_data_raw[expected_cols]

    # Convert all columns to numeric, handling any whitespace or formatting issues
    for col in expected_cols:
        ref_data_raw[col] = pd.to_numeric(ref_data_raw[col], errors='coerce')

    # Remove rows with any NaN values
    ref_data_raw = ref_data_raw.dropna()
    print(f"   🔧 Datos limpiados: {len(ref_data_raw)} muestras válidas")

    # Get predictions for reference data using API
    print("   🔮 Obteniendo predicciones de referencia desde API...")
    ref_data = call_api_predict(ref_data_raw, API_URL)

    # Handle case where API fails for reference data
    if ref_data[PRED_HEATING_COL].isna().any():
        print("   ⚠️ API falló para datos de referencia → usando predicciones simuladas")
        # Use actual target values as baseline "predictions" with small noise
        ref_data[PRED_HEATING_COL] = ref_data[TARGETS["heating"]].astype(float) + np.random.normal(0, 1.5, len(ref_data))
        ref_data[PRED_COOLING_COL] = ref_data[TARGETS["cooling"]].astype(float) + np.random.normal(0, 1.0, len(ref_data))
    else:
        print("   ✅ Predicciones de referencia obtenidas exitosamente")

    ref_data = ref_data.dropna()
    print(f"   ✅ Datos de referencia procesados: {len(ref_data)} muestras")


    # Generate current (monitoring) data with drift
    print("📊 Generando datos actuales con drift simulado...")
    current_raw = generate_raw_data(N_SAMPLES, "drift")
    current_with_preds = call_api_predict(current_raw, API_URL)

    # Handle API failure
    api_failed = current_with_preds[PRED_HEATING_COL].isna().any()
    current_data = generate_targets(
        current_with_preds.drop(columns=[PRED_HEATING_COL, PRED_COOLING_COL], errors="ignore"),
        "drift"
    )

    if api_failed:
        print("⚠️ API falló → simulando predicciones degradadas")
        current_data[PRED_HEATING_COL] = current_data["Y1"] + np.random.normal(15,5,len(current_data))
        current_data[PRED_COOLING_COL] = current_data["Y2"] + np.random.normal(-10,4,len(current_data))
    else:
        print("✅ API funcionó correctamente")
        current_data[PRED_HEATING_COL] = current_with_preds[PRED_HEATING_COL]
        current_data[PRED_COOLING_COL] = current_with_preds[PRED_COOLING_COL]

    current_data = current_data.dropna()
    print(f"   ✅ Datos actuales: {len(current_data)} muestras\n")

    # =========================
    # PERFORMANCE ANALYSIS
    # =========================
    print("📈 === ANÁLISIS DE PERFORMANCE ===")

    # Calculate baseline performance metrics
    ref_metrics_heating = calculate_performance_metrics(
        ref_data[TARGETS["heating"]],
        ref_data[PRED_HEATING_COL]
    )

    ref_metrics_cooling = calculate_performance_metrics(
        ref_data[TARGETS["cooling"]],
        ref_data[PRED_COOLING_COL]
    )

    # Calculate current performance metrics
    curr_metrics_heating = calculate_performance_metrics(
        current_data[TARGETS["heating"]],
        current_data[PRED_HEATING_COL]
    )

    curr_metrics_cooling = calculate_performance_metrics(
        current_data[TARGETS["cooling"]],
        current_data[PRED_COOLING_COL]
    )

    # Analyze performance drift
    perf_degradation_h, perf_alerts_h = analyze_performance_drift(
        ref_metrics_heating, curr_metrics_heating, "Heating"
    )

    perf_degradation_c, perf_alerts_c = analyze_performance_drift(
        ref_metrics_cooling, curr_metrics_cooling, "Cooling"
    )

    # Display performance results
    print("\n🔥 HEATING MODEL (Y1):")
    print(f"   Reference  → RMSE: {ref_metrics_heating['rmse']:.4f} | MAE: {ref_metrics_heating['mae']:.4f} | R²: {ref_metrics_heating['r2']:.4f}")
    print(f"   Current    → RMSE: {curr_metrics_heating['rmse']:.4f} | MAE: {curr_metrics_heating['mae']:.4f} | R²: {curr_metrics_heating['r2']:.4f}")
    print(f"   Change     → RMSE: {perf_degradation_h.get('rmse_change', 0):.2f}% | MAE: {perf_degradation_h.get('mae_change', 0):.2f}% | R²: {perf_degradation_h.get('r2_change', 0):.2f}%")

    print("\n❄️ COOLING MODEL (Y2):")
    print(f"   Reference  → RMSE: {ref_metrics_cooling['rmse']:.4f} | MAE: {ref_metrics_cooling['mae']:.4f} | R²: {ref_metrics_cooling['r2']:.4f}")
    print(f"   Current    → RMSE: {curr_metrics_cooling['rmse']:.4f} | MAE: {curr_metrics_cooling['mae']:.4f} | R²: {curr_metrics_cooling['r2']:.4f}")
    print(f"   Change     → RMSE: {perf_degradation_c.get('rmse_change', 0):.2f}% | MAE: {perf_degradation_c.get('mae_change', 0):.2f}% | R²: {perf_degradation_c.get('r2_change', 0):.2f}%")

    # =========================
    # DATA DRIFT ANALYSIS
    # =========================
    print("\n🌊 === ANÁLISIS DE DATA DRIFT ===")

    # Detect data drift
    print("🔍 Detectando drift en features...")
    try:
        drift_results, drift_success = detect_data_drift(ref_data, current_data)
        drift_alerts = generate_drift_alerts(drift_results, drift_success)
        print("   ✅ Análisis de drift completado")
    except Exception as e:
        print(f"   ⚠️ Error en análisis de drift: {str(e)}")
        drift_alerts = [f"⚠️ Error en detección de drift: {str(e)}"]

    # =========================
    # GENERATE ALERTS AND REPORTS
    # =========================
    print("\n🚨 === ALERTAS Y RECOMENDACIONES ===")

    # Combine all alerts
    all_alerts = perf_alerts_h + perf_alerts_c + drift_alerts

    if len(all_alerts) == 0:
        print("✅ No se detectaron problemas críticos")
    else:
        print("⚠️ ALERTAS DETECTADAS:")
        for alert in all_alerts:
            print(f"   {alert}")

    # Generate recommendations
    recommendations = generate_recommendations(all_alerts, perf_degradation_h, perf_degradation_c)

    print(f"\n💡 === RECOMENDACIONES ===")
    for rec in recommendations:
        print(rec)

    # =========================
    # GENERATE VISUALIZATIONS
    # =========================
    print(f"\n📊 === GENERANDO VISUALIZACIONES ===")

    # Generate timestamp for file naming
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        fig = create_visualizations(
            ref_data, current_data,
            ref_metrics_heating, curr_metrics_heating,
            ref_metrics_cooling, curr_metrics_cooling
        )

        # Save the plot
        plot_filename = f"drift_monitoring_report_{timestamp_file}.png"
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"   ✅ Gráficos guardados en: {plot_filename}")
        plt.show()

    except Exception as e:
        print(f"   ⚠️ Error generando visualizaciones: {str(e)}")

    # =========================
    # SAVE DETAILED REPORT
    # =========================
    print(f"\n💾 === GUARDANDO REPORTE DETALLADO ===")

    report = {
        "timestamp": timestamp,
        "summary": {
            "api_status": "failed" if api_failed else "success",
            "total_alerts": len(all_alerts),
            "critical_issues": len([a for a in all_alerts if "🚨" in a]),
            "warnings": len([a for a in all_alerts if "⚠️" in a])
        },
        "performance_metrics": {
            "heating": {
                "reference": ref_metrics_heating,
                "current": curr_metrics_heating,
                "degradation": perf_degradation_h
            },
            "cooling": {
                "reference": ref_metrics_cooling,
                "current": curr_metrics_cooling,
                "degradation": perf_degradation_c
            }
        },
        "alerts": all_alerts,
        "recommendations": recommendations,
        "thresholds": {
            "performance": PERFORMANCE_THRESHOLDS,
            "drift": DRIFT_THRESHOLDS
        }
    }

    # Save report to JSON
    report_filename = f"drift_monitoring_report_{timestamp_file}.json"
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"   ✅ Reporte guardado en: {report_filename}")

    # =========================
    # FINAL SUMMARY
    # =========================
    print(f"\n🎯 === RESUMEN FINAL ===")
    print(f"📊 Total de alertas: {len(all_alerts)}")
    print(f"🚨 Problemas críticos: {len([a for a in all_alerts if '🚨' in a])}")
    print(f"⚠️ Advertencias: {len([a for a in all_alerts if '⚠️' in a])}")

    if len([a for a in all_alerts if "🚨" in a]) > 0:
        print(f"\n🛑 ACCIÓN INMEDIATA REQUERIDA")
        print(f"   El modelo muestra degradación crítica de performance o drift significativo")
        print(f"   Revisa las recomendaciones y considera reentrenar el modelo")
    elif len(all_alerts) > 0:
        print(f"\n⚠️ MONITOREO CONTINUO RECOMENDADO")
        print(f"   Se detectaron algunos problemas que requieren atención")
    else:
        print(f"\n✅ MODELO EN BUEN ESTADO")
        print(f"   Continúa el monitoreo regular")

    print(f"\n🏁 Monitoreo completado - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
