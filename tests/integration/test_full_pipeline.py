# tests/integration/test_full_pipeline.py

import pytest
import pandas as pd
from pathlib import Path
from typing import Dict

from proyecto_final.preprocessing.prediction_pipeline import create_prediction_pipeline
from proyecto_final.modeling.predictor import MultiTargetPredictor 

# --- Pruebas de Integración ---

def test_prediction_pipeline_integrity(dummy_training_data, common_cols):
    """
    Verifica que el pipeline de predicción preserve el número de filas 
    cuando se usa 'clip' o 'none', lo cual es crucial para la inferencia.
    """
    X_train = dummy_training_data["X_train"]
    
    # Pipeline con 'clip' (debe mantener todas las filas)
    pipeline_clip = create_prediction_pipeline(
        numeric_cols=common_cols["numeric"],
        categorical_cols=common_cols["categorical"],
        handle_outliers='clip'
    )
    X_processed_clip = pipeline_clip.fit_transform(X_train)
    assert len(X_processed_clip) == len(X_train)
    
    # Pipeline con 'none' (debe mantener todas las filas)
    pipeline_none = create_prediction_pipeline(
        numeric_cols=common_cols["numeric"],
        categorical_cols=common_cols["categorical"],
        handle_outliers='none'
    )
    X_processed_none = pipeline_none.fit_transform(X_train)
    assert len(X_processed_none) == len(X_train)

def test_end_to_end_prediction_flow(mock_predictor, dummy_training_data, tmp_path):
    """
    Prueba el flujo completo: 
    Inicialización del Predictor (mockeado) -> Predicción en datos nuevos.
    """
    X_new_raw = dummy_training_data["X_new_raw"]
    
    # Se simulan los paths de los artefactos
    model_paths = {
        "heating": tmp_path / "heating_model.joblib",
        "cooling": tmp_path / "cooling_model.joblib"
    }
    pipeline_path = tmp_path / "initial_cleaning_pipeline.joblib"
    transformer_path = tmp_path / "encoding_scaling_transformer.joblib"
    
    # Se inicializa el predictor, que usará los Mocks
    multi_predictor = MultiTargetPredictor(
        model_paths=model_paths,
        preprocessing_pipeline_path=pipeline_path,
        transformer_path=transformer_path
    )
    
    # Ejecutar la predicción
    predictions_df = multi_predictor.predict(X_new_raw)
    
    # Verificaciones
    assert isinstance(predictions_df, pd.DataFrame)
    assert len(predictions_df) == len(X_new_raw)
    assert "heating" in predictions_df.columns
    assert "cooling" in predictions_df.columns
    
    # El MockModel predice 15.0 para una sola fila
    assert predictions_df["heating"].iloc[0] == 15.0