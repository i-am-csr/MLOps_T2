# tests/unit/test_predictor.py

import pytest
import pandas as pd
from pathlib import Path
from typing import Dict
from proyecto_final.modeling.predictor import Predictor, MultiTargetPredictor 

# Los Mocks de Predictor y MultiTargetPredictor están en conftest.py

@pytest.fixture
def dummy_predictor(mock_predictor) -> Predictor:
    """Instancia de Predictor usando el mock."""
    return Predictor(
        model_path=Path("mock/model.joblib"),
        preprocessing_pipeline_path=Path("mock/pipeline.joblib"),
        transformer_path=Path("mock/transformer.joblib")
    )

@pytest.fixture
def dummy_multi_predictor(mock_predictor) -> MultiTargetPredictor:
    """Instancia de MultiTargetPredictor usando el mock."""
    return MultiTargetPredictor(
        model_paths={
            "heating": Path("mock/heating_model.joblib"),
            "cooling": Path("mock/cooling_model.joblib")
        },
        preprocessing_pipeline_path=Path("mock/pipeline.joblib"),
        transformer_path=Path("mock/transformer.joblib")
    )

def test_predictor_predict_dataframe(dummy_predictor: Predictor, dummy_training_data: Dict):
    """Verifica la predicción con un DataFrame de entrada."""
    X_new_raw = dummy_training_data["X_new_raw"]
    predictions = dummy_predictor.predict(X_new_raw)
    
    # El mock predice 15.0 para una sola fila
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(X_new_raw)
    assert predictions.iloc[0] == 15.0

def test_multitarget_predictor_predict_dict(dummy_multi_predictor: MultiTargetPredictor):
    """Verifica la predicción multiobjetivo con lista de diccionarios (formato API)."""
    data = [
        {"X1": 10.0, "X2": "a"}, 
        {"X1": 20.0, "X2": "b"}
    ]
    results = dummy_multi_predictor.predict_dict(data)
    
    # El mock predice [20.0, 30.0] para 2 filas
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0] == {"heating": 20.0, "cooling": 20.0}