# tests/unit/test_trainer_metrics.py

import pytest
import pandas as pd
from proyecto_final.modeling.trainer import ModelTrainer
from pathlib import Path

# Se usa el mock_trainer que aísla ModelTrainer de mlflow y joblib

@pytest.fixture
def mock_trainer(dummy_train_config):
    """Instancia mockeada de ModelTrainer para probar métodos auxiliares."""
    # Se usa Path.cwd() para el path del modelo
    trainer = ModelTrainer(
        config=dummy_train_config,
        experiment_name="test_experiment",
        models_dir=Path.cwd() / "mock_models_dir"
    )
    return trainer

def test_compute_metrics_perfect_fit(mock_trainer: ModelTrainer):
    """Verifica el cálculo de métricas para un ajuste perfecto."""
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = pd.Series([10.0, 20.0, 30.0])
    
    metrics = mock_trainer._compute_metrics(y_true, y_pred)
    
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)

def test_compute_metrics_r2_zero(mock_trainer: ModelTrainer):
    """Verifica el cálculo de R² para una predicción constante (R² ≈ 0)."""
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = pd.Series([20.0, 20.0, 20.0]) # Predicción igual a la media
    
    metrics = mock_trainer._compute_metrics(y_true, y_pred)
    
    assert metrics["r2"] == pytest.approx(0.0)