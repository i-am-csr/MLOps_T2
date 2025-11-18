# tests/conftest.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Importar la configuración real (se necesita para el tipo)
from proyecto_final.data.schemas import CleaningConfig

# --- Fixtures de Configuración y Mocking de Clases ---

class MockBaseConfig:
    """Clase base para simular objetos de configuración anidados (ej. TrainConfig)."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, MockBaseConfig(**value))
            else:
                setattr(self, key, value)

@pytest.fixture
def dummy_cleaning_config() -> CleaningConfig:
    """Fixture para la configuración de limpieza de datos."""
    return CleaningConfig(
        threshold=0.7,
        numeric_cols=["X1", "X3", "X5", "X7"],
        categorical_cols=["X6", "X8"]
    )

@pytest.fixture
def dummy_train_config() -> MockBaseConfig:
    """Fixture para una configuración de entrenamiento mockeada."""
    return MockBaseConfig(
        model={
            "name": "RandomForest",
            "library": "sklearn",
            "params": {"n_estimators": 10, "random_state": 42},
        },
        split={"test_size": 0.2, "random_state": 42},
        hpo={"enabled": False, "search": "grid", "cv": 3, "n_jobs": 1}
    )

@pytest.fixture
def common_cols():
    """Definición de columnas para uso general en pruebas."""
    return {
        "numeric": ["X1", "X3", "X5", "X7"],
        "categorical": ["X6", "X8"],
    }

# --- Fixtures de Datos ---

@pytest.fixture
def dummy_dataframe() -> pd.DataFrame:
    """Fixture para un DataFrame de prueba con varios tipos de datos y NaNs."""
    data = {
        "X1": [10.0, 20.0, 30.0, 40.0, 50.0, 1000.0, -100.0, np.nan],  # Numeric, outliers, NaN
        "X2": ["1", "2", "3", "4", "5", "6", "7", "8"],  # String-numeric
        "X3": [1.1, 2.2, np.nan, 4.4, 5.5, 6.6, 7.7, 8.8],  # Numeric, NaN
        "X4": ["$10", "$20", "$30", "$40", "$50", "$60", "$70", "$80"],  # Currency
        "X5": [0, 0, 0, 1, 1, 1, 2, 2], # Numeric (low variance)
        "X6": ["A", "B", "A", "C", "A", "B", "Z", "Z"],  # Categorical, Rare ('C')
        "X7": [5, 6, 7, 8, 9, 10, 11, 12], # Another numeric
        "X8": [1, 1, 2, 2, 3, 3, 4, 4], # Categorical (converted to numeric/category)
        "Y1": [25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0], # Target 1 (Heating Load)
        "Y2": [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]  # Target 2 (Cooling Load)
    }
    df = pd.DataFrame(data)
    # Introduce un duplicado exacto
    df.loc[8] = df.loc[0]
    return df

@pytest.fixture
def dummy_training_data(dummy_dataframe: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Fixture para datos de entrenamiento y prueba simples."""
    # Usar datos limpios y sin duplicados/NaNs en targets
    df = dummy_dataframe.drop(index=[5, 6, 8]).reset_index(drop=True)
    
    X = df.drop(columns=["Y1", "Y2"])
    y_heating = df["Y1"]
    y_cooling = df["Y2"]
    
    # Split simple para propósito de prueba
    X_train = X.iloc[:4].copy()
    X_test = X.iloc[4:].copy()
    y_train = pd.DataFrame({'Y1': y_heating.iloc[:4], 'Y2': y_cooling.iloc[:4]})
    y_test = pd.DataFrame({'Y1': y_heating.iloc[4:], 'Y2': y_cooling.iloc[4:]})

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        # Nueva muestra para predecir, simulando datos nuevos
        "X_new_raw": X_test.head(1).copy().rename(index={4: 99}).reset_index(drop=True)
    }

# --- Mocking de Clases para Integración/Unidad de Predicción ---

@pytest.fixture
def mock_predictor(monkeypatch) -> None:
    """
    Mockea las clases Predictor y MultiTargetPredictor para que no lean archivos 
    y usen modelos/transformadores simulados. También mockea joblib y mlflow.
    """
    
    # 1. Mock de Predictor para simular la predicción sin archivos reales
    class MockPredictorImpl:
        def __init__(self, model_path: Path, preprocessing_pipeline_path: Path, transformer_path: Path):
            self.model_path = model_path
            self.preprocessing_pipeline_path = preprocessing_pipeline_path
            self.transformer_path = transformer_path
            # Atributos simulados para que la inicialización del Predictor sea exitosa
            self.model = self._create_mock_model()
            self.preprocessing_pipeline = self.PassThroughTransformer()
            self.transformer = self._create_mock_transformer()

        class PassThroughTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None): return self
            def transform(self, X): return X.copy()

        class MockTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None): return self
            def transform(self, X): 
                N = X.shape[0]
                return np.arange(N * 6).reshape(N, 6) 
            def get_feature_names_out(self): 
                # Nombres de características simuladas después de OHE y Scaler
                return ["X1", "X3", "X5", "X7", "X6_A", "X8_1"]
        
        class MockModel(BaseEstimator):
            def predict(self, X: np.ndarray) -> np.ndarray:
                # Retorna predicciones determinísticas para la prueba
                if len(X) == 1:
                    return np.array([15.0])
                else:
                    return np.array([20.0, 30.0])

        def _create_mock_model(self): return self.MockModel()
        def _create_mock_transformer(self): return self.MockTransformer()

        def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
            X_cleaned = self.preprocessing_pipeline.transform(X)
            X_transformed = self.transformer.transform(X_cleaned)
            feature_names = self.transformer.get_feature_names_out()
            return pd.DataFrame(X_transformed, columns=feature_names, index=X_cleaned.index)

        def predict(self, X: Union[pd.DataFrame, List[Dict]]) -> pd.Series:
            if isinstance(X, list):
                X = pd.DataFrame(X)
            X_processed = self.preprocess(X)
            predictions = self.model.predict(X_processed)
            return pd.Series(predictions, index=X.index, name="prediction")

        def predict_dict(self, X: Union[pd.DataFrame, List[Dict]]) -> List[Dict]:
            predictions = self.predict(X)
            return [{"prediction": float(p)} for p in predictions]

    # 2. Mock de MultiTargetPredictor
    class MockMultiTargetPredictorImpl:
        def __init__(self, model_paths: Dict[str, Path], *args, **kwargs):
            self.predictors = {t: MockPredictorImpl(p, Path("pipe"), Path("trans")) for t, p in model_paths.items()}

        def predict(self, X: Union[pd.DataFrame, List[Dict]]) -> pd.DataFrame:
            if isinstance(X, list): X = pd.DataFrame(X)
            predictions = {t: p.predict(X).rename(t) for t, p in self.predictors.items()}
            return pd.DataFrame(predictions)
            
        def predict_dict(self, X: Union[pd.DataFrame, List[Dict]]) -> List[Dict]:
            return self.predict(X).to_dict(orient="records")

    # 3. Aplicar Mocks al módulo real
    monkeypatch.setattr("proyecto_final.modeling.predictor.Predictor", MockPredictorImpl)
    monkeypatch.setattr("proyecto_final.modeling.predictor.MultiTargetPredictor", MockMultiTargetPredictorImpl)
    
    # 4. Mockear dependencias externas
    def mock_dump(*args, **kwargs): pass
    
    # Usar un iterador para joblib.load para devolver los 3 objetos mockeados en orden (model, pipeline, transformer).
    mock_model = MockPredictorImpl.MockModel()
    mock_pipeline = MockPredictorImpl.PassThroughTransformer()
    mock_transformer = MockPredictorImpl.MockTransformer()
    mock_loader_iterator = iter([mock_model, mock_pipeline, mock_transformer])
    
    def mock_load(path):
        return next(mock_loader_iterator)
        
    monkeypatch.setattr("joblib.load", mock_load)
    monkeypatch.setattr("joblib.dump", mock_dump)

    # Mock MLflow para evitar comunicación con el servidor 
    class MockMLflow:
        def start_run(self, *args, **kwargs):
            class MockRunInfo: run_id = "mock_run_id_123"
            class MockRun:
                info = MockRunInfo()
                def __enter__(self): return self
                def __exit__(self, exc_type, exc_val, exc_tb): pass
            return MockRun()
        def set_tracking_uri(self, *args, **kwargs): pass
        def set_experiment(self, *args, **kwargs): pass
        def set_tag(self, *args, **kwargs): pass
        def log_params(self, *args, **kwargs): pass
        def log_metrics(self, *args, **kwargs): pass
        def log_artifact(self, *args, **kwargs): pass
        def get_artifact_uri(self, *args, **kwargs): return "file:///mock/path/artifacts"
        class sklearn:
            @staticmethod
            def log_model(*args, **kwargs): pass
        class models:
            @staticmethod
            def infer_signature(*args, **kwargs): pass
            
    # Reemplazamos la importación en los módulos que lo usan para evitar el TypeError
    monkeypatch.setattr("proyecto_final.modeling.trainer.mlflow", MockMLflow())
    monkeypatch.setattr("proyecto_final.modeling.trainer.mlflow.sklearn", MockMLflow.sklearn)
    monkeypatch.setattr("proyecto_final.modeling.trainer.mlflow.models", MockMLflow.models)


# Mock loguru para suprimir salida
@pytest.fixture(autouse=True)
def disable_loguru(monkeypatch):
    """Fixture para deshabilitar la salida de loguru durante las pruebas."""
    class MockLogger:
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
    monkeypatch.setattr("loguru.logger", MockLogger())