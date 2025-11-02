"""
Standalone preprocessing script.

This script runs only the preprocessing pipeline without training models.
Useful for data preparation and pipeline testing.

Usage:
    python scripts/run_preprocessing.py
"""

from loguru import logger

from proyecto_final.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR
)
from proyecto_final.data.preprocessing import PreprocessingOrchestrator


def main():
    """
    Run complete preprocessing workflow.

    This function executes:
    1. Initial data cleaning (type conversion, imputation, outliers, etc.)
    2. Feature engineering (encoding, scaling)
    3. Train-test split
    4. Save processed datasets and pipelines
    """
    logger.info("=" * 70)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 70)

    raw_data_file = RAW_DATA_DIR / "energy_efficiency_modified.csv"

    orchestrator = PreprocessingOrchestrator(
        raw_data_path=raw_data_file,
        interim_path=INTERIM_DATA_DIR,
        processed_path=PROCESSED_DATA_DIR,
        models_path=MODELS_DIR,
        test_size=0.2,
        random_state=42
    )

    # Definir features que queremos usar (basado en EDA)
    # Nota: X2, X4 se excluyen por alta correlación con X1, X3
    numeric_features = ["X1", "X3", "X5", "X7"]
    categorical_features = ["X6", "X8"]
    all_features = numeric_features + categorical_features
    
    # Targets
    targets = ["Y1", "Y2"]
    
    # Para limpieza inicial: incluir features + targets
    # (el dataset tiene más columnas como X2, X4, mixed_type_col que ignoramos)
    all_numeric_cols = numeric_features + targets
    
    X_train, X_test, y_train, y_test = orchestrator.run_complete_workflow(
        numeric_cols=all_numeric_cols,
        categorical_cols=categorical_features,
        feature_cols=all_features,
        target_cols=targets,
        numeric_feature_cols=numeric_features,
        categorical_feature_cols=categorical_features
    )

    logger.info("=" * 70)
    logger.info("PREPROCESSING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info("Artifacts saved:")
    logger.info(f"  - {INTERIM_DATA_DIR}/energy_efficiency_interim_clean.csv")
    logger.info(f"  - {PROCESSED_DATA_DIR}/energy_efficiency_train_prepared.csv")
    logger.info(f"  - {PROCESSED_DATA_DIR}/energy_efficiency_test_prepared.csv")
    logger.info(f"  - {MODELS_DIR}/initial_cleaning_pipeline.joblib")
    logger.info(f"  - {MODELS_DIR}/encoding_scaling_transformer.joblib")


if __name__ == "__main__":
    main()
