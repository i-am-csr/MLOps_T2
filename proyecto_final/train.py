"""
Main training script with complete preprocessing and MLflow integration.

This script orchestrates the complete machine learning workflow:
1. Data loading and preprocessing
2. Feature engineering
3. Model training with hyperparameter optimization
4. MLflow experiment tracking
5. Model and artifact saving

Usage:
    python train.py --config configs/xgb.yaml --run-name my_experiment
"""

from argparse import ArgumentParser
from json import dumps

from loguru import logger

from proyecto_final.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    MLFLOW,
    DATA,
    train_config_from_yaml,
    setup_mlflow
)
from proyecto_final.data.data_loader import DataLoader
from proyecto_final.data.preprocessing import PreprocessingOrchestrator
from proyecto_final.modeling.trainer import ModelTrainer


def main():
    """
    Main training pipeline.

    Steps:
        1. Parse command-line arguments
        2. Setup MLflow tracking
        3. Load and preprocess data
        4. Train models for all targets
        5. Log results and save artifacts
    """
    ap = ArgumentParser(description="Train Energy Efficiency models with complete pipeline.")
    ap.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g., configs/xgb.yaml)"
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help="Optional custom run name (default: timestamp)"
    )
    ap.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing and use existing processed data"
    )
    args = ap.parse_args()

    logger.info("=" * 70)
    logger.info("ENERGY EFFICIENCY MODEL TRAINING PIPELINE")
    logger.info("=" * 70)

    setup_mlflow()
    logger.info(f"MLflow tracking URI: {MLFLOW.tracking_uri}")
    logger.info(f"MLflow experiment: {MLFLOW.experiment}")

    config = train_config_from_yaml(args.config)
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Library: {config.model.library}")
    logger.info(f"HPO enabled: {config.hpo.enabled}")

    if not args.skip_preprocessing:
        logger.info("[Pipeline] Running preprocessing...")

        raw_data_file = RAW_DATA_DIR / "energy_efficiency_modified.csv"

        orchestrator = PreprocessingOrchestrator(
            raw_data_path=raw_data_file,
            interim_path=INTERIM_DATA_DIR,
            processed_path=PROCESSED_DATA_DIR,
            models_path=MODELS_DIR,
            test_size=config.split.test_size,
            random_state=config.split.random_state
        )

        # Definir features y targets (basado en análisis EDA)
        # Nota: X2, X4 excluidos por alta correlación con X1, X3
        numeric_features = ["X1", "X3", "X5", "X7"]
        categorical_features = ["X6", "X8"]
        all_features = numeric_features + categorical_features
        targets = ["Y1", "Y2"]
        
        # Para limpieza inicial: features + targets
        all_numeric_cols = numeric_features + targets
        
        X_train, X_test, y_train, y_test = orchestrator.run_complete_workflow(
            numeric_cols=all_numeric_cols,
            categorical_cols=categorical_features,
            feature_cols=all_features,
            target_cols=targets,
            numeric_feature_cols=numeric_features,
            categorical_feature_cols=categorical_features
        )
    else:
        logger.info("[Pipeline] Loading preprocessed data...")

        train_file = PROCESSED_DATA_DIR / "energy_efficiency_train_prepared.csv"
        test_file = PROCESSED_DATA_DIR / "energy_efficiency_test_prepared.csv"

        train_data = DataLoader.load_csv(train_file)
        test_data = DataLoader.load_csv(test_file)

        feature_cols = [c for c in train_data.columns if c not in ["Y1", "Y2"]]
        X_train = train_data[feature_cols]
        y_train = train_data[["Y1", "Y2"]]
        X_test = test_data[feature_cols]
        y_test = test_data[["Y1", "Y2"]]

    trainer = ModelTrainer(
        config=config,
        experiment_name=MLFLOW.experiment,
        models_dir=MODELS_DIR
    )

    results = trainer.train_all_targets(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_mapping=DATA.targets
    )

    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Results summary:")
    print(dumps(
        {k: {"metrics": v["metrics"]} for k, v in results.items()},
        indent=2
    ))


if __name__ == "__main__":
    main()