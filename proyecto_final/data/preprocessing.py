"""
High-level preprocessing orchestrator.

This module provides the PreprocessingOrchestrator class that coordinates
the complete preprocessing workflow for the Energy Efficiency dataset,
including data loading, cleaning, feature engineering, and saving artifacts.
"""

from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from proyecto_final.data.data_loader import DataLoader
from proyecto_final.preprocessing.pipeline import (
    create_preprocessing_pipeline,
    create_encoding_scaling_pipeline,
    save_pipeline
)


class PreprocessingOrchestrator:
    """
    Orchestrates the complete preprocessing workflow.

    This class coordinates data loading, preprocessing, train-test split,
    encoding, scaling, and artifact saving. It ensures a consistent and
    reproducible preprocessing workflow.

    Attributes:
        raw_data_path: Path to raw data file
        interim_path: Path for interim processed data
        processed_path: Path for final processed data
        models_path: Path for saving preprocessing artifacts
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        raw_data_path: Path,
        interim_path: Path,
        processed_path: Path,
        models_path: Path,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize PreprocessingOrchestrator.

        Args:
            raw_data_path: Path to raw data CSV
            interim_path: Directory for interim files
            processed_path: Directory for processed files
            models_path: Directory for saving pipelines
            test_size: Test set proportion (default: 0.2)
            random_state: Random seed (default: 42)
        """
        self.raw_data_path = raw_data_path
        self.interim_path = interim_path
        self.processed_path = processed_path
        self.models_path = models_path
        self.test_size = test_size
        self.random_state = random_state

        self.interim_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)

    def run_initial_cleaning(
        self,
        numeric_cols: list,
        categorical_cols: list,
        columns_to_drop: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Run initial cleaning pipeline.

        This method loads raw data and applies initial cleaning steps:
        type conversion, missing value imputation, outlier removal,
        categorical cleaning, and duplicate removal.

        Args:
            numeric_cols: List of numeric column names
            categorical_cols: List of categorical column names
            columns_to_drop: Optional list of columns to remove

        Returns:
            Cleaned DataFrame

        Example:
            >>> df_clean = orchestrator.run_initial_cleaning(
            ...     numeric_cols=['X1', 'X3', 'X5', 'X7'],
            ...     categorical_cols=['X6', 'X8'],
            ...     columns_to_drop=['X2', 'X4', 'mixed_type_col']
            ... )
        """
        logger.info("=" * 70)
        logger.info("STEP 1: INITIAL DATA CLEANING")
        logger.info("=" * 70)

        df_raw = DataLoader.load_csv(self.raw_data_path)
        logger.info(f"Raw data shape: {df_raw.shape}")

        pipeline = create_preprocessing_pipeline(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            columns_to_drop=columns_to_drop,
            iqr_multiplier=7.0,
            imputation_strategy="median",
            categorical_threshold=0.01
        )

        df_clean = pipeline.fit_transform(df_raw)
        logger.info(f"Cleaned data shape: {df_clean.shape}")

        interim_file = self.interim_path / "energy_efficiency_interim_clean.csv"
        DataLoader.save_csv(df_clean, interim_file)

        pipeline_file = self.models_path / "initial_cleaning_pipeline.joblib"
        save_pipeline(pipeline, str(pipeline_file))

        return df_clean

    def run_feature_engineering(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        target_cols: list,
        numeric_cols: list,
        categorical_cols: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run feature engineering and train-test split.

        This method performs:
        1. Train-test split (to prevent data leakage)
        2. Encoding and scaling (fit on train, transform on both)
        3. Save transformed datasets and transformers

        Args:
            df: Cleaned DataFrame
            feature_cols: List of feature column names
            target_cols: List of target column names
            numeric_cols: List of numeric feature names
            categorical_cols: List of categorical feature names

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)

        Example:
            >>> X_train, X_test, y_train, y_test = orchestrator.run_feature_engineering(
            ...     df=df_clean,
            ...     feature_cols=['X1', 'X3', 'X5', 'X7', 'X6', 'X8'],
            ...     target_cols=['Y1', 'Y2'],
            ...     numeric_cols=['X1', 'X3', 'X5', 'X7'],
            ...     categorical_cols=['X6', 'X8']
            ... )
        """
        logger.info("=" * 70)
        logger.info("STEP 2: FEATURE ENGINEERING & TRAIN-TEST SPLIT")
        logger.info("=" * 70)

        X = df[feature_cols].copy()
        y = df[target_cols].copy()

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Targets shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

        encoder_scaler = create_encoding_scaling_pipeline(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols
        )

        logger.info("[FitTransform] Fitting encoder/scaler on training data...")
        X_train_transformed = encoder_scaler.fit_transform(X_train)
        logger.info("[Transform] Transforming test data...")
        X_test_transformed = encoder_scaler.transform(X_test)

        feature_names = encoder_scaler.get_feature_names_out()
        X_train_transformed = pd.DataFrame(
            X_train_transformed,
            columns=feature_names,
            index=X_train.index
        )
        X_test_transformed = pd.DataFrame(
            X_test_transformed,
            columns=feature_names,
            index=X_test.index
        )

        logger.info(f"Transformed train shape: {X_train_transformed.shape}")
        logger.info(f"Transformed test shape: {X_test_transformed.shape}")

        train_data = pd.concat([X_train_transformed, y_train], axis=1)
        test_data = pd.concat([X_test_transformed, y_test], axis=1)

        train_file = self.processed_path / "energy_efficiency_train_prepared.csv"
        test_file = self.processed_path / "energy_efficiency_test_prepared.csv"

        DataLoader.save_csv(train_data, train_file)
        DataLoader.save_csv(test_data, test_file)

        transformer_file = self.models_path / "encoding_scaling_transformer.joblib"
        save_pipeline(encoder_scaler, str(transformer_file))

        return X_train_transformed, X_test_transformed, y_train, y_test

    def run_complete_workflow(
        self,
        numeric_cols: list,
        categorical_cols: list,
        feature_cols: list,
        target_cols: list,
        numeric_feature_cols: list,
        categorical_feature_cols: list,
        columns_to_drop: Optional[list] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run complete preprocessing workflow.

        This is a convenience method that executes both cleaning and
        feature engineering steps in sequence.

        Args:
            numeric_cols: Numeric columns in raw data
            categorical_cols: Categorical columns in raw data
            feature_cols: Final feature columns for training
            target_cols: Target column names
            numeric_feature_cols: Numeric features for encoding
            categorical_feature_cols: Categorical features for encoding
            columns_to_drop: Optional columns to remove (default: None)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)

        Example:
            >>> # Simple usage (no column dropping)
            >>> X_train, X_test, y_train, y_test = orchestrator.run_complete_workflow(
            ...     numeric_cols=['X1', 'X3', 'X5', 'X7', 'Y1', 'Y2'],
            ...     categorical_cols=['X6', 'X8'],
            ...     feature_cols=['X1', 'X3', 'X5', 'X7', 'X6', 'X8'],
            ...     target_cols=['Y1', 'Y2'],
            ...     numeric_feature_cols=['X1', 'X3', 'X5', 'X7'],
            ...     categorical_feature_cols=['X6', 'X8']
            ... )
        """
        logger.info("=" * 70)
        logger.info("COMPLETE PREPROCESSING WORKFLOW")
        logger.info("=" * 70)

        df_clean = self.run_initial_cleaning(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            columns_to_drop=columns_to_drop
        )

        X_train, X_test, y_train, y_test = self.run_feature_engineering(
            df=df_clean,
            feature_cols=feature_cols,
            target_cols=target_cols,
            numeric_cols=numeric_feature_cols,
            categorical_cols=categorical_feature_cols
        )

        logger.info("=" * 70)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Final training set: {X_train.shape}")
        logger.info(f"Final test set: {X_test.shape}")
        logger.info(f"Targets: {target_cols}")

        return X_train, X_test, y_train, y_test

