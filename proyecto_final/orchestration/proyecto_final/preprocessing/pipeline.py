"""
Preprocessing pipeline factory for Energy Efficiency dataset.

This module provides functions to build complete preprocessing pipelines
using sklearn's Pipeline and ColumnTransformer. The pipelines handle
data cleaning, feature engineering, encoding, and scaling.
"""

from typing import List, Optional

import joblib
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from proyecto_final.preprocessing.transformers import (
    TypeConverter,
    MissingValueImputer,
    OutlierRemover,
    CategoricalCleaner,
    FeatureSelector,
    DuplicateRemover
)


def create_preprocessing_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    columns_to_drop: Optional[List[str]] = None,
    iqr_multiplier: float = 7.0,
    imputation_strategy: str = "median",
    categorical_threshold: float = 0.01
) -> Pipeline:
    """
    Create complete preprocessing pipeline.

    This function builds a sklearn Pipeline that performs all preprocessing
    steps in the correct order:
    1. Type conversion
    2. Missing value imputation
    3. Outlier removal (IQR)
    4. Categorical cleaning
    5. Duplicate removal
    6. Feature selection
    7. Encoding and scaling

    Args:
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        columns_to_drop: Optional list of columns to remove (e.g., highly correlated)
        iqr_multiplier: IQR multiplier for outlier detection (default: 7.0)
        imputation_strategy: Strategy for missing values ('median' or 'drop')
        categorical_threshold: Minimum frequency for valid categories (default: 0.01)

    Returns:
        Configured sklearn Pipeline

    Example:
        >>> pipeline = create_preprocessing_pipeline(
        ...     numeric_cols=['X1', 'X3', 'X5', 'X7'],
        ...     categorical_cols=['X6', 'X8'],
        ...     columns_to_drop=['X2', 'X4']
        ... )
        >>> X_train_transformed = pipeline.fit_transform(X_train)
    """
    logger.info("[Pipeline] Creating preprocessing pipeline")

    preprocessing_steps = [
        ("type_converter", TypeConverter(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols
        )),
        ("missing_imputer", MissingValueImputer(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            strategy=imputation_strategy
        )),
        ("outlier_remover", OutlierRemover(
            numeric_cols=numeric_cols,
            iqr_multiplier=iqr_multiplier
        )),
        ("categorical_cleaner", CategoricalCleaner(
            categorical_cols=categorical_cols,
            threshold=categorical_threshold
        )),
        ("duplicate_remover", DuplicateRemover()),
        ("feature_selector", FeatureSelector(
            columns_to_drop=columns_to_drop
        ))
    ]

    pipeline = Pipeline(steps=preprocessing_steps)

    logger.info(f"[Pipeline] Created pipeline with {len(preprocessing_steps)} steps")

    return pipeline


def create_encoding_scaling_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Create encoding and scaling transformer.

    This function creates a ColumnTransformer that applies:
    - MinMaxScaler to numeric columns
    - OneHotEncoder to categorical columns

    This transformer should be applied AFTER train-test split to prevent
    data leakage.

    Args:
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names

    Returns:
        Configured ColumnTransformer

    Example:
        >>> transformer = create_encoding_scaling_pipeline(
        ...     numeric_cols=['X1', 'X3'],
        ...     categorical_cols=['X6', 'X8']
        ... )
        >>> X_train_encoded = transformer.fit_transform(X_train)
    """
    logger.info("[Pipeline] Creating encoding & scaling transformer")

    transformers = []

    if numeric_cols:
        numeric_transformer = Pipeline(steps=[
            ("scaler", MinMaxScaler(feature_range=(0, 1)))
        ])
        transformers.append(("num", numeric_transformer, numeric_cols))

    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ])
        transformers.append(("cat", categorical_transformer, categorical_cols))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=True
    )

    logger.info(f"[Pipeline] Transformer created with {len(transformers)} components")

    return column_transformer


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """
    Save pipeline to disk using joblib.

    Args:
        pipeline: Sklearn Pipeline to save
        path: File path for saving

    Example:
        >>> save_pipeline(pipeline, "models/preprocessing_pipeline.joblib")
    """
    joblib.dump(pipeline, path)
    logger.info(f"[Pipeline] Saved pipeline to: {path}")


def load_pipeline(path: str) -> Pipeline:
    """
    Load pipeline from disk.

    Args:
        path: File path to load from

    Returns:
        Loaded sklearn Pipeline

    Example:
        >>> pipeline = load_pipeline("models/preprocessing_pipeline.joblib")
    """
    pipeline = joblib.load(path)
    logger.info(f"[Pipeline] Loaded pipeline from: {path}")
    return pipeline

