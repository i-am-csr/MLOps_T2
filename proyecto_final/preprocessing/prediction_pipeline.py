"""
Preprocessing pipeline for prediction/inference.

This module provides functions to create preprocessing pipelines optimized
for prediction on new data, where we want to keep ALL rows (no removal).
"""

from typing import List

from loguru import logger
from sklearn.pipeline import Pipeline

from proyecto_final.preprocessing.transformers import (
    TypeConverter,
    MissingValueImputer,
    OutlierRemover,
    CategoricalCleaner,
    DuplicateRemover
)


def create_prediction_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    handle_outliers: str = "none",
    iqr_multiplier: float = 7.0,
    imputation_strategy: str = "median",
    categorical_threshold: float = 0.01
) -> Pipeline:
    """
    Create preprocessing pipeline for prediction/inference.

    This pipeline is optimized for prediction on new data, ensuring ALL
    input rows generate predictions. Unlike the training pipeline, this
    can skip outlier removal entirely.

    Args:
        numeric_cols: List of numeric column names to process
        categorical_cols: List of categorical column names to process
        handle_outliers: How to handle outliers in prediction:
            - 'none': Don't process outliers (recommended for robust models)
            - 'clip': Cap values to learned bounds (conservative approach)
            - 'remove': Remove outliers (NOT recommended for prediction)
        iqr_multiplier: IQR multiplier for outlier detection (default: 7.0)
        imputation_strategy: Strategy for missing values ('median' or 'drop')
        categorical_threshold: Minimum frequency for valid categories (default: 0.01)

    Returns:
        Configured sklearn Pipeline for prediction

    Example:
        >>> # Recommended: No outlier handling (let model decide)
        >>> # Only specify the columns you want to use
        >>> pipeline = create_prediction_pipeline(
        ...     numeric_cols=['X1', 'X3', 'X5', 'X7'],  # Only features to use
        ...     categorical_cols=['X6', 'X8'],
        ...     handle_outliers='none'  # ← No clipping/removal
        ... )
        >>> pipeline.fit(X_train)  # Learn parameters from training
        >>> X_new_processed = pipeline.transform(X_new)  # Process new data
        >>> # X_new_processed has same number of rows as X_new ✓
    """
    logger.info(f"[PredictionPipeline] Creating pipeline with handle_outliers='{handle_outliers}'")
    
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
    ]
    
    # Only add OutlierRemover if explicitly requested
    if handle_outliers == "clip":
        logger.info("[PredictionPipeline] Adding OutlierRemover with mode='clip'")
        preprocessing_steps.append(
            ("outlier_remover", OutlierRemover(
                numeric_cols=numeric_cols,
                iqr_multiplier=iqr_multiplier,
                mode="clip"
            ))
        )
    elif handle_outliers == "remove":
        logger.warning(
            "[PredictionPipeline] Using mode='remove' for prediction is NOT recommended. "
            "This will drop rows and lose predictions."
        )
        preprocessing_steps.append(
            ("outlier_remover", OutlierRemover(
                numeric_cols=numeric_cols,
                iqr_multiplier=iqr_multiplier,
                mode="remove"
            ))
        )
    elif handle_outliers == "none":
        logger.info("[PredictionPipeline] Skipping OutlierRemover (recommended for robust models)")
        # Don't add OutlierRemover at all
    else:
        raise ValueError(
            f"Invalid handle_outliers: '{handle_outliers}'. "
            "Must be 'none', 'clip', or 'remove'."
        )
    
    # Add remaining steps
    preprocessing_steps.extend([
        ("categorical_cleaner", CategoricalCleaner(
            categorical_cols=categorical_cols,
            threshold=categorical_threshold
        )),
        ("duplicate_remover", DuplicateRemover())
    ])

    return Pipeline(steps=preprocessing_steps)

