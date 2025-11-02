"""
Preprocessing module for Energy Efficiency dataset.

This module contains custom transformers and preprocessing utilities
for building sklearn pipelines.

Custom Transformers:
    TypeConverter: Convert column types and handle invalid entries
    MissingValueImputer: Impute missing values (median/mode)
    OutlierRemover: Remove outliers using IQR method
    CategoricalCleaner: Clean rare categorical values
    FeatureSelector: Select features by removing correlations
    DuplicateRemover: Remove duplicate rows

Pipeline Factories:
    create_preprocessing_pipeline: Create complete preprocessing pipeline
    create_encoding_scaling_pipeline: Create encoding/scaling transformer
    save_pipeline: Save pipeline to disk
    load_pipeline: Load pipeline from disk

Example:
    >>> from proyecto_final.preprocessing import (
    ...     TypeConverter,
    ...     create_preprocessing_pipeline
    ... )
    >>> pipeline = create_preprocessing_pipeline(
    ...     numeric_cols=['X1', 'X3'],
    ...     categorical_cols=['X6', 'X8']
    ... )
    >>> df_clean = pipeline.fit_transform(df_raw)
"""

# Import transformers
from proyecto_final.preprocessing.transformers import (
    TypeConverter,
    MissingValueImputer,
    OutlierRemover,
    CategoricalCleaner,
    FeatureSelector,
    DuplicateRemover
)

# Import pipeline utilities
from proyecto_final.preprocessing.pipeline import (
    create_preprocessing_pipeline,
    create_encoding_scaling_pipeline,
    save_pipeline,
    load_pipeline
)
from proyecto_final.preprocessing.prediction_pipeline import (
    create_prediction_pipeline
)

__all__ = [
    # Transformers
    "TypeConverter",
    "MissingValueImputer",
    "OutlierRemover",
    "CategoricalCleaner",
    "FeatureSelector",
    "DuplicateRemover",
    # Pipeline utilities
    "create_preprocessing_pipeline",
    "create_prediction_pipeline",
    "create_encoding_scaling_pipeline",
    "save_pipeline",
    "load_pipeline"
]

