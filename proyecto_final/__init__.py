"""
Energy Efficiency ML Project - MLOps Implementation.

This package provides a complete MLOps solution for predicting heating and
cooling loads of buildings based on their geometric and physical characteristics.

Key Features:
    - Complete preprocessing pipeline with sklearn transformers
    - Model training with MLflow tracking
    - Hyperparameter optimization
    - DVC data versioning
    - Production-ready prediction API

Modules:
    config: Central configuration management
    data: Data loading and preprocessing
    preprocessing: Custom sklearn transformers and pipelines
    modeling: Training and prediction with MLflow integration
    scripts: Utility scripts for orchestration

Example:
    >>> from proyecto_final.config import MODELS_DIR
    >>> from proyecto_final.modeling.trainer import ModelTrainer
    >>> # Train models with MLflow tracking
    >>> trainer = ModelTrainer(config, "MyExperiment", MODELS_DIR)

Version: 2.0.0
Author: MLOps Team
License: MIT
"""

from proyecto_final import config  # noqa: F401

__version__ = "2.0.0"
__author__ = "MLOps Team"
__license__ = "MIT"

__all__ = [
    "config",
    "__version__",
    "__author__",
    "__license__"
]
