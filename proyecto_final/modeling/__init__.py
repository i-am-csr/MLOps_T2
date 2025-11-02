"""
Modeling module for training and prediction.

This module provides classes and utilities for:
- Model training with MLflow tracking
- Hyperparameter optimization
- Model evaluation
- Making predictions with preprocessing pipelines

Classes:
    ModelTrainer: Orchestrates model training with MLflow
    Predictor: Single-target prediction
    MultiTargetPredictor: Multi-target prediction
"""

from proyecto_final.modeling.trainer import ModelTrainer
from proyecto_final.modeling.predictor import Predictor, MultiTargetPredictor

__all__ = [
    "ModelTrainer",
    "Predictor",
    "MultiTargetPredictor"
]

