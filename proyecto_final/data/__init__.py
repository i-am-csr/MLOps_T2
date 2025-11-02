"""
Data module for loading, cleaning, and preprocessing.

This module provides utilities for:
- Loading and saving data files (CSV, Parquet)
- Data cleaning and validation
- Preprocessing orchestration
- Schema validation with Pydantic

Classes:
    DataLoader: Static methods for I/O operations
    DataCleaner: Legacy data cleaning utilities
    PreprocessingOrchestrator: Complete preprocessing workflow
    CleaningConfig: Pydantic schema for configuration
"""

from proyecto_final.data.data_loader import DataLoader
from proyecto_final.data.clean_data import DataCleaner
from proyecto_final.data.preprocessing import PreprocessingOrchestrator
from proyecto_final.data.schemas import CleaningConfig

__all__ = [
    "DataLoader",
    "DataCleaner",
    "PreprocessingOrchestrator",
    "CleaningConfig"
]

