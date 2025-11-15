# tests/unit/test_transformers.py

import pandas as pd
import numpy as np
import pytest
from proyecto_final.preprocessing.transformers import (
    TypeConverter, MissingValueImputer, OutlierRemover, CategoricalCleaner, DuplicateRemover
)

@pytest.fixture
def clean_dummy_dataframe(dummy_dataframe: pd.DataFrame):
    """DataFrame para probar imputadores y limpiadores."""
    df = dummy_dataframe.drop(index=[8]).reset_index(drop=True) # Eliminar duplicado (8 filas)
    df.loc[0, "X1"] = np.nan # NaN forzado (1)
    df.loc[1, "X6"] = np.nan # NaN forzado (2)
    # NaNs originales: X3 (fila 2) y X1/X3 (fila 7).
    # Filas con NaN: 0, 1, 2, 7.
    return df

# --- MissingValueImputer Tests ---

def test_imputer_median_strategy(clean_dummy_dataframe: pd.DataFrame, common_cols):
    """Verifica la imputación con mediana/moda."""
    imputer = MissingValueImputer(
        numeric_cols=common_cols["numeric"],
        categorical_cols=common_cols["categorical"],
        strategy="median"
    )
    imputer.fit(clean_dummy_dataframe)
    
    X_transformed = imputer.transform(clean_dummy_dataframe)
    
    median_x1 = clean_dummy_dataframe["X1"].drop(0).median()
    assert X_transformed.loc[0, "X1"] == pytest.approx(median_x1)
    
    mode_x6 = clean_dummy_dataframe["X6"].drop(1).mode()[0]
    assert X_transformed.loc[1, "X6"] == mode_x6
    
def test_imputer_drop_strategy(clean_dummy_dataframe: pd.DataFrame):
    """Verifica la eliminación de filas con estrategia 'drop'."""
    # 4 filas contienen NaN (0, 1, 2, 7). 8 - 4 = 4 filas restantes.
    imputer = MissingValueImputer(strategy="drop")
    X_transformed = imputer.transform(clean_dummy_dataframe)
    
    # Se esperaba 6, el valor correcto es 4.
    assert len(X_transformed) == 4

# --- OutlierRemover Tests ---

def test_outlier_remover_clip_mode(dummy_dataframe: pd.DataFrame, common_cols):
    """Verifica el recorte de valores outliers."""
    remover = OutlierRemover(numeric_cols=common_cols["numeric"], iqr_multiplier=1.5, mode="clip")
    remover.fit(dummy_dataframe)
    
    lower, upper = remover.bounds_["X1"]
    
    X_transformed = remover.transform(dummy_dataframe)
    
    assert len(X_transformed) == 9
    assert X_transformed.loc[5, "X1"] == pytest.approx(upper)

# --- CategoricalCleaner Tests ---

def test_categorical_cleaner_replace_rare(dummy_dataframe: pd.DataFrame, common_cols):
    """Verifica el reemplazo de categorías raras (threshold=0.2)."""
    cleaner = CategoricalCleaner(categorical_cols=common_cols["categorical"], threshold=0.2)
    cleaner.fit(dummy_dataframe.drop(index=[8]))
    
    X_transformed = cleaner.transform(dummy_dataframe)
    
    assert X_transformed.loc[3, "X6"] == 'A'
    assert X_transformed.loc[6, "X6"] == 'Z'

# --- DuplicateRemover Test ---

def test_duplicate_remover_transform(dummy_dataframe: pd.DataFrame):
    """Verifica la eliminación de filas duplicadas."""
    remover = DuplicateRemover()
    X_transformed = remover.transform(dummy_dataframe)
    
    assert len(X_transformed) == 8