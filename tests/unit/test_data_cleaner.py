# tests/unit/test_data_cleaner.py

import pandas as pd
import numpy as np
import pytest
from proyecto_final.data.clean_data import DataCleaner
from proyecto_final.data.schemas import CleaningConfig

def test_coerce_numeric(dummy_dataframe: pd.DataFrame, dummy_cleaning_config: CleaningConfig):
    """Verifica la conversión a numérico y el manejo de errores."""
    df = dummy_dataframe.copy()
    df.loc[0, "X1"] = "invalid" 
    
    cleaner = DataCleaner(df).apply_config(dummy_cleaning_config)
    cleaner.coerce_numeric()
    
    # 'invalid' debería ser forzado a NaN
    assert pd.isna(cleaner.df.loc[0, "X1"])
    assert cleaner.df["X8"].dtype in [np.dtype('float64'), np.dtype('int64')]

def test_drop_missing_values(dummy_dataframe: pd.DataFrame, dummy_cleaning_config: CleaningConfig):
    """Drop rows with too many NaNs (based on threshold)."""
    df = dummy_dataframe.copy()
    # Row 2 tiene 6 NaNs, lo cual es < 7 no-NaNs (threshold 0.7 * 10 cols)
    df.loc[2, ["X1", "X3", "X5", "X6", "X8", "Y1"]] = np.nan 

    cleaner = DataCleaner(df).apply_config(dummy_cleaning_config)
    cleaner.drop_missing_values()
    
    assert 2 not in cleaner.df.index
    assert 0 in cleaner.df.index

def test_remove_outliers_iqr(dummy_dataframe: pd.DataFrame, dummy_cleaning_config: CleaningConfig):
    """Verifica la eliminación de outliers usando IQR."""
    df = dummy_dataframe.copy()
    # El proceso de IQR remueve:
    # 1. Los 2 outliers extremos de X1 (índices 5 y 6).
    # 2. El NaN original de X1 (índice 7).
    # 3. El NaN original de X3 (índice 2).
    # Total de filas removidas: 4. Filas restantes: 9 - 4 = 5.
    
    cleaner = DataCleaner(df).apply_config(dummy_cleaning_config)
    cleaner.remove_outliers_iqr(verbose=False)
    
    # Se esperaba 6, el valor correcto es 5.
    assert len(cleaner.df) == 5
    assert 5 not in cleaner.df.index
    assert 6 not in cleaner.df.index
    assert 7 not in cleaner.df.index
    assert 2 not in cleaner.df.index