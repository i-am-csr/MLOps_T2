# tests/unit/test_data_loader.py

import pytest
import pandas as pd
from pathlib import Path
from proyecto_final.data.data_loader import DataLoader

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """DataFrame de muestra para guardar/cargar."""
    return pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})

def test_save_and_load_csv(sample_dataframe: pd.DataFrame, tmp_path: Path):
    """Verifica que DataLoader pueda guardar y cargar CSV."""
    file_path = tmp_path / "test_data.csv"
    
    # Guardar CSV
    saved_path = DataLoader.save_csv(sample_dataframe, file_path)
    
    # Cargar CSV
    loaded_df = DataLoader.load_csv(file_path)
    
    pd.testing.assert_frame_equal(sample_dataframe, loaded_df)