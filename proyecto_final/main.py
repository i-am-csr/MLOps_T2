from pathlib import Path

from loguru import logger

from proyecto_final.data.clean_data import DataCleaner
from proyecto_final.data.schemas import CleaningConfig
from proyecto_final.data.data_loader import DataLoader
import pandas as pd

# Resolve project root (main.py is inside proyecto_final/proyecto_final)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw" / "energy_efficiency_modified_dropped.csv"

# Load dataset (previous path pointed to processed + wrong filename)
# If you intended a processed file, switch RAW_DATA to the desired processed path.
df = DataLoader.load_csv(RAW_DATA)

config = CleaningConfig(
    threshold=0.5,
    numeric_cols=["X1", "X2", "X3", "X4", "X5", "X7", "Y1", "Y2"],
    categorical_cols=["X6", "X8"],
)

valid_categories = {
    "X6": [2, 3, 4, 5],
    "X8": [0, 1, 2, 3, 4, 5]
}

# Run cleaning pipeline and get both versions
cleaner = (
    DataCleaner(df)
    .apply_config(config)
    .coerce_numeric()
)

df_drop, df_fill = cleaner.split_versions()

# print("Clean_drop:", df_drop.shape)
# print("Clean_fill:", df_fill.shape)

cleaner_drop = (
    DataCleaner(df_drop)
    .apply_config(config)
    .remove_outliers_iqr()
    .filter_valid_categories(valid_categories)
)
df_drop_final = cleaner_drop.get()


cleaner_fill = (
    DataCleaner(df_fill)
    .apply_config(config)
    .remove_outliers_iqr()
    .fill_continuous_with_median()
    .fill_categorical_with_mode()
    .filter_valid_categories(valid_categories)
)
df_fill_final = cleaner_fill.get()

logger.info(f"Clean_drop_final: {df_drop_final.shape}")
logger.info(f"Clean_fill_final: {df_fill_final.shape}")
DataLoader.save_csv(df_drop_final, DATA_DIR / "processed" / "cleaned_drop.csv")
DataLoader.save_csv(df_fill_final, DATA_DIR / "processed" / "cleaned_fill.csv")