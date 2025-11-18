from pathlib import Path
import pandas as pd
from loguru import logger


class DataLoader:
    """
    DataLoader is a utility class for loading and saving tabular data files.
    It provides static methods to read and write CSV and Parquet files using pandas,
    with logging for traceability and automatic directory creation for saving files.
    """

    @staticmethod
    def load_csv(path: Path) -> pd.DataFrame:
        """
        Load a CSV file from the specified path into a pandas DataFrame.

        Args:
            path (Path): Path to the CSV file to load.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
        logger.info(f"Loading CSV: {path}")
        return pd.read_csv(path)

    @staticmethod
    def save_csv(df: pd.DataFrame, path: Path) -> Path:
        """
        Save a pandas DataFrame to a CSV file at the specified path.
        Automatically creates parent directories if they do not exist.

        Args:
            df (pd.DataFrame): DataFrame to save.
            path (Path): Path to save the CSV file.

        Returns:
            Path: The path where the CSV file was saved.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved CSV: {path}")
        return path

    @staticmethod
    def save_parquet(df: pd.DataFrame, path: Path) -> Path:
        """
        Save a pandas DataFrame to a Parquet file at the specified path.
        Automatically creates parent directories if they do not exist.

        Args:
            df (pd.DataFrame): DataFrame to save.
            path (Path): Path to save the Parquet file.

        Returns:
            Path: The path where the Parquet file was saved.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Saved Parquet: {path}")
        return path


__all__ = ["DataLoader"]

