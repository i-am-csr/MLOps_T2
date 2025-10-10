from proyecto_final.data.schemas import CleaningConfig
import pandas as pd
from typing import Optional
from loguru import logger


class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df_original = df.copy()
        self.df = df.copy()
        self.config: Optional[CleaningConfig] = None

    def apply_config(self, config: CleaningConfig):
        self.config = config
        return self

    def coerce_numeric(self):
        """Convert to numeric, invalids → NaN."""
        for col in self.config.numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        for col in self.config.categorical_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        return self

    def drop_missing_values(self):
        """Drop rows with too many NaNs (based on threshold)."""
        min_non_na = int(self.config.threshold * self.df.shape[1])
        self.df = self.df.dropna(thresh=min_non_na)
        return self

    def clean_currency(self, col: str):
        """Remove $ symbol and cast to float."""
        self.df[col] = self.df[col].str.replace("$", "", regex=False).astype(float)
        return self

    def clip_outliers(self, col: str, lower=0, upper=100):
        """Clip column values to [lower, upper]."""
        self.df[col] = self.df[col].clip(lower, upper)
        return self

    def fill_continuous_with_median(self):
        """Fill numeric columns with median."""
        for col in self.config.numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        return self

    def fill_categorical_with_mode(self):
        """Fill categorical columns with mode."""
        for col in self.config.categorical_cols:
            if not self.df[col].mode().empty:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        return self

    def split_versions(self):
        """
        Create two versions:
        - df_drop: drop rows with any NaNs
        - df_fill: fill numeric with median, categorical with mode
        """
        df_base = self.df.copy()

        # Drop version
        df_drop = df_base.dropna().reset_index(drop=True)

        # Fill version
        df_fill = df_base.copy()
        for col in self.config.numeric_cols:
            df_fill[col] = df_fill[col].fillna(df_fill[col].median())

        for col in self.config.categorical_cols:
            if not df_fill[col].mode().empty:
                df_fill[col] = df_fill[col].fillna(df_fill[col].mode()[0])

        return df_drop, df_fill

    def remove_outliers_iqr(self, verbose: bool = True):
        """
        Remove outliers using the IQR method on numeric columns.
        Updates self.df in-place.

        Args:
            verbose (bool): Print number of outliers removed per column.

        Returns:
            self: Enables method chaining
        """
        df_iqr = self.df.copy()
        outlier_counts = {}
        rows_before = df_iqr.shape[0]

        for col in self.config.numeric_cols:
            Q1 = df_iqr[col].quantile(0.25)
            Q3 = df_iqr[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            # Count and filter
            n_outliers = ((df_iqr[col] < lower) | (df_iqr[col] > upper)).sum()
            outlier_counts[col] = n_outliers
            df_iqr = df_iqr[(df_iqr[col] >= lower) & (df_iqr[col] <= upper)]

        rows_after = df_iqr.shape[0]
        self.df = df_iqr

        if verbose:
            logger.info(f"[IQR] Rows before: {rows_before}, after: {rows_after}, removed: {rows_before - rows_after}")
            for col, count in outlier_counts.items():
                logger.info(f"[IQR] {col}: {count} outliers removed")

        return self

    def filter_valid_categories(self, valid_values: dict, verbose: bool = True):
        """
        Filter rows by valid values in categorical columns.

        Args:
            valid_values (dict): mapping of {column_name: list_of_valid_values}
            verbose (bool): whether to print stats on how many rows were dropped

        Returns:
            self
        """
        initial_rows = self.df.shape[0]

        for col, valid in valid_values.items():
            if col in self.df.columns:
                before = self.df.shape[0]
                self.df = self.df[self.df[col].isin(valid)]
                after = self.df.shape[0]
                if verbose:
                    logger.info(f"[Category Filter] {col}: removed {before - after} rows")

        final_rows = self.df.shape[0]
        if verbose:
            logger.info(f"[Category Filter] Total rows removed: {initial_rows - final_rows}")

        return self

    def get(self):
        """Get the current version of the working dataframe."""
        return self.df
