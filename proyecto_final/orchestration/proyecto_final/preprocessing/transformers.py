"""
Custom sklearn transformers for preprocessing pipeline.

This module implements custom transformers that follow sklearn's
BaseEstimator and TransformerMixin interfaces to enable seamless
integration with sklearn Pipeline.
"""

from typing import List, Optional, Dict, Union

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin


class TypeConverter(BaseEstimator, TransformerMixin):
    """
    Convert columns to numeric types and handle invalid entries.

    This transformer converts specified columns to numeric types,
    coercing invalid entries to NaN. It's the first step in the
    preprocessing pipeline.

    Attributes:
        numeric_cols: List of columns to convert to numeric
        categorical_cols: List of categorical column names
    """

    def __init__(
        self,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None
    ):
        """
        Initialize TypeConverter.

        Args:
            numeric_cols: List of numeric column names
            categorical_cols: List of categorical column names
        """
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit transformer (no-op for this transformer).

        Args:
            X: Input DataFrame
            y: Target variable (unused)

        Returns:
            self: Fitted transformer
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns to appropriate types.

        Args:
            X: Input DataFrame

        Returns:
            Transformed DataFrame with corrected types
        """
        X = X.copy()

        for col in self.numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        for col in self.categorical_cols:
            if col in X.columns:
                # Safeguard: Don't convert to categorical if it's already in numeric_cols
                if col not in self.numeric_cols:
                    X[col] = pd.to_numeric(X[col], errors="coerce").astype("category")
                else:
                    logger.warning(f"[TypeConverter] Column '{col}' is in both numeric_cols and categorical_cols. Treating as numeric.")
                    X[col] = pd.to_numeric(X[col], errors="coerce")

        return X


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values using median for numeric and mode for categorical.

    This transformer learns imputation values from training data and applies
    them to both training and test sets to prevent data leakage.

    Attributes:
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        strategy: Strategy for imputation ('median' or 'drop')
        numeric_impute_values_: Learned median values for numeric columns
        categorical_impute_values_: Learned mode values for categorical columns
    """

    def __init__(
        self,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        strategy: str = "median"
    ):
        """
        Initialize MissingValueImputer.

        Args:
            numeric_cols: List of numeric column names
            categorical_cols: List of categorical column names
            strategy: Imputation strategy ('median' or 'drop')
        """
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
        self.strategy = strategy
        self.numeric_impute_values_: Dict[str, float] = {}
        self.categorical_impute_values_: Dict[str, Union[float, str]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn imputation values from training data.

        Args:
            X: Training DataFrame
            y: Target variable (unused)

        Returns:
            self: Fitted transformer
        """
        if self.strategy == "median":
            for col in self.numeric_cols:
                if col in X.columns:
                    self.numeric_impute_values_[col] = X[col].median()

            for col in self.categorical_cols:
                if col in X.columns and not X[col].mode().empty:
                    self.categorical_impute_values_[col] = X[col].mode()[0]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation to data.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with imputed missing values
        """
        X = X.copy()

        if self.strategy == "median":
            for col, value in self.numeric_impute_values_.items():
                if col in X.columns:
                    X[col] = X[col].fillna(value)

            for col, value in self.categorical_impute_values_.items():
                if col in X.columns:
                    # Handle categorical dtype columns that might not have the impute value in their categories
                    if pd.api.types.is_categorical_dtype(X[col]):
                        # Add the impute value to categories if it's not already there
                        if value not in X[col].cat.categories:
                            X[col] = X[col].cat.add_categories([value])
                    X[col] = X[col].fillna(value)
        elif self.strategy == "drop":
            X = X.dropna()

        return X


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Remove or clip outliers using IQR method.

    This transformer identifies outliers from numeric columns using the
    Interquartile Range (IQR) method. It has two modes:
    - 'remove': Removes outlier rows (for training)
    - 'clip': Clips values to bounds (for prediction)

    Attributes:
        numeric_cols: List of numeric column names
        iqr_multiplier: IQR multiplier for outlier bounds (default: 7.0)
        mode: 'remove' to drop rows or 'clip' to cap values
        bounds_: Learned outlier bounds per column
    """

    def __init__(
        self,
        numeric_cols: Optional[List[str]] = None,
        iqr_multiplier: float = 7.0,
        mode: str = "remove"
    ):
        """
        Initialize OutlierRemover.

        Args:
            numeric_cols: List of numeric column names
            iqr_multiplier: Multiplier for IQR bounds (default: 7.0 for lenient)
            mode: 'remove' to drop outlier rows (training) or
                  'clip' to cap values to bounds (prediction)
        """
        self.numeric_cols = numeric_cols or []
        self.iqr_multiplier = iqr_multiplier
        self.mode = mode
        self.bounds_: Dict[str, tuple] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Calculate outlier bounds from training data.

        Args:
            X: Training DataFrame
            y: Target variable (unused)

        Returns:
            self: Fitted transformer
        """
        for col in self.numeric_cols:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.iqr_multiplier * IQR
                upper = Q3 + self.iqr_multiplier * IQR
                self.bounds_[col] = (lower, upper)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove or clip outliers based on learned bounds.

        Behavior depends on mode:
        - 'remove': Removes rows with outliers (for training)
        - 'clip': Clips values to bounds (for prediction)

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with outliers handled according to mode
        """
        X = X.copy()
        initial_rows = len(X)

        if self.mode == "remove":
            # Training mode: remove entire rows with outliers
            for col, (lower, upper) in self.bounds_.items():
                if col in X.columns:
                    mask = (X[col] >= lower) & (X[col] <= upper)
                    X = X[mask]

            removed = initial_rows - len(X)
            if removed > 0:
                logger.info(f"[OutlierRemover] Removed {removed} rows ({removed/initial_rows*100:.2f}%)")

            return X.reset_index(drop=True)

        elif self.mode == "clip":
            # Prediction mode: clip values to bounds (keep all rows)
            clipped_count = 0
            for col, (lower, upper) in self.bounds_.items():
                if col in X.columns:
                    before = X[col].copy()
                    X[col] = X[col].clip(lower=lower, upper=upper)
                    clipped = (before != X[col]).sum()
                    clipped_count += clipped

            if clipped_count > 0:
                logger.info(f"[OutlierRemover] Clipped {clipped_count} outlier values")

            return X

        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'remove' or 'clip'.")


class CategoricalCleaner(BaseEstimator, TransformerMixin):
    """
    Clean categorical variables by replacing rare categories.

    This transformer identifies rare categories (below threshold) and replaces
    them with the mode. This helps reduce noise and dimensionality.

    Attributes:
        categorical_cols: List of categorical column names
        threshold: Minimum frequency threshold for valid categories
        valid_categories_: Learned valid categories per column
        mode_values_: Learned mode values per column
    """

    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        threshold: float = 0.01
    ):
        """
        Initialize CategoricalCleaner.

        Args:
            categorical_cols: List of categorical column names
            threshold: Minimum frequency for valid category (default: 0.01 = 1%)
        """
        self.categorical_cols = categorical_cols or []
        self.threshold = threshold
        self.valid_categories_: Dict[str, List] = {}
        self.mode_values_: Dict[str, Union[float, str]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Identify valid categories and mode from training data.

        Args:
            X: Training DataFrame
            y: Target variable (unused)

        Returns:
            self: Fitted transformer
        """
        for col in self.categorical_cols:
            if col in X.columns:
                freq = X[col].value_counts(normalize=True)
                valid = freq[freq >= self.threshold].index.tolist()
                self.valid_categories_[col] = valid

                mode = X[col].mode()
                if not mode.empty:
                    self.mode_values_[col] = mode[0]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace rare categories with mode.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with cleaned categorical columns
        """
        X = X.copy()
        for col in self.categorical_cols:
            if col in X.columns and col in self.valid_categories_:
                valid_cats = set(self.valid_categories_[col])
                unique_vals = set(X[col].unique())
                unseen = unique_vals - valid_cats

                if unseen:
                    logger.warning(
                        f"[CategoricalCleaner][DEBUG] Column '{col}' has unseen values: {unseen}. "
                        f"Valid categories: {valid_cats}"
                    )

        for col in self.categorical_cols:
            if col in X.columns and col in self.valid_categories_:
                valid_cats = self.valid_categories_[col]
                mode_val = self.mode_values_.get(col)

                if mode_val is not None:
                    mask = ~X[col].isin(valid_cats)
                    n_replaced = mask.sum()
                    if n_replaced > 0:
                        # Handle categorical dtype: add mode_val to categories if needed
                        if pd.api.types.is_categorical_dtype(X[col]):
                            if mode_val not in X[col].cat.categories:
                                X[col] = X[col].cat.add_categories([mode_val])
                        X.loc[mask, col] = mode_val
                        logger.info(f"[CategoricalCleaner] {col}: Replaced {n_replaced} rare values with mode")

                X[col] = X[col].astype("category")
                if hasattr(X[col].cat, "remove_unused_categories"):
                    X[col] = X[col].cat.remove_unused_categories()

        return X



class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Remove highly correlated features.

    This transformer removes features that are highly correlated with others
    to reduce multicollinearity and improve model stability.

    Attributes:
        columns_to_drop: List of column names to drop
    """

    def __init__(self, columns_to_drop: Optional[List[str]] = None):
        """
        Initialize FeatureSelector.

        Args:
            columns_to_drop: List of column names to remove
        """
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit transformer (no-op for this transformer).

        Args:
            X: Input DataFrame
            y: Target variable (unused)

        Returns:
            self: Fitted transformer
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop specified columns.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with selected features
        """
        cols_to_drop = [c for c in self.columns_to_drop if c in X.columns]
        if cols_to_drop:
            logger.info(f"[FeatureSelector] Dropping columns: {cols_to_drop}")
        return X.drop(columns=cols_to_drop, errors="ignore")


class DuplicateRemover(BaseEstimator, TransformerMixin):
    """
    Remove duplicate rows from dataset.

    This transformer identifies and removes exact duplicate rows to ensure
    data quality and prevent overfitting.
    """

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit transformer (no-op for this transformer).

        Args:
            X: Input DataFrame
            y: Target variable (unused)

        Returns:
            self: Fitted transformer
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with duplicates removed
        """
        initial = len(X)
        X = X.drop_duplicates().reset_index(drop=True)
        removed = initial - len(X)

        if removed > 0:
            logger.info(f"[DuplicateRemover] Removed {removed} duplicate rows")

        return X

