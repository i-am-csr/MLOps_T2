"""
Model prediction orchestrator with preprocessing pipeline.

This module provides the Predictor class for making predictions using
trained models and preprocessing pipelines.
"""

from pathlib import Path
from typing import Union, List, Dict

import joblib
import pandas as pd
from loguru import logger


class Predictor:
    """
    Handles predictions using trained models and preprocessing pipelines.

    This class loads saved preprocessing artifacts and trained models to
    make predictions on new data, ensuring consistent preprocessing.

    Attributes:
        model_path: Path to saved model file
        preprocessing_pipeline_path: Path to initial cleaning pipeline
        transformer_path: Path to encoding/scaling transformer
        model: Loaded sklearn model
        preprocessing_pipeline: Loaded preprocessing pipeline
        transformer: Loaded encoding/scaling transformer
    """

    def __init__(
        self,
        model_path: Path,
        preprocessing_pipeline_path: Path,
        transformer_path: Path
    ):
        """
        Initialize Predictor with model and preprocessing artifacts.

        Args:
            model_path: Path to trained model (.joblib)
            preprocessing_pipeline_path: Path to preprocessing pipeline (.joblib)
            transformer_path: Path to encoding/scaling transformer (.joblib)

        Example:
            >>> predictor = Predictor(
            ...     model_path=Path("models/xgboost_heating_model.joblib"),
            ...     preprocessing_pipeline_path=Path("models/initial_cleaning_pipeline.joblib"),
            ...     transformer_path=Path("models/encoding_scaling_transformer.joblib")
            ... )
        """
        self.model_path = model_path
        self.preprocessing_pipeline_path = preprocessing_pipeline_path
        self.transformer_path = transformer_path

        logger.info(f"[Predictor] Loading model from: {model_path}")
        self.model = joblib.load(model_path)

        logger.info(f"[Predictor] Loading preprocessing pipeline from: {preprocessing_pipeline_path}")
        self.preprocessing_pipeline = joblib.load(preprocessing_pipeline_path)

        logger.info(f"[Predictor] Loading transformer from: {transformer_path}")
        self.transformer = joblib.load(transformer_path)

        logger.info("[Predictor] All artifacts loaded successfully")

    # def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Apply preprocessing pipeline to input data.
    #
    #     This method applies both the initial cleaning pipeline and the
    #     encoding/scaling transformer to prepare data for prediction.
    #
    #     Args:
    #         X: Raw input DataFrame
    #
    #     Returns:
    #         Preprocessed DataFrame ready for prediction
    #
    #     Example:
    #         >>> X_processed = predictor.preprocess(X_raw)
    #     """
    #     logger.info(f"[Preprocess] Input shape: {X.shape}")
    #
    #     X_cleaned = self.preprocessing_pipeline.transform(X)
    #     logger.info(f"[Preprocess] After cleaning: {X_cleaned.shape}")
    #
    #     X_transformed = self.transformer.transform(X_cleaned)
    #     logger.info(f"[Preprocess] After encoding/scaling: {X_transformed.shape}")
    #
    #     if hasattr(X_transformed, "shape") and len(X_transformed.shape) == 2:
    #         feature_names = self.transformer.get_feature_names_out()
    #         X_transformed = pd.DataFrame(
    #             X_transformed,
    #             columns=feature_names,
    #             index=X_cleaned.index
    #         )
    #
    #     return X_transformed

    # python
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"[Preprocess] Input shape: {getattr(X, 'shape', None)}")
        X = X.copy()

        # Convert ALL columns to non-categorical dtypes upfront
        for col in X.columns:
            if X[col].dtype == 'category':
                X[col] = X[col].astype(object)

        def _force_non_categorical(df):
            """Recursively ensure no categorical dtypes remain."""
            if isinstance(df, pd.DataFrame):
                for col in df.select_dtypes(include=['category']).columns:
                    df[col] = df[col].astype(object)
            return df

        def _as_dataframe(arr, index):
            if hasattr(arr, "shape") and len(arr.shape) == 2:
                try:
                    cols = self.transformer.get_feature_names_out()
                except Exception:
                    cols = [f"f_{i}" for i in range(arr.shape[1])]
                return pd.DataFrame(arr, columns=cols, index=index)
            return arr

        # Initialize variables to avoid UnboundLocalError
        X_cleaned = None
        X_transformed = None

        # Apply preprocessing with categorical error handling
        max_retries = 3
        for attempt in range(max_retries):
            try:
                X_cleaned = self.preprocessing_pipeline.transform(X)

                # Force non-categorical if result is DataFrame
                X_cleaned = _force_non_categorical(X_cleaned)

                logger.info(f"[Preprocess] After cleaning: {getattr(X_cleaned, 'shape', None)}")

                X_transformed = self.transformer.transform(X_cleaned)
                logger.info(f"[Preprocess] After encoding/scaling: {getattr(X_transformed, 'shape', None)}")

                break  # Success, exit retry loop

            except (TypeError, ValueError) as e:
                if "Cannot setitem on a Categorical with a new category" in str(e):
                    logger.warning(f"[Preprocess] Attempt {attempt + 1}: Categorical error: {e}")

                    if attempt == max_retries - 1:
                        # Last resort: convert everything to object/numeric
                        logger.warning("[Preprocess] Final attempt: converting all non-numeric to object")
                        for col in X.columns:
                            if not pd.api.types.is_numeric_dtype(X[col]):
                                X[col] = X[col].astype(str).astype(object)
                            else:
                                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                    else:
                        # Progressive fixes
                        logger.warning(f"[Preprocess] Retry {attempt + 1}: forcing object dtypes")
                        for col in X.columns:
                            if not pd.api.types.is_numeric_dtype(X[col]):
                                X[col] = X[col].astype(object)
                else:
                    raise

        # Check if preprocessing succeeded
        if X_transformed is None:
            raise RuntimeError("Preprocessing failed after all retry attempts")

        # Ensure DataFrame output with feature names and original index
        result = _as_dataframe(X_transformed,
                               index=(X_cleaned.index if isinstance(X_cleaned, pd.DataFrame) else X.index))
        return result

    def predict(self, X: Union[pd.DataFrame, List[Dict]]) -> pd.Series:
        """
        Make predictions on new data.

        This method handles the complete prediction workflow:
        1. Convert input to DataFrame if needed
        2. Apply preprocessing pipeline
        3. Make predictions
        4. Return results

        Args:
            X: Input data (DataFrame or list of dicts)

        Returns:
            Predictions as pandas Series

        Example:
            >>> predictions = predictor.predict(X_new)
            >>> print(predictions)
        """
        if isinstance(X, list):
            X = pd.DataFrame(X)

        logger.info(f"[Predict] Making predictions for {len(X)} samples")

        X_processed = self.preprocess(X)

        predictions = self.model.predict(X_processed)

        logger.info(f"[Predict] Generated {len(predictions)} predictions")

        return pd.Series(predictions, index=X.index, name="prediction")

    def predict_dict(self, X: Union[pd.DataFrame, List[Dict]]) -> List[Dict]:
        """
        Make predictions and return as list of dictionaries.

        Convenience method that returns predictions in dictionary format
        for easier serialization (e.g., JSON).

        Args:
            X: Input data (DataFrame or list of dicts)

        Returns:
            List of dictionaries with predictions

        Example:
            >>> results = predictor.predict_dict(X_new)
            >>> print(results)
            [{"prediction": 15.5}, {"prediction": 20.3}, ...]
        """
        predictions = self.predict(X)
        return [{"prediction": float(p)} for p in predictions]


class MultiTargetPredictor:
    """
    Handles predictions for multiple targets (heating and cooling).

    This class manages multiple Predictor instances, one for each target
    variable, enabling simultaneous predictions.

    Attributes:
        predictors: Dictionary mapping target names to Predictor instances
    """

    def __init__(
        self,
        model_paths: Dict[str, Path],
        preprocessing_pipeline_path: Path,
        transformer_path: Path
    ):
        """
        Initialize MultiTargetPredictor.

        Args:
            model_paths: Dict mapping target names to model paths
                        e.g., {"heating": Path("..."), "cooling": Path("...")}
            preprocessing_pipeline_path: Path to preprocessing pipeline
            transformer_path: Path to encoding/scaling transformer

        Example:
            >>> predictor = MultiTargetPredictor(
            ...     model_paths={
            ...         "heating": Path("models/xgboost_heating_model.joblib"),
            ...         "cooling": Path("models/xgboost_cooling_model.joblib")
            ...     },
            ...     preprocessing_pipeline_path=Path("models/initial_cleaning_pipeline.joblib"),
            ...     transformer_path=Path("models/encoding_scaling_transformer.joblib")
            ... )
        """
        self.predictors = {}

        for target, model_path in model_paths.items():
            logger.info(f"[MultiTarget] Initializing predictor for target: {target}")
            self.predictors[target] = Predictor(
                model_path=model_path,
                preprocessing_pipeline_path=preprocessing_pipeline_path,
                transformer_path=transformer_path
            )

        logger.info(f"[MultiTarget] Initialized {len(self.predictors)} predictors")

    def predict(self, X: Union[pd.DataFrame, List[Dict]]) -> pd.DataFrame:
        """
        Make predictions for all targets.

        Args:
            X: Input data (DataFrame or list of dicts)

        Returns:
            DataFrame with predictions for all targets

        Example:
            >>> predictions = multi_predictor.predict(X_new)
            >>> print(predictions)
                   heating  cooling
            0        15.5     21.3
            1        20.3     28.1
        """
        if isinstance(X, list):
            X = pd.DataFrame(X)

        logger.info(f"[MultiTarget] Making predictions for {len(X)} samples across {len(self.predictors)} targets")

        predictions = {}
        for target, predictor in self.predictors.items():
            predictions[target] = predictor.predict(X)

        result = pd.DataFrame(predictions)

        logger.info("[MultiTarget] Predictions complete")

        return result


    def predict_dict(self, X: Union[pd.DataFrame, List[Dict]]) -> List[Dict]:
        """
        Make predictions and return as list of dictionaries.

        Args:
            X: Input data (DataFrame or list of dicts)

        Returns:
            List of dictionaries with predictions for all targets

        Example:
            >>> results = multi_predictor.predict_dict(X_new)
            >>> print(results)
            [{"heating": 15.5, "cooling": 21.3}, {"heating": 20.3, "cooling": 28.1}]
        """
        predictions_df = self.predict(X)
        return predictions_df.to_dict(orient="records")

