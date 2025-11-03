"""
Model training orchestrator with MLflow integration.

This module provides the ModelTrainer class that handles model training,
hyperparameter optimization, evaluation, and MLflow tracking for the
Energy Efficiency regression task.
"""

from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import joblib
import mlflow
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import traceback

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not installed. XGBoost models will not be available.")

from proyecto_final.config import TrainConfig


class ModelTrainer:
    """
    Orchestrates model training with MLflow tracking.

    This class handles the complete training workflow including:
    - Model instantiation
    - Hyperparameter optimization (Grid/Random Search)
    - Model training
    - Evaluation metrics computation
    - MLflow experiment tracking
    - Model and artifact saving

    Attributes:
        config: Training configuration
        experiment_name: MLflow experiment name
        models_dir: Directory for saving models
    """

    def __init__(
        self,
        config: TrainConfig,
        experiment_name: str,
        models_dir: Path
    ):
        """
        Initialize ModelTrainer.

        Args:
            config: Training configuration with model params, split config, HPO config
            experiment_name: Name for MLflow experiment
            models_dir: Directory path for saving model artifacts
        """
        self.config = config
        self.experiment_name = experiment_name
        self.models_dir = models_dir

        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Respect project-level MLflow configuration (set via proyecto_final.config.setup_mlflow).
        # Only fall back to a local `mlruns/` file store if no tracking URI is configured.
        try:
            from proyecto_final.config import MLFLOW, PROJ_ROOT
        except Exception:
            MLFLOW = None
            PROJ_ROOT = Path(__file__).resolve().parents[2]

        # If MLFLOW.tracking_uri is empty or falsy, use a local file-backed mlruns directory
        if not (MLFLOW and getattr(MLFLOW, "tracking_uri", None)):
            repo_root = PROJ_ROOT
            mlruns_dir = repo_root / "mlruns"
            mlruns_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(f"file://{mlruns_dir.resolve()}")

        mlflow.set_experiment(self.experiment_name)
        logger.info(f"[Trainer] Initialized with experiment: {self.experiment_name}")

    def _create_estimator(self) -> Any:
        """
        Create sklearn estimator based on configuration.

        Returns:
            Configured sklearn estimator

        Raises:
            ValueError: If library is not supported
            RuntimeError: If XGBoost is requested but not installed
        """
        library = self.config.model.library
        params = self.config.model.params

        if library == "sklearn":
            return RandomForestRegressor(**params)
        elif library == "xgboost_sklearn":
            if not HAS_XGB:
                raise RuntimeError("XGBoost not installed")
            return XGBRegressor(**params)
        else:
            raise ValueError(f"Unsupported library: {library}")

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> Dict[str, float]:
        """
        Compute regression metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dictionary with MAE, RMSE, and R² metrics
        """
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(mean_squared_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }

    def _run_hyperparameter_optimization(
        self,
        estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[Any, Optional[Dict]]:
        """
        Perform hyperparameter optimization.

        Args:
            estimator: Base sklearn estimator
            X_train: Training features
            y_train: Training target

        Returns:
            Tuple of (best_estimator, hpo_results)
            hpo_results is None if HPO is disabled
        """
        hpo = self.config.hpo

        if not hpo.enabled:
            logger.info("[HPO] Hyperparameter optimization disabled")
            estimator.fit(X_train, y_train)
            return estimator, None

        logger.info(f"[HPO] Starting {hpo.search} search with {hpo.cv}-fold CV")

        param_grid = hpo.param_grid or {}

        if hpo.search == "grid":
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=hpo.cv,
                scoring=hpo.scoring,
                n_jobs=hpo.n_jobs,
                refit=True,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=hpo.n_iter,
                cv=hpo.cv,
                scoring=hpo.scoring,
                n_jobs=hpo.n_jobs,
                random_state=self.config.split.random_state,
                refit=True,
                verbose=1
            )

        search.fit(X_train, y_train)

        logger.info(f"[HPO] Best score: {search.best_score_:.4f}")
        logger.info(f"[HPO] Best params: {search.best_params_}")

        hpo_results = {
            "best_cv_score": float(search.best_score_),
            "best_params": search.best_params_
        }

        return search.best_estimator_, hpo_results

    def train_single_target(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        target_name: str,
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train model for a single target variable.

        This method orchestrates:
        1. Model creation
        2. Hyperparameter optimization (if enabled)
        3. Training
        4. Evaluation
        5. MLflow logging
        6. Model saving

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            target_name: Name of target variable (for logging)
            run_name: Optional custom run name

        Returns:
            Dictionary with metrics and model info

        Example:
            >>> results = trainer.train_single_target(
            ...     X_train, X_test, y_train, y_test,
            ...     target_name="heating"
            ... )
        """
        logger.info("=" * 70)
        logger.info(f"TRAINING MODEL FOR TARGET: {target_name}")
        logger.info("=" * 70)

        run_name = run_name or f"{self.config.model.name}-{target_name}"

        with mlflow.start_run(run_name=run_name) as run:
            try:
                mlflow.set_tag("target", target_name)
                mlflow.set_tag("model_library", self.config.model.library)

                base_estimator = self._create_estimator()

                best_estimator, hpo_results = self._run_hyperparameter_optimization(
                    base_estimator, X_train, y_train
                )

                y_pred = best_estimator.predict(X_test)
                metrics = self._compute_metrics(y_test, y_pred)

                logger.info(f"[Metrics] MAE: {metrics['mae']:.4f}")
                logger.info(f"[Metrics] RMSE: {metrics['rmse']:.4f}")
                logger.info(f"[Metrics] R²: {metrics['r2']:.4f}")

                flat_params = {
                    f"model__{k}": v
                    for k, v in self.config.model.params.items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                }

                mlflow.log_params({
                    **flat_params,
                    "test_size": self.config.split.test_size,
                    "random_state": self.config.split.random_state,
                    "hpo_enabled": self.config.hpo.enabled
                })

                if hpo_results:
                    mlflow.log_params({
                        f"hpo_best__{k}": v
                        for k, v in hpo_results["best_params"].items()
                    })
                    mlflow.log_metric("hpo_best_cv_score", hpo_results["best_cv_score"])

                mlflow.log_metrics(metrics)

                try:
                    from mlflow.models.signature import infer_signature
                    signature = infer_signature(X_test, y_pred)
                    input_example = X_test.head(2)
                except Exception as e:
                    logger.warning(f"Could not infer signature: {e}")
                    signature = None
                    input_example = None

                # Ensure we have a local copy of the model first (always writable)
                model_path = self.models_dir / f"{self.config.model.name}_{target_name}_model.joblib"
                joblib.dump(best_estimator, model_path)
                logger.info(f"[Save] Local model copy saved to: {model_path}")

                # Try to use mlflow's sklearn model logger. If the configured
                # MLflow artifact store is not writable from this environment
                # (common when a remote server has an artifact root like '/root'),
                # fall back to logging the saved file as an artifact.
                print("artifact_uri:", mlflow.get_artifact_uri()+",.")
                try:
                    mlflow.sklearn.log_model(
                        sk_model=best_estimator,
                        # artifact_path="model",
                        # name=self.config.model.name,
                        signature=signature,
                        input_example=input_example
                    )
                except Exception as e:
                    logger.warning(f"mlflow.sklearn.log_model failed: {e}. Falling back to manual artifact logging.")
                    try:
                        mlflow.log_artifact(str(model_path), artifact_path="model")
                    except Exception as e2:
                        logger.warning(f"mlflow.log_artifact also failed: {e2}. Model is available locally at {model_path}.")

                run_id = run.info.run_id
                logger.info(f"[MLflow] Run ID: {run_id}")
            except Exception as e:
                mlflow.set_tag("train_status", "error")
                mlflow.log_param("failure_reason", str(e)[:250])
                mlflow.log_text(traceback.format_exc(), f"failures/{self.config.model.name}_trace.txt")
                # re-raise if you want the outer pipeline to fail;
                # if you want the parent run to succeed even if a child fails, DO NOT re-raise here.
                raise

        return {
             "target": target_name,
             "metrics": metrics,
             "hpo_results": hpo_results,
             "run_id": run_id,
             "model_path": str(model_path)
         }

    def train_all_targets(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        target_mapping: Dict[str, str]
    ) -> Dict[str, Dict]:
        """
        Train models for all target variables.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets (all)
            y_test: Test targets (all)
            target_mapping: Dict mapping target names to column names
                           e.g., {"heating": "Y1", "cooling": "Y2"}

        Returns:
            Dictionary with results for each target

        Example:
            >>> results = trainer.train_all_targets(
            ...     X_train, X_test, y_train, y_test,
            ...     target_mapping={"heating": "Y1", "cooling": "Y2"}
            ... )
        """
        logger.info("=" * 70)
        logger.info("TRAINING MODELS FOR ALL TARGETS")
        logger.info("=" * 70)

        all_results = {}

        for label, col in target_mapping.items():
            results = self.train_single_target(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train[col],
                y_test=y_test[col],
                target_name=label
            )
            all_results[label] = results

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE FOR ALL TARGETS")
        logger.info("=" * 70)

        for label, res in all_results.items():
            logger.info(f"{label}: MAE={res['metrics']['mae']:.4f}, "
                       f"RMSE={res['metrics']['rmse']:.4f}, "
                       f"R²={res['metrics']['r2']:.4f}")

        return all_results

