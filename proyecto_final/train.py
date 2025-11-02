from argparse import ArgumentParser
from json import dumps
from typing import Dict, Optional

from numpy import sqrt
from pandas import DataFrame, read_csv, to_numeric
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from config import (
    DATA,
    MLFLOW,
    TrainConfig,
    train_config_from_yaml,
    nowstamp,
    setup_mlflow,
    to_numeric_df
)

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from mlflow import start_run, log_params, log_metrics
from mlflow.sklearn import log_model
from mlflow.models.signature import infer_signature

def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}

def make_estimator(library: str, params: Dict):
    if library == "sklearn":
        return RandomForestRegressor(**params)
    elif library == "xgboost_sklearn":
        if not _HAS_XGB:
            raise RuntimeError("XGBoost not installed")
        return XGBRegressor(**params)
    else:
        raise ValueError(f"Unsupported library: {library}")
    
class Trainer:
    def __init__(self, cfg: TrainConfig, run_name: Optional[str] = None):
        self.cfg = cfg
        self.stamp = run_name or nowstamp()
        setup_mlflow()

    def _load_df(self) -> DataFrame:
        df = read_csv(DATA.csv_path)
        req = DATA.features + list(DATA.targets.values())
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in input: {missing}")
        return df
    
    def _split(self, df: DataFrame, target_col: str):
        X = to_numeric_df(df[DATA.features])
        y = to_numeric(df[target_col], errors="coerce")
        return train_test_split(
            X, y,
            test_size=self.cfg.split.test_size,
            random_state=self.cfg.split.random_state
        )
    
    def _train(self, est, Xtr, ytr):
        h = self.cfg.hpo
        if not h.enabled:
            est.fit(Xtr, ytr)
            return est, None

        grid = h.param_grid or {}
        if h.search == "grid":
            search = GridSearchCV(
                est, param_grid = grid, cv = h.cv,
                scoring = h.scoring, n_jobs = h.n_jobs, refit = True, verbose = 1
            )
        else:
            search = RandomizedSearchCV(
                est, param_distributions = grid, n_iter = h.n_iter, cv = h.cv,
                scoring = h.scoring, n_jobs = h.n_jobs, random_state = self.cfg.split.random_state,
                refit = True, verbose = 1
            )
        search.fit(Xtr, ytr)
        return search.best_estimator_, {
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
        }
    
    def train_target(self, label: str, target_col: str):
        df = self._load_df()
        Xtr, Xte, ytr, yte = self._split(df, target_col)

        base = make_estimator(self.cfg.model.library, self.cfg.model.params)
        final_est, hpo = self._train(base, Xtr, ytr)
        yhat = final_est.predict(Xte)
        met = regression_metrics(yte, yhat)

        run_name = f"{self.cfg.model.name}-{label}:{self.stamp}"
        with start_run(run_name = run_name):
            flat_params = {f"model__{k}": v for k, v in self.cfg.model.params.items()
                           if isinstance(v, (int, float, str, bool, type(None)))}
            log_params({
                **flat_params,
                "library": self.cfg.model.library,
                "model_name": self.cfg.model.name,
                "target": target_col,
                "split_test_size": self.cfg.split.test_size,
                "split_random_state": self.cfg.split.random_state,
            })
            if hpo:
                log_params({f"hpo_best__{k}": v for k, v in hpo["best_params"].items()})
                log_metrics("hpo_best_cv_score", hpo["best_score"])
            log_metrics(met)

            try:
                sig = infer_signature(Xte, yhat)
                example = Xte.head(2)
            except Exception:
                sig, example = None, None

            log_model(
                sk_model = final_est,
                artifact_path = "model",
                signature = sig,
                input_example = example,
                registered_model_name = (MLFLOW.register_name or None),
            )

        print(f"Model run: {run_name}")
        print(f"Metrics: {dumps(met)}")

        return {"metrics": met, "hpo": hpo}

    def train_all(self):
        results = {}
        for label, target_col in DATA.targets.items():
            results[label] = self.train_target(label, target_col)
        return results

def main():
    ap = ArgumentParser(description="Train model.")
    ap.add_argument("--config", required=True, help="YAML with model/split/hpo(optional) parameters only")
    ap.add_argument("--run-name", default=None, help="Optional run name (else timestamp)")
    args = ap.parse_args()

    cfg = train_config_from_yaml(args.config)
    trainer = Trainer(cfg, run_name=args.run_name)
    out = trainer.train_all()
    print(dumps(out, indent=2))

if __name__ == "__main__":
    main()