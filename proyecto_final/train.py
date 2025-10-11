import argparse, json, os, sys, time, yaml, joblib, platform
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

PROCESSED_DATA_PATH = "../data/processed/energy_efficiency_prepared.csv"
MODELS_DIR = "../models/"

FEATURES = ["X1","X3","X5","X7","X6_3","X6_4","X6_5","X8_1","X8_2","X8_3","X8_4","X8_5"]

TARGET_HEATING = "Y1" # Heating_Load
TARGET_COOLING = "Y2" # Cooling_Load

def nowstamp():
    return time.strftime("%Y-%m-%d_%H-%M-%S")

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_env(path):
    try:
        import sklearn, pandas as _pd
        lines = [
            f"python: {platform.python_version()}",
            f"platform: {platform.platform()}",
            f"pandas: {_pd.__version__}",
            f"scikit-learn: {sklearn.__version__}",
            f"xgboost: {xgb.__version__ if _HAS_XGB else 'not-installed'}",
            f"joblib: {joblib.__version__}",
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines))
    except Exception as e:
        with open(path, "w") as f:
            f.write(f"env capture failed: {e}\n")

def regression_metrics(y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}

# TODO: Create class to send parameters
def train_model(X, y, test_size, seed, model_name, stamp, params, lib):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    run_dir = os.path.join(MODELS_DIR, model_name, stamp)
    os.makedirs(run_dir, exist_ok=True)

    if lib == "sklearn":
        reg = RandomForestRegressor(**params)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        joblib.dump(reg, os.path.join(run_dir, "model.joblib"))
    elif lib == "xgboost_sklearn":
        if not _HAS_XGB:
            raise RuntimeError("xgboost not installed but library='xgboost_sklearn' in config.")

        reg = xgb.XGBRegressor(**params)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        joblib.dump(reg, os.path.join(run_dir, "model.joblib"))
    else:
        raise ValueError(f"Unsupported library: {lib}")

    met = regression_metrics(y_test, y_pred)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(met, f, indent=2)

    print(f"[OK] run saved to: {run_dir}")
    print(f"Test metrics: {met}")

    return met, run_dir

def save_run_data(run_dir, cfg, split_cfg, stamp, target, model_name, met):
    with open(os.path.join(run_dir, "params.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(run_dir, "split.yaml"), "w") as f:
        yaml.safe_dump(split_cfg, f, sort_keys=False)

    signature = {"feature_cols": FEATURES, "target": target, "task": "regression"}
    with open(os.path.join(run_dir, "signature.json"), "w") as f:
        json.dump(signature, f, indent=2)

    with open(os.path.join(run_dir, "model_card.md"), "w") as f:
        f.write(f"# {model_name} ({stamp})\n\n")
        f.write(f"- Task: regression\n")
        f.write(f"- Data: {PROCESSED_DATA_PATH}\n")
        f.write(f"- Test metrics: {json.dumps(met)}\n")

    save_env(os.path.join(run_dir, "environment.txt"))

def main(cfg_path, split_path, run_name=None):
    cfg = load_yaml(cfg_path)
    split_cfg = load_yaml(split_path)["split"]

    model_name = cfg["model"]["name"]
    lib        = cfg["model"]["library"]
    params     = cfg["model"]["params"]

    test_size  = float(split_cfg.get("test_size", 0.2))
    seed       = int(split_cfg.get("random_state", 42))

    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: The file was not found at the path '{PROCESSED_DATA_PATH}'.")
        return 1

    #TODO: For now, assume input CSV is correct. Add schema validations later
    X = df[FEATURES]
    y_heating = df[TARGET_HEATING]
    y_cooling = df[TARGET_COOLING]

    stamp = run_name or nowstamp()

    model_name_h = f"{model_name}-heating"
    met_h, run_dir_h = train_model(
        X, y_heating, test_size, seed, model_name_h, stamp, params, lib
    )
    save_run_data(
        run_dir_h, cfg, split_cfg, stamp, TARGET_HEATING, model_name_h, met_h
    )

    model_name_c = f"{model_name}-cooling"
    met_c, run_dir_c = train_model(
        X, y_cooling, test_size, seed, model_name_c, stamp, params, lib
    )
    save_run_data(
        run_dir_c, cfg, split_cfg, stamp, TARGET_COOLING, model_name_c, met_c
    )
    
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Model YAML (e.g., configs/xgb.yaml)")
    ap.add_argument("--split",  required=True, help="Split YAML (e.g., configs/split.yaml)")
    ap.add_argument("--run-name", default=None, help="Optional name instead of timestamp")
    args = ap.parse_args()
    sys.exit(main(args.config, args.split, run_name=args.run_name))
