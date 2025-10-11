import argparse, json, os, sys, joblib
import numpy as np
import pandas as pd

# TODO: Move variables to config.py
MODEL_DIR = "../models/xgboost-heating/2025-10-11_21-20-05"

def load_signature(model_dir: str):
    sig_path = os.path.join(model_dir, "signature.json")
    if not os.path.exists(sig_path):
        raise FileNotFoundError(f"signature.json not found in {model_dir}")
    with open(sig_path, "r") as f:
        return json.load(f)

def load_model(model_dir: str):
    mdl_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(mdl_path):
        raise FileNotFoundError(f"model.joblib not found in {model_dir}")
    return joblib.load(mdl_path)

def read_input(args):
    if args.csv:
        return pd.read_csv(args.csv)
    if args.json:
        with open(args.json, "r") as f:
            payload = json.load(f)
        rows = payload.get("rows", payload)
        return pd.DataFrame(rows)
    
    payload = json.load(sys.stdin)
    rows = payload.get("rows", payload)
    return pd.DataFrame(rows)

def align_columns(df: pd.DataFrame, feature_cols: list[str]):
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[feature_cols]

def predict(model_dir: str, input_df: pd.DataFrame):
    sig = load_signature(model_dir)
    if sig.get("task", "regression") != "regression":
        raise ValueError("This predict script expects a regression model.")
    
    #TODO: A CSV with preprocessed data is expected. Add features engineering
    # later.
    feat_cols = sig["feature_cols"]
    df = align_columns(input_df.copy(), feat_cols)

    model = load_model(model_dir)
    y = model.predict(df)
    return [float(v) for v in np.asarray(y).ravel().tolist()]

def main():
    ap = argparse.ArgumentParser(description="Predict with a saved regression model (no preprocessing).")
    ap.add_argument("--csv", help="Input CSV with feature columns matching training signature.")
    ap.add_argument("--json", help="Input JSON: {'rows': [ {feat: val}, ... ]} or a list of dicts.")
    ap.add_argument("--out", help="Optional output file to write JSON predictions.")
    args = ap.parse_args()

    #TODO: Add input validation
    df = read_input(args)
    preds = predict(MODEL_DIR, df)
    out_obj = {"predictions": preds}
    out_s = json.dumps(out_obj, indent=2)

    if args.out:
        with open(args.out, "w") as f:
            f.write(out_s + "\n")
        print(f"[OK] wrote predictions to {args.out}")
    else:
        print(out_s)

if __name__ == "__main__":
    main()
