from argparse import ArgumentParser
from json import loads, dumps
from typing import List
from pathlib import Path

from numpy import asarray, nan
from pandas import read_csv, DataFrame
from mlflow.pyfunc import load_model

from config import (
    DATA,
    DEFAULT_MODEL_URI,
    setup_mlflow,
    to_numeric_df
)

def _align(df: DataFrame, features: List[str]) -> DataFrame:
    for c in features:
        if c not in df.columns:
            df[c] = nan
    return df[features]

def _read_input(args) -> DataFrame:
    if args.csv:
        return read_csv(args.csv)
    if args.json:
        payload = loads(Path(args.json).read_text(encoding = "utf-8"))
        rows = payload.get("rows", payload)
        return DataFrame(rows)

class Predictor:
    def __init__(self, model_uri: str):
        setup_mlflow()
        self.model = load_model(model_uri)
        self.features = DATA.features

    def predict(self, rows: DataFrame | list[dict]) -> list[float]:
        df = rows if isinstance(rows, DataFrame) else DataFrame(rows)
        df = _align(df.copy(), self.features)
        df = to_numeric_df(df)
        y = self.model.predict(df)
        return [float(v) for v in asarray(y).ravel().tolist()]
    
def main():
    ap = ArgumentParser(description = "Predict from an MLflow model.")
    ap.add_argument("--model-uri", default = DEFAULT_MODEL_URI, help = "MLflow model URI.")
    ap.add_argument("--csv", help = "Input CSV")
    ap.add_argument("--json", help = "JSON file (instead of CSV)")
    ap.add_argument("--out", help = "Output file for predictions (JSON). Defaults to stdout.")
    args = ap.parse_args()

    if not args.model_uri:
        raise SystemExit("No model URI provided. DEFAULT_MODEL_URI not set.")

    df = _read_input(args)
    predictor = Predictor(model_uri = args.model_uri)
    preds = predictor.predict(df)

    out = {"predictions": preds}
    out_s = dumps(out, indent = 2)
    if args.out:
        Path(args.out).write_text(out_s + "\n", encoding = "utf-8")
    else:
        print(out_s)

if __name__ == "__main__":
    main()