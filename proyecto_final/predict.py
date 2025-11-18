"""
Main prediction script with complete preprocessing pipeline.

This script loads trained models and preprocessing pipelines to make
predictions on new data.

Usage:
    python predict.py --input data.csv --target heating --output predictions.json
    python predict.py --input data.csv --target both --output predictions.json
"""

from argparse import ArgumentParser
from json import loads, dumps
from pathlib import Path

import pandas as pd
from loguru import logger

from proyecto_final.config import MODELS_DIR
from proyecto_final.modeling.predictor import Predictor, MultiTargetPredictor
from sklearn.pipeline import Pipeline
from loguru import logger
import joblib
import pandas as pd

def load_input_data(csv_path: str = None, json_path: str = None) -> pd.DataFrame:
    """
    Load input data from CSV or JSON file.

    Args:
        csv_path: Path to CSV file
        json_path: Path to JSON file

    Returns:
        DataFrame with input data

    Raises:
        ValueError: If neither csv_path nor json_path is provided
    """
    if csv_path:
        logger.info(f"Loading input from CSV: {csv_path}")
        return pd.read_csv(csv_path)
    elif json_path:
        logger.info(f"Loading input from JSON: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = loads(f.read())
        rows = data.get("rows", data)
        return pd.DataFrame(rows)
    else:
        raise ValueError("Either --csv or --json must be provided")


def main():
    """
    Main prediction pipeline.

    Steps:
        1. Parse command-line arguments
        2. Load input data
        3. Load models and preprocessing pipelines
        4. Make predictions
        5. Save results
    """
    ap = ArgumentParser(description="Make predictions using trained models.")
    ap.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file"
    )
    ap.add_argument(
        "--target",
        choices=["heating", "cooling", "both"],
        default="both",
        help="Target to predict (default: both)"
    )
    ap.add_argument(
        "--model-name",
        default="xgboost",
        help="Model name prefix (default: xgboost)"
    )
    ap.add_argument(
        "--output",
        help="Output file for predictions (JSON). If not provided, prints to stdout"
    )
    args = ap.parse_args()

    logger.info("=" * 70)
    logger.info("ENERGY EFFICIENCY MODEL PREDICTION PIPELINE")
    logger.info("=" * 70)

    df_input = load_input_data(csv_path=args.input)
    logger.info(f"Loaded {len(df_input)} samples")

    preprocessing_pipeline_path = MODELS_DIR / "initial_cleaning_pipeline.joblib"
    transformer_path = MODELS_DIR / "encoding_scaling_transformer.joblib"



    pipe = joblib.load(MODELS_DIR / "initial_cleaning_pipeline.joblib")
    df = df_input.copy()

    if isinstance(pipe, Pipeline):
        X_tmp = df.copy()
        for name, step in pipe.named_steps.items():
            logger.info(f"Applying step: {name} ({type(step).__name__})")
            try:
                X_tmp = step.transform(X_tmp)
            except TypeError as e:
                logger.error(f"TypeError in step '{name}': {e}")
                # Inspect problematic columns
                for col in X_tmp.columns:
                    if pd.api.types.is_categorical_dtype(X_tmp[col]):
                        cats = set(X_tmp[col].cat.categories)
                        vals = set(X_tmp[col].unique())
                        bad_vals = vals - cats
                        if bad_vals:
                            logger.error(f"Column '{col}' has values not in categories: {bad_vals}")
                raise
    else:
        # no Pipeline, it’s a single transformer
        pipe.transform(df)

    if args.target == "both":
        logger.info("[Predict] Using MultiTargetPredictor for both targets")

        model_paths = {
            "heating": MODELS_DIR / f"{args.model_name}_heating_model.joblib",
            "cooling": MODELS_DIR / f"{args.model_name}_cooling_model.joblib"
        }

        predictor = MultiTargetPredictor(
            model_paths=model_paths,
            preprocessing_pipeline_path=preprocessing_pipeline_path,
            transformer_path=transformer_path
        )

        predictions = predictor.predict_dict(df_input)

    else:
        logger.info(f"[Predict] Using single Predictor for target: {args.target}")

        model_path = MODELS_DIR / f"{args.model_name}_{args.target}_model.joblib"

        predictor = Predictor(
            model_path=model_path,
            preprocessing_pipeline_path=preprocessing_pipeline_path,
            transformer_path=transformer_path
        )

        predictions_series = predictor.predict(df_input)
        predictions = [{args.target: float(p)} for p in predictions_series]

    logger.info(f"[Predict] Generated {len(predictions)} predictions")

    output_data = {
        "target": args.target,
        "num_predictions": len(predictions),
        "predictions": predictions
    }

    output_json = dumps(output_data, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json + "\n", encoding="utf-8")
        logger.info(f"[Save] Predictions saved to: {args.output}")
    else:
        print(output_json)

    logger.info("=" * 70)
    logger.info("PREDICTION PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()