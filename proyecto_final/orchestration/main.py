from typing import List, Literal, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path

from predictor import Predictor, MultiTargetPredictor

# -----------------------------------------------------------------------------
# Metadata: API and model artifact info (for portability + documentation)
# -----------------------------------------------------------------------------
API_VERSION = "0.1.0"
MODEL_NAME = "xgboost"
MODEL_VERSION = "0.1.0"

# Logical URI for model
MODEL_ARTIFACT_URI = f"models:/energy-efficiency/{MODEL_NAME}/{MODEL_VERSION}"

# In Docker, models are packaged under /app/models (see Dockerfile)
MODELS_DIR = Path("/app/models")
BEST_MODEL = "xgboost"

# -----------------------------------------------------------------------------
# Request / Response schemas
# -----------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """
    Input schema for POST /predict.
    'rows' must be list of objects with the same feature names
    used during model training.
    """
    target: Literal["heating", "cooling", "both"] = Field(
        "both",
        description="Which target to predict: heating, cooling or both."
    )
    rows: List[Dict[str, Any]] = Field(
        ...,
        description="List of input samples as dictionaries."
    )

    class Config:
        schema_extra = {
            "example": {
                "target": "both",
                "rows": [
                    {
                        "X1": 0.98,
                        "X2": 514.5,
                        "X3": 294.0,
                        "X4": 110.25,
                        "X5": 7.0,
                        "X6": 2.0,
                        "X7": 0.0,
                        "X8": 0.0
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """
    Output schema for POST /predict.
    """
    target: str = Field(..., description="Target that was predicted.")
    num_predictions: int = Field(..., description="Number of predictions.")
    predictions: List[Dict[str, float]] = Field(
        ..., description="List of prediction dictionaries per row."
    )
    model_artifact_uri: str = Field(
        ...,
        description="Versioned location of the model artifact.",
        example=MODEL_ARTIFACT_URI,
    )
    api_version: str = Field(
        ...,
        description="Version of this FastAPI service.",
        example=API_VERSION,
    )


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Energy Efficiency Prediction API",
    version=API_VERSION,
    description=(
        "FastAPI service to predict heating/cooling.\n\n"
        f"Artifact: {MODEL_ARTIFACT_URI}.\n"
        "Models (.joblib) are packaged inside Docker image in /app/models."
    ),
)


@app.on_event("startup")
def load_paths():
    """
    Initialize shared paths to preprocessing and model artifacts.
    All artifacts are packaged as .joblib files under /app/models.
    """
    app.state.preprocessing_pipeline_path = MODELS_DIR / "initial_cleaning_pipeline.joblib"
    app.state.transformer_path = MODELS_DIR / "encoding_scaling_transformer.joblib"


@app.get("/", summary="Healthcheck")
def read_root():
    """
    Simple healthcheck endpoint.
    """
    return {
        "message": "Energy efficiency model API is running.",
        "api_version": API_VERSION,
        "model_artifact_uri": MODEL_ARTIFACT_URI,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict heating/cooling energy demand",
    description=(
        "Takes a list of rows as input "
        "and returns predictions for heating, cooling or both."
    ),
)
def predict(request: PredictionRequest):
    # Basic validation
    if not request.rows:
        raise HTTPException(
            status_code=400,
            detail="Input 'rows' must contain at least one sample."
        )

    df_input = pd.DataFrame(request.rows)

    if df_input.empty:
        raise HTTPException(
            status_code=400,
            detail="Input 'rows' produced an empty DataFrame. "
                   "Check JSON structure."
        )

    preprocessing_pipeline_path = app.state.preprocessing_pipeline_path
    transformer_path = app.state.transformer_path

    if request.target == "both":
        # Multi-target mode -> use MultiTargetPredictor and two .joblib models
        model_paths = {
            "heating": MODELS_DIR / f"{BEST_MODEL}_heating_model.joblib",
            "cooling": MODELS_DIR / f"{BEST_MODEL}_cooling_model.joblib",
        }

        predictor = MultiTargetPredictor(
            model_paths=model_paths,
            preprocessing_pipeline_path=preprocessing_pipeline_path,
            transformer_path=transformer_path,
        )

        predictions = predictor.predict_dict(df_input)

    else:
        # Single target mode
        model_path = MODELS_DIR / f"{BEST_MODEL}_{request.target}_model.joblib"

        predictor = Predictor(
            model_path=model_path,
            preprocessing_pipeline_path=preprocessing_pipeline_path,
            transformer_path=transformer_path,
        )

        try:
            predictions_series = predictor.predict(df_input)
        except Exception as exc:
            # Basic error handling with a clear message
            raise HTTPException(
                status_code=500,
                detail=f"Error during prediction: {exc}",
            ) from exc

        predictions = [{request.target: float(p)} for p in predictions_series]

    return PredictionResponse(
        target=request.target,
        num_predictions=len(predictions),
        predictions=predictions,
        model_artifact_uri=MODEL_ARTIFACT_URI,
        api_version=API_VERSION,
    )


# Optional: local dev
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)