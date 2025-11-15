import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger

# Importa tu clase MultiTargetPredictor y los esquemas
from proyecto_final.modeling.predictor import MultiTargetPredictor
from proyecto_final.data.schemas import InputSchema, PredictionOutput, BatchPredictionOutput

# --- Configuración de Rutas ---

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATHS = {
    "heating": MODEL_DIR / "xgb_heating_load_model.joblib",
    "cooling": MODEL_DIR / "xgb_cooling_load_model.joblib"
}

PREPROCESSING_PIPELINE_PATH = MODEL_DIR / "preprocessing_pipeline.joblib"
TRANSFORMER_PATH = MODEL_DIR / "transformer.joblib"
# -----------------------------

# Crea la app de FastAPI
app = FastAPI(
    title="API de Predicción de Carga Energética",
    description="Proyecto final de MLOps. API para predecir Carga de Calefacción (Heating Load) y Carga de Refrigeración (Cooling Load).",
    version="1.0.0"
)

# Variable global para el predictor
predictor: MultiTargetPredictor | None = None

@app.on_event("startup")
async def startup_event():
    """Carga el MultiTargetPredictor al iniciar la aplicación."""
    global predictor
    try:
        logger.info("Iniciando carga de modelos y pipelines...")
        predictor = MultiTargetPredictor(
            model_paths=MODEL_PATHS,
            preprocessing_pipeline_path=PREPROCESSING_PIPELINE_PATH,
            transformer_path=TRANSFORMER_PATH
        )
        logger.info("Modelos y pipelines cargados exitosamente.")
    except Exception as e:
        logger.error(f"Error crítico al cargar el MultiTargetPredictor: {e}")
        # Si falla al iniciar, la app no será funcional
        raise RuntimeError(f"Fallo crítico: No se pudo cargar el modelo. {e}")

@app.get("/", summary="Endpoint de Salud")
def read_root():
    """Verifica que el servicio esté online."""
    return {"status": "ok", "message": "Servicio de MLOps funcionando"}

@app.post("/predict",
          response_model=PredictionOutput,
          summary="Realiza una predicción para una sola entrada")
def post_predict(data: InputSchema):
    """
    Recibe los 8 features de un edificio y retorna la predicción.

    - **data**: Un objeto JSON con los 8 features requeridos.

    Retorna:
    - Un objeto JSON con las predicciones 'heating' y 'cooling'.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado. Revisa los logs del servidor.")

    try:
        # Convierte la entrada Pydantic a un DataFrame de 1 fila
        input_df = pd.DataFrame([data.model_dump()])

        # Realiza la predicción (devuelve un DataFrame)
        predictions_df = predictor.predict(input_df)

        # Extrae la primera (y única) predicción
        result = predictions_df.to_dict(orient="records")[0]
        
        return PredictionOutput(heating=result["heating"], cooling=result["cooling"])

    except Exception as e:
        # Manejo de errores durante la predicción
        logger.error(f"Error durante la predicción individual: {e}")
        raise HTTPException(status_code=400, detail=f"Error durante la predicción: {str(e)}")

@app.post("/predict_batch",
          response_model=BatchPredictionOutput,
          summary="Realiza predicciones para un lote de entradas")
def post_predict_batch(data: List[InputSchema]):
    """
    Recibe una lista de objetos JSON con los 8 features y retorna las predicciones.

    - **data**: Una lista de objetos JSON, cada uno con 8 features.

    Retorna:
    - Un objeto JSON con una lista 'predictions' y un campo 'errors'.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado. Revisa los logs del servidor.")
    
    if not data:
         raise HTTPException(status_code=400, detail="No se proporcionaron datos de entrada.")

    try:
        # Convierte la lista de entradas Pydantic a un DataFrame
        input_data = [item.model_dump() for item in data]
        input_df = pd.DataFrame(input_data)

        # Realiza la predicción usando tu clase MultiTargetPredictor
        predictions_df = predictor.predict(input_df)

        # Convierte el DataFrame de predicciones a una lista de dicts
        results = predictions_df.to_dict(orient="records")
        
        # Empaqueta en el modelo de salida
        return BatchPredictionOutput(
            predictions=[PredictionOutput(**res) for res in results]
        )

    except Exception as e:
        # Manejo de errores durante la predicción
        logger.error(f"Error durante la predicción por lotes: {e}")
        raise HTTPException(status_code=400, detail=f"Error durante la predicción en lote: {str(e)}")