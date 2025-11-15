from pydantic import BaseModel, Field, field_validator
from typing import List, Any, Dict


class CleaningConfig(BaseModel):
    threshold: float = Field(..., ge=0, le=1)
    numeric_cols: List[str]
    categorical_cols: List[str]

    @field_validator("numeric_cols", "categorical_cols")
    @classmethod
    def check_non_empty(cls, v):  # type: ignore[no-untyped-def]
        if not v:
            raise ValueError("Column list must not be empty")
        return v

# --- Esquema de entrada para la API ---
class InputSchema(BaseModel):
    Relative_Compactness: float = Field(..., ge=0, le=1)
    Surface_Area: float
    Wall_Area: float
    Roof_Area: float
    Overall_Height: float
    Orientation: int = Field(..., ge=2, le=5) # Asumiendo 2,3,4,5
    Glazing_Area: float
    Glazing_Area_Distribution: int

    class Config:
        # Ejemplo de datos para la UI de Swagger
        json_schema_extra = {
            "example": {
                "Relative_Compactness": 0.98,
                "Surface_Area": 514.5,
                "Wall_Area": 294.0,
                "Roof_Area": 110.25,
                "Overall_Height": 7.0,
                "Orientation": 2,
                "Glazing_Area": 0.0,
                "Glazing_Area_Distribution": 0
            }
        }

# --- Esquema de salida para la API ---
# Este esquema es para un solo ítem de predicción
class PredictionOutput(BaseModel):
    heating: float
    cooling: float

# Este esquema es para la salida de predicción por lotes
class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    errors: Any = None