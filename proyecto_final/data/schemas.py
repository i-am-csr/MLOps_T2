from pydantic import BaseModel, Field, field_validator
from typing import List


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
