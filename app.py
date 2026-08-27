"""
app.py — Stage 4.1: FastAPI inference service.
Implement GET /health and POST /predict, then demonstrate it from the operations notebook.

Run:  uvicorn app:app --reload
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
THRESHOLD = 0.5

app = FastAPI(title="Hospital Readmission Predictor", version="1.0")


class PredictRequest(BaseModel):
    features: dict = Field(..., description="Raw encounter features (column -> value)")


# TODO (Stage 4.1):
# - Lazily load artifacts/best_model.pkl and artifacts/input_columns.json.
# - GET  /health  -> {"status": ..., "model_loaded": bool}.
# - POST /predict -> align the incoming features to the training columns (missing -> NaN; the
#   pipeline imputes), return {"readmission_probability", "readmitted_30d", "threshold"}.
#   Validate the payload (reject an empty features dict with 422) and log every prediction to
#   artifacts/predictions.log (governance). Note: JSON has no NaN, so clients send null / omit fields.
