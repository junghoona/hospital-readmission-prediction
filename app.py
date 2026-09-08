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
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
THRESHOLD = 0.5

app = FastAPI(title="Hospital Readmission Predictor", version="1.0")

# - Lazily load artifacts/best_model.pkl and artifacts/input_columns.json.
# Loaded once at import.
# Try/except allows a missing artifact to be reported,
# instead of an ImportError that no endpoint is alive to explain.
try:
    MODEL = joblib.load(ARTIFACTS / "best_model.pkl")
    COLUMNS = json.loads((ARTIFACTS / "input_columns.json").read_text())
except Exception:
    MODEL, COLUMNS = None, None


class PredictRequest(BaseModel):
    features: dict = Field(..., description="Raw encounter features (column -> value)")

    @field_validator("features")
    @classmethod
    def features_not_empty(cls, value):
        if not value:
            raise ValueError("features must not be empty")
        return value


# (Stage 4.1):
# - GET  /health  -> {"status": ..., "model_loaded": bool}.
@app.get("/health")
def health():
    return {"status": "ok" if MODEL is not None else "degraded",
            "model_loaded": MODEL is not None}

# - POST /predict -> align the incoming features to the training columns (missing -> NaN; the
#   pipeline imputes), return {"readmission_probability", "readmitted_30d", "threshold"}.
@app.post("/predict")
def predict(request: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model artifacts not loaded")

    # Align to the training schema: exact columns, exact order.
    # Unknown keys are dropped, absent ones NaN and are filled by the pipeline's imputers
    # using statistics learned on TRAIN. JSON has no NaN, so clients send null.
    supplied = [col for col in COLUMNS if col in request.features]
    row = pd.DataFrame([{col: request.features.get(col, None) for col in COLUMNS}])[COLUMNS]

    try:
        probability = float(MODEL.predict_proba(row)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not score payload: {e}")

    label = int(probability >= THRESHOLD)
    response = {"readmission_probability": round(probability, 4),
                "readmitted_30d": label,
                "threshold": THRESHOLD}
    
    #   Validate the payload (reject an empty features dict with 422) and log every prediction to
    #   artifacts/predictions.log (governance). Note: JSON has no NaN, so clients send null / omit fields.
    # Governance trail: records the decision, not the patient.
    # Wrapped because logging must never take down serving.
    try:
        with open(ARTIFACTS / "predictions.log", "a") as handle:
            handle.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "readmission_probability": response["readmission_probability"],
                "readmitted_30d": label,
                "threshold": THRESHOLD,
                "n_features_supplied": len(supplied),
                "n_features_expected": len(COLUMNS), 
            }) + "\n")
    except Exception:
        pass

    return response
