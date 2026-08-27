"""tests/test_app.py — CI smoke tests for the inference API."""
import json
import math
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

import app as app_module

ART = Path(__file__).resolve().parents[1] / "artifacts"
client = TestClient(app_module.app)


def _sample_features():
    """A realistic JSON payload: NaN -> None (clients send null/omit, never NaN)."""
    cols = json.loads((ART / "input_columns.json").read_text())
    ref = pd.read_csv(ART / "reference_sample.csv")
    row = ref[cols].iloc[0].to_dict()
    return {k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in row.items()}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_returns_probability():
    r = client.post("/predict", json={"features": _sample_features()})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["readmission_probability"] <= 1.0
    assert body["readmitted_30d"] in (0, 1)


def test_predict_empty_payload_rejected():
    r = client.post("/predict", json={"features": {}})
    assert r.status_code == 422
