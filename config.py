"""
config.py — central configuration for the Hospital Readmission MLOps pipeline.

Dataset: Diabetes 130-US Hospitals (1999-2008). Task: predict 30-day readmission
(binary). Everything tunable (paths, target, columns, model params, MLflow) lives here.
"""
import os
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

RAW_CSV    = DATA_DIR / "diabetic_data.csv"
RANDOM_STATE = 42
TEST_SIZE  = 0.20
VAL_SIZE   = 0.20            # of the train split

# Target: readmitted within 30 days -> 1, else 0
TARGET = "readmitted_30d"

# MLflow
MLFLOW_TRACKING_URI = os.environ.get(
    #"MLFLOW_TRACKING_URI", f"file://{(BASE_DIR / 'mlruns').as_posix()}")
    "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = "hospital_readmission"
REGISTERED_MODEL  = "readmission_classifier"

# Columns to drop outright: identifiers + ~97% missing + constant medication cols
DROP_COLS = [
    "encounter_id", "patient_nbr",   # identifiers
    "weight",                         # 96.9% missing
    "payer_code",                     # 39.6% missing, administrative (not clinical)
    "examide", "citoglipton",         # constant ('No' for every row)
    "metformin-rosiglitazone", "metformin-pioglitazone",  # near-constant
    "glimepiride-pioglitazone", "acetohexamide", "troglitazone", "tolbutamide",
]

# Discharge dispositions meaning expired / hospice — patient cannot be readmitted.
# Removing them prevents label leakage / invalid negatives.
EXPIRED_HOSPICE_DISPOSITIONS = [11, 13, 14, 19, 20, 21]

# Medication columns whose values are in {No, Steady, Up, Down}
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "tolazamide", "insulin", "glyburide-metformin",
    "glipizide-metformin",
]

# XGBoost / LogReg hyper-parameters (kept modest for reproducible CPU runs)
XGB_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.9,
    colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=-1,
    eval_metric="auc", tree_method="hist",
)
