"""
data_prep.py — Stage 2: data cleaning, feature engineering, leakage-safe
preprocessing and splitting for the Hospital Readmission pipeline.

Implement the functions below; they are imported by Data_Preparation.ipynb and by
src/train.py. Expected end state: a model frame of ~69,973 rows at ~9% positive, and a
stratified train/val/test split with the preprocessor fit on the TRAIN split only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import config as cfg

ID_CODE_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]


def load_raw():
    """TODO: read cfg.RAW_CSV with na_values=['?'] so '?' becomes NaN; return the DataFrame."""
    raw = pd.read_csv(cfg.RAW_CSV, na_values=["?"])
    return raw


def clean(df):
    """TODO (Stage 2.1): keep the FIRST encounter per patient (drop_duplicates('patient_nbr'))
    to prevent patient-level leakage; drop expired/hospice discharges
    (discharge_disposition_id in cfg.EXPIRED_HOSPICE_DISPOSITIONS); build the binary target
    cfg.TARGET = 1 if readmitted == '<30' else 0; drop the cfg.DROP_COLS that are present.
    Return the cleaned frame (~69,973 rows, ~9% positive)."""
    df = df.copy() # Copy the dataframe so original df doesn't get mutated.

    # These encounters cannot be readmitted, so they'd be invalid negatives
    df = df[~df["discharge_disposition_id"].isin(cfg.EXPIRED_HOSPICE_DISPOSITIONS)]

    # Sort by encounter_id first so "first" is deterministic (chronological), not just row order
    df = df.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first")

    # Binary Target: readmitted within 30 days -> 1, else 0
    df[cfg.TARGET] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # Drop the columns to drop
    df = df.drop(columns=df.columns.intersection(cfg.DROP_COLS))

    return df.reset_index(drop=True)


def engineer_features(df):
    """TODO (Stage 2.2): group ICD9 diag_1/2/3 into clinical categories; map the age band to a
    numeric midpoint; service_utilization = number_outpatient + number_emergency + number_inpatient;
    num_med_changes = count of 'Up'/'Down' across cfg.MED_COLS; collapse high-cardinality
    medical_specialty to top-k + 'Other'. Return the engineered frame."""
    raise NotImplementedError


def get_feature_lists(df):
    """TODO: return (numeric_cols, categorical_cols), excluding cfg.TARGET."""
    raise NotImplementedError


def build_preprocessor(numeric, categorical):
    """TODO (Stage 2.3): return ONE ColumnTransformer — numeric: median SimpleImputer + StandardScaler;
    categorical: constant SimpleImputer + OneHotEncoder(handle_unknown='ignore'). Do NOT fit it here
    (it is fit on the TRAIN split only)."""
    raise NotImplementedError


def get_model_frame():
    """TODO: return engineer_features(clean(load_raw()))."""
    raise NotImplementedError


def get_splits(df=None):
    """TODO (Stage 2.4): create a STRATIFIED train/val/test split using cfg.RANDOM_STATE,
    cfg.TEST_SIZE and cfg.VAL_SIZE (val taken from the train split). Return
    X_train, X_val, X_test, y_train, y_val, y_test. The class ratio must be preserved across splits;
    the preprocessor is fit on TRAIN only (in the notebook / train.py), never on the full data."""
    raise NotImplementedError
