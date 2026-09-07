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
    """ read cfg.RAW_CSV with na_values=['?'] so '?' becomes NaN; return the DataFrame."""
    raw = pd.read_csv(cfg.RAW_CSV, na_values=["?"])
    return raw


def clean(df):
    """ (Stage 2.1): keep the FIRST encounter per patient (drop_duplicates('patient_nbr'))
    to prevent patient-level leakage; drop expired/hospice discharges
    (discharge_disposition_id in cfg.EXPIRED_HOSPICE_DISPOSITIONS); build the binary target
    cfg.TARGET = 1 if readmitted == '<30' else 0; drop the cfg.DROP_COLS that are present.
    Return the cleaned frame (~69,973 rows, ~9% positive)."""
    df = df.copy() # Copy the dataframe so original df doesn't get mutated.

    # Sort by encounter_id first so "first" is deterministic (chronological), not just row order
    df = df.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first")

    # These encounters cannot be readmitted, so they'd be invalid negatives
    df = df[~df["discharge_disposition_id"].isin(cfg.EXPIRED_HOSPICE_DISPOSITIONS)]

    # Binary Target: readmitted within 30 days -> 1, else 0
    df[cfg.TARGET] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # Drop the columns to drop
    df = df.drop(columns=df.columns.intersection(cfg.DROP_COLS))

    return df.reset_index(drop=True)


""" Helper function that assigns clinical category value to ICD-9 grouping """
def _icd9_category(code):
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"
    try:
        num = float(code)
    except ValueError:
        return "Other"
    if 250 <= num < 251:
        return "Diabetes"
    if (390 <= num <= 459) or num == 785:
        return "Circulatory"
    if (460 <= num <= 519) or num == 786:
        return "Respiratory"
    if (520 <= num <= 579) or num == 787:
        return "Digestive"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if (580 <= num <= 629) or num == 788:
        return "Genitourinary"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"


def engineer_features(df):
    """(Stage 2.2): group ICD9 diag_1/2/3 into clinical categories; map the age band to a
    numeric midpoint; service_utilization = number_outpatient + number_emergency + number_inpatient;
    num_med_changes = count of 'Up'/'Down' across cfg.MED_COLS; collapse high-cardinality
    medical_specialty to top-k + 'Other'. Return the engineered frame."""
    df = df.copy()

    # ICD-9 grouping into clinical categories
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_category"] = df[col].apply(_icd9_category)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    # age midpoint
    bounds = df["age"].str.extract(r"\[(\d+)-(\d+)\)").astype(float)
    df["age"] = bounds.mean(axis=1)

    # total prior utilization
    df["service_utilization"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    # how many meds were actively adjusted this stay
    med_cols = df.columns.intersection(cfg.MED_COLS)
    df["num_med_changes"] = df[med_cols].isin(["Up", "Down"]).sum(axis=1)

    # collapse medical_specialty to top-10 + Other (49% missing, high cardinality)
    top_specialties = df["medical_specialty"].value_counts().nlargest(10).index
    df["medical_specialty"] = (
        df["medical_specialty"].where(df["medical_specialty"].isin(top_specialties), "Other").fillna("Other")
    )

    return df


def get_feature_lists(df):
    """Return (numeric_cols, categorical_cols), excluding cfg.TARGET."""
    features = df.drop(columns=[cfg.TARGET])

    numeric = [col for col in features.select_dtypes(include="number").columns
               if col not in ID_CODE_COLS]
    categorical = [col for col in features.columns if col not in numeric]

    return numeric, categorical


def build_preprocessor(numeric, categorical):
    """(Stage 2.3): return ONE ColumnTransformer — numeric: median SimpleImputer + StandardScaler;
    categorical: constant SimpleImputer + OneHotEncoder(handle_unknown='ignore'). Do NOT fit it here
    (it is fit on the TRAIN split only)."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])


def get_model_frame():
    """ Raw -> cleaned -> engineered """
    return engineer_features(clean(load_raw()))


def get_splits(df=None):
    """create a STRATIFIED train/val/test split using cfg.RANDOM_STATE,
    cfg.TEST_SIZE and cfg.VAL_SIZE (val taken from the train split). Return
    X_train, X_val, X_test, y_train, y_val, y_test. The class ratio must be preserved across splits;
    the preprocessor is fit on TRAIN only (in the notebook / train.py), never on the full data."""
    if df is None:
        df = get_model_frame()

    X = df.drop(columns=[cfg.TARGET])
    y = df[cfg.TARGET]

    # Step 1: hold out the test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE,
        stratify=y,
        random_state=cfg.RANDOM_STATE,
    )

    # Step 2: carve validation out of what's left (20% of it -> 16% overall)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=cfg.VAL_SIZE,
        stratify=y_temp, # stratify on the remaining labels
        random_state=cfg.RANDOM_STATE,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
