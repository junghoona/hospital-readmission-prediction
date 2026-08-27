"""
train.py — Stage 3: train a baseline + advanced model, track with MLflow,
register and promote the best. Imported by Model_Development_and_Tracking.ipynb.

Run:  python -m src.train
"""
import config as cfg


def train_and_log():
    """TODO (Stage 3):
    - Get splits + feature lists from src.data_prep; compute scale_pos_weight = neg / pos.
    - Build two FULL sklearn Pipelines (preprocessor + classifier):
        * LogisticRegression(class_weight='balanced')           (baseline)
        * XGBClassifier(scale_pos_weight=..., **cfg.XGB_PARAMS)  (advanced)
    - For each model: fit on train, evaluate on validation (src.evaluate.compute_metrics),
      and log params + metrics + the model to the MLflow experiment cfg.MLFLOW_EXPERIMENT.
    - Pick the best by validation ROC-AUC, evaluate on the test split, then register it
      (cfg.REGISTERED_MODEL) and promote to the 'production' alias.
    - Save artifacts: best_model.pkl, metrics.json, reference_sample.csv, input_columns.json.
    Return (best_model_name, test_metrics). Expected validation ROC-AUC ~0.65."""
    raise NotImplementedError


if __name__ == "__main__":
    train_and_log()
