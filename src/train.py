"""
train.py — Stage 3: train a baseline + advanced model, track with MLflow,
register and promote the best. Imported by Model_Development_and_Tracking.ipynb.

Run:  python -m src.train
"""
import config as cfg, src.data_prep as dp
import json, joblib, mlflow
from src.evaluate import compute_metrics
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from mlflow import MlflowClient


def train_and_log():
    """
    - app.py needs best_model.pkl + input_columns.json
    - monitoring.py needs reference_sample.csv + best_model.pkl
    """
    """(Stage 3):
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
    fe = dp.get_model_frame()
    numeric, categorical = dp.get_feature_lists(fe)
    X_train, X_val, X_test, y_train, y_val, y_test = dp.get_splits(fe)

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())

    candidates = {
        "logistic_regression": Pipeline([
            ("prep", dp.build_preprocessor(numeric, categorical)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=cfg.RANDOM_STATE))
        ]),
        "xgboost": Pipeline([
            ("prep", dp.build_preprocessor(numeric, categorical)),
            ("clf", XGBClassifier(scale_pos_weight=neg / pos, **cfg.XGB_PARAMS))
        ])
    }

    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT)

    results = {}
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            metrics = compute_metrics(y_val, model.predict_proba(X_val)[:, 1])
            mlflow.log_param("model_type", name)
            mlflow.log_params(model.named_steps["clf"].get_params())
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
            info = mlflow.sklearn.log_model(model, name="model",
                                            input_example=X_train.head(5))
            results[name] = {"model": model, "metrics": metrics,
                             "model_uri": info.model_uri}

    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best = results[best_name]["model"]
    test_metrics = compute_metrics(y_test, best.predict_proba(X_test)[:, 1])

    # register + promote
    mv = mlflow.register_model(results[best_name]["model_uri"], cfg.REGISTERED_MODEL)
    MlflowClient().set_registered_model_alias(cfg.REGISTERED_MODEL, "production", mv.version)

    # artifacts for Stage 4
    artifact = cfg.ARTIFACT_DIR
    joblib.dump(best, artifact / "best_model.pkl")
    (artifact / "metrics.json").write_text(json.dumps(
        {"best_model": best_name, "version": mv.version,
         "validation": results[best_name]["metrics"], "test": test_metrics}, indent=2))
    (artifact / "input_columns.json").write_text(json.dumps(list(X_train.columns), indent=2))
    X_train.sample(min(5000, len(X_train)),
                   random_state=cfg.RANDOM_STATE).to_csv(artifact / "reference_sample.csv",
                                                         index=False)

    print(f"best={best_name} version{mv.version} test={test_metrics}")
    return best_name, test_metrics


if __name__ == "__main__":
    train_and_log()
