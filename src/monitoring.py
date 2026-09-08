"""
monitoring.py — Stage 4.4: data + prediction drift monitoring.
Compares a production 'current' batch against the saved training reference.

Run:  python -m src.monitoring
"""
import json

import joblib
import numpy as np, pandas as pd
from evidently import Report, Dataset
from evidently.presets import DataDriftPreset

import config as cfg


def run_monitoring():
    """(Stage 4.4):
    - Load artifacts/reference_sample.csv (training baseline) and artifacts/current_batch.csv.
    - Feature drift: run Evidently DataDriftPreset(reference vs current) and save drift_report.html;
      read number_of_drifted_columns / number_of_columns / dataset_drift.
    - Prediction drift: load artifacts/best_model.pkl and compute the PSI between the reference and
      current predicted probabilities.
    - Write artifacts/drift_summary.json and return the summary dict.
    Expected (reference run): ~6/39 features drifted, dataset_drift False, prediction PSI ~0.46."""
    artifact = cfg.ARTIFACT_DIR
    reference = pd.read_csv(artifact / "reference_sample.csv")

    # Reindex to the reference schema: raises loudly if columns ever diverge,
    # which is the failure that previously went silent as a phantom diag_1 column.
    current = pd.read_csv(artifact / "current_batch.csv")[reference.columns]

    # ---------- 4.4.1 feature drift ----------
    # Feature drift: run Evidently DataDriftPreset(reference vs current) and save drift_report.html;
    # read number_of_drifted_columns / number_of_columns / dataset_drift.
    run = Report([DataDriftPreset()]).run(
        current_data=Dataset.from_pandas(current),
        reference_data=Dataset.from_pandas(reference),
    )
    run.save_html(str(artifact / "drift_report.html"))

    drifted_columns, n_columns = [], 0
    share_drifted, share_threshold = 0.0, 0.5
    for metric in run.dict()["metrics"]:
      name = metric["metric_name"]
      if name.startswith("DriftedColumnsCount"):
        share_drifted = metric["value"]["share"]
        share_threshold = metric["config"]["drift_share"]
      elif name.startswith("ValueDrift"):
        n_columns += 1
        threshold = metric["config"]["threshold"]
        # Evidently 0.7 picks a method per column and the direction differs:
        # p-value tests (chi-square, K-S) drift BELOW the threshold,
        # distance measures drift ABOVE it
        if "p_value" in metric["config"]["method"]:
          is_drifted = metric["value"] < threshold
        else:
          is_drifted = metric["value"] >= threshold
        if is_drifted:
          drifted_columns.append(metric["config"]["column"])

    # ---------- 4.4.2 prediction drift ----------
    model = joblib.load(artifact / "best_model.pkl")
    columns = json.loads((artifact / "input_columns.json").read_text())
    ref_scores = model.predict_proba(reference[columns])[:, 1]
    cur_scores = model.predict_proba(current[columns])[:, 1]

    # PSI. Bin edges come from the REFERENCE quantiles so the reference is uniform
    # by construction and every difference is attributable to the current batch.
    # np.unique collapses duplicate edges; outer edges open to +/-inf so unseen
    # extremes still land in a bin; the clip keeps log() finite on empty bins.
    edges = np.unique(np.quantile(ref_scores, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.clip(np.histogram(ref_scores, edges)[0] / len(ref_scores), 1e-6, None)
    cur_pct = np.clip(np.histogram(cur_scores, edges)[0] / len(cur_scores), 1e-6, None)
    prediction_psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    summary = {
        "n_columns": n_columns,
        "n_drifted_columns": len(drifted_columns),
        "share_drifted": round(share_drifted, 4),
        "dataset_drift": bool(share_drifted >= share_threshold),
        "drifted_columns": drifted_columns,
        "prediction_psi": round(prediction_psi, 4),
        "ref_mean_score": round(float(ref_scores.mean()), 4),
        "cur_mean_score": round(float(cur_scores.mean()), 4),
    }
    (artifact / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_monitoring()
