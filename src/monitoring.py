"""
monitoring.py — Stage 4.4: data + prediction drift monitoring.
Compares a production 'current' batch against the saved training reference.

Run:  python -m src.monitoring
"""
import config as cfg


def run_monitoring():
    """TODO (Stage 4.4):
    - Load artifacts/reference_sample.csv (training baseline) and artifacts/current_batch.csv.
    - Feature drift: run Evidently DataDriftPreset(reference vs current) and save drift_report.html;
      read number_of_drifted_columns / number_of_columns / dataset_drift.
    - Prediction drift: load artifacts/best_model.pkl and compute the PSI between the reference and
      current predicted probabilities.
    - Write artifacts/drift_summary.json and return the summary dict.
    Expected (reference run): ~6/39 features drifted, dataset_drift False, prediction PSI ~0.46."""
    raise NotImplementedError


if __name__ == "__main__":
    run_monitoring()
