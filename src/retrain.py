"""
retrain.py (STARTER) — Stage 4.5: multi-signal retraining trigger + workflow.

Run:  python -m src.retrain
"""
import config as cfg

PSI_THRESHOLD = 0.20            # representative example for instruction
DRIFT_SHARE_THRESHOLD = 0.30   # representative example for instruction


def decide(summary):
    """TODO: return a list of human-readable reasons if any trigger fires —
    prediction_psi > PSI_THRESHOLD, share_drifted > DRIFT_SHARE_THRESHOLD, or dataset_drift is True.
    An empty list means no retraining is needed."""
    raise NotImplementedError


def run_retraining_workflow():
    """TODO (Stage 4.5): read artifacts/drift_summary.json; if decide() returns reasons, retrain
    (call src.train.train_and_log, which registers + promotes a new version) and record
    artifacts/retraining_decision.json with the reasons + new metrics. Return the decision dict.
    In production the retrain would use freshly LABELLED data; here it re-runs the pipeline to
    demonstrate the automated workflow + versioning."""
    raise NotImplementedError


if __name__ == "__main__":
    run_retraining_workflow()
