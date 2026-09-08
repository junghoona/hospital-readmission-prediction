"""
retrain.py (STARTER) — Stage 4.5: multi-signal retraining trigger + workflow.

Run:  python -m src.retrain
"""
import json
from datetime import datetime, timezone
from mlflow import MlflowClient
import config as cfg

PSI_THRESHOLD = 0.20            # representative example for instruction
DRIFT_SHARE_THRESHOLD = 0.30   # representative example for instruction


def decide(summary):
    """Return a list of human-readable reasons if any trigger fires —
    prediction_psi > PSI_THRESHOLD, share_drifted > DRIFT_SHARE_THRESHOLD, or dataset_drift is True.
    An empty list means no retraining is needed."""
    reasons = []
    if summary["prediction_psi"] > PSI_THRESHOLD:
        reasons.append(
            f"prediction PSI {summary['prediction_psi']} > {PSI_THRESHOLD} "
            f"(mean score {summary['ref_mean_score']} -> {summary['cur_mean_score']})"
        )
    if summary["share_drifted"] > DRIFT_SHARE_THRESHOLD:
        reasons.append(
            f"drifted feature share {summary['share_drifted']} > {DRIFT_SHARE_THRESHOLD} "
            f"({summary['n_drifted_columns']}/{summary['n_columns']} columns)"
        )
    if summary.get("dataset_drift"):
        reasons.append("Evidently flagged dataset-level drift")
    return reasons


def run_retraining_workflow():
    """(Stage 4.5): read artifacts/drift_summary.json; if decide() returns reasons, retrain
    (call src.train.train_and_log, which registers + promotes a new version) and record
    artifacts/retraining_decision.json with the reasons + new metrics. Return the decision dict.
    In production the retrain would use freshly LABELLED data; here it re-runs the pipeline to
    demonstrate the automated workflow + versioning."""
    artifact = cfg.ARTIFACT_DIR
    summary = json.loads((artifact / "drift_summary.json").read_text())
    reasons = decide(summary)

    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "triggered": bool(reasons),
        "reasons": reasons,
        "signals": {
            "prediction_psi": summary["prediction_psi"],
            "share_drifted": summary["share_drifted"],
            "dataset_drift": summary["dataset_drift"],
        },
        "thresholds": {"psi": PSI_THRESHOLD, "drift_share": DRIFT_SHARE_THRESHOLD},
    }

    if reasons:
        # imported here, not at module scope: pulling in train drags
        # xgboost and sklearn, which a no-trigger run has no reason to pay for.
        from src.train import train_and_log

        best_name, test_metrics = train_and_log()
        decision["retrained_model"] = best_name
        decision["new_version"] = MlflowClient().get_model_version_by_alias(
            cfg.REGISTERED_MODEL, "production").version
        decision["new_test_metrics"] = test_metrics
    else:
        decision["action"] = "no retraining required"

    (artifact / "retraining_decision.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    return decision 


if __name__ == "__main__":
    run_retraining_workflow()
