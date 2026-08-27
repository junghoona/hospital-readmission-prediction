"""
generate_current_batch.py — simulate a post-deployment 'current' batch with drift.

In production you'd receive new encounters over time. Here we take a fresh sample and
apply realistic distributional shifts (longer stays, more medications/inpatient visits,
older skew, more circulatory diagnoses) so the monitoring stage has genuine drift to
detect — exactly the pattern used to demonstrate drift -> retraining.

Run:  python -m src.generate_current_batch
"""
import numpy as np
import pandas as pd
import config as cfg
from src import data_prep as dp


def main(n=5000, seed=cfg.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    df = dp.get_model_frame().drop(columns=[cfg.TARGET])
    cur = df.sample(min(n, len(df)), random_state=seed + 1).copy()

    # --- inject drift ---
    cur["time_in_hospital"] = (cur["time_in_hospital"] + rng.integers(1, 4, len(cur))).clip(1, 14)
    cur["num_medications"] = (cur["num_medications"] * rng.normal(1.25, 0.1, len(cur))).round().clip(1, 81)
    cur["number_inpatient"] = (cur["number_inpatient"] + rng.integers(0, 2, len(cur)))
    cur["age"] = (cur["age"] + 10).clip(5, 95)                 # older skew
    cur["service_utilization"] = (cur["number_outpatient"]
                                  + cur["number_emergency"] + cur["number_inpatient"])
    # shift diagnosis mix toward Circulatory
    flip = rng.random(len(cur)) < 0.25
    cur.loc[flip, "diag_1"] = "Circulatory"

    cur.to_csv(cfg.ARTIFACT_DIR / "current_batch.csv", index=False)
    print(f"wrote current_batch.csv ({len(cur)} rows) with injected drift")


if __name__ == "__main__":
    main()
