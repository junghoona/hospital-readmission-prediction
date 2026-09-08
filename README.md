# Hospital Readmission Prediction — MLOps Pipeline

This project was completed as an MLOps Capstone project for UpGrad. This is a guided healthcare MLOps pipeline predicting 30-day readmission on the Diabetes 130-US Hospitals dataset (1999–2008). The project covers data preparation, model development with MLflow tracking, and operations: a FastAPI inference service, containerisation, CI, drift monitoring, automated retraining and governance.

---

## Quickstart

```bash
# 1. environment (Python 3.12 — see "Why 3.12" below)
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt     # or: python -m pip install -r requirements.txt
python -c "import numpy, sklearn, xgboost; print(numpy.__version__)"    # verify installed package version

brew install libomp                  # macOS only; XGBoost needs OpenMP at runtime

# 2. verify
python -c "import xgboost, mlflow, evidently; print('ok')"

# 3. train + serve
python -m src.train                  # writes artifacts/, registers a model version
python -m pytest tests/ -q           # expect: 3 passed
python -m uvicorn app:app --reload   # http://localhost:8000/docs
```

Invoke tools as `python -m <tool>` rather than as bare commands — see
[Console scripts](#console-scripts-use-python--m) below.

Then run the three notebooks in order (see [Running the project](#running-the-project)).

---

## Setup

### Requirements

| | version | why |
|---|---|---|
| Python | **3.12** | `numpy==2.0.2` ships no cp313 wheels |
| libomp | any | XGBoost links OpenMP at runtime (macOS) |
| Docker | any | Stage 4.2 evidence only |

### Why 3.12 specifically

Installing on Python 3.13 fails with:

```
error: metadata-generation-failed
╰─> numpy
```

`numpy==2.0.2` publishes wheels for cp39–cp312 only; cp313 support starts at
numpy 2.1.0. On 3.13 pip finds no wheel, falls back to a source build, and dies
at metadata generation. Do **not** loosen the numpy pin to work around this —
`scikit-learn==1.6.1`, `xgboost==2.1.4` and `pandas==2.3.3` are all pinned
against numpy 2.0.x. Use 3.12 instead.

If you see `XGBoostError: Library not loaded: @rpath/libomp.dylib`, run
`brew install libomp`. That is a separate problem from the pip failure above.

### Data

Place `diabetic_data.csv` in `data/` (Kaggle: Diabetes 130-US Hospitals).
Expected shape: 101,766 rows × 50 columns.

### Jupyter kernel

Select the `.venv` interpreter in the kernel picker, then confirm inside the
notebook before running anything:

```python
import sys; print(sys.executable)
# must end in .../hospital-readmission-prediction/.venv/bin/python
```

A kernel pointing at a system Python will fail to import the pinned packages.

### Configuration

Everything tunable lives in `config.py` — paths, target name, dropped columns,
split sizes, `RANDOM_STATE = 42`, XGBoost params and the MLflow URIs.

MLflow tracking defaults to `sqlite:///mlflow.db`. **Keep the SQLite backend:**
the Model Registry requires a database and will not work against a `file://`
store, so Stage 3.5 and Stage 4.6 both depend on it.

---

## Running the project

Run the notebooks **top to bottom** (Restart & Run All), and run the stages
**in order** — each depends on artifacts the previous one wrote. Cherry-picking
cells produces non-sequential execution counts, which a marker can see.

### Stages 1–2 · `Data_Preparation.ipynb`

EDA, cleaning, feature engineering, leakage-safe preprocessing and splitting.
Logic lives in `src/data_prep.py`; the notebook imports and calls it.

```bash
jupyter lab Data_Preparation.ipynb     # or open in VS Code
```

Pipeline: `load_raw` → `clean` → `engineer_features` → `get_feature_lists` →
`build_preprocessor` → `get_splits`.

Two invariants worth knowing, because breaking either fails silently downstream:

- **Order inside `clean()`** — deduplicate to the first encounter per patient
  *before* dropping expired/hospice discharges. The reverse order yields 69,990
  rows instead of 69,973.
- **`engineer_features` transforms in place** — `age` becomes a numeric midpoint
  and `diag_1/2/3` become ICD-9 category strings under their *original names*.
  Renaming them to `age_midpoint` / `diag_*_category` breaks
  `src/generate_current_batch.py` and `src/monitoring.py`, which address those
  columns directly. The failure is silent: `.loc` creates a phantom column
  rather than raising.

### Stage 3 · `src/train.py` + `Model_Development_and_Tracking.ipynb`

```bash
python -m src.train
```

Trains a logistic-regression baseline and an XGBoost model as full sklearn
Pipelines (preprocessor + classifier), logs both to MLflow, selects on
validation ROC-AUC, registers the winner and promotes it to the `production`
alias. Takes several minutes.

Writes four artifacts consumed downstream:

| file | consumed by |
|---|---|
| `best_model.pkl` | `app.py`, `src/monitoring.py` |
| `input_columns.json` | `app.py` |
| `reference_sample.csv` | `src/monitoring.py` |
| `metrics.json` | report / evidence |

Then run the notebook, which reproduces the comparison, plots, MLflow runs and
registry history.

Inspect runs in the UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```

### Stage 4 · `app.py`, `src/monitoring.py`, `src/retrain.py` + `Operations_Monitoring_and_Evidence.ipynb`

Run Stage 3 first — Stage 4 reads its artifacts.

```bash
python -m src.generate_current_batch   # simulate a post-deployment batch with drift
python -m src.monitoring               # Evidently feature drift + prediction PSI
python -m src.retrain                  # retrain if a trigger fires; writes decision record
pytest tests/ -q                       # API smoke tests
uvicorn app:app --reload               # interactive docs at /docs
```

Or run `Operations_Monitoring_and_Evidence.ipynb`, which drives all of it. Cell
dependencies, in order:

```
cell 3  → creates artifacts/predictions.log      (needed by the governance cell)
cell 12 → defines `summary`, regenerates the drift batch
cell 16 → retrains, registers a new version      (shows up in the history table)
cell 19 → reads predictions.log + registry
```

Cell 16 retrains from scratch and takes several minutes. Don't interrupt it —
a partial run leaves an orphaned MLflow run.

**Subprocess calls from notebooks:** use `sys.executable`, not a bare command.

```python
subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], ...)
```

A bare `pytest` resolves against the kernel's `PATH`, which is not guaranteed
to include `.venv/bin`.

### Evidence that can't be produced inside the notebook

**Docker (4.2.2).** Start Docker Desktop first. Build *after* retraining — the
image copies `artifacts/`, so an earlier build ships a stale model.

```bash
docker build -t readmission-api .
docker run -d --rm -p 8000:8000 --name api readmission-api
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"features": {"age": 75.0, "time_in_hospital": 5, "num_medications": 18, "number_inpatient": 2, "diag_1": "Circulatory"}}'
docker stop api
```

The second curl deliberately sends 5 of 39 fields — it demonstrates that the
pipeline's imputers fill the rest, over real HTTP.

**CI (4.3.2).** The workflow only runs once pushed:

```bash
git add .github/ && git commit -m "CI workflow" && git push
```

Then screenshot the green check from the Actions tab.

---

## Expected results

Use these to verify your run. Values are from a clean pass with
`RANDOM_STATE = 42`.

| stage | metric | value |
|---|---|---|
| 1 | raw dataset | 101,766 × 50 |
| 2 | after `clean()` | **69,973** rows, **8.97%** positive |
| 2 | model frame | **39** features |
| 2 | splits | 44,782 train / 11,196 val / 13,995 test |
| 3 | val ROC-AUC | ~**0.654** (both models, within noise) |
| 3 | test ROC-AUC | ~0.644 |
| 4 | tests | **3 passed** |
| 4 | feature drift | **6 / 39** (15.4%), `dataset_drift=False` |
| 4 | prediction PSI | ~**0.26** → retraining trigger fires |

Final `artifacts/` contents:

```
best_model.pkl        current_batch.csv     drift_report.html
drift_summary.json    input_columns.json    metrics.json
predictions.log       reference_sample.csv  retraining_decision.json
```

A validation ROC-AUC near 0.65 is the honest ceiling for this dataset —
administrative billing records lack the social and clinical detail that drives
readmission. Do not tune toward a higher number.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `metadata-generation-failed × numpy` | installing on Python 3.13 | use the 3.12 venv |
| `Library not loaded: @rpath/libomp.dylib` | OpenMP missing | `brew install libomp` |
| `ModuleNotFoundError: No module named 'config'` | running a module from inside `src/` | run from the project root: `python -m src.train` |
| Monitoring reports `0/0` columns | PSI block indented inside the metrics loop | dedent it to function-body level |
| Drift report shows ~31/39 drifted | one comparison applied to every method | p-value tests drift *below* threshold, distances *above* |
| Registry table shows `aliases: []` | `search_model_versions` doesn't populate aliases | read `client.get_registered_model(name).aliases` |
| `FileNotFoundError: predictions.log` | governance cell run before the API cell | Restart & Run All |
| Tests return 404 | no routes registered | check the `@app.get` / `@app.post` decorators in `app.py` |

---

## Project structure

```
config.py                  paths, target, column lists, params, MLflow URIs
app.py                     Stage 4.1 — FastAPI inference service
Dockerfile                 Stage 4.2 — container image
.github/workflows/ci.yml   Stage 4.3 — install deps, pytest, docker build
data/diabetic_data.csv     raw dataset
artifacts/                 model, columns, reference sample, drift + decision records
src/
  data_prep.py             Stage 2 — clean, engineer, preprocess, split
  train.py                 Stage 3 — train, track, register, promote
  evaluate.py              shared imbalance-aware metrics (provided)
  generate_current_batch.py  simulate a drifted production batch (provided)
  monitoring.py            Stage 4.4 — Evidently feature drift + prediction PSI
  retrain.py               Stage 4.5 — multi-signal trigger + retraining workflow
tests/test_app.py          API smoke tests (provided)
Data_Preparation.ipynb                    Stages 1–2
Model_Development_and_Tracking.ipynb      Stage 3
Operations_Monitoring_and_Evidence.ipynb  Stage 4
```

---

## File ownership (what to change vs leave alone)

- **Provided — no modification expected:** `config.py`, `requirements.txt`,
  `Dockerfile`, `tests/`, `src/evaluate.py`, `src/generate_current_batch.py`.
- **You build:** the 3 notebooks; `src/data_prep.py`, `src/train.py`,
  `src/monitoring.py`, `src/retrain.py`, `app.py`, `.github/workflows/ci.yml`;
  and the MLOps report.

Each notebook stage opens with a Markdown sub-task checklist — every sub-task
shows its ID and marks (e.g. `2.1.1 — Missing value handling [2]`) so you can
see exactly what each mark rewards; the code cells below carry short `# TODO`
pointers.

---
