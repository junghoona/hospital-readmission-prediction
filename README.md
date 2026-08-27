# Hospital Readmission Prediction — Starter

Build a **guided healthcare MLOps** pipeline for 30-day readmission prediction on the
Diabetes 130-US Hospitals dataset. 

Complete every `# TODO` in the three notebooks and in the `.py` modules they import.

## File ownership (what to change vs leave alone)
- **Provided — no modification expected (supporting assets):** `config.py` (paths, target, column
  lists, params), `requirements.txt`, `Dockerfile`, `.github/workflows/ci.yml`, `tests/`,
  `src/evaluate.py`, `src/generate_current_batch.py`.
- **You build (deliverables):** the 3 notebooks; `src/data_prep.py`, `src/train.py`,
`src/monitoring.py`, `src/retrain.py`, `app.py`; and the MLOps report.

## How the tasks are laid out
Each notebook stage opens with a **Markdown sub-task checklist** — every sub-task shows its ID and marks
(e.g. `2.1.1 — Missing value handling [2]`) so you can see exactly what each mark rewards; the code
cells below carry only short `# TODO` pointers.

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt          # on Mac also: brew install libomp
Place `diabetic_data.csv` in `data/` (Kaggle: Diabetes 130-US Hospitals).

## Build order (one stage per week)
    # Week 1-2  (Stage 1-2)
    Data_Preparation.ipynb            + src/data_prep.py
    # Week 3    (Stage 3)
    python -m src.train ;  Model_Development_and_Tracking.ipynb
    # Week 4    (Stage 4)
    python -m src.generate_current_batch ; python -m src.monitoring ; python -m src.retrain
    pytest tests/ -q ; uvicorn app:app --reload ; Operations_Monitoring_and_Evidence.ipynb

## What you submit (files only — no ZIP, no GitHub URL)
Upload these individual files; all 100 marks are graded from them:
1. `Hospital_Readmission_MLOps_Report` as **PDF/DOCX** (Stages 1, 3, 4 + Stage-4 evidence).
2. `Data_Preparation.ipynb` (executed, with outputs).
3. `Model_Development_and_Tracking.ipynb` (executed, with outputs).
4. `Operations_Monitoring_and_Evidence.ipynb` (executed, with outputs).
5. `app.py`.

**Stage-4 operational evidence** (Docker build, CI green check, MLflow UI, drift report, prediction
log) must be **captured inside the operations notebook and/or the report** as screenshots, code
output and summaries — we do not accept repositories, ZIPs, HTML or JSON files. 
