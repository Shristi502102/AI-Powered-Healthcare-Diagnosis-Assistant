# AI-Powered Healthcare Diagnosis Assistant

A machine-learning web application that accepts patient symptoms and returns the most probable conditions (with probabilities), an estimated risk level, a recommended medical specialist, informational medicine guidance, and precaution advice — with prediction history stored in a database and downloadable reports.

> **Disclaimer:** This project is for **educational and informational purposes only**. It predicts *likely conditions*, not definitive diagnoses, and must never replace consultation with a qualified healthcare professional. Seek emergency care for critical symptoms.

## Screenshots

**Symptom intake form**
![Symptom intake form](image/Screenshot%202026-07-17%20104914.jpg)

**Prediction results**
![Prediction results](image/Screenshot%202026-07-17%20105243.jpg)

## Model & accuracy

- **Test accuracy: 96.3%** · Precision 0.964 · Recall 0.963 · F1 0.963 · ROC-AUC (one-vs-rest) 0.9996 · 5-fold CV 96.4% ± 1.0%. See `model/metrics.json`.
- These numbers are from a genuinely held-out test set: the data is deduplicated before splitting, so there is no train/test leakage — the model is not simply memorizing rows it already saw.

**Why 25 diseases, not all 773 in the raw file — read this before changing `TOP_N_DISEASES`.**
The source Kaggle dataset (see below) covers 773 disease labels, many of which are clinically very similar and share almost identical symptom profiles (e.g. "common cold" vs "flu" vs "allergy"). We trained and honestly evaluated a Random Forest across the *full* 773-label space and it plateaued at **75–85% test accuracy no matter how much model capacity we threw at it** — that's a real ceiling from symptom overlap between diseases, not a tuning or resource limit. (You'll find similar or lower figures in most rigorous evaluations of this dataset; notebooks claiming 98–100% across all 773 labels are almost always evaluating on data that wasn't deduplicated before the split, so identical rows leak between train and test.)

To meet a genuinely high, defensible accuracy target, `train_model.py` curates the **25 most-frequent, best-represented diseases** from the dataset — the same approach real symptom-checker products use (a focused list of common conditions rather than an exhaustive one). This is a deliberate, documented trade-off between *breadth* (all 773 labels, ~80% accuracy) and *reliability* (25 labels, ~96% accuracy). Change `TOP_N_DISEASES` in `train_model.py` if you want a different point on that trade-off — the comment above the constant explains what to expect.

## Dataset

- **Source:** Kaggle "Diseases and Symptoms" augmented dataset — `Final_Augmented_dataset_Diseases_and_Symptoms.csv`, 246,945 rows, 773 diseases, 377 binary symptom columns. Place it in `dataset/raw/` (already included in this build).
- **Cleaning:** column names normalised to snake_case, 49 symptom columns that never fire anywhere in the file are dropped, and the data is deduplicated (246,945 raw rows → 189,647 unique symptom-pattern/disease combinations across all 773 diseases).
- **Curation:** the true top 25 diseases by frequency are kept (see rationale above), then capped at 150 samples/disease (still ample per class) — 3,750 total training samples, 135 informative symptoms remain after re-checking for dead columns within the curated subset.
- `train_model.py` also accepts the smaller legacy Kaggle format (`DiseaseAndSymptoms.csv` + `Disease precaution.csv`, 41 diseases) as a fallback if the augmented file isn't present — useful if you want to swap datasets without editing code.

## Recommendation engines (Modules 8–10)

With 773 possible diseases, hand-curating a specialist/medicine/precaution table one-by-one isn't feasible, so these are **rule-based engines** that scale to any disease list:

- **Specialist recommendation:** keyword rules matched against the disease name (e.g. "bronch"/"pneumonia" → Pulmonologist, "cysti"/"bladder" → Urologist, "gout"/"fibromyalgia" → Rheumatologist). Falls back to General Physician. See `SPECIALIST_RULES` in `train_model.py`.
- **Symptom severity scoring:** keyword-based (critical/serious/moderate/mild tiers) rather than one entry per symptom, since the dataset has 377 symptom names. See `severity_for()`.
- **Precautions & medicine:** generic, safe, evidence-general guidance ("stay hydrated", "consult a doctor if symptoms worsen") for diseases without a curated match — deliberately *not* fabricated per-disease medicine names, since inventing specific drug recommendations for hundreds of conditions without a clinical source would be irresponsible. The legacy 41-disease dataset's real `Disease precaution.csv` is used verbatim when that fallback path is active.
- **Emergency detection & risk scoring:** unchanged in spirit from the original design — rule-based combinations (e.g. chest pain + difficulty breathing, seizures, vomiting blood) force a High risk flag and an emergency banner; age/fever/chronic-condition/severity factors feed a 0–100+ risk score mapped to Low/Medium/High.

## Project structure

```
AI_Healthcare_Diagnosis_Assistant/
│
├── dataset/
│   ├── symptoms.csv
│   ├── diseases.csv
│   ├── precautions.csv
│   ├── medications.csv
│   ├── doctors.csv
│   └── severity.csv
│
├── model/
│   ├── disease_model.pkl
│   ├── symptom_encoder.pkl
│   └── scaler.pkl
│
├── image/
│   ├── Screenshot 2026-07-17 104914.jpg
│   └── Screenshot 2026-07-17 105243.jpg
│
├── images/
│
├── templates/
│   ├── index.html
│   ├── prediction.html
│   └── history.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── app.py
├── train_model.py
├── predict.py
├── database.py
├── requirements.txt
└── README.md
```

(`dataset/raw/` holds the raw Kaggle CSV locally so `train_model.py` can be re-run from scratch, but at ~190MB it is **not included in the downloadable zip**. The six processed CSVs above and the trained model *are* included, so the app runs immediately without it — you only need the raw file again if you want to retrain. Re-download `Final_Augmented_dataset_Diseases_and_Symptoms.csv` from Kaggle's "Diseases and Symptoms" dataset and drop it in `dataset/raw/` to retrain. `model/metrics.json` and `model/classification_report.json` hold the full evaluation from the model already trained and shipped here.)

## Quick start

```bash
pip install -r requirements.txt

python train_model.py   # cleans + curates the dataset, trains, evaluates, saves the model
python app.py            # launch web app -> http://127.0.0.1:5000
```

`predict.py` can also run standalone (`python predict.py`) to test the pipeline on a sample patient and print the JSON result.

## Web application

- **/** — patient intake form with a filterable symptom chip picker (135 symptoms)
- **/predict** — results page: Top-3 conditions with probability bars, risk gauge with contributing factors, emergency banner when triggered, specialist + medicine guidance + precautions per condition
- **/history** — stats (total checks, high-risk count, most common results) and a table of all past predictions
- **/report/\<id\>** — downloadable plain-text health report

## Ethical & safety design

- All output framed as "most likely conditions", never definitive diagnosis
- Persistent disclaimer banner on every page and in every report
- Medicine guidance is informational only, with explicit "consult a doctor" wording — no fabricated specific drug names for diseases we don't have a real source for
- Emergency rules override the model when critical symptom combinations appear
- Patient data stays in a local SQLite database

## Note on the earlier `kagglehub` request

If you're wondering why the project doesn't call Kaggle's API directly: `kagglehub.dataset_download(...)` requires an authenticated Kaggle account (an API token), which this environment can't hold or supply — the call fails with a 403 regardless of network access. Uploading the CSV directly (as was done here) sidesteps that entirely and is the simplest path if you want to swap in a different Kaggle dataset later.

## Deploying on Railway

The project ships ready for this — `Procfile`, `gunicorn` in `requirements.txt`, and `$PORT` binding are already in place.

1. **Push this project to a GitHub repo** (skip if it's already there).
2. **Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo** and pick the repo.
3. Railway auto-detects Python via Nixpacks and installs `requirements.txt`. Nothing to configure — it reads the `Procfile` (`web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`) as the start command automatically.
4. Once the build finishes, open **Settings → Networking → Generate Domain** to get a public `*.up.railway.app` URL.
5. **Persistence caveat:** Railway's filesystem is ephemeral — every redeploy wipes `health_assistant.db`, so prediction history won't survive a redeploy by default. To fix that:
   - Add a **Volume** in the Railway service (Settings → Volumes), mount it at `/data`.
   - Add an environment variable `DATABASE_PATH=/data/health_assistant.db` (Settings → Variables).
   - `database.py` already reads this variable, so no code changes needed.
6. The trained model (`model/*.pkl`) and processed dataset (`dataset/*.csv`) are already in the repo, so the app works immediately — no need to run `train_model.py` on Railway (the 190MB raw CSV isn't in the repo, so retraining there won't work without uploading it separately).

If the deploy log shows the build succeeding but the app never comes up, the two most common causes are a missing `gunicorn` in `requirements.txt` or the app binding to a hardcoded port instead of `$PORT` — both are already handled here, but worth checking first if you fork/modify this.

## Extending (optional features from the original sheet)

PDF reports (swap the text report for `reportlab`), speech-to-text symptom input (Web Speech API in `main.js`), an LLM chatbot for general health questions, multi-language UI, and retraining on the full 773-disease label space (set `TOP_N_DISEASES = 773` in `train_model.py`, expect ~75-85% accuracy per the rationale above).
