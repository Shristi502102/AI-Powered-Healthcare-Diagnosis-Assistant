"""
predict.py
----------
Prediction pipeline: input validation -> encoding -> model inference ->
Top-N disease probabilities -> risk engine -> emergency detection ->
recommendation engine (specialist, medicines, precautions).

Can also be run standalone:  python predict.py
"""

import os
import joblib
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
DATASET = os.path.join(BASE, "dataset")

DISCLAIMER = ("This tool is for educational and informational purposes only. "
              "It does not provide a medical diagnosis. Always consult a "
              "qualified healthcare professional.")

# ---------------------------------------------------------------------------
# Load artifacts once at import time
# ---------------------------------------------------------------------------
_model = joblib.load(os.path.join(MODEL_DIR, "disease_model.pkl"))
_encoder = joblib.load(os.path.join(MODEL_DIR, "symptom_encoder.pkl"))
_schema = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
SYMPTOM_COLS = _schema["symptom_cols"]
FEATURE_COLS = _schema["feature_cols"]
SEVERITY_MAP = _schema["severity_map"]

_precautions = pd.read_csv(os.path.join(DATASET, "precautions.csv")).set_index("disease")
_medications = pd.read_csv(os.path.join(DATASET, "medications.csv")).set_index("disease")
_doctors = pd.read_csv(os.path.join(DATASET, "doctors.csv")).set_index("disease")

# Symptom combinations that indicate a possible emergency
EMERGENCY_RULES = [
    ({"sharp_chest_pain", "difficulty_breathing"},
     "Chest pain with difficulty breathing — possible cardiac/respiratory emergency."),
    ({"sharp_chest_pain", "irregular_heartbeat"},
     "Chest pain with irregular heartbeat — possible cardiac emergency."),
    ({"hemoptysis"}, "Coughing up blood requires urgent medical evaluation."),
    ({"vomiting_blood"}, "Vomiting blood is a medical emergency."),
    ({"seizures"}, "Seizure activity requires immediate medical attention."),
    ({"focal_weakness", "slurring_words"},
     "Sudden weakness with slurred speech — possible stroke. Act F.A.S.T. and seek emergency care."),
    ({"loss_of_sensation", "slurring_words"},
     "Sudden numbness with slurred speech — possible stroke. Seek emergency care immediately."),
    ({"blood_in_stool"}, "Blood in stool requires prompt medical evaluation."),
    ({"rectal_bleeding"}, "Rectal bleeding requires prompt medical evaluation."),
]


# ---------------------------------------------------------------------------
# Validation & encoding
# ---------------------------------------------------------------------------
def validate_input(patient):
    """Validate raw patient input. Returns (clean_dict, errors_list)."""
    errors = []
    clean = {}

    clean["name"] = str(patient.get("name", "Anonymous")).strip()[:80] or "Anonymous"

    try:
        clean["age"] = int(patient.get("age", 0))
        if not 0 < clean["age"] <= 120:
            errors.append("Age must be between 1 and 120.")
    except (TypeError, ValueError):
        errors.append("Age must be a number.")

    clean["gender"] = str(patient.get("gender", "Other")).strip().title()

    try:
        clean["weight"] = float(patient.get("weight", 0) or 0)
        clean["height"] = float(patient.get("height", 0) or 0)  # metres
    except (TypeError, ValueError):
        clean["weight"], clean["height"] = 0.0, 0.0
        errors.append("Weight and height must be numbers.")

    try:
        clean["temperature"] = float(patient.get("temperature", 0) or 0)  # °C
    except (TypeError, ValueError):
        clean["temperature"] = 0.0

    try:
        clean["duration_days"] = int(patient.get("duration_days", 1) or 1)
    except (TypeError, ValueError):
        clean["duration_days"] = 1

    symptoms = patient.get("symptoms", [])
    if isinstance(symptoms, str):
        symptoms = [s.strip() for s in symptoms.split(",") if s.strip()]
    clean["symptoms"] = [s for s in symptoms if s in SYMPTOM_COLS]
    unknown = [s for s in symptoms if s not in SYMPTOM_COLS]
    if unknown:
        errors.append(f"Unknown symptoms ignored: {', '.join(unknown[:5])}")
    if len(clean["symptoms"]) < 1:
        errors.append("Please select at least one valid symptom.")

    existing = patient.get("existing_diseases", [])
    if isinstance(existing, str):
        existing = [e.strip() for e in existing.split(",") if e.strip()]
    clean["existing_diseases"] = [e.lower() for e in existing]

    return clean, errors


def encode(clean):
    """Multi-hot encode symptoms + engineered features -> model input row."""
    vec = {s: (1 if s in clean["symptoms"] else 0) for s in SYMPTOM_COLS}
    vec["symptom_count"] = sum(vec.values())
    vec["severity_score"] = sum(SEVERITY_MAP.get(s, 2) for s in clean["symptoms"])
    return pd.DataFrame([vec])[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Risk engine (rule-based, per project sheet Module 7)
# ---------------------------------------------------------------------------
def compute_risk(clean):
    score, factors = 0, []

    if clean["age"] >= 65:
        score += 20; factors.append("Age 65+ (+20)")
    elif clean["age"] <= 5:
        score += 15; factors.append("Young child (+15)")

    if clean["temperature"] >= 39.5:
        score += 25; factors.append("Very high fever ≥39.5°C (+25)")
    elif clean["temperature"] >= 38.0:
        score += 10; factors.append("Fever ≥38°C (+10)")

    if "difficulty_breathing" in clean["symptoms"]:
        score += 40; factors.append("Breathing difficulty (+40)")
    if any(s in clean["symptoms"] for s in
           ("sharp_chest_pain", "burning_chest_pain", "chest_tightness")):
        score += 25; factors.append("Chest pain (+25)")

    chronic = {"diabetes": 20, "heart disease": 25, "hypertension": 15,
               "asthma": 15, "kidney disease": 20, "cancer": 25}
    for cond, pts in chronic.items():
        if any(cond in e for e in clean["existing_diseases"]):
            score += pts; factors.append(f"Existing {cond} (+{pts})")

    sev = sum(SEVERITY_MAP.get(s, 2) for s in clean["symptoms"])
    if sev >= 15:
        score += 15; factors.append(f"High symptom severity ({sev}) (+15)")

    if clean["duration_days"] >= 7:
        score += 10; factors.append("Symptoms ≥7 days (+10)")

    # BMI factor
    bmi = None
    if clean["weight"] > 0 and clean["height"] > 0:
        bmi = round(clean["weight"] / (clean["height"] ** 2), 1)
        if bmi >= 35 or bmi < 16:
            score += 10; factors.append(f"BMI {bmi} outside healthy range (+10)")

    level = "Low" if score <= 30 else "Medium" if score <= 60 else "High"
    return {"score": score, "level": level, "factors": factors, "bmi": bmi}


def check_emergency(symptoms):
    hits = [msg for combo, msg in EMERGENCY_RULES if combo.issubset(set(symptoms))]
    return hits


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------
def recommendations_for(disease):
    rec = {"specialist": "General Physician", "medicine": "Consult a doctor",
           "precautions": []}
    if disease in _doctors.index:
        rec["specialist"] = _doctors.loc[disease, "specialist"]
    if disease in _medications.index:
        rec["medicine"] = _medications.loc[disease, "medicine"]
    if disease in _precautions.index:
        row = _precautions.loc[disease]
        rec["precautions"] = [str(v) for v in row.values if pd.notna(v)]
    return rec


# ---------------------------------------------------------------------------
# Main prediction entry point
# ---------------------------------------------------------------------------
def predict(patient, top_n=3):
    """Full pipeline. Returns a result dict or {'errors': [...]}."""
    clean, errors = validate_input(patient)
    if any("symptom" in e.lower() and "ignored" not in e.lower() for e in errors) \
            or "Age must be a number." in errors:
        return {"errors": errors}

    X = encode(clean)
    proba = _model.predict_proba(X)[0]
    order = proba.argsort()[::-1][:top_n]

    predictions = []
    for idx in order:
        disease = _encoder.inverse_transform([idx])[0]
        predictions.append({
            "disease": disease,
            "probability": round(float(proba[idx]) * 100, 1),
            **recommendations_for(disease),
        })

    risk = compute_risk(clean)
    emergencies = check_emergency(clean["symptoms"])
    if emergencies:
        risk["level"], risk["score"] = "High", max(risk["score"], 80)

    return {
        "patient": clean,
        "predictions": predictions,
        "risk": risk,
        "emergencies": emergencies,
        "warnings": errors,          # non-fatal warnings (e.g. ignored symptoms)
        "disclaimer": DISCLAIMER,
    }


def get_all_symptoms():
    """List of all model symptoms (for UI dropdowns)."""
    return SYMPTOM_COLS


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = {
        "name": "Test Patient", "age": 24, "gender": "Male",
        "weight": 72, "height": 1.75, "temperature": 38.6,
        "symptoms": ["anxiety_and_nervousness", "shortness_of_breath", "dizziness",
                     "chest_tightness", "palpitations"],
        "duration_days": 2, "existing_diseases": [],
    }
    import json
    print(json.dumps(predict(sample), indent=2))
