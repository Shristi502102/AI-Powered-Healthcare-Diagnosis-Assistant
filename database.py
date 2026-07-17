"""
database.py
-----------
SQLite persistence layer for the AI Healthcare Diagnosis Assistant.

Tables:
    patients     - patient details per submission
    predictions  - prediction results linked to a patient
"""

import os
import json
import sqlite3
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DATABASE_PATH", os.path.join(BASE, "health_assistant.db"))
# On Railway (or any container host with an ephemeral filesystem), the SQLite
# file is wiped on every redeploy/restart unless it lives on a mounted Volume.
# Add a Volume in Railway, mount it at e.g. /data, and set the env var
# DATABASE_PATH=/data/health_assistant.db to persist history across deploys.


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            weight REAL,
            height REAL,
            bmi REAL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL REFERENCES patients(id),
            symptoms TEXT NOT NULL,           -- JSON list
            temperature REAL,
            duration_days INTEGER,
            existing_diseases TEXT,           -- JSON list
            top_disease TEXT NOT NULL,
            probability REAL NOT NULL,
            all_predictions TEXT NOT NULL,    -- JSON list of top-N
            risk_level TEXT NOT NULL,
            risk_score INTEGER NOT NULL,
            specialist TEXT,
            emergency INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """)


def save_prediction(result):
    """Persist a prediction result dict (from predict.predict())."""
    p = result["patient"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top = result["predictions"][0]
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO patients (name, age, gender, weight, height, bmi, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (p["name"], p["age"], p["gender"], p["weight"], p["height"],
             result["risk"].get("bmi"), now))
        patient_id = cur.lastrowid
        conn.execute(
            """INSERT INTO predictions
               (patient_id, symptoms, temperature, duration_days, existing_diseases,
                top_disease, probability, all_predictions, risk_level, risk_score,
                specialist, emergency, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (patient_id, json.dumps(p["symptoms"]), p["temperature"],
             p["duration_days"], json.dumps(p["existing_diseases"]),
             top["disease"], top["probability"],
             json.dumps([{"disease": x["disease"], "probability": x["probability"]}
                         for x in result["predictions"]]),
             result["risk"]["level"], result["risk"]["score"],
             top["specialist"], 1 if result["emergencies"] else 0, now))
        return patient_id


def get_history(limit=50):
    """Recent prediction history joined with patient details."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT pr.id, pa.name, pa.age, pa.gender, pr.symptoms,
                      pr.top_disease, pr.probability, pr.all_predictions,
                      pr.risk_level, pr.risk_score, pr.specialist,
                      pr.emergency, pr.created_at
               FROM predictions pr
               JOIN patients pa ON pa.id = pr.patient_id
               ORDER BY pr.id DESC LIMIT ?""", (limit,)).fetchall()
    history = []
    for r in rows:
        d = dict(r)
        d["symptoms"] = json.loads(d["symptoms"])
        d["all_predictions"] = json.loads(d["all_predictions"])
        history.append(d)
    return history


def get_stats():
    """Simple aggregate stats for the history page."""
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
        high = conn.execute(
            "SELECT COUNT(*) c FROM predictions WHERE risk_level='High'").fetchone()["c"]
        common = conn.execute(
            """SELECT top_disease, COUNT(*) c FROM predictions
               GROUP BY top_disease ORDER BY c DESC LIMIT 5""").fetchall()
    return {"total": total, "high_risk": high,
            "common": [(r["top_disease"], r["c"]) for r in common]}


if __name__ == "__main__":
    init_db()
    print(f"Database initialised at {DB_PATH}")
