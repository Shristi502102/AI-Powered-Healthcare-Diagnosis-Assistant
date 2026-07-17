"""
app.py
------
Flask web application for the AI Healthcare Diagnosis Assistant.

Routes:
    GET  /            - symptom input form
    POST /predict     - run prediction, show results
    GET  /history     - prediction history + stats
    GET  /report/<id> - downloadable plain-text report

Run:  python app.py   ->  http://127.0.0.1:5000
"""

import io
import json
import os
from datetime import datetime
from flask import Flask, render_template, request, send_file, redirect, url_for

os.environ.setdefault("FLASK_SKIP_DOTENV", "1")

import predict as engine
import database

app = Flask(__name__)
database.init_db()


def pretty(symptom):
    return symptom.replace("_", " ").title()


app.jinja_env.filters["pretty"] = pretty


@app.route("/")
def index():
    symptoms = [{"key": s, "label": pretty(s)} for s in engine.get_all_symptoms()]
    return render_template("index.html", symptoms=symptoms)


@app.route("/predict", methods=["POST"])
def do_predict():
    patient = {
        "name": request.form.get("name", ""),
        "age": request.form.get("age", ""),
        "gender": request.form.get("gender", ""),
        "weight": request.form.get("weight", 0),
        "height": request.form.get("height", 0),
        "temperature": request.form.get("temperature", 0),
        "duration_days": request.form.get("duration_days", 1),
        "symptoms": request.form.getlist("symptoms"),
        "existing_diseases": request.form.get("existing_diseases", ""),
    }
    result = engine.predict(patient, top_n=3)

    if "errors" in result:
        symptoms = [{"key": s, "label": pretty(s)} for s in engine.get_all_symptoms()]
        return render_template("index.html", symptoms=symptoms,
                               errors=result["errors"], form=patient)

    record_id = database.save_prediction(result)
    return render_template("prediction.html", r=result, record_id=record_id,
                           now=datetime.now().strftime("%d %b %Y, %H:%M"))


@app.route("/history")
def history():
    return render_template("history.html",
                           history=database.get_history(),
                           stats=database.get_stats())


@app.route("/report/<int:record_id>")
def report(record_id):
    for rec in database.get_history(limit=1000):
        if rec["id"] == record_id:
            lines = [
                "=" * 58,
                "     AI HEALTHCARE DIAGNOSIS ASSISTANT - HEALTH REPORT",
                "=" * 58,
                f"Report ID   : {rec['id']}",
                f"Date        : {rec['created_at']}",
                f"Patient     : {rec['name']}  ({rec['age']} yrs, {rec['gender']})",
                f"Symptoms    : {', '.join(pretty(s) for s in rec['symptoms'])}",
                "-" * 58,
                "LIKELY CONDITIONS (not a diagnosis):",
            ]
            for p in rec["all_predictions"]:
                lines.append(f"   - {p['disease']:<28} {p['probability']}%")
            lines += [
                "-" * 58,
                f"Risk level  : {rec['risk_level']} (score {rec['risk_score']})",
                f"Specialist  : {rec['specialist']}",
                f"Emergency   : {'YES - seek immediate care' if rec['emergency'] else 'No'}",
                "-" * 58,
                "DISCLAIMER: This report is for educational and informational",
                "purposes only and is NOT a medical diagnosis. Always consult",
                "a qualified healthcare professional.",
                "=" * 58,
            ]
            buf = io.BytesIO("\n".join(lines).encode())
            return send_file(buf, as_attachment=True,
                             download_name=f"health_report_{record_id}.txt",
                             mimetype="text/plain")
    return redirect(url_for("history"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    app.run(debug=debug, host="0.0.0.0", port=port)
