# ==========================================
# HEART ATTACK PREDICTION - FLASK APP
# ==========================================

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

BASE = os.path.dirname(__file__)
model       = joblib.load(os.path.join(BASE, "models/best_model.pkl"))
scaler      = joblib.load(os.path.join(BASE, "models/scaler.pkl"))
col_medians = joblib.load(os.path.join(BASE, "models/col_medians.pkl"))

# Feature order must match training
FEATURE_ORDER = ['age', 'sex', 'cp', 'trestbps', 'chol',
                 'fbs', 'restecg', 'thalach', 'exang',
                 'oldpeak', 'slope', 'ca', 'thal']

# Required fields (cannot be skipped — model needs these to be useful)
REQUIRED = {'age', 'sex', 'cp', 'thalach', 'exang'}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        row     = {}
        skipped = []

        for col in FEATURE_ORDER:
            val = request.form.get(col, '').strip()
            if val == '':
                row[col] = float(col_medians[col])
                skipped.append(col)
            else:
                row[col] = float(val)

        data        = pd.DataFrame([row])[FEATURE_ORDER]
        data_scaled = scaler.transform(data)
        prediction  = model.predict(data_scaled)[0]
        probability = model.predict_proba(data_scaled)[0]

        risk_pct = round(probability[1] * 100, 1)

        if risk_pct < 30:
            level = 'low'
        elif risk_pct < 60:
            level = 'medium'
        else:
            level = 'high'

        result_text = "High Risk of Heart Attack" if prediction == 1 else "Low Risk of Heart Attack"

        return render_template("result.html",
                               prediction=result_text,
                               risk_pct=risk_pct,
                               level=level,
                               skipped=skipped)

    except Exception as e:
        return render_template("result.html",
                               prediction=f"Error: {str(e)}",
                               risk_pct=None,
                               level='error',
                               skipped=[])

if __name__ == '__main__':
    app.run(debug=True)
