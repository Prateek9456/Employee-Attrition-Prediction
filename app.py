from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import shap

from database import SessionLocal
from models import Employee, Prediction
from schemas import EmployeeInput

from feature_pipeline import apply_feature_engineering
from model_registry import load_active_model
from retrain import retrain_pipeline

from drift import compute_drift_score

app = FastAPI()

# ---------------- CORS ---------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODEL ---------------- #
bundle = load_active_model()

model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]
label_encoders = bundle["encoders"]

explainer = shap.TreeExplainer(model)


# ---------------- ROOT ---------------- #
@app.get("/")
def home():
    return {"message": "Employee Attrition API Running"}


# ---------------- PREDICT ---------------- #
@app.post("/predict")
def predict(data: EmployeeInput):

    db = SessionLocal()

    df = pd.DataFrame([data.dict()])
    df = apply_feature_engineering(df)

    # Safe encoding
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except Exception:
                df[col] = 0

    df = df[feature_names]
    X = scaler.transform(df)

    prob = float(model.predict_proba(X)[0][1])

    # Save employee
    employee = Employee(
        age=data.Age,
        department=data.Department,
        job_role=data.JobRole,
        monthly_income=data.MonthlyIncome,
        years_at_company=data.YearsAtCompany
    )

    db.add(employee)
    db.commit()
    db.refresh(employee)

    # Save prediction
    prediction = Prediction(
        employee_id=employee.id,
        risk_probability=prob,
        risk_level="High" if prob > 0.5 else "Low",
        model_version="active"
    )

    db.add(prediction)
    db.commit()

    # ---------------- SHAP (NO DB STORAGE) ---------------- #
    shap_vals = explainer.shap_values(X)

    if isinstance(shap_vals, list):
        shap_values = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    else:
        shap_values = shap_vals[0]

    feature_impact = dict(zip(feature_names, shap_values))

    top_features = sorted(
        feature_impact.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    return {
        "attrition_probability": prob,
        "top_factors": [
            {"feature": f, "impact": float(v)} for f, v in top_features
        ]
    }


# ---------------- DRIFT ALERT ---------------- #
@app.get("/governance/drift/alerts")
def drift_alerts():

    drift_score = compute_drift_score()

    DRIFT_THRESHOLD = 0.6

    if drift_score > DRIFT_THRESHOLD:
        retrain_pipeline()

    return {
        "drift_score": drift_score,
        "status": "unstable" if drift_score > DRIFT_THRESHOLD else "healthy"
    }