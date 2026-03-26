import pandas as pd
import joblib

from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from database import SessionLocal
from models import Employee, Prediction
from feature_pipeline import apply_feature_engineering
from model_registry import save_model, set_active_model, get_active_metrics


def fetch_training_data():
    db: Session = SessionLocal()

    employees = db.query(Employee).all()
    predictions = db.query(Prediction).all()

    df_emp = pd.DataFrame([e.__dict__ for e in employees])
    df_pred = pd.DataFrame([p.__dict__ for p in predictions])

    if df_emp.empty or df_pred.empty:
        return pd.DataFrame()

    df = df_emp.merge(df_pred, left_on="id", right_on="employee_id")

    if "risk_probability" not in df.columns:
        return pd.DataFrame()

    df["target"] = (df["risk_probability"] > 0.5).astype(int)

    return df


def preprocess(df):
    try:
        df = apply_feature_engineering(df)
    except Exception:
        print("⚠️ Feature engineering skipped due to missing columns")

    if "target" not in df.columns:
        return None, None

    y = df["target"]
    X = df.drop(columns=["target"], errors="ignore")

    if X.shape[1] < 3:
        print("❌ Not enough features")
        return None, None

    return X, y


def train_model(X, y):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)
    return model


def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, probs)


def retrain_pipeline():
    print("🚀 Starting retraining pipeline...")

    df = fetch_training_data()

    if df.empty:
        print("❌ No data available")
        return

    X, y = preprocess(df)

    if X is None:
        print("❌ Skipping retraining")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    new_auc = evaluate(model, X_test, y_test)
    print(f"New Model AUC: {new_auc}")

    old_metrics = get_active_metrics()
    old_auc = old_metrics["auc"] if old_metrics else 0

    if new_auc > old_auc:
        print("✅ New model better → promoting")

        version = save_model(
            model=model,
            scaler=joblib.load("scaler.pkl"),
            feature_names=joblib.load("feature_names.pkl"),
            encoders=joblib.load("label_encoders.pkl"),
            metrics={"auc": float(new_auc)}
        )

        set_active_model(version)
        print(f"🔥 Model promoted: {version}")
    else:
        print("❌ Model rejected")