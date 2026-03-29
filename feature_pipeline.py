import pandas as pd
import joblib

# Lazy loading (prevents startup crash)
scaler = None
label_encoders = None
feature_names = None


def load_artifacts():
    global scaler, label_encoders, feature_names

    if scaler is None:
        scaler = joblib.load("scaler.pkl")

    if label_encoders is None:
        label_encoders = joblib.load("label_encoders.pkl")

    if feature_names is None:
        feature_names = joblib.load("feature_names.pkl")


def transform_input(input_data: dict):
    load_artifacts()

    df = pd.DataFrame([input_data])

    # 🔹 Apply label encoding
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except:
                df[col] = 0  # fallback for unseen category

    # 🔹 Ensure feature order
    df = df[feature_names]

    # 🔹 Scale
    df_scaled = scaler.transform(df)

    return df_scaled