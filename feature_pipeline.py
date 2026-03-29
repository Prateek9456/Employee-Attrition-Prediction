import pandas as pd
import joblib

# Load artifacts
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_names = joblib.load("feature_names.pkl")


def transform_input(input_data: dict):
    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # 🔹 Apply label encoding
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except:
                # Handle unseen categories
                df[col] = 0

    # 🔹 Ensure correct feature order
    df = df[feature_names]

    # 🔹 Apply scaling
    df_scaled = scaler.transform(df)

    return df_scaled