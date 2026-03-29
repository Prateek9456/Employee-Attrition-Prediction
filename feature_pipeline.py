import pandas as pd
import joblib

# Lazy loading
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


# 🔥 FEATURE ENGINEERING
def create_features(df):
    df = df.copy()

    try:
        df["IncomeToAgeRatio"] = df["MonthlyIncome"] / (df["Age"] + 1)
        df["ExperienceToPromotionRatio"] = df["TotalWorkingYears"] / (df["YearsSinceLastPromotion"] + 1)
        df["SatisfactionWorkloadRatio"] = df["JobSatisfaction"] / (df["WorkLifeBalance"] + 1)

        df["PromotionGap"] = df["YearsSinceLastPromotion"] - df["YearsInCurrentRole"]
        df["CareerStagnation"] = df["YearsInCurrentRole"] / (df["TotalWorkingYears"] + 1)

    except Exception as e:
        print("Feature Engineering Error:", e)

    return df


def transform_input(input_data: dict):
    load_artifacts()

    df = pd.DataFrame([input_data])

    print("RAW INPUT:", df)

    # ✅ STEP 1: Feature Engineering
    df = create_features(df)

    print("AFTER FEATURE ENGINEERING:", df.columns.tolist())

    # ✅ STEP 2: Encoding
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except:
                df[col] = 0

    # ✅ STEP 3: HANDLE MISSING FEATURES (CRITICAL FIX)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # ✅ STEP 4: ORDER FEATURES
    df = df[feature_names]

    # ✅ STEP 5: SCALING
    df_scaled = scaler.transform(df)

    print("FINAL FEATURES SHAPE:", df_scaled.shape)

    return df_scaled