import joblib
from feature_pipeline import transform_input

# Load model
model = joblib.load("model.pkl")


def predict_and_explain(input_data: dict):
    try:
        # 🔹 Transform input
        features = transform_input(input_data)

        print("TRANSFORMED FEATURES:", features)

        # 🔹 Predict probability
        prob = model.predict_proba(features)[0][1]

        print("PREDICTION:", prob)

        return {
            "status": "success",
            "attrition_probability": float(prob),
            "top_factors": []
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }