import joblib
from feature_pipeline import transform_input

model = None


def load_model():
    global model
    if model is None:
        model = joblib.load("model.pkl")
    return model


def predict_and_explain(input_data: dict):
    try:
        model = load_model()

        features = transform_input(input_data)

        prob = model.predict_proba(features)[0][1]

        print("PREDICTION:", prob)

        return {
            "status": "success",
            "attrition_probability": float(prob),
            "top_factors": []
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "status": "error",
            "message": str(e)
        }