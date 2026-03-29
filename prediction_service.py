# prediction_service.py

def predict_and_explain(data: dict):
    """
    Dummy / existing model wrapper
    Replace model logic if needed
    """

    # 🔥 YOUR MODEL CALL HERE
    # Example:
    probability = 0.65  # <-- replace with model.predict_proba()

    return {
        "status": "success",
        "attrition_probability": probability,
        "top_factors": []
    }