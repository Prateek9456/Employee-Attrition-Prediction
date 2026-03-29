# app.py

from fastapi import FastAPI
from prediction_service import predict_and_explain
from llm_helper import generate_chat_response

app = FastAPI()


@app.get("/")
def home():
    return {"message": "API Running"}


# -------------------------
# PREDICT
# -------------------------
@app.post("/predict")
def predict(data: dict):
    return predict_and_explain(data)


# -------------------------
# CHAT
# -------------------------
@app.post("/chat")
def chat_endpoint(request: dict):
    try:
        messages = request.get("messages", [])
        employee_data = request.get("employee_data", None)
        probability = request.get("probability", None)

        reply = generate_chat_response(messages, employee_data, probability)

        return {
            "status": "success",
            "reply": reply
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}